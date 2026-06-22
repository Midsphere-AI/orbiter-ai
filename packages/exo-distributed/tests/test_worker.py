"""Tests for Worker task execution lifecycle."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exo.distributed.cancel import CancellationToken  # pyright: ignore[reportMissingImports]
from exo.distributed.models import (  # pyright: ignore[reportMissingImports]
    TaskPayload,
    TaskResult,
    TaskStatus,
)
from exo.distributed.worker import (  # pyright: ignore[reportMissingImports]
    Worker,
    _deserialize_messages,
    _generate_worker_id,
)
from exo.types import ExoError  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# _generate_worker_id
# ---------------------------------------------------------------------------


class TestGenerateWorkerId:
    def test_format(self) -> None:
        wid = _generate_worker_id()
        parts = wid.split("-")
        # hostname-pid-suffix  (hostname may contain dashes, so at least 3 parts)
        assert len(parts) >= 3

    def test_unique(self) -> None:
        ids = {_generate_worker_id() for _ in range(50)}
        assert len(ids) == 50


# ---------------------------------------------------------------------------
# Worker.__init__
# ---------------------------------------------------------------------------


class TestWorkerInit:
    def test_defaults(self) -> None:
        w = Worker("redis://localhost:6379")
        assert w._redis_url == "redis://localhost:6379"
        assert w._concurrency == 1
        assert w._queue_name == "exo:tasks"
        assert w._heartbeat_ttl == 30
        assert w._worker_id  # auto-generated
        assert w.tasks_processed == 0
        assert w.tasks_failed == 0

    def test_custom_params(self) -> None:
        w = Worker(
            "redis://host:1234",
            worker_id="my-worker",
            concurrency=4,
            queue_name="custom:q",
            heartbeat_ttl=60,
        )
        assert w.worker_id == "my-worker"
        assert w._concurrency == 4
        assert w._queue_name == "custom:q"
        assert w._heartbeat_ttl == 60

    def test_auto_worker_id(self) -> None:
        w1 = Worker("redis://localhost")
        w2 = Worker("redis://localhost")
        assert w1.worker_id != w2.worker_id


# ---------------------------------------------------------------------------
# Worker._reconstruct_agent
# ---------------------------------------------------------------------------


class TestWorkerReconstructAgent:
    def test_agent_config(self) -> None:
        w = Worker("redis://localhost")
        config = {"name": "test-agent", "model": "openai:gpt-4o"}
        with patch("exo.agent.Agent") as mock_agent_cls:
            mock_agent_cls.from_dict.return_value = "mock-agent"
            result = w._reconstruct_agent(config)
            mock_agent_cls.from_dict.assert_called_once_with(config)
            assert result == "mock-agent"

    def test_swarm_config(self) -> None:
        w = Worker("redis://localhost")
        config = {
            "agents": [{"name": "a1", "model": "openai:gpt-4o"}],
            "mode": "workflow",
        }
        with patch("exo.swarm.Swarm") as mock_swarm_cls:
            mock_swarm_cls.from_dict.return_value = "mock-swarm"
            result = w._reconstruct_agent(config)
            mock_swarm_cls.from_dict.assert_called_once_with(config)
            assert result == "mock-swarm"


# ---------------------------------------------------------------------------
# Worker._execute_task
# ---------------------------------------------------------------------------


class TestWorkerExecuteTask:
    @pytest.mark.asyncio
    async def test_successful_execution(self) -> None:
        w = Worker("redis://localhost", worker_id="w1")
        w._broker = AsyncMock()
        w._store = AsyncMock()
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-1",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="hello",
            detailed=False,
        )

        async def _fake_run_agent(agent: object, t: TaskPayload, token: CancellationToken) -> str:
            return "Hello!"

        # Mock the agent reconstruction and streaming
        mock_agent = MagicMock()
        with (
            patch.object(w, "_reconstruct_agent", return_value=mock_agent),
            patch.object(w, "_run_agent", side_effect=_fake_run_agent),
            patch.object(w, "_listen_for_cancel", new_callable=AsyncMock),
        ):
            await w._execute_task(task)

        # Verify status transitions
        calls = w._store.set_status.call_args_list
        assert calls[0].args[1] == TaskStatus.RUNNING
        assert calls[1].args[1] == TaskStatus.COMPLETED
        assert calls[1].kwargs["result"] == {"output": "Hello!"}

        # Verify ack
        w._broker.ack.assert_called_once_with("task-1")
        assert w.tasks_processed == 1
        assert w.tasks_failed == 0

    @pytest.mark.asyncio
    async def test_failed_execution_with_retries(self) -> None:
        w = Worker("redis://localhost", worker_id="w1")
        w._broker = AsyncMock()
        w._broker.max_retries = 3
        w._store = AsyncMock()
        w._store.get_status.return_value = TaskResult(
            task_id="task-2", status=TaskStatus.RUNNING, retries=0
        )
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-2",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="fail",
        )

        with (
            patch.object(w, "_reconstruct_agent", side_effect=ValueError("bad config")),
            patch.object(w, "_listen_for_cancel", new_callable=AsyncMock),
        ):
            await w._execute_task(task)

        # Should set FAILED then RETRYING
        status_calls = w._store.set_status.call_args_list
        statuses = [c.args[1] for c in status_calls]
        assert TaskStatus.RUNNING in statuses
        assert TaskStatus.FAILED in statuses
        assert TaskStatus.RETRYING in statuses

        # Should nack (re-queue) since retries < max
        w._broker.nack.assert_called_once_with("task-2")
        assert w.tasks_failed == 1

    @pytest.mark.asyncio
    async def test_failed_execution_max_retries_exhausted(self) -> None:
        w = Worker("redis://localhost", worker_id="w1")
        w._broker = AsyncMock()
        w._broker.max_retries = 3
        w._store = AsyncMock()
        w._store.get_status.return_value = TaskResult(
            task_id="task-3", status=TaskStatus.RUNNING, retries=3
        )
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-3",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="fail",
        )

        with (
            patch.object(w, "_reconstruct_agent", side_effect=RuntimeError("crash")),
            patch.object(w, "_listen_for_cancel", new_callable=AsyncMock),
        ):
            await w._execute_task(task)

        # Should ack (not nack) since retries exhausted
        w._broker.ack.assert_called_once_with("task-3")
        w._broker.nack.assert_not_called()
        assert w.tasks_failed == 1

    @pytest.mark.asyncio
    async def test_failed_execution_store_read_failure_acks_not_nacks(self) -> None:
        """When get_status() fails, treat as retries-exhausted and ack (not nack).

        A broken store must not trigger infinite nack/retry storm.
        """
        w = Worker("redis://localhost", worker_id="w1")
        w._broker = AsyncMock()
        w._broker.max_retries = 3
        w._store = AsyncMock()
        # Simulate store read failure
        w._store.get_status.side_effect = RuntimeError("Redis connection lost")
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-store-fail",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="fail",
        )

        with (
            patch.object(w, "_reconstruct_agent", side_effect=RuntimeError("crash")),
            patch.object(w, "_listen_for_cancel", new_callable=AsyncMock),
        ):
            await w._execute_task(task)

        # Must ack (not nack) so the message doesn't loop forever
        w._broker.ack.assert_called_once_with("task-store-fail")
        w._broker.nack.assert_not_called()

    @pytest.mark.asyncio
    async def test_current_task_id_tracked(self) -> None:
        w = Worker("redis://localhost", worker_id="w1")
        w._broker = AsyncMock()
        w._store = AsyncMock()
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-4",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="hello",
        )

        captured_active: set[str] = set()

        async def capture_run(agent: object, t: TaskPayload, token: CancellationToken) -> str:
            nonlocal captured_active
            captured_active = set(w._active_task_ids)
            return "done"

        with (
            patch.object(w, "_reconstruct_agent", return_value=MagicMock()),
            patch.object(w, "_run_agent", side_effect=capture_run),
            patch.object(w, "_listen_for_cancel", new_callable=AsyncMock),
        ):
            await w._execute_task(task)

        assert "task-4" in captured_active
        assert "task-4" not in w._active_task_ids  # cleared after execution


# ---------------------------------------------------------------------------
# Worker._run_agent
# ---------------------------------------------------------------------------


class TestWorkerRunAgent:
    @pytest.mark.asyncio
    async def test_streams_and_publishes_events(self) -> None:
        w = Worker("redis://localhost", worker_id="w1")
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-5",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="hello",
            detailed=True,
        )

        from exo.types import StatusEvent, TextEvent  # pyright: ignore[reportMissingImports]

        events = [
            TextEvent(text="Hello", agent_name="agent"),
            TextEvent(text=" world", agent_name="agent"),
            StatusEvent(status="completed", agent_name="agent", message="done"),
        ]

        mock_agent = MagicMock()

        # Patch exo.runner.run (the actual module where run is defined)
        # so the local import inside _run_agent picks up our mock
        mock_run = MagicMock()

        async def _fake_stream_gen(*a: object, **kw: object) -> object:
            for ev in events:
                yield ev

        mock_run.stream = _fake_stream_gen

        token = CancellationToken()
        with patch("exo.runner.run", mock_run):
            result = await w._run_agent(mock_agent, task, token)

        assert result == "Hello world"
        assert w._publisher.publish.call_count == 3

    @pytest.mark.asyncio
    async def test_collects_only_text_events(self) -> None:
        w = Worker("redis://localhost", worker_id="w1")
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-6",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="hello",
        )

        from exo.types import (  # pyright: ignore[reportMissingImports]
            StatusEvent,
            TextEvent,
            ToolCallEvent,
        )

        events = [
            ToolCallEvent(tool_name="search", tool_call_id="tc-1", agent_name="agent"),
            TextEvent(text="result", agent_name="agent"),
            StatusEvent(status="completed", agent_name="agent", message="done"),
        ]

        mock_agent = MagicMock()

        mock_run = MagicMock()

        async def _fake_stream_gen(*a: object, **kw: object) -> object:
            for ev in events:
                yield ev

        mock_run.stream = _fake_stream_gen

        token = CancellationToken()
        with patch("exo.runner.run", mock_run):
            result = await w._run_agent(mock_agent, task, token)

        # Only TextEvent text should be collected
        assert result == "result"

    @pytest.mark.asyncio
    async def test_terminal_status_event_published_when_not_detailed(self) -> None:
        """When detailed=False, _run_agent must publish a terminal StatusEvent.

        EventSubscriber.subscribe() exits only on a terminal StatusEvent.
        When detailed=False, run.stream() never emits one, so the worker must
        publish it itself to avoid leaving the subscriber hung forever.
        """
        w = Worker("redis://localhost", worker_id="w1")
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-terminal-nondtailed",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="hello",
            detailed=False,  # <-- key: no StatusEvent from run.stream()
        )

        from exo.types import StatusEvent, TextEvent  # pyright: ignore[reportMissingImports]

        # Simulate a stream that emits only TextEvents (what detailed=False produces)
        events = [
            TextEvent(text="Hello", agent_name="agent"),
        ]

        mock_agent = MagicMock()
        mock_agent.name = "agent"

        mock_run = MagicMock()

        async def _fake_stream_gen(*a: object, **kw: object) -> object:
            for ev in events:
                yield ev

        mock_run.stream = _fake_stream_gen

        token = CancellationToken()
        with patch("exo.runner.run", mock_run):
            result = await w._run_agent(mock_agent, task, token)

        assert result == "Hello"

        # One TextEvent published + one terminal StatusEvent appended by worker
        assert w._publisher.publish.call_count == 2
        _task_id, last_event = w._publisher.publish.call_args_list[-1].args
        assert isinstance(last_event, StatusEvent)
        assert last_event.status == "completed"

    @pytest.mark.asyncio
    async def test_no_duplicate_terminal_status_when_detailed(self) -> None:
        """When detailed=True, run.stream() emits StatusEvent; worker must NOT double-publish."""
        w = Worker("redis://localhost", worker_id="w1")
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-terminal-detailed",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="hello",
            detailed=True,
        )

        from exo.types import StatusEvent, TextEvent  # pyright: ignore[reportMissingImports]

        # Simulate what run.stream() produces with detailed=True
        events = [
            TextEvent(text="Hi", agent_name="agent"),
            StatusEvent(status="completed", agent_name="agent", message="done"),
        ]

        mock_agent = MagicMock()
        mock_agent.name = "agent"

        mock_run = MagicMock()

        async def _fake_stream_gen(*a: object, **kw: object) -> object:
            for ev in events:
                yield ev

        mock_run.stream = _fake_stream_gen

        token = CancellationToken()
        with patch("exo.runner.run", mock_run):
            result = await w._run_agent(mock_agent, task, token)

        assert result == "Hi"
        # Worker should publish exactly the events from run.stream() — no extra one
        assert w._publisher.publish.call_count == 2
        _task_id, last_event = w._publisher.publish.call_args_list[-1].args
        assert isinstance(last_event, StatusEvent)
        assert last_event.status == "completed"


# ---------------------------------------------------------------------------
# Worker._claim_loop
# ---------------------------------------------------------------------------


class TestWorkerClaimLoop:
    @pytest.mark.asyncio
    async def test_loops_until_shutdown(self) -> None:
        w = Worker("redis://localhost", worker_id="w1")
        w._broker = AsyncMock()

        call_count = 0

        async def fake_claim(worker_id: str, *, timeout: float = 2.0) -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                w._shutdown_event.set()
            return None

        w._broker.claim = fake_claim  # type: ignore[assignment]
        await w._claim_loop()
        assert call_count >= 3

    @pytest.mark.asyncio
    async def test_executes_claimed_task(self) -> None:
        w = Worker("redis://localhost", worker_id="w1")
        w._broker = AsyncMock()

        task = TaskPayload(
            task_id="task-7",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="hello",
        )

        calls = 0

        async def fake_claim(worker_id: str, *, timeout: float = 2.0) -> TaskPayload | None:
            nonlocal calls
            calls += 1
            if calls == 1:
                return task
            w._shutdown_event.set()
            return None

        w._broker.claim = fake_claim  # type: ignore[assignment]

        with patch.object(w, "_execute_task", new_callable=AsyncMock) as mock_exec:
            await w._claim_loop()
            mock_exec.assert_called_once_with(task)


# ---------------------------------------------------------------------------
# Worker.stop
# ---------------------------------------------------------------------------


class TestWorkerStop:
    @pytest.mark.asyncio
    async def test_stop_sets_shutdown_event(self) -> None:
        w = Worker("redis://localhost")
        assert not w._shutdown_event.is_set()
        await w.stop()
        assert w._shutdown_event.is_set()


# ---------------------------------------------------------------------------
# Worker.start integration
# ---------------------------------------------------------------------------


class TestWorkerStart:
    @pytest.mark.asyncio
    async def test_start_connects_and_runs(self) -> None:
        w = Worker("redis://localhost", worker_id="w1")
        w._broker = AsyncMock()
        w._store = AsyncMock()
        w._publisher = AsyncMock()

        # Immediately signal shutdown so start() exits
        w._shutdown_event.set()

        with (
            patch.object(w, "_claim_loop", new_callable=AsyncMock),
            patch.object(w, "_heartbeat_loop", new_callable=AsyncMock),
            patch("asyncio.get_running_loop") as mock_loop,
        ):
            mock_loop.return_value = MagicMock()
            mock_loop.return_value.add_signal_handler = MagicMock()
            await w.start()

        w._broker.connect.assert_called_once()
        w._store.connect.assert_called_once()
        w._publisher.connect.assert_called_once()
        w._broker.disconnect.assert_called_once()
        w._store.disconnect.assert_called_once()
        w._publisher.disconnect.assert_called_once()


# ---------------------------------------------------------------------------
# _deserialize_messages
# ---------------------------------------------------------------------------


class TestDeserializeMessages:
    def test_user_message(self) -> None:
        from exo.types import UserMessage  # pyright: ignore[reportMissingImports]

        result = _deserialize_messages([{"role": "user", "content": "hello"}])
        assert len(result) == 1
        assert isinstance(result[0], UserMessage)
        assert result[0].content == "hello"

    def test_assistant_message(self) -> None:
        from exo.types import AssistantMessage  # pyright: ignore[reportMissingImports]

        result = _deserialize_messages([{"role": "assistant", "content": "hi there"}])
        assert len(result) == 1
        assert isinstance(result[0], AssistantMessage)
        assert result[0].content == "hi there"

    def test_system_message(self) -> None:
        from exo.types import SystemMessage  # pyright: ignore[reportMissingImports]

        result = _deserialize_messages([{"role": "system", "content": "Be helpful"}])
        assert len(result) == 1
        assert isinstance(result[0], SystemMessage)
        assert result[0].content == "Be helpful"

    def test_tool_result(self) -> None:
        from exo.types import ToolResult  # pyright: ignore[reportMissingImports]

        result = _deserialize_messages(
            [
                {
                    "role": "tool",
                    "tool_call_id": "tc-1",
                    "tool_name": "search",
                    "content": "found it",
                }
            ]
        )
        assert len(result) == 1
        assert isinstance(result[0], ToolResult)
        assert result[0].tool_call_id == "tc-1"
        assert result[0].content == "found it"

    def test_mixed_messages(self) -> None:
        from exo.types import (  # pyright: ignore[reportMissingImports]
            AssistantMessage,
            SystemMessage,
            UserMessage,
        )

        raw = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = _deserialize_messages(raw)
        assert len(result) == 3
        assert isinstance(result[0], SystemMessage)
        assert isinstance(result[1], UserMessage)
        assert isinstance(result[2], AssistantMessage)

    def test_unknown_role_raises(self) -> None:
        with pytest.raises(ExoError, match="Unknown message role"):
            _deserialize_messages([{"role": "unknown", "content": "bad"}])

    def test_empty_list(self) -> None:
        result = _deserialize_messages([])
        assert result == []


# ---------------------------------------------------------------------------
# Worker._run_agent passes messages
# ---------------------------------------------------------------------------


class TestWorkerRunAgentMessages:
    @pytest.mark.asyncio
    async def test_passes_deserialized_messages(self) -> None:
        """_run_agent passes deserialized messages from task.messages to run.stream."""
        w = Worker("redis://localhost", worker_id="w1")
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-msg",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="hello",
            messages=[
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "previous question"},
                {"role": "assistant", "content": "previous answer"},
            ],
        )

        from exo.types import TextEvent  # pyright: ignore[reportMissingImports]

        events = [TextEvent(text="response", agent_name="agent")]
        mock_agent = MagicMock()
        mock_run = MagicMock()

        captured_kwargs: dict = {}

        async def _fake_stream_gen(*a: object, **kw: object) -> object:
            captured_kwargs.update(kw)
            for ev in events:
                yield ev

        mock_run.stream = _fake_stream_gen

        from exo.distributed.cancel import (  # pyright: ignore[reportMissingImports]
            CancellationToken,
        )

        token = CancellationToken()
        with patch("exo.runner.run", mock_run):
            result = await w._run_agent(mock_agent, task, token)

        assert result == "response"
        # Verify messages were passed (not None)
        assert captured_kwargs["messages"] is not None
        assert len(captured_kwargs["messages"]) == 3

    @pytest.mark.asyncio
    async def test_empty_messages_passes_none(self) -> None:
        """_run_agent passes None when task.messages is empty."""
        w = Worker("redis://localhost", worker_id="w1")
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-empty",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="hello",
            messages=[],
        )

        from exo.types import TextEvent  # pyright: ignore[reportMissingImports]

        events = [TextEvent(text="ok", agent_name="agent")]
        mock_agent = MagicMock()
        mock_run = MagicMock()

        captured_kwargs: dict = {}

        async def _fake_stream_gen(*a: object, **kw: object) -> object:
            captured_kwargs.update(kw)
            for ev in events:
                yield ev

        mock_run.stream = _fake_stream_gen

        from exo.distributed.cancel import (  # pyright: ignore[reportMissingImports]
            CancellationToken,
        )

        token = CancellationToken()
        with patch("exo.runner.run", mock_run):
            await w._run_agent(mock_agent, task, token)

        assert captured_kwargs["messages"] is None


# ---------------------------------------------------------------------------
# Feature 3: on_task_done
# ---------------------------------------------------------------------------


class TestOnTaskDone:
    @pytest.mark.asyncio
    async def test_completed_status(self) -> None:
        """on_task_done receives COMPLETED status on success."""
        captured: list[tuple[str, TaskStatus, str | None, str | None]] = []

        class MyWorker(Worker):
            async def on_task_done(
                self,
                task: TaskPayload,
                status: TaskStatus,
                result: str | None,
                error: str | None,
            ) -> None:
                captured.append((task.task_id, status, result, error))

        w = MyWorker("redis://localhost", worker_id="w1")
        w._broker = AsyncMock()
        w._store = AsyncMock()
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-done-1",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="hello",
        )

        async def _fake_run(agent: object, t: TaskPayload, token: object) -> str:
            return "result!"

        with (
            patch.object(w, "_reconstruct_agent", return_value=MagicMock()),
            patch.object(w, "_run_agent", side_effect=_fake_run),
            patch.object(w, "_listen_for_cancel", new_callable=AsyncMock),
        ):
            await w._execute_task(task)

        assert len(captured) == 1
        assert captured[0] == ("task-done-1", TaskStatus.COMPLETED, "result!", None)

    @pytest.mark.asyncio
    async def test_failed_status(self) -> None:
        """on_task_done receives FAILED status on error."""
        captured: list[tuple[str, TaskStatus, str | None, str | None]] = []

        class MyWorker(Worker):
            async def on_task_done(
                self,
                task: TaskPayload,
                status: TaskStatus,
                result: str | None,
                error: str | None,
            ) -> None:
                captured.append((task.task_id, status, result, error))

        w = MyWorker("redis://localhost", worker_id="w1")
        w._broker = AsyncMock()
        w._broker.max_retries = 0
        w._store = AsyncMock()
        w._store.get_status.return_value = TaskResult(
            task_id="task-done-2", status=TaskStatus.RUNNING, retries=0
        )
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-done-2",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="fail",
        )

        with (
            patch.object(w, "_reconstruct_agent", side_effect=ValueError("boom")),
            patch.object(w, "_listen_for_cancel", new_callable=AsyncMock),
        ):
            await w._execute_task(task)

        assert len(captured) == 1
        assert captured[0][0] == "task-done-2"
        assert captured[0][1] == TaskStatus.FAILED
        assert captured[0][2] is None
        assert "boom" in captured[0][3]

    @pytest.mark.asyncio
    async def test_exception_in_on_task_done_doesnt_crash(self) -> None:
        """Exception in on_task_done is logged, not raised."""

        class MyWorker(Worker):
            async def on_task_done(self, *args: object, **kwargs: object) -> None:
                raise RuntimeError("callback exploded")

        w = MyWorker("redis://localhost", worker_id="w1")
        w._broker = AsyncMock()
        w._store = AsyncMock()
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-done-3",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="hello",
        )

        async def _fake_run(agent: object, t: TaskPayload, token: object) -> str:
            return "ok"

        with (
            patch.object(w, "_reconstruct_agent", return_value=MagicMock()),
            patch.object(w, "_run_agent", side_effect=_fake_run),
            patch.object(w, "_listen_for_cancel", new_callable=AsyncMock),
        ):
            # Should not raise
            await w._execute_task(task)

        assert w.tasks_processed == 1

    @pytest.mark.asyncio
    async def test_default_is_noop(self) -> None:
        """Default on_task_done does nothing."""
        w = Worker("redis://localhost")
        task = TaskPayload(task_id="x", input="y")
        # Should not raise
        await w.on_task_done(task, TaskStatus.COMPLETED, "ok", None)


# ---------------------------------------------------------------------------
# Feature 5: Worker Provider Factory
# ---------------------------------------------------------------------------


class TestWorkerProviderFactory:
    def test_init_stores_factory(self) -> None:
        def my_factory(model: str) -> str:
            return f"provider-for-{model}"

        w = Worker("redis://localhost", provider_factory=my_factory)
        assert w._provider_factory is my_factory

    def test_init_none_by_default(self) -> None:
        w = Worker("redis://localhost")
        assert w._provider_factory is None

    @pytest.mark.asyncio
    async def test_factory_called_with_model(self) -> None:
        """Provider factory is called with the model string and passed to run.stream."""
        factory_calls: list[str] = []
        mock_provider = MagicMock()

        def my_factory(model: str) -> MagicMock:
            factory_calls.append(model)
            return mock_provider

        w = Worker("redis://localhost", worker_id="w1", provider_factory=my_factory)
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-pf-1",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="hello",
        )

        from exo.types import TextEvent  # pyright: ignore[reportMissingImports]

        events = [TextEvent(text="ok", agent_name="agent")]
        mock_agent = MagicMock()
        mock_agent.model = "openai:gpt-4o"
        mock_agent.memory = None

        mock_run = MagicMock()
        captured_kwargs: dict = {}

        async def _fake_stream_gen(*a: object, **kw: object) -> object:
            captured_kwargs.update(kw)
            for ev in events:
                yield ev

        mock_run.stream = _fake_stream_gen

        token = CancellationToken()
        with patch("exo.runner.run", mock_run):
            await w._run_agent(mock_agent, task, token)

        assert factory_calls == ["openai:gpt-4o"]
        assert captured_kwargs["provider"] is mock_provider

    @pytest.mark.asyncio
    async def test_none_factory_passes_none_provider(self) -> None:
        """When no factory is set, provider=None is passed."""
        w = Worker("redis://localhost", worker_id="w1")
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-pf-2",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="hello",
        )

        from exo.types import TextEvent  # pyright: ignore[reportMissingImports]

        events = [TextEvent(text="ok", agent_name="agent")]
        mock_agent = MagicMock()
        mock_agent.model = "openai:gpt-4o"
        mock_agent.memory = None

        mock_run = MagicMock()
        captured_kwargs: dict = {}

        async def _fake_stream_gen(*a: object, **kw: object) -> object:
            captured_kwargs.update(kw)
            for ev in events:
                yield ev

        mock_run.stream = _fake_stream_gen

        token = CancellationToken()
        with patch("exo.runner.run", mock_run):
            await w._run_agent(mock_agent, task, token)

        assert captured_kwargs["provider"] is None


# ---------------------------------------------------------------------------
# Feature 4: Memory Hydration
# ---------------------------------------------------------------------------


class TestWorkerMemoryHydration:
    @pytest.mark.asyncio
    async def test_no_memory_config_no_store(self) -> None:
        """Without memory config, no store is created."""
        w = Worker("redis://localhost", worker_id="w1")
        w._broker = AsyncMock()
        w._store = AsyncMock()
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-mem-1",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="hello",
            metadata={},
        )

        mock_agent = MagicMock()
        # Verify memory is not set
        mock_agent.memory = None

        async def _fake_run(agent: object, t: TaskPayload, token: object) -> str:
            return "ok"

        with (
            patch.object(w, "_reconstruct_agent", return_value=mock_agent),
            patch.object(w, "_run_agent", side_effect=_fake_run),
            patch.object(w, "_listen_for_cancel", new_callable=AsyncMock),
        ):
            await w._execute_task(task)

        # agent.memory should not have been set
        assert mock_agent.memory is None

    @pytest.mark.asyncio
    async def test_short_term_store_created(self) -> None:
        """Memory config with short_term backend creates a ShortTermMemory."""
        w = Worker("redis://localhost", worker_id="w1")
        w._broker = AsyncMock()
        w._store = AsyncMock()
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-mem-2",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="hello",
            metadata={
                "memory": {
                    "backend": "short_term",
                    "scope": {"user_id": "u1", "session_id": "s1"},
                }
            },
        )

        captured_agent: list = []

        async def _fake_run(agent: object, t: TaskPayload, token: object) -> str:
            captured_agent.append(agent)
            return "ok"

        mock_agent = MagicMock()

        with (
            patch.object(w, "_reconstruct_agent", return_value=mock_agent),
            patch.object(w, "_run_agent", side_effect=_fake_run),
            patch.object(w, "_listen_for_cancel", new_callable=AsyncMock),
        ):
            await w._execute_task(task)

        from exo.memory.short_term import ShortTermMemory  # pyright: ignore[reportMissingImports]

        assert isinstance(mock_agent.memory, ShortTermMemory)

    @pytest.mark.asyncio
    async def test_user_input_saved_as_human_memory(self) -> None:
        """The task input is saved as HumanMemory in the store."""
        w = Worker("redis://localhost", worker_id="w1")
        w._broker = AsyncMock()
        w._store = AsyncMock()
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-mem-3",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="What is 2+2?",
            metadata={
                "memory": {
                    "backend": "short_term",
                    "scope": {"user_id": "u1"},
                }
            },
        )

        async def _fake_run(agent: object, t: TaskPayload, token: object) -> str:
            return "4"

        mock_agent = MagicMock()

        with (
            patch.object(w, "_reconstruct_agent", return_value=mock_agent),
            patch.object(w, "_run_agent", side_effect=_fake_run),
            patch.object(w, "_listen_for_cancel", new_callable=AsyncMock),
        ):
            await w._execute_task(task)

        # Check the store has a HumanMemory item
        store = mock_agent.memory
        items = store._items
        human_items = [i for i in items if i.memory_type == "human"]
        assert len(human_items) == 1
        assert human_items[0].content == "What is 2+2?"


# ---------------------------------------------------------------------------
# Feature 6: Conversation History (memory_items_to_messages)
# ---------------------------------------------------------------------------


class TestMemoryItemsToMessages:
    def test_human_to_user_message(self) -> None:
        from exo.distributed.memory import memory_items_to_messages
        from exo.memory.base import HumanMemory
        from exo.types import UserMessage  # pyright: ignore[reportMissingImports]

        items = [HumanMemory(content="Hello")]
        result = memory_items_to_messages(items)
        assert len(result) == 1
        assert isinstance(result[0], UserMessage)
        assert result[0].content == "Hello"

    def test_ai_to_assistant_message(self) -> None:
        from exo.distributed.memory import memory_items_to_messages
        from exo.memory.base import AIMemory
        from exo.types import AssistantMessage  # pyright: ignore[reportMissingImports]

        items = [AIMemory(content="Hi there")]
        result = memory_items_to_messages(items)
        assert len(result) == 1
        assert isinstance(result[0], AssistantMessage)
        assert result[0].content == "Hi there"

    def test_ai_with_tool_calls(self) -> None:
        from exo.distributed.memory import memory_items_to_messages
        from exo.memory.base import AIMemory
        from exo.types import AssistantMessage  # pyright: ignore[reportMissingImports]

        items = [
            AIMemory(
                content="Calling tool",
                tool_calls=[{"id": "tc-1", "name": "search", "arguments": '{"q":"test"}'}],
            )
        ]
        result = memory_items_to_messages(items)
        assert len(result) == 1
        assert isinstance(result[0], AssistantMessage)
        assert len(result[0].tool_calls) == 1
        assert result[0].tool_calls[0].name == "search"

    def test_tool_to_tool_result(self) -> None:
        from exo.distributed.memory import memory_items_to_messages
        from exo.memory.base import ToolMemory
        from exo.types import ToolResult  # pyright: ignore[reportMissingImports]

        items = [
            ToolMemory(
                content="found it",
                tool_call_id="tc-1",
                tool_name="search",
            )
        ]
        result = memory_items_to_messages(items)
        assert len(result) == 1
        assert isinstance(result[0], ToolResult)
        assert result[0].content == "found it"
        assert result[0].tool_call_id == "tc-1"

    def test_system_to_system_message(self) -> None:
        from exo.distributed.memory import memory_items_to_messages
        from exo.memory.base import SystemMemory
        from exo.types import SystemMessage  # pyright: ignore[reportMissingImports]

        items = [SystemMemory(content="Be helpful")]
        result = memory_items_to_messages(items)
        assert len(result) == 1
        assert isinstance(result[0], SystemMessage)
        assert result[0].content == "Be helpful"

    def test_mixed_items(self) -> None:
        from exo.distributed.memory import memory_items_to_messages
        from exo.memory.base import AIMemory, HumanMemory
        from exo.types import (  # pyright: ignore[reportMissingImports]
            AssistantMessage,
            UserMessage,
        )

        items = [
            HumanMemory(content="Hello"),
            AIMemory(content="Hi!"),
        ]
        result = memory_items_to_messages(items)
        assert len(result) == 2
        assert isinstance(result[0], UserMessage)
        assert isinstance(result[1], AssistantMessage)

    def test_empty_list(self) -> None:
        from exo.distributed.memory import memory_items_to_messages

        assert memory_items_to_messages([]) == []

    def test_error_tool_memory(self) -> None:
        from exo.distributed.memory import memory_items_to_messages
        from exo.memory.base import ToolMemory
        from exo.types import ToolResult  # pyright: ignore[reportMissingImports]

        items = [
            ToolMemory(
                content="timeout",
                tool_call_id="tc-1",
                tool_name="fetch",
                is_error=True,
            )
        ]
        result = memory_items_to_messages(items)
        assert isinstance(result[0], ToolResult)
        assert result[0].error == "timeout"


# ---------------------------------------------------------------------------
# Feature 6: Conversation history loading in _run_agent
# ---------------------------------------------------------------------------


class TestConversationHistoryLoading:
    @pytest.mark.asyncio
    async def test_no_memory_no_history(self) -> None:
        """When agent has no memory, no history is loaded."""
        w = Worker("redis://localhost", worker_id="w1")
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-hist-1",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="hello",
        )

        from exo.types import TextEvent  # pyright: ignore[reportMissingImports]

        events = [TextEvent(text="ok", agent_name="agent")]
        mock_agent = MagicMock()
        mock_agent.memory = None
        mock_agent.model = "openai:gpt-4o"

        mock_run = MagicMock()
        captured_kwargs: dict = {}

        async def _fake_stream_gen(*a: object, **kw: object) -> object:
            captured_kwargs.update(kw)
            for ev in events:
                yield ev

        mock_run.stream = _fake_stream_gen

        token = CancellationToken()
        with patch("exo.runner.run", mock_run):
            await w._run_agent(mock_agent, task, token)

        assert captured_kwargs["messages"] is None

    @pytest.mark.asyncio
    async def test_memory_search_failure_graceful(self) -> None:
        """When memory search raises, execution continues with a warning."""
        w = Worker("redis://localhost", worker_id="w1")
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-hist-2",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="hello",
            metadata={"memory": {"scope": {"user_id": "u1"}}},
        )

        from exo.types import TextEvent  # pyright: ignore[reportMissingImports]

        events = [TextEvent(text="ok", agent_name="agent")]
        mock_agent = MagicMock()
        mock_agent.model = "openai:gpt-4o"

        # Memory that fails on search
        mock_memory = AsyncMock()
        mock_memory.search = AsyncMock(side_effect=RuntimeError("db down"))
        mock_agent.memory = mock_memory

        mock_run = MagicMock()
        captured_kwargs: dict = {}

        async def _fake_stream_gen(*a: object, **kw: object) -> object:
            captured_kwargs.update(kw)
            for ev in events:
                yield ev

        mock_run.stream = _fake_stream_gen

        token = CancellationToken()
        with patch("exo.runner.run", mock_run):
            result = await w._run_agent(mock_agent, task, token)

        # Should still complete despite search failure
        assert result == "ok"


# ---------------------------------------------------------------------------
# Task timeout enforcement (P0)
# ---------------------------------------------------------------------------


class TestWorkerTaskTimeout:
    @pytest.mark.asyncio
    async def test_timeout_raises_exo_error(self) -> None:
        """When _run_agent exceeds timeout_seconds, ExoError is raised."""
        w = Worker("redis://localhost", worker_id="w1")
        w._broker = AsyncMock()
        w._broker.max_retries = 3
        w._store = AsyncMock()
        w._store.get_status = AsyncMock(
            return_value=TaskResult(task_id="task-timeout-1", status=TaskStatus.RUNNING, retries=0)
        )
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-timeout-1",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="hello",
            timeout_seconds=0.01,  # very short timeout
        )

        async def _slow_run(agent: object, t: TaskPayload, token: object) -> str:
            import asyncio

            await asyncio.sleep(10)  # will be cancelled by wait_for
            return "never"

        with (
            patch.object(w, "_reconstruct_agent", return_value=MagicMock()),
            patch.object(w, "_run_agent", side_effect=_slow_run),
            patch.object(w, "_listen_for_cancel", new_callable=AsyncMock),
        ):
            await w._execute_task(task)

        # Should have set FAILED and acked (exhausted retries, since retries=0 and max=3)
        status_calls = w._store.set_status.call_args_list
        statuses = [c.args[1] for c in status_calls]
        assert TaskStatus.RUNNING in statuses
        assert TaskStatus.FAILED in statuses
        assert w.tasks_failed == 1

    @pytest.mark.asyncio
    async def test_zero_timeout_is_unlimited(self) -> None:
        """timeout_seconds=0 means no timeout limit (wait_for gets None)."""
        w = Worker("redis://localhost", worker_id="w1")
        w._broker = AsyncMock()
        w._store = AsyncMock()
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-timeout-2",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="hello",
            timeout_seconds=0,  # zero → no timeout enforcement
        )

        async def _fast_run(agent: object, t: TaskPayload, token: object) -> str:
            return "done"

        with (
            patch.object(w, "_reconstruct_agent", return_value=MagicMock()),
            patch.object(w, "_run_agent", side_effect=_fast_run),
            patch.object(w, "_listen_for_cancel", new_callable=AsyncMock),
        ):
            await w._execute_task(task)

        assert w.tasks_processed == 1


# ---------------------------------------------------------------------------
# PEL-safe ack on Redis failure in error path (P0)
# ---------------------------------------------------------------------------


class TestWorkerPELSafeAck:
    @pytest.mark.asyncio
    async def test_ack_called_even_when_set_status_fails(self) -> None:
        """The message must be acked even if the FAILED status update throws."""
        w = Worker("redis://localhost", worker_id="w1")
        w._broker = AsyncMock()
        w._broker.max_retries = 0

        # set_status always raises after the first call (RUNNING succeeds)
        call_count = 0

        async def flaky_set_status(*args: object, **kwargs: object) -> None:
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise ConnectionError("Redis down")

        w._store = AsyncMock()
        w._store.set_status = flaky_set_status
        # get_status also fails (retry count unreadable)
        w._store.get_status = AsyncMock(side_effect=ConnectionError("Redis down"))
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-pel-1",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="fail",
        )

        with (
            patch.object(w, "_reconstruct_agent", side_effect=RuntimeError("boom")),
            patch.object(w, "_listen_for_cancel", new_callable=AsyncMock),
        ):
            await w._execute_task(task)

        # Despite store failures, the broker ack must still have been called
        w._broker.ack.assert_called_once_with("task-pel-1")

    @pytest.mark.asyncio
    async def test_nack_called_when_retries_remain(self) -> None:
        """When retries remain, nack is called even if intermediate store ops fail."""
        w = Worker("redis://localhost", worker_id="w1")
        w._broker = AsyncMock()
        w._broker.max_retries = 3
        w._store = AsyncMock()
        w._store.get_status = AsyncMock(
            return_value=TaskResult(task_id="task-pel-2", status=TaskStatus.RUNNING, retries=0)
        )
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-pel-2",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="fail",
        )

        with (
            patch.object(w, "_reconstruct_agent", side_effect=RuntimeError("boom")),
            patch.object(w, "_listen_for_cancel", new_callable=AsyncMock),
        ):
            await w._execute_task(task)

        w._broker.nack.assert_called_once_with("task-pel-2")
        w._broker.ack.assert_not_called()


# ---------------------------------------------------------------------------
# _listen_for_cancel reconnect on ConnectionError (P1)
# ---------------------------------------------------------------------------


class TestListenForCancelReconnect:
    @pytest.mark.asyncio
    async def test_reconnects_after_connection_error(self) -> None:
        """_listen_for_cancel rebuilds the connection on aioredis.ConnectionError."""

        import redis.asyncio as aioredis

        w = Worker("redis://localhost", worker_id="w1")
        token = CancellationToken()

        call_count = 0

        async def _fake_get_message(
            ignore_subscribe_messages: bool = False, timeout: float = 1.0
        ) -> dict | None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise aioredis.ConnectionError("disconnected")
            # On reconnect, deliver cancel signal
            return {"type": "message", "data": "cancel"}

        mock_pubsub = AsyncMock()
        mock_pubsub.get_message = _fake_get_message
        mock_pubsub.subscribe = AsyncMock()
        mock_pubsub.unsubscribe = AsyncMock()
        mock_pubsub.aclose = AsyncMock()

        mock_redis = MagicMock()
        mock_redis.pubsub.return_value = mock_pubsub
        mock_redis.aclose = AsyncMock()

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            # Should complete without raising (reconnects and receives cancel)
            await w._listen_for_cancel("task-reconnect", token)

        assert token.cancelled
