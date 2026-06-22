"""End-to-end tests for the durable agent loop (AgentLoopWorkflow).

These run against ``temporalio``'s in-memory time-skipping test server, with a
registered mock provider so the durable activities can resolve a model without
real API calls.  They exercise the real workflow + activities on a real Worker.
"""

from __future__ import annotations

from typing import Any

import pytest

from exo.distributed.temporal import HAS_TEMPORAL  # pyright: ignore[reportMissingImports]

pytestmark = pytest.mark.skipif(not HAS_TEMPORAL, reason="temporalio not installed")


# ---------------------------------------------------------------------------
# Mock provider registered under the "mock" provider name
# ---------------------------------------------------------------------------


def _register_mock_provider() -> None:
    from exo.models.provider import ModelProvider, model_registry
    from exo.models.types import ModelResponse
    from exo.types import ToolCall, Usage

    if "mock" in model_registry.list_all():
        return

    @model_registry.register("mock")
    class _MockProvider(ModelProvider):  # type: ignore[misc]
        """Returns one tool call, then a final answer once a tool result exists."""

        async def complete(
            self,
            messages: list[Any],
            *,
            tools: list[dict[str, Any]] | None = None,
            temperature: float | None = None,
            max_tokens: int | None = None,
        ) -> ModelResponse:
            has_tool_result = any(getattr(m, "role", None) == "tool" for m in messages)
            if has_tool_result:
                return ModelResponse(content="final answer", tool_calls=[], usage=Usage())
            return ModelResponse(
                content="",
                tool_calls=[ToolCall(id="c1", name="echo", arguments='{"text": "hi"}')],
                usage=Usage(),
            )

        async def stream(self, messages: list[Any], **kwargs: Any) -> Any:  # pragma: no cover
            yield  # type: ignore[misc]


def _register_multi_step_provider() -> None:
    """Provider that loops N times (based on tool-result count) before finishing."""
    from exo.models.provider import ModelProvider, model_registry
    from exo.models.types import ModelResponse
    from exo.types import ToolCall, Usage

    if "mock_multi" in model_registry.list_all():
        return

    @model_registry.register("mock_multi")
    class _MultiStepProvider(ModelProvider):  # type: ignore[misc]
        """Returns a tool call for the first 2 turns, then a final answer."""

        async def complete(
            self,
            messages: list[Any],
            *,
            tools: list[dict[str, Any]] | None = None,
            temperature: float | None = None,
            max_tokens: int | None = None,
        ) -> ModelResponse:
            tool_results = [m for m in messages if getattr(m, "role", None) == "tool"]
            if len(tool_results) >= 2:
                return ModelResponse(content="multi-step done", tool_calls=[], usage=Usage())
            call_id = f"c{len(tool_results) + 1}"
            return ModelResponse(
                content="",
                tool_calls=[ToolCall(id=call_id, name="echo", arguments='{"text": "step"}')],
                usage=Usage(),
            )

        async def stream(self, messages: list[Any], **kwargs: Any) -> Any:  # pragma: no cover
            yield  # type: ignore[misc]


def _register_sentinel_provider() -> None:
    """Provider that loops until it sees a user message containing 'STOP'."""
    from exo.models.provider import ModelProvider, model_registry
    from exo.models.types import ModelResponse
    from exo.types import ToolCall, Usage

    if "mock_sentinel" in model_registry.list_all():
        return

    @model_registry.register("mock_sentinel")
    class _SentinelProvider(ModelProvider):  # type: ignore[misc]
        """Issues a tool call each turn until any user message contains 'STOP'."""

        async def complete(
            self,
            messages: list[Any],
            *,
            tools: list[dict[str, Any]] | None = None,
            temperature: float | None = None,
            max_tokens: int | None = None,
        ) -> ModelResponse:
            # Inspect raw content — messages may be model objects or plain dicts
            for m in messages:
                role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)
                content = (
                    getattr(m, "content", None)
                    or (m.get("content") if isinstance(m, dict) else None)
                    or ""
                )
                if role == "user" and "STOP" in str(content):
                    return ModelResponse(content="sentinel triggered", tool_calls=[], usage=Usage())
            return ModelResponse(
                content="",
                tool_calls=[ToolCall(id="s1", name="echo", arguments='{"text": "loop"}')],
                usage=Usage(),
            )

        async def stream(self, messages: list[Any], **kwargs: Any) -> Any:  # pragma: no cover
            yield  # type: ignore[misc]


def _register_mock2_provider() -> None:
    """Second distinct mock provider returning a unique answer."""
    from exo.models.provider import ModelProvider, model_registry
    from exo.models.types import ModelResponse
    from exo.types import ToolCall, Usage

    if "mock2" in model_registry.list_all():
        return

    @model_registry.register("mock2")
    class _Mock2Provider(ModelProvider):  # type: ignore[misc]
        """Returns a tool call once, then a distinctive 'mock2 answer'."""

        async def complete(
            self,
            messages: list[Any],
            *,
            tools: list[dict[str, Any]] | None = None,
            temperature: float | None = None,
            max_tokens: int | None = None,
        ) -> ModelResponse:
            has_tool_result = any(getattr(m, "role", None) == "tool" for m in messages)
            if has_tool_result:
                return ModelResponse(content="mock2 answer", tool_calls=[], usage=Usage())
            return ModelResponse(
                content="",
                tool_calls=[ToolCall(id="m2c1", name="echo", arguments='{"text": "m2"}')],
                usage=Usage(),
            )

        async def stream(self, messages: list[Any], **kwargs: Any) -> Any:  # pragma: no cover
            yield  # type: ignore[misc]


def _register_infinite_provider() -> None:
    """Provider that always returns a tool call — never finishes on its own."""
    from exo.models.provider import ModelProvider, model_registry
    from exo.models.types import ModelResponse
    from exo.types import ToolCall, Usage

    if "mock_infinite" in model_registry.list_all():
        return

    @model_registry.register("mock_infinite")
    class _InfiniteProvider(ModelProvider):  # type: ignore[misc]
        """Always issues a tool call — relies on cancel/max_steps to stop."""

        async def complete(
            self,
            messages: list[Any],
            *,
            tools: list[dict[str, Any]] | None = None,
            temperature: float | None = None,
            max_tokens: int | None = None,
        ) -> ModelResponse:
            n = sum(1 for m in messages if getattr(m, "role", None) == "tool")
            return ModelResponse(
                content="",
                tool_calls=[ToolCall(id=f"inf{n}", name="echo", arguments='{"text": "inf"}')],
                usage=Usage(),
            )

        async def stream(self, messages: list[Any], **kwargs: Any) -> Any:  # pragma: no cover
            yield  # type: ignore[misc]


from exo import tool  # noqa: E402


@tool
def echo(text: str) -> str:
    """Echo tool — module-level so Agent.to_dict() can serialize it."""
    return f"echo: {text}"


def _build_agent_config(model: str = "mock:test") -> dict[str, Any]:
    from exo.agent import Agent

    agent = Agent(name="durable-test", model=model, tools=[echo], instructions="be brief")
    return agent.to_dict()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def _run_worker_and_workflow(loop_input: dict[str, Any]) -> tuple[str, Any]:
    """Start a worker + run AgentLoopWorkflow, returning (result, handle)."""
    import uuid

    from temporalio.client import Client
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Worker

    from exo.distributed.temporal.activities import agent_llm_turn, agent_tool_call
    from exo.distributed.temporal.executor import build_workflow_runner
    from exo.distributed.temporal.workflows import AgentLoopWorkflow

    _register_mock_provider()

    async with await WorkflowEnvironment.start_time_skipping() as env:
        client: Client = env.client
        task_queue = f"durable-{uuid.uuid4().hex[:8]}"
        async with Worker(
            client,
            task_queue=task_queue,
            workflows=[AgentLoopWorkflow],
            activities=[agent_llm_turn, agent_tool_call],
            workflow_runner=build_workflow_runner(),
        ):
            result = await client.execute_workflow(
                AgentLoopWorkflow.run,
                loop_input,
                id=f"wf-{uuid.uuid4().hex[:8]}",
                task_queue=task_queue,
            )
            return result, None


async def test_durable_loop_completes_with_tool_call() -> None:
    """One tool round-trip then a final answer."""
    cfg = _build_agent_config()
    result, _ = await _run_worker_and_workflow(
        {"agent_config": cfg, "input": "say hi", "max_steps": 5}
    )
    assert result == "final answer"


async def test_durable_loop_signal_query_update() -> None:
    """Signal injection, live query, and steering update against a running run."""
    import uuid

    from temporalio.client import Client
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Worker

    from exo.distributed.temporal.activities import agent_llm_turn, agent_tool_call
    from exo.distributed.temporal.executor import build_workflow_runner
    from exo.distributed.temporal.workflows import AgentLoopWorkflow

    _register_mock_provider()
    cfg = _build_agent_config()

    async with await WorkflowEnvironment.start_time_skipping() as env:
        client: Client = env.client
        task_queue = f"durable-{uuid.uuid4().hex[:8]}"
        async with Worker(
            client,
            task_queue=task_queue,
            workflows=[AgentLoopWorkflow],
            activities=[agent_llm_turn, agent_tool_call],
            workflow_runner=build_workflow_runner(),
        ):
            handle = await client.start_workflow(
                AgentLoopWorkflow.run,
                {"agent_config": cfg, "input": "go", "max_steps": 5},
                id=f"wf-{uuid.uuid4().hex[:8]}",
                task_queue=task_queue,
            )
            result = await handle.result()
            assert result == "final answer"

            progress = await handle.query(AgentLoopWorkflow.get_progress)
            assert progress["done"] is True
            assert progress["step"] >= 1


# ---------------------------------------------------------------------------
# Test 1: Multi-step tool loop
# ---------------------------------------------------------------------------


async def test_multi_step_tool_loop() -> None:
    """Provider loops 2 tool round-trips before finishing; asserts step >= 2."""
    import uuid

    from temporalio.client import Client
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Worker

    from exo.distributed.temporal.activities import agent_llm_turn, agent_tool_call
    from exo.distributed.temporal.executor import build_workflow_runner
    from exo.distributed.temporal.workflows import AgentLoopWorkflow

    _register_multi_step_provider()
    cfg = _build_agent_config(model="mock_multi:test")

    async with await WorkflowEnvironment.start_time_skipping() as env:
        client: Client = env.client
        task_queue = f"durable-{uuid.uuid4().hex[:8]}"
        async with Worker(
            client,
            task_queue=task_queue,
            workflows=[AgentLoopWorkflow],
            activities=[agent_llm_turn, agent_tool_call],
            workflow_runner=build_workflow_runner(),
        ):
            handle = await client.start_workflow(
                AgentLoopWorkflow.run,
                {"agent_config": cfg, "input": "start", "max_steps": 10},
                id=f"wf-{uuid.uuid4().hex[:8]}",
                task_queue=task_queue,
            )
            result = await handle.result()
            assert result == "multi-step done"

            progress = await handle.query(AgentLoopWorkflow.get_progress)
            assert progress["done"] is True
            assert progress["step"] >= 2, f"expected >= 2 steps, got {progress['step']}"


# ---------------------------------------------------------------------------
# Test 2: Signal injection mid-run affects the run
# ---------------------------------------------------------------------------


async def test_signal_injection_mid_run() -> None:
    """inject_message signal with 'STOP' causes provider to return a final answer."""
    import uuid

    from temporalio.client import Client
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Worker

    from exo.distributed.temporal.activities import agent_llm_turn, agent_tool_call
    from exo.distributed.temporal.executor import build_workflow_runner
    from exo.distributed.temporal.workflows import AgentLoopWorkflow

    _register_sentinel_provider()
    cfg = _build_agent_config(model="mock_sentinel:test")

    async with await WorkflowEnvironment.start_time_skipping() as env:
        client: Client = env.client
        task_queue = f"durable-{uuid.uuid4().hex[:8]}"
        async with Worker(
            client,
            task_queue=task_queue,
            workflows=[AgentLoopWorkflow],
            activities=[agent_llm_turn, agent_tool_call],
            workflow_runner=build_workflow_runner(),
        ):
            handle = await client.start_workflow(
                AgentLoopWorkflow.run,
                # max_steps=20 gives us room; the sentinel will stop it early
                {"agent_config": cfg, "input": "begin", "max_steps": 20},
                id=f"wf-{uuid.uuid4().hex[:8]}",
                task_queue=task_queue,
            )
            # Inject the sentinel message — the provider will see it before the
            # next LLM turn and return a final answer instead of another tool call.
            await handle.signal(AgentLoopWorkflow.inject_message, "STOP now")
            result = await handle.result()
            assert result == "sentinel triggered"


# ---------------------------------------------------------------------------
# Test 3: steer update — ack + validator rejects empty / completed run
# ---------------------------------------------------------------------------


async def test_steer_update_ack_and_validator() -> None:
    """steer returns an ack dict; validator rejects empty/whitespace instruction.

    The "reject steering a completed run" path is not exercised through the
    Temporal test server here because the server returns an RPCError (not a
    WorkflowUpdateFailedError) when execute_update is called on an already-closed
    workflow — the rejection is instead verified directly via the workflow's
    ``_validate_steer`` method in unit tests.  This test focuses on the
    input-validation path reachable via a live workflow.
    """
    import uuid

    from temporalio.client import Client, WorkflowUpdateFailedError
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Worker

    from exo.distributed.temporal.activities import agent_llm_turn, agent_tool_call
    from exo.distributed.temporal.executor import build_workflow_runner
    from exo.distributed.temporal.workflows import AgentLoopWorkflow

    # Use the infinite provider so the workflow stays open long enough for
    # the update validator tests; max_steps=3 as a backstop.
    _register_infinite_provider()
    cfg = _build_agent_config(model="mock_infinite:test")

    async with await WorkflowEnvironment.start_time_skipping() as env:
        client: Client = env.client
        task_queue = f"durable-{uuid.uuid4().hex[:8]}"
        async with Worker(
            client,
            task_queue=task_queue,
            workflows=[AgentLoopWorkflow],
            activities=[agent_llm_turn, agent_tool_call],
            workflow_runner=build_workflow_runner(),
        ):
            handle = await client.start_workflow(
                AgentLoopWorkflow.run,
                {"agent_config": cfg, "input": "go", "max_steps": 3},
                id=f"wf-{uuid.uuid4().hex[:8]}",
                task_queue=task_queue,
            )

            # Validator must reject empty instruction on a running workflow.
            with pytest.raises(WorkflowUpdateFailedError):
                await handle.execute_update(AgentLoopWorkflow.steer, "")

            # Validator must reject whitespace-only instruction.
            with pytest.raises(WorkflowUpdateFailedError):
                await handle.execute_update(AgentLoopWorkflow.steer, "   ")

            # A valid steer instruction must succeed and return an ack dict.
            ack = await handle.execute_update(AgentLoopWorkflow.steer, "be concise")
            assert isinstance(ack, dict)
            assert "accepted_at_step" in ack
            assert "queued" in ack

            # Let the backstop (max_steps=3) finish the run.
            await handle.result()

            # Verify the _validate_steer logic directly: a done=True workflow
            # should raise ValueError when steer is validated on it.
            wf_instance = AgentLoopWorkflow()
            wf_instance._done = True
            with pytest.raises(ValueError, match="completed"):
                wf_instance._validate_steer("valid instruction")

            # And empty/whitespace also raises ValueError.
            wf_instance._done = False
            with pytest.raises(ValueError, match="non-empty"):
                wf_instance._validate_steer("")
            with pytest.raises(ValueError, match="non-empty"):
                wf_instance._validate_steer("   ")


# ---------------------------------------------------------------------------
# Test 4: set_model update — validator rejects bad model; good model accepted
# ---------------------------------------------------------------------------


async def test_set_model_update() -> None:
    """set_model validator rejects bad format; a valid model string is accepted."""
    import uuid

    from temporalio.client import Client, WorkflowUpdateFailedError
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Worker

    from exo.distributed.temporal.activities import agent_llm_turn, agent_tool_call
    from exo.distributed.temporal.executor import build_workflow_runner
    from exo.distributed.temporal.workflows import AgentLoopWorkflow

    _register_mock_provider()
    _register_mock2_provider()
    cfg = _build_agent_config()

    async with await WorkflowEnvironment.start_time_skipping() as env:
        client: Client = env.client
        task_queue = f"durable-{uuid.uuid4().hex[:8]}"
        async with Worker(
            client,
            task_queue=task_queue,
            workflows=[AgentLoopWorkflow],
            activities=[agent_llm_turn, agent_tool_call],
            workflow_runner=build_workflow_runner(),
        ):
            wf_id = f"wf-{uuid.uuid4().hex[:8]}"
            handle = await client.start_workflow(
                AgentLoopWorkflow.run,
                {"agent_config": cfg, "input": "go", "max_steps": 10},
                id=wf_id,
                task_queue=task_queue,
            )

            # Validator must reject a model string without a colon.
            with pytest.raises(WorkflowUpdateFailedError):
                await handle.execute_update(AgentLoopWorkflow.set_model, "nonsense")

            # Validator must reject an empty model string.
            with pytest.raises(WorkflowUpdateFailedError):
                await handle.execute_update(AgentLoopWorkflow.set_model, "")

            # A well-formed model string must be accepted and returned.
            # Note: we target mock2 so the remaining turn uses mock2's provider.
            # The race between the update and the in-flight LLM turn means we
            # can't guarantee the final output is always "mock2 answer" (it
            # depends on whether the turn has already started), so we only
            # assert the update itself succeeds.
            new_model = await handle.execute_update(AgentLoopWorkflow.set_model, "mock2:x")
            assert new_model == "mock2:x"

            # Let the run finish (whichever provider finishes it).
            result = await handle.result()
            assert result in ("final answer", "mock2 answer")


# ---------------------------------------------------------------------------
# Test 5: continue_as_new — low threshold; run completes; total_steps accumulates
# ---------------------------------------------------------------------------


async def test_continue_as_new_completes_and_accumulates_steps() -> None:
    """Very low continue_as_new_threshold forces rollover; run still finishes."""
    import uuid

    from temporalio.client import Client
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Worker

    from exo.distributed.temporal.activities import agent_llm_turn, agent_tool_call
    from exo.distributed.temporal.executor import build_workflow_runner
    from exo.distributed.temporal.workflows import AgentLoopWorkflow

    _register_multi_step_provider()
    cfg = _build_agent_config(model="mock_multi:test")

    async with await WorkflowEnvironment.start_time_skipping() as env:
        client: Client = env.client
        task_queue = f"durable-{uuid.uuid4().hex[:8]}"
        async with Worker(
            client,
            task_queue=task_queue,
            workflows=[AgentLoopWorkflow],
            activities=[agent_llm_turn, agent_tool_call],
            workflow_runner=build_workflow_runner(),
        ):
            result = await client.execute_workflow(
                AgentLoopWorkflow.run,
                {
                    "agent_config": cfg,
                    "input": "start",
                    "max_steps": 10,
                    # Set threshold to 1 so each step triggers a continue-as-new.
                    "continue_as_new_threshold": 1,
                },
                id=f"wf-{uuid.uuid4().hex[:8]}",
                task_queue=task_queue,
            )
            assert result == "multi-step done"


# ---------------------------------------------------------------------------
# Test 6: cancel signal — workflow returns rather than hanging
# ---------------------------------------------------------------------------


async def test_cancel_signal_stops_run() -> None:
    """cancel signal causes the loop to exit; get_progress shows cancel_requested."""
    import uuid

    from temporalio.client import Client
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Worker

    from exo.distributed.temporal.activities import agent_llm_turn, agent_tool_call
    from exo.distributed.temporal.executor import build_workflow_runner
    from exo.distributed.temporal.workflows import AgentLoopWorkflow

    _register_infinite_provider()
    cfg = _build_agent_config(model="mock_infinite:test")

    async with await WorkflowEnvironment.start_time_skipping() as env:
        client: Client = env.client
        task_queue = f"durable-{uuid.uuid4().hex[:8]}"
        async with Worker(
            client,
            task_queue=task_queue,
            workflows=[AgentLoopWorkflow],
            activities=[agent_llm_turn, agent_tool_call],
            workflow_runner=build_workflow_runner(),
        ):
            handle = await client.start_workflow(
                AgentLoopWorkflow.run,
                # max_steps=5 is a hard backstop so the test can't hang even if
                # cancel somehow misses (time-skipping env makes this safe).
                {"agent_config": cfg, "input": "go", "max_steps": 5},
                id=f"wf-{uuid.uuid4().hex[:8]}",
                task_queue=task_queue,
            )
            await handle.signal(AgentLoopWorkflow.cancel)
            # Workflow must return (not hang) after cancel.
            _result = await handle.result()

            progress = await handle.query(AgentLoopWorkflow.get_progress)
            assert progress["cancel_requested"] is True
            assert progress["done"] is True


# ---------------------------------------------------------------------------
# Test 7: Swarm config rejected by agent_llm_turn activity
# ---------------------------------------------------------------------------


async def test_swarm_config_rejected_by_activity() -> None:
    """A Swarm-shaped agent_config raises in the activity (via ActivityEnvironment)."""
    from temporalio.testing import ActivityEnvironment

    from exo.agent import Agent
    from exo.distributed.temporal.activities import agent_llm_turn
    from exo.swarm import Swarm

    _register_mock_provider()

    agent_a = Agent(name="a", model="mock:test", tools=[echo])
    agent_b = Agent(name="b", model="mock:test", tools=[echo])
    swarm = Swarm(agents=[agent_a, agent_b])
    swarm_cfg = swarm.to_dict()
    assert "agents" in swarm_cfg  # sanity-check the fixture

    env = ActivityEnvironment()
    with pytest.raises(Exception, match=r"AgentLoopWorkflow|Swarm|single"):
        await env.run(
            agent_llm_turn,
            {
                "agent_config": swarm_cfg,
                "messages": [{"role": "user", "content": "hi"}],
                "max_retries": 1,
            },
        )


# ---------------------------------------------------------------------------
# Test 8: Executor control plane (inject_message, get_progress, steer)
# ---------------------------------------------------------------------------


async def test_executor_control_plane() -> None:
    """TemporalExecutor control-plane methods work against a live worker env."""
    import uuid

    from temporalio.client import Client
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Worker

    from exo.distributed.temporal.activities import agent_llm_turn, agent_tool_call
    from exo.distributed.temporal.executor import TemporalExecutor, build_workflow_runner
    from exo.distributed.temporal.workflows import AgentLoopWorkflow

    _register_mock_provider()
    cfg = _build_agent_config()
    task_id = f"task-{uuid.uuid4().hex[:8]}"

    async with await WorkflowEnvironment.start_time_skipping() as env:
        client: Client = env.client
        task_queue = f"durable-{uuid.uuid4().hex[:8]}"
        async with Worker(
            client,
            task_queue=task_queue,
            workflows=[AgentLoopWorkflow],
            activities=[agent_llm_turn, agent_tool_call],
            workflow_runner=build_workflow_runner(),
        ):
            executor = TemporalExecutor(durable=True)
            # Bypass the real connect() — inject the test client directly.
            executor._client = client

            wf_id = executor.workflow_id_for(task_id)
            assert wf_id == f"exo-task-{task_id}"

            # Start the workflow manually so we can exercise the control plane.
            handle = await client.start_workflow(
                AgentLoopWorkflow.run,
                {"agent_config": cfg, "input": "go", "max_steps": 5},
                id=wf_id,
                task_queue=task_queue,
            )

            # get_progress returns a sensible dict before completion.
            progress = await executor.get_progress(task_id)
            assert isinstance(progress, dict)
            assert "done" in progress

            # inject_message fires without error (fire-and-forget signal).
            await executor.inject_message(task_id, "extra context")

            # steer returns an ack dict (may race with completion — if the run
            # finishes before steer arrives the validator rejects it, which is
            # also correct behaviour; we accept either outcome).
            try:
                ack = await executor.steer(task_id, "be concise")
                assert isinstance(ack, dict)
                assert "accepted_at_step" in ack
            except Exception:
                # steer rejected because run completed — both are valid.
                pass

            # Let the workflow finish.
            result = await handle.result()
            assert result == "final answer"

            # After completion, get_progress shows done=True.
            final_progress = await executor.get_progress(task_id)
            assert final_progress["done"] is True
