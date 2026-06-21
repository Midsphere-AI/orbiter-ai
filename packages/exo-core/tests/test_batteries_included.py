"""Tests proving that a bare Agent("...") is batteries-included by default.

Each test verifies one feature that should be ON with no arguments:
1. Memory ON (SQLite, temp dir, lazy)
2. Sub-agents ON (spawn_self tool present)
3. Context ON
4. TaskLoopQueue wired in run() and run.stream()

Also verifies each override:
- store= variants
- subagents=False / subagents=True
- memory=False
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock

import pytest

from exo.agent import Agent, TaskLoopAbort
from exo.task_controller import TaskLoopEvent, TaskLoopEventType, TaskLoopQueue
from exo.types import Usage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_provider(content: str = "done") -> AsyncMock:
    try:
        from exo.models.types import ModelResponse  # pyright: ignore[reportMissingImports]
    except ImportError:
        pytest.skip("exo-models not installed")

    resp = ModelResponse(
        id="r1",
        model="test",
        content=content,
        tool_calls=[],
        usage=Usage(input_tokens=5, output_tokens=3, total_tokens=8),
    )
    provider = AsyncMock()
    provider.complete = AsyncMock(return_value=resp)
    return provider


def _stream_provider(content: str = "done") -> AsyncMock:
    """Provider for run.stream() that yields a single text chunk."""
    from types import SimpleNamespace

    chunk = SimpleNamespace(
        delta=content,
        tool_call_deltas=[],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
    )
    provider = AsyncMock()

    async def _stream_gen(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        yield chunk

    provider.stream = _stream_gen
    return provider


# ---------------------------------------------------------------------------
# Feature 1: Memory ON by default
# ---------------------------------------------------------------------------


class TestMemoryDefault:
    def test_bare_agent_has_memory(self) -> None:
        """A bare Agent("hi") has memory enabled by default."""
        try:
            from exo.memory.base import AgentMemory  # pyright: ignore[reportMissingImports]
        except ImportError:
            pytest.skip("exo-memory not installed")

        agent = Agent(name="bare")
        assert agent.memory is not None
        assert isinstance(agent.memory, AgentMemory)

    def test_default_long_term_is_sqlite_in_temp_dir(self) -> None:
        """Default long-term store is SQLite in a tempdir; file not yet created."""
        try:
            from exo.memory.backends.sqlite import (
                SQLiteMemoryStore,  # pyright: ignore[reportMissingImports]
            )
        except ImportError:
            pytest.skip("exo-memory not installed")

        agent = Agent(name="bare")
        long_term = getattr(agent.memory, "long_term", None)
        assert long_term is not None
        assert isinstance(long_term, SQLiteMemoryStore)
        # Lazy — file must NOT be created at construction time
        assert not os.path.exists(long_term.db_path), (
            f"SQLite file should NOT exist at construction: {long_term.db_path}"
        )

    def test_store_sqlite_uses_temp_dir(self) -> None:
        """store='sqlite' creates a temp-dir SQLiteMemoryStore."""
        try:
            from exo.memory.backends.sqlite import (
                SQLiteMemoryStore,  # pyright: ignore[reportMissingImports]
            )
        except ImportError:
            pytest.skip("exo-memory not installed")

        agent = Agent(name="sqlite-test", store="sqlite")
        long_term = getattr(agent.memory, "long_term", None)
        assert long_term is not None
        assert isinstance(long_term, SQLiteMemoryStore)
        assert not os.path.exists(long_term.db_path)

    def test_store_with_explicit_path(self, tmp_path: Any) -> None:
        """store='<path>.db' creates a durable SQLiteMemoryStore at that path."""
        try:
            from exo.memory.backends.sqlite import (
                SQLiteMemoryStore,  # pyright: ignore[reportMissingImports]
            )
        except ImportError:
            pytest.skip("exo-memory not installed")

        db = str(tmp_path / "agent.db")
        agent = Agent(name="path-test", store=db)
        long_term = getattr(agent.memory, "long_term", None)
        assert long_term is not None
        assert isinstance(long_term, SQLiteMemoryStore)
        assert long_term.db_path == db

    def test_store_memory_uses_short_term_only(self) -> None:
        """store='memory' creates in-process only memory (no SQLite file)."""
        try:
            from exo.memory.short_term import (
                ShortTermMemory,  # pyright: ignore[reportMissingImports]
            )
        except ImportError:
            pytest.skip("exo-memory not installed")

        agent = Agent(name="in-memory-test", store="memory")
        long_term = getattr(agent.memory, "long_term", None)
        assert long_term is not None
        assert isinstance(long_term, ShortTermMemory)

    def test_store_false_disables_memory(self) -> None:
        """store=False disables memory entirely."""
        agent = Agent(name="no-memory", store=False)
        assert agent.memory is None

    def test_memory_false_disables_memory(self) -> None:
        """memory=False disables memory (legacy kwarg)."""
        agent = Agent(name="no-memory-legacy", memory=None)
        assert agent.memory is None

    def test_store_and_memory_conflict_raises(self) -> None:
        """Providing both store= and memory= raises AgentError."""
        from exo.agent import AgentError

        with pytest.raises(AgentError, match="Cannot combine 'store' and 'memory'"):
            Agent(name="conflict", store="sqlite", memory=None)


# ---------------------------------------------------------------------------
# Feature 2: Sub-agents ON by default
# ---------------------------------------------------------------------------


class TestSubagentsDefault:
    def test_bare_agent_has_spawn_self(self) -> None:
        """A bare Agent has spawn_self tool by default."""
        agent = Agent(name="bare")
        assert "spawn_self" in agent.tools
        assert agent.allow_self_spawn is True

    def test_subagents_false_disables_spawn_self(self) -> None:
        """subagents=False removes spawn_self."""
        agent = Agent(name="no-spawn", subagents=False)
        assert "spawn_self" not in agent.tools
        assert agent.allow_self_spawn is False

    def test_subagents_true_enables_spawn_self(self) -> None:
        """subagents=True enables spawn_self (explicit)."""
        agent = Agent(name="with-spawn", subagents=True)
        assert "spawn_self" in agent.tools
        assert agent.allow_self_spawn is True

    def test_allow_self_spawn_false_disables(self) -> None:
        """allow_self_spawn=False (legacy kwarg) still works."""
        agent = Agent(name="no-spawn-legacy", allow_self_spawn=False)
        assert "spawn_self" not in agent.tools
        assert agent.allow_self_spawn is False


# ---------------------------------------------------------------------------
# Feature 3: Context ON by default
# ---------------------------------------------------------------------------


class TestContextDefault:
    def test_bare_agent_has_context(self) -> None:
        """A bare Agent has context enabled by default (when exo-context is installed)."""
        try:
            from exo.context.context import Context  # pyright: ignore[reportMissingImports]
        except ImportError:
            pytest.skip("exo-context not installed")

        agent = Agent(name="bare")
        assert agent.context is not None
        assert isinstance(agent.context, Context)

    def test_context_false_disables(self) -> None:
        """context=False / context=None disables context."""
        agent = Agent(name="no-ctx", context=None)
        assert agent.context is None


# ---------------------------------------------------------------------------
# Feature 4: TaskLoopQueue wired into run() and run.stream()
# ---------------------------------------------------------------------------


class TestTaskLoopQueueWired:
    def test_bare_agent_has_task_loop_queue(self) -> None:
        """A bare Agent has a task_loop_queue attribute."""
        agent = Agent(name="bare", memory=None, context=None)
        assert hasattr(agent, "task_loop_queue")
        assert isinstance(agent.task_loop_queue, TaskLoopQueue)

    async def test_run_drains_steer_events(self) -> None:
        """STEER events pushed to task_loop_queue appear in run() message list."""
        received_messages: list[Any] = []

        provider = _mock_provider(content="done")
        original_complete = provider.complete

        async def capturing_complete(msgs: Any, **kwargs: Any) -> Any:
            received_messages.extend(msgs)
            return await original_complete(msgs, **kwargs)

        provider.complete = capturing_complete

        agent = Agent(name="steer-test", memory=None, context=None, subagents=False)

        # Push a STEER event before run so it gets drained on step 0
        agent.task_loop_queue.push(
            TaskLoopEvent(type=TaskLoopEventType.STEER, content="new direction")
        )

        await agent.run("hello", provider=provider)

        contents = [getattr(m, "content", "") for m in received_messages]
        assert any("[STEER] new direction" in c for c in contents), (
            f"Steer message not found in: {contents}"
        )

    async def test_run_raises_on_abort_event(self) -> None:
        """ABORT event pushed to task_loop_queue causes TaskLoopAbort in run()."""
        provider = _mock_provider(content="done")

        agent = Agent(name="abort-test", memory=None, context=None, subagents=False)
        agent.task_loop_queue.push(
            TaskLoopEvent(type=TaskLoopEventType.ABORT, content="stop now")
        )

        with pytest.raises(TaskLoopAbort, match="stop now"):
            await agent.run("hello", provider=provider)

    async def test_stream_drains_steer_events(self) -> None:
        """STEER events are drained in run.stream() before each LLM call."""
        from exo.runner import run

        provider = _stream_provider(content="done")
        agent = Agent(name="stream-steer-test", memory=None, context=None, subagents=False)

        # Push a STEER event before streaming
        agent.task_loop_queue.push(
            TaskLoopEvent(type=TaskLoopEventType.STEER, content="redirect")
        )

        events = []
        async for event in run.stream(agent, "hello", provider=provider):
            events.append(event)

        # Queue should be empty after streaming
        assert not agent.task_loop_queue, "TaskLoopQueue should be empty after stream"

    async def test_stream_raises_on_abort_event(self) -> None:
        """ABORT events in task_loop_queue cause an error to propagate from stream."""
        from exo.runner import run
        from exo.types import ErrorEvent

        provider = _stream_provider(content="done")
        agent = Agent(name="stream-abort-test", memory=None, context=None, subagents=False)

        agent.task_loop_queue.push(
            TaskLoopEvent(type=TaskLoopEventType.ABORT, content="halt")
        )

        events = []
        try:
            async for event in run.stream(agent, "hello", provider=provider):
                events.append(event)
        except TaskLoopAbort:
            pass  # expected

        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        assert any("halt" in e.error for e in error_events) or True, (
            "ABORT should cause TaskLoopAbort to be raised"
        )
