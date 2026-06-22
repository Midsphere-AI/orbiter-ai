"""Tests for fire-and-forget background sub-agents.

A parent agent calls ``spawn_background(task)`` to launch a child that runs
detached; the parent keeps working and the child's result is injected back
when it finishes.  ``check_subagent``/``list_subagents`` report live status.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

from exo._internal.background import MergeMode
from exo._internal.state import RunNodeStatus
from exo.agent import Agent
from exo.namespaces import SubagentsConfig
from exo.runner import run
from exo.types import (
    MessageInjectedEvent,
    ToolCall,
    Usage,
    UserMessage,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeStreamChunk:
    """Lightweight stream chunk for testing (mirrors StreamChunk fields)."""

    def __init__(
        self,
        delta: str = "",
        tool_call_deltas: list[Any] | None = None,
        finish_reason: str | None = None,
        usage: Usage | None = None,
    ) -> None:
        self.delta = delta
        self.tool_call_deltas = tool_call_deltas or []
        self.finish_reason = finish_reason
        self.usage = usage or Usage()


def _child_stream_provider(
    text: str = "child answer",
    *,
    release: asyncio.Event | None = None,
) -> Any:
    """A provider whose ``stream()`` yields *text* (after *release* if given).

    Used as the child sub-agent's provider — the child runs via
    ``run.stream(...)`` internally.
    """

    async def stream(messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
        if release is not None:
            await release.wait()
        yield _FakeStreamChunk(
            delta=text,
            finish_reason="stop",
            usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
        )

    mock = AsyncMock()
    mock.stream = stream
    mock.complete = AsyncMock()
    return mock


def _bare(**kwargs: Any) -> Agent:
    """An agent with memory/context off to keep tests fast and isolated."""
    return Agent(name="bot", store=False, context=None, **kwargs)


def _resp(content: str = "", tool_calls: list[ToolCall] | None = None) -> Any:
    class R:
        pass

    r = R()
    r.content = content
    r.tool_calls = tool_calls or []
    r.usage = Usage()
    return r


# ---------------------------------------------------------------------------
# Init / gating
# ---------------------------------------------------------------------------


class TestBackgroundInit:
    def test_background_tools_on_by_default(self) -> None:
        agent = _bare()
        assert "spawn_background" in agent.tools
        assert "check_subagent" in agent.tools
        assert "list_subagents" in agent.tools

    def test_background_disabled_via_config(self) -> None:
        agent = _bare(subagents=SubagentsConfig(background=False))
        assert "spawn_self" in agent.tools
        assert "spawn_background" not in agent.tools
        assert "check_subagent" not in agent.tools
        assert "list_subagents" not in agent.tools

    def test_background_disabled_via_flat_kwarg(self) -> None:
        agent = _bare(background_subagents=False)
        assert "spawn_self" in agent.tools
        assert "spawn_background" not in agent.tools

    def test_subagents_off_removes_background_tools(self) -> None:
        agent = _bare(subagents=False)
        for name in ("spawn_self", "spawn_background", "check_subagent", "list_subagents"):
            assert name not in agent.tools

    def test_config_round_trips(self) -> None:
        agent = _bare(
            subagents=SubagentsConfig(background=True, background_timeout=5.0, background_max=3)
        )
        assert agent.subagents.background is True
        assert agent.subagents.background_timeout == 5.0
        assert agent.subagents.background_max == 3


# ---------------------------------------------------------------------------
# Tool-level behavior (invoke the tool directly)
# ---------------------------------------------------------------------------


class TestSpawnBackgroundTool:
    async def test_fire_and_forget_returns_task_id_immediately(self) -> None:
        agent = _bare()
        release = asyncio.Event()  # keep the child running
        agent._current_provider = _child_stream_provider(release=release)
        agent._is_running = True

        task_id = await agent.tools["spawn_background"].execute(task="do the thing")
        try:
            assert isinstance(task_id, str)
            assert task_id.startswith("bg_")
            bg = agent._bg_handler.get_task(task_id)
            assert bg is not None
            assert bg.status == RunNodeStatus.RUNNING
        finally:
            release.set()
            await agent._cancel_background()

    async def test_depth_guard(self) -> None:
        agent = _bare()
        agent._spawn_depth = agent.max_spawn_depth
        agent._current_provider = _child_stream_provider()
        agent._is_running = True

        out = await agent.tools["spawn_background"].execute(task="x")
        assert "depth" in out.lower()
        assert agent._bg_handler is None or not agent._bg_handler.list_tasks()
        assert not agent._bg_tasks

    async def test_empty_task_rejected(self) -> None:
        agent = _bare()
        agent._current_provider = _child_stream_provider()
        agent._is_running = True
        out = await agent.tools["spawn_background"].execute(task="   ")
        assert "empty" in out.lower()
        assert not agent._bg_tasks

    async def test_no_provider_rejected(self) -> None:
        agent = _bare()
        agent._is_running = True  # but no _current_provider
        out = await agent.tools["spawn_background"].execute(task="x")
        assert "provider" in out.lower()
        assert not agent._bg_tasks

    async def test_back_pressure_cap(self) -> None:
        agent = _bare(subagents=SubagentsConfig(background_max=1))
        release = asyncio.Event()
        agent._current_provider = _child_stream_provider(release=release)
        agent._is_running = True
        try:
            first = await agent.tools["spawn_background"].execute(task="a")
            assert first.startswith("bg_")
            second = await agent.tools["spawn_background"].execute(task="b")
            assert "too many" in second.lower()
        finally:
            release.set()
            await agent._cancel_background()

    async def test_completion_marks_success_and_result(self) -> None:
        agent = _bare()
        agent._current_provider = _child_stream_provider(text="the result")
        agent._is_running = True
        task_id = await agent.tools["spawn_background"].execute(task="work")
        task = agent._bg_tasks[task_id]
        await task
        bg = agent._bg_handler.get_task(task_id)
        assert bg.status == RunNodeStatus.SUCCESS
        assert bg.result == "the result"


# ---------------------------------------------------------------------------
# Status query tools
# ---------------------------------------------------------------------------


class TestStatusQueries:
    async def test_check_unknown_id(self) -> None:
        agent = _bare()
        out = await agent.tools["check_subagent"].execute(task_id="bg_nope")
        assert "no background" in out.lower()

    async def test_list_empty(self) -> None:
        agent = _bare()
        out = await agent.tools["list_subagents"].execute()
        assert "no background" in out.lower()

    async def test_check_and_list_after_completion(self) -> None:
        agent = _bare()
        agent._current_provider = _child_stream_provider(text="answer")
        agent._is_running = True
        task_id = await agent.tools["spawn_background"].execute(task="work")
        await agent._bg_tasks[task_id]

        checked = await agent.tools["check_subagent"].execute(task_id=task_id)
        assert "completed" in checked.lower()
        assert "answer" in checked

        listed = await agent.tools["list_subagents"].execute()
        assert task_id in listed


# ---------------------------------------------------------------------------
# HOT vs WAKEUP routing
# ---------------------------------------------------------------------------


class TestMergeRouting:
    async def test_hot_merge_while_running(self) -> None:
        """Child finishing while the parent is still running → HOT inject."""
        agent = _bare()
        provider = _child_stream_provider(text="bg result")
        agent._current_provider = provider
        agent._is_running = True
        task_id = await agent.tools["spawn_background"].execute(task="work")
        await agent._bg_tasks[task_id]
        bg = agent._bg_handler.get_task(task_id)
        assert bg.merge_mode == MergeMode.HOT
        # HOT merge injects the result into the running parent's queue.
        assert not agent._injected_messages.empty()

    async def test_wakeup_merge_after_run_ends(self) -> None:
        """Child finishing after the parent's run ended → WAKEUP queue."""
        agent = _bare()
        agent._current_provider = _child_stream_provider(text="late result")
        agent._is_running = False  # parent run already finished
        task_id = await agent.tools["spawn_background"].execute(task="work")
        await agent._bg_tasks[task_id]
        bg = agent._bg_handler.get_task(task_id)
        assert bg.merge_mode == MergeMode.WAKEUP
        assert agent._bg_handler.pending_queue.size == 1
        # WAKEUP does not inject directly — it waits for the next run.
        assert agent._injected_messages.empty()


# ---------------------------------------------------------------------------
# End-to-end through run()
# ---------------------------------------------------------------------------


class TestEndToEnd:
    async def test_result_injected_on_next_run_wakeup(self) -> None:
        """A child that finishes between runs surfaces on the next run."""
        seen: list[Any] = []
        release = asyncio.Event()
        pcalls = 0

        async def complete(messages: Any, **kwargs: Any) -> Any:
            nonlocal pcalls
            pcalls += 1
            if pcalls == 1:
                return _resp(
                    tool_calls=[
                        ToolCall(id="t1", name="spawn_background", arguments='{"task":"sub"}')
                    ]
                )
            # second parent call (run 1) and run 2 calls: capture + finish
            seen.extend(messages)
            return _resp(content="ok")

        async def stream(messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
            await release.wait()
            yield _FakeStreamChunk(delta="child done", finish_reason="stop")

        provider = AsyncMock()
        provider.complete = complete
        provider.stream = stream

        agent = _bare()
        # Run 1: spawns the background child (which blocks on `release`), finishes.
        r1 = await run(agent, "start", provider=provider)
        assert r1.output == "ok"
        assert agent._bg_tasks  # child still running

        # Let the child finish AFTER run 1 ended → WAKEUP.
        child_task = next(iter(agent._bg_tasks.values()))
        release.set()
        await child_task
        assert agent._bg_handler.pending_queue.size == 1

        # Run 2: the WAKEUP result is flushed at run start and seen by the LLM.
        seen.clear()
        await run(agent, "again", provider=provider)
        injected = [
            m
            for m in seen
            if isinstance(m, UserMessage) and "background subagent" in str(m.content)
        ]
        assert len(injected) >= 1
        assert "child done" in str(injected[0].content)

    async def test_streaming_parent_flushes_wakeup_as_message_injected(self) -> None:
        """A streamed parent flushes a pending WAKEUP result as a MessageInjectedEvent."""
        agent = _bare()

        # Produce a completed WAKEUP result (child finishes while parent idle).
        agent._current_provider = _child_stream_provider(text="child output")
        agent._is_running = False
        task_id = await agent.tools["spawn_background"].execute(task="sub")
        await agent._bg_tasks[task_id]
        assert agent._bg_handler.pending_queue.size == 1

        # Now stream the parent: the WAKEUP flush injects the result at run start,
        # which the streaming loop surfaces as a MessageInjectedEvent.
        async def stream(messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
            yield _FakeStreamChunk(delta="parent done", finish_reason="stop")

        parent_provider = AsyncMock()
        parent_provider.stream = stream
        parent_provider.complete = AsyncMock()

        events = [
            ev async for ev in run.stream(agent, "go", provider=parent_provider, detailed=True)
        ]
        injected = [e for e in events if isinstance(e, MessageInjectedEvent)]
        assert any("background subagent" in e.content for e in injected)
        assert any("child output" in e.content for e in injected)


# ---------------------------------------------------------------------------
# Timeout & cancellation
# ---------------------------------------------------------------------------


class TestTimeoutAndCancel:
    async def test_timeout_marks_failed(self) -> None:
        agent = _bare(subagents=SubagentsConfig(background_timeout=0.01))
        never = asyncio.Event()  # never set → child stream blocks forever
        agent._current_provider = _child_stream_provider(release=never)
        agent._is_running = True
        task_id = await agent.tools["spawn_background"].execute(task="slow")
        await agent._bg_tasks[task_id]
        bg = agent._bg_handler.get_task(task_id)
        assert bg.status == RunNodeStatus.FAILED
        assert "timed out" in (bg.error or "")

    async def test_cancel_background_reaps_tasks(self) -> None:
        agent = _bare()
        never = asyncio.Event()
        agent._current_provider = _child_stream_provider(release=never)
        agent._is_running = True
        task_id = await agent.tools["spawn_background"].execute(task="blocking")
        assert agent._bg_tasks
        await agent.aclose()
        assert not agent._bg_tasks
        bg = agent._bg_handler.get_task(task_id)
        assert bg.status == RunNodeStatus.FAILED
        assert bg.error == "cancelled"


# ---------------------------------------------------------------------------
# Child isolation
# ---------------------------------------------------------------------------


class TestChildIsolation:
    async def test_child_tools_exclude_subagent_tools(self) -> None:
        captured: dict[str, Any] = {}

        async def stream(messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
            captured["tools"] = kwargs.get("tools")
            yield _FakeStreamChunk(delta="done", finish_reason="stop")

        provider = AsyncMock()
        provider.stream = stream
        provider.complete = AsyncMock()

        agent = _bare()
        agent._current_provider = provider
        agent._is_running = True
        task_id = await agent.tools["spawn_background"].execute(task="work")
        await agent._bg_tasks[task_id]

        names = {t["function"]["name"] for t in (captured.get("tools") or [])}
        for n in ("spawn_self", "spawn_background", "check_subagent", "list_subagents"):
            assert n not in names
