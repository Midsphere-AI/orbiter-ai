"""Tests for exo.parallel — run_parallel / stream_parallel.

These replace the exo-harness test_harness_parallel.py tests that tested
the same parallel dispatch capability through the (now-deleted) HarnessContext.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest

from exo.agent import Agent
from exo.parallel import (
    SubAgentError,
    SubAgentResult,
    SubAgentStatus,
    SubAgentTask,
    run_parallel,
    stream_parallel,
)
from exo.types import AgentOutput, ErrorEvent, StatusEvent, StreamEvent, TextEvent, Usage

# ---------------------------------------------------------------------------
# Mock provider helpers
# ---------------------------------------------------------------------------


def _make_provider(responses: list[AgentOutput]) -> Any:
    """Create a mock provider returning pre-defined AgentOutput values."""
    call_count = 0

    async def complete(messages: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        resp = responses[min(call_count, len(responses) - 1)]
        call_count += 1

        class FakeResponse:
            content = resp.text
            tool_calls = resp.tool_calls
            usage = resp.usage

        return FakeResponse()

    mock = AsyncMock()
    mock.complete = complete
    return mock


class _FakeStreamChunk:
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


def _make_stream_provider(stream_rounds: list[list[_FakeStreamChunk]]) -> Any:
    call_count = 0

    async def stream(messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
        nonlocal call_count
        chunks = stream_rounds[min(call_count, len(stream_rounds) - 1)]
        call_count += 1
        for c in chunks:
            yield c

    mock = AsyncMock()
    mock.stream = stream
    mock.complete = AsyncMock()
    return mock


# ---------------------------------------------------------------------------
# Tests: run_parallel — basic
# ---------------------------------------------------------------------------


class TestRunParallelBasic:
    async def test_empty_tasks(self) -> None:
        results = await run_parallel([])
        assert results == []

    async def test_single_task(self) -> None:
        agent = Agent(name="solo")
        provider = _make_provider([AgentOutput(text="done")])
        tasks = [SubAgentTask(agent=agent, input="test", provider=provider)]
        results = await run_parallel(tasks, continue_on_error=True)

        assert len(results) == 1
        assert results[0].status == SubAgentStatus.SUCCESS
        assert results[0].output == "done"
        assert results[0].agent_name == "solo"

    async def test_two_agents(self) -> None:
        agent_a = Agent(name="alpha")
        agent_b = Agent(name="beta")
        provider = _make_provider([AgentOutput(text="hello")])
        tasks = [
            SubAgentTask(agent=agent_a, input="Hi"),
            SubAgentTask(agent=agent_b, input="Hi"),
        ]
        results = await run_parallel(tasks, provider=provider, continue_on_error=True)

        assert len(results) == 2
        assert all(r.status == SubAgentStatus.SUCCESS for r in results)
        assert all(r.output == "hello" for r in results)

    async def test_results_in_task_order(self) -> None:
        agent_a = Agent(name="first")
        agent_b = Agent(name="second")
        provider = _make_provider([AgentOutput(text="ok")])
        tasks = [
            SubAgentTask(agent=agent_a, input="test"),
            SubAgentTask(agent=agent_b, input="test"),
        ]
        results = await run_parallel(tasks, provider=provider, continue_on_error=True)

        assert results[0].agent_name == "first"
        assert results[1].agent_name == "second"

    async def test_custom_name_override(self) -> None:
        agent = Agent(name="real_name")
        provider = _make_provider([AgentOutput(text="ok")])
        tasks = [SubAgentTask(agent=agent, input="test", name="custom_label", provider=provider)]
        results = await run_parallel(tasks, continue_on_error=True)

        assert results[0].agent_name == "custom_label"

    async def test_duplicate_names_raises(self) -> None:
        agent_a = Agent(name="dup")
        agent_b = Agent(name="dup")
        tasks = [
            SubAgentTask(agent=agent_a, input="a"),
            SubAgentTask(agent=agent_b, input="b"),
        ]
        with pytest.raises(ValueError, match="Duplicate"):
            await run_parallel(tasks)

    async def test_per_task_provider_override(self) -> None:
        """Per-task provider takes precedence over the default provider."""
        agent = Agent(name="agent")
        task_provider = _make_provider([AgentOutput(text="from_task_provider")])
        default_provider = _make_provider([AgentOutput(text="from_default")])
        tasks = [SubAgentTask(agent=agent, input="test", provider=task_provider)]
        results = await run_parallel(tasks, provider=default_provider, continue_on_error=True)

        assert results[0].output == "from_task_provider"


# ---------------------------------------------------------------------------
# Tests: run_parallel — error handling
# ---------------------------------------------------------------------------


class TestRunParallelErrors:
    async def test_continue_on_error_partial_results(self) -> None:
        good_agent = Agent(name="good")
        bad_agent = Agent(name="bad")

        good_provider = _make_provider([AgentOutput(text="ok")])
        bad_provider = AsyncMock()
        bad_provider.complete = AsyncMock(side_effect=RuntimeError("boom"))

        tasks = [
            SubAgentTask(agent=good_agent, input="test", provider=good_provider),
            SubAgentTask(agent=bad_agent, input="test", provider=bad_provider),
        ]
        results = await run_parallel(tasks, continue_on_error=True)

        assert len(results) == 2
        good_r = next(r for r in results if r.agent_name == "good")
        bad_r = next(r for r in results if r.agent_name == "bad")
        assert good_r.status == SubAgentStatus.SUCCESS
        assert bad_r.status == SubAgentStatus.FAILED
        assert bad_r.error is not None

    async def test_fail_fast_raises_sub_agent_error(self) -> None:
        good_agent = Agent(name="good")
        bad_agent = Agent(name="bad")

        bad_provider = AsyncMock()
        bad_provider.complete = AsyncMock(side_effect=RuntimeError("boom"))
        good_provider = _make_provider([AgentOutput(text="ok")])

        tasks = [
            SubAgentTask(agent=bad_agent, input="test", provider=bad_provider),
            SubAgentTask(agent=good_agent, input="test", provider=good_provider),
        ]
        with pytest.raises(SubAgentError) as exc_info:
            await run_parallel(tasks, continue_on_error=False)

        assert "bad" in exc_info.value.failed_agents
        assert len(exc_info.value.results) == 2

    async def test_all_agents_fail(self) -> None:
        agent_a = Agent(name="a")
        agent_b = Agent(name="b")

        bad_provider = AsyncMock()
        bad_provider.complete = AsyncMock(side_effect=RuntimeError("fail"))

        tasks = [
            SubAgentTask(agent=agent_a, input="test", provider=bad_provider),
            SubAgentTask(agent=agent_b, input="test", provider=bad_provider),
        ]
        results = await run_parallel(tasks, continue_on_error=True)

        assert all(r.status == SubAgentStatus.FAILED for r in results)

    async def test_per_agent_timeout(self) -> None:
        fast_agent = Agent(name="fast")
        slow_agent = Agent(name="slow")

        fast_provider = _make_provider([AgentOutput(text="quick")])

        async def slow_complete(messages: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(10)

        slow_provider = AsyncMock()
        slow_provider.complete = slow_complete

        tasks = [
            SubAgentTask(agent=fast_agent, input="test", provider=fast_provider),
            SubAgentTask(agent=slow_agent, input="test", provider=slow_provider, timeout=0.1),
        ]
        results = await run_parallel(tasks, continue_on_error=True)

        fast_r = next(r for r in results if r.agent_name == "fast")
        slow_r = next(r for r in results if r.agent_name == "slow")
        assert fast_r.status == SubAgentStatus.SUCCESS
        assert slow_r.status == SubAgentStatus.TIMED_OUT
        assert slow_r.elapsed_seconds > 0


# ---------------------------------------------------------------------------
# Tests: run_parallel — max_concurrency
# ---------------------------------------------------------------------------


class TestRunParallelConcurrency:
    async def test_limits_parallel_execution(self) -> None:
        """max_concurrency=1 forces sequential execution."""
        execution_order: list[str] = []
        call_count = 0

        agent_a = Agent(name="a")
        agent_b = Agent(name="b")

        async def ordered_complete(messages: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            current = call_count
            call_count += 1
            execution_order.append(f"start_{current}")
            await asyncio.sleep(0.01)
            execution_order.append(f"end_{current}")

            class FakeResponse:
                content = f"result_{current}"
                tool_calls: list[Any] = []  # noqa: RUF012
                usage = Usage()

            return FakeResponse()

        provider = AsyncMock()
        provider.complete = ordered_complete

        tasks = [
            SubAgentTask(agent=agent_a, input="test"),
            SubAgentTask(agent=agent_b, input="test"),
        ]
        results = await run_parallel(tasks, provider=provider, continue_on_error=True, max_concurrency=1)

        assert len(results) == 2
        # With max_concurrency=1, first ends before second starts
        assert execution_order[1] == "end_0"
        assert execution_order[2] == "start_1"


# ---------------------------------------------------------------------------
# Tests: stream_parallel
# ---------------------------------------------------------------------------


class TestStreamParallel:
    async def test_empty_tasks(self) -> None:
        events = [e async for e in stream_parallel([])]
        assert events == []

    async def test_events_from_multiple_agents(self) -> None:
        agent_a = Agent(name="alpha")
        agent_b = Agent(name="beta")

        provider_a = _make_stream_provider([[_FakeStreamChunk(delta="A")]])
        provider_b = _make_stream_provider([[_FakeStreamChunk(delta="B")]])

        tasks = [
            SubAgentTask(agent=agent_a, input="test", provider=provider_a),
            SubAgentTask(agent=agent_b, input="test", provider=provider_b),
        ]
        events: list[StreamEvent] = []
        async for event in stream_parallel(tasks, continue_on_error=True):
            events.append(event)

        # StatusEvent(starting) + TextEvent + StatusEvent(completed) per agent
        status_events = [e for e in events if isinstance(e, StatusEvent)]
        starting = [e for e in status_events if e.status == "starting"]
        completed = [e for e in status_events if e.status == "completed"]
        assert len(starting) == 2
        assert len(completed) == 2

        text_events = [e for e in events if isinstance(e, TextEvent)]
        assert len(text_events) == 2
        texts = {e.text for e in text_events}
        assert "A" in texts
        assert "B" in texts

    async def test_error_event_on_failure(self) -> None:
        agent = Agent(name="failing")
        bad_provider = AsyncMock()
        bad_provider.stream = AsyncMock(side_effect=RuntimeError("crash"))

        tasks = [SubAgentTask(agent=agent, input="test", provider=bad_provider)]
        events: list[StreamEvent] = []
        async for event in stream_parallel(tasks, continue_on_error=True):
            events.append(event)

        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        assert len(error_events) >= 1
        assert error_events[0].agent_name == "failing"

    async def test_duplicate_names_raises(self) -> None:
        agent_a = Agent(name="dup")
        agent_b = Agent(name="dup")
        tasks = [
            SubAgentTask(agent=agent_a, input="a"),
            SubAgentTask(agent=agent_b, input="b"),
        ]
        with pytest.raises(ValueError, match="Duplicate"):
            async for _ in stream_parallel(tasks):
                pass


# ---------------------------------------------------------------------------
# Tests: SubAgentResult properties
# ---------------------------------------------------------------------------


class TestSubAgentResult:
    def test_success_result(self) -> None:
        r = SubAgentResult(
            agent_name="test",
            status=SubAgentStatus.SUCCESS,
            output="hello",
            elapsed_seconds=1.5,
        )
        assert r.agent_name == "test"
        assert r.status == SubAgentStatus.SUCCESS
        assert r.output == "hello"
        assert r.error is None
        assert r.elapsed_seconds == 1.5

    def test_failed_result(self) -> None:
        exc = RuntimeError("boom")
        r = SubAgentResult(
            agent_name="test",
            status=SubAgentStatus.FAILED,
            error=exc,
        )
        assert r.status == SubAgentStatus.FAILED
        assert r.error is exc
        assert r.output == ""

    def test_timed_out_result(self) -> None:
        r = SubAgentResult(agent_name="test", status=SubAgentStatus.TIMED_OUT)
        assert r.status == SubAgentStatus.TIMED_OUT

    def test_cancelled_result(self) -> None:
        r = SubAgentResult(agent_name="test", status=SubAgentStatus.CANCELLED)
        assert r.status == SubAgentStatus.CANCELLED


# ---------------------------------------------------------------------------
# Tests: SubAgentError
# ---------------------------------------------------------------------------


class TestSubAgentError:
    def test_error_carries_results(self) -> None:
        results = [
            SubAgentResult(agent_name="a", status=SubAgentStatus.SUCCESS, output="ok"),
            SubAgentResult(
                agent_name="b",
                status=SubAgentStatus.FAILED,
                error=RuntimeError("fail"),
            ),
        ]
        err = SubAgentError(
            "test failure",
            results=results,
            failed_agents=["b"],
        )
        assert err.results == results
        assert err.failed_agents == ["b"]
        assert "test failure" in str(err)
