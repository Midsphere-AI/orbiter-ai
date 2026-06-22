"""Tests for the Error DX foundation: structured ExoError + async-noise helpers."""

import asyncio

import pytest

from exo._internal.errors import describe_exception_group, unwrap_exception_group
from exo._internal.resilience import is_cancellation, retry_async
from exo.types import ExoError


class ProviderError(ExoError):
    """Sample subclass used to confirm subclassing keeps the structured API."""


# --------------------------------------------------------------------------- #
# ExoError: backward compatibility
# --------------------------------------------------------------------------- #


def test_plain_message_is_unchanged():
    err = ExoError("something went wrong")
    assert str(err) == "something went wrong"
    assert err.message == "something went wrong"
    assert err.context == {}
    assert err.hint is None
    assert err.doc is None


def test_empty_construction():
    assert str(ExoError()) == ""


def test_subclass_inherits_structured_api():
    err = ProviderError("boom", hint="set the key")
    assert isinstance(err, ExoError)
    assert err.hint == "set the key"


# --------------------------------------------------------------------------- #
# ExoError: rich rendering
# --------------------------------------------------------------------------- #


def test_full_rendering():
    err = ProviderError(
        "Anthropic request failed after 3 retries.",
        context={"model": "anthropic:claude-opus-4", "agent": "researcher"},
        hint="Check ANTHROPIC_API_KEY is set and the model id is correct.",
        doc="https://docs.exo.dev/providers#anthropic",
    )
    rendered = str(err)
    assert "Anthropic request failed after 3 retries." in rendered
    assert "where: model='anthropic:claude-opus-4'  agent='researcher'" in rendered
    assert "→ Check ANTHROPIC_API_KEY is set and the model id is correct." in rendered
    assert "docs: https://docs.exo.dev/providers#anthropic" in rendered


def test_with_context_mutates_and_returns_self():
    err = ExoError("bad result")
    returned = err.with_context(tool="web_search", agent="researcher")
    assert returned is err
    assert err.context == {"tool": "web_search", "agent": "researcher"}
    assert "tool='web_search'" in str(err)


def test_chaining_preserves_cause():
    root = ValueError("underlying")
    try:
        try:
            raise root
        except ValueError as exc:
            raise ExoError("wrapped", hint="do the thing") from exc
    except ExoError as err:
        assert err.__cause__ is root


# --------------------------------------------------------------------------- #
# Async-noise helpers
# --------------------------------------------------------------------------- #


def test_unwrap_single_child_group():
    leaf = ValueError("real cause")
    eg = ExceptionGroup("wrapper", [leaf])
    assert unwrap_exception_group(eg) is leaf


def test_unwrap_nested_single_child_groups():
    leaf = ExoError("real cause")
    nested = ExceptionGroup("outer", [ExceptionGroup("inner", [leaf])])
    assert unwrap_exception_group(nested) is leaf


def test_unwrap_leaves_multi_error_group_intact():
    eg = ExceptionGroup("two", [ValueError("a"), TypeError("b")])
    assert unwrap_exception_group(eg) is eg


def test_unwrap_passes_through_plain_exception():
    err = ValueError("plain")
    assert unwrap_exception_group(err) is err


def test_describe_exception_group_summarizes():
    eg = ExceptionGroup("many", [ValueError("a"), TypeError("b")])
    summary = describe_exception_group(eg)
    assert summary.startswith("2 errors:")
    assert "ValueError: a" in summary
    assert "TypeError: b" in summary


def test_describe_exception_group_truncates():
    eg = ExceptionGroup("many", [ValueError(str(i)) for i in range(5)])
    summary = describe_exception_group(eg)
    assert summary.startswith("5 errors:")
    assert "+2 more" in summary


# --------------------------------------------------------------------------- #
# Resilience: is_cancellation
# --------------------------------------------------------------------------- #


def test_is_cancellation_bare():
    assert is_cancellation(asyncio.CancelledError())


def test_is_cancellation_nested_in_group():
    # CancelledError is a BaseException, so it lives in a BaseExceptionGroup
    # (this is exactly the shape a TaskGroup raises on external cancellation).
    eg = BaseExceptionGroup("g", [asyncio.CancelledError()])
    assert is_cancellation(eg)


def test_is_cancellation_false_for_normal_error():
    assert not is_cancellation(ValueError("x"))
    assert not is_cancellation(ExceptionGroup("g", [ValueError("a"), TypeError("b")]))


# --------------------------------------------------------------------------- #
# Resilience: retry_async
# --------------------------------------------------------------------------- #


async def test_retry_succeeds_first_try():
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        return "ok"

    assert await retry_async(fn, attempts=3) == "ok"
    assert calls == 1


async def test_retry_recovers_after_transient_failures():
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ConnectionError("transient")
        return "ok"

    assert await retry_async(fn, attempts=3, base_delay=0, jitter=0) == "ok"
    assert calls == 3


async def test_retry_reraises_last_error_after_exhaustion():
    async def fn():
        raise ConnectionError("still down")

    with pytest.raises(ConnectionError, match="still down"):
        await retry_async(fn, attempts=2, base_delay=0, jitter=0)


async def test_retry_never_retries_cancellation():
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await retry_async(fn, attempts=5, base_delay=0, jitter=0)
    assert calls == 1  # cancellation propagated immediately, no retries


async def test_retry_respects_retry_on_filter():
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        raise ValueError("not retryable here")

    with pytest.raises(ValueError):
        await retry_async(fn, attempts=5, base_delay=0, jitter=0, retry_on=(ConnectionError,))
    assert calls == 1  # ValueError not in retry_on → raised on first attempt


# --------------------------------------------------------------------------- #
# Cancellation must propagate out of tool execution (P0 regression guard)
# --------------------------------------------------------------------------- #


async def test_run_cancellation_propagates():
    """Cancelling an in-flight run must raise CancelledError — never return a result.

    Guards the charter rule that a broad ``except`` on the tool-execution path
    must re-raise cancellation rather than convert it into a normal/error tool
    result and let the agent carry on as if nothing happened.
    """
    from exo.agent import Agent
    from exo.runner import run
    from exo.tool import tool
    from exo.types import ToolCall, Usage

    started = asyncio.Event()

    @tool
    async def slow() -> str:
        started.set()
        await asyncio.Event().wait()  # block forever until cancelled
        return "done"

    class _Provider:
        async def complete(self, messages, **kwargs):
            class _Resp:
                content = ""
                tool_calls = [ToolCall(id="tc1", name="slow", arguments="{}")]
                usage = Usage()

            return _Resp()

    agent = Agent(name="cancel_probe", tools=[slow])
    task = asyncio.create_task(run(agent, "go", provider=_Provider()))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
