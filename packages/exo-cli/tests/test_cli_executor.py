"""Tests for exo_cli.executor — local agent execution."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from rich.console import Console as RichConsole

from exo_cli.executor import ExecutionResult, ExecutorError, LocalExecutor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_agent(name: str = "test-agent") -> MagicMock:
    agent = MagicMock()
    agent.name = name
    return agent


def _mock_run_result(
    output: str = "Hello!",
    steps: int = 1,
    prompt_tokens: int = 10,
    output_tokens: int = 5,
) -> MagicMock:
    result = MagicMock()
    result.output = output
    result.steps = steps
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.output_tokens = output_tokens
    usage.total_tokens = prompt_tokens + output_tokens
    result.usage = usage
    return result


def _silent_console() -> RichConsole:
    return RichConsole(file=MagicMock(), no_color=True)


# ---------------------------------------------------------------------------
# ExecutorError
# ---------------------------------------------------------------------------


class TestExecutorError:
    def test_is_exception(self) -> None:
        exc = ExecutorError("boom")
        assert isinstance(exc, Exception)
        assert str(exc) == "boom"


# ---------------------------------------------------------------------------
# ExecutionResult
# ---------------------------------------------------------------------------


class TestExecutionResultCreation:
    def test_defaults(self) -> None:
        r = ExecutionResult(output="hi")
        assert r.output == "hi"
        assert r.steps == 0
        assert r.elapsed == 0.0
        assert r.usage == {}
        assert r.raw is None

    def test_full(self) -> None:
        raw = object()
        r = ExecutionResult(
            output="ok",
            steps=3,
            elapsed=1.5,
            usage={"total_tokens": 100},
            raw=raw,
        )
        assert r.output == "ok"
        assert r.steps == 3
        assert r.elapsed == 1.5
        assert r.usage == {"total_tokens": 100}
        assert r.raw is raw

    def test_usage_is_copy(self) -> None:
        r = ExecutionResult(output="x", usage={"total_tokens": 5})
        u = r.usage
        u["total_tokens"] = 999
        assert r.usage["total_tokens"] == 5


class TestExecutionResultSummary:
    def test_basic(self) -> None:
        r = ExecutionResult(output="x", steps=2)
        assert "2 step(s)" in r.summary()

    def test_with_elapsed(self) -> None:
        r = ExecutionResult(output="x", steps=1, elapsed=2.3)
        s = r.summary()
        assert "1 step(s)" in s
        assert "2.3s" in s

    def test_with_tokens(self) -> None:
        r = ExecutionResult(output="x", steps=1, usage={"total_tokens": 150})
        s = r.summary()
        assert "150 tokens" in s

    def test_repr(self) -> None:
        r = ExecutionResult(output="hi", steps=1)
        assert "ExecutionResult" in repr(r)
        assert "hi" in repr(r)


# ---------------------------------------------------------------------------
# LocalExecutor — init
# ---------------------------------------------------------------------------


class TestLocalExecutorInit:
    def test_defaults(self) -> None:
        agent = _mock_agent()
        ex = LocalExecutor(agent=agent)
        assert ex.agent is agent
        assert ex.timeout == 0.0
        assert ex.verbose is False

    def test_custom(self) -> None:
        agent = _mock_agent()
        console = _silent_console()
        ex = LocalExecutor(
            agent=agent,
            provider="prov",
            timeout=30.0,
            max_retries=5,
            console=console,
            verbose=True,
        )
        assert ex.timeout == 30.0
        assert ex.verbose is True

    def test_repr(self) -> None:
        agent = _mock_agent("my-agent")
        ex = LocalExecutor(agent=agent, timeout=10.0)
        r = repr(ex)
        assert "LocalExecutor" in r
        assert "my-agent" in r
        assert "10.0" in r


# ---------------------------------------------------------------------------
# LocalExecutor — execute
# ---------------------------------------------------------------------------


class TestLocalExecutorExecute:
    async def test_basic_execution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = _mock_agent()
        raw = _mock_run_result("Hello world", steps=2)

        mock_run = AsyncMock(return_value=raw)
        monkeypatch.setattr("exo_cli.executor.run", mock_run, raising=False)
        # Patch the import inside execute
        import exo_cli.executor as mod

        monkeypatch.setattr(mod, "__name__", mod.__name__)  # dummy
        # We need to mock the lazy import inside execute
        import exo.runner  # pyright: ignore[reportMissingImports]

        monkeypatch.setattr(exo.runner, "run", mock_run)

        ex = LocalExecutor(agent=agent, console=_silent_console())
        result = await ex.execute("hi")

        assert result.output == "Hello world"
        assert result.steps == 2
        assert result.elapsed > 0
        assert result.usage["total_tokens"] == 15
        assert result.raw is raw

    async def test_with_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = _mock_agent()
        raw = _mock_run_result()
        mock_run = AsyncMock(return_value=raw)
        import exo.runner  # pyright: ignore[reportMissingImports]

        monkeypatch.setattr(exo.runner, "run", mock_run)

        provider = MagicMock()
        ex = LocalExecutor(agent=agent, provider=provider, console=_silent_console())
        await ex.execute("test")

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["provider"] is provider

    async def test_with_messages(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = _mock_agent()
        raw = _mock_run_result()
        mock_run = AsyncMock(return_value=raw)
        import exo.runner  # pyright: ignore[reportMissingImports]

        monkeypatch.setattr(exo.runner, "run", mock_run)

        msgs = [{"role": "user", "content": "prior"}]
        ex = LocalExecutor(agent=agent, console=_silent_console())
        await ex.execute("test", messages=msgs)

        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["messages"] is msgs

    async def test_verbose_prints_summary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = _mock_agent()
        raw = _mock_run_result(steps=3)
        mock_run = AsyncMock(return_value=raw)
        import exo.runner  # pyright: ignore[reportMissingImports]

        monkeypatch.setattr(exo.runner, "run", mock_run)

        console = _silent_console()
        ex = LocalExecutor(agent=agent, console=console, verbose=True)
        result = await ex.execute("test")

        assert result.steps == 3
        # Verbose mode should have called console.print with summary
        assert console.file.write.called  # type: ignore[union-attr]

    async def test_no_usage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = _mock_agent()
        raw = MagicMock()
        raw.output = "ok"
        raw.steps = 1
        raw.usage = None
        mock_run = AsyncMock(return_value=raw)
        import exo.runner  # pyright: ignore[reportMissingImports]

        monkeypatch.setattr(exo.runner, "run", mock_run)

        ex = LocalExecutor(agent=agent, console=_silent_console())
        result = await ex.execute("test")
        assert result.usage == {}


# ---------------------------------------------------------------------------
# LocalExecutor — timeout
# ---------------------------------------------------------------------------


class TestLocalExecutorTimeout:
    async def test_timeout_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = _mock_agent()

        async def slow_run(*args: Any, **kwargs: Any) -> None:
            await asyncio.sleep(10)

        import exo.runner  # pyright: ignore[reportMissingImports]

        monkeypatch.setattr(exo.runner, "run", slow_run)

        ex = LocalExecutor(agent=agent, timeout=0.05, console=_silent_console())
        with pytest.raises(ExecutorError, match="timed out"):
            await ex.execute("test")

    async def test_no_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = _mock_agent()
        raw = _mock_run_result()
        mock_run = AsyncMock(return_value=raw)
        import exo.runner  # pyright: ignore[reportMissingImports]

        monkeypatch.setattr(exo.runner, "run", mock_run)

        ex = LocalExecutor(agent=agent, timeout=0.0, console=_silent_console())
        result = await ex.execute("test")
        assert result.output == "Hello!"


# ---------------------------------------------------------------------------
# LocalExecutor — error handling
# ---------------------------------------------------------------------------


class TestLocalExecutorErrors:
    async def test_agent_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = _mock_agent()
        mock_run = AsyncMock(side_effect=RuntimeError("LLM crashed"))
        import exo.runner  # pyright: ignore[reportMissingImports]

        monkeypatch.setattr(exo.runner, "run", mock_run)

        ex = LocalExecutor(agent=agent, console=_silent_console())
        with pytest.raises(ExecutorError, match="Agent execution failed"):
            await ex.execute("test")

    async def test_error_preserves_cause(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = _mock_agent()
        cause = ValueError("bad input")
        mock_run = AsyncMock(side_effect=cause)
        import exo.runner  # pyright: ignore[reportMissingImports]

        monkeypatch.setattr(exo.runner, "run", mock_run)

        ex = LocalExecutor(agent=agent, console=_silent_console())
        with pytest.raises(ExecutorError) as exc_info:
            await ex.execute("test")
        assert exc_info.value.__cause__ is cause


# ---------------------------------------------------------------------------
# LocalExecutor — stream
# ---------------------------------------------------------------------------


class TestLocalExecutorStream:
    async def test_basic_stream(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = _mock_agent()

        event1 = MagicMock()
        event1.text = "Hello "
        event2 = MagicMock()
        event2.text = "world"

        async def mock_stream(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            yield event1
            yield event2

        mock_run = MagicMock()
        mock_run.stream = mock_stream
        import exo.runner  # pyright: ignore[reportMissingImports]

        monkeypatch.setattr(exo.runner, "run", mock_run)

        ex = LocalExecutor(agent=agent, console=_silent_console())
        chunks: list[str] = []
        async for chunk in ex.stream("test"):
            chunks.append(chunk)

        assert chunks == ["Hello ", "world"]

    async def test_stream_no_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = _mock_agent()

        event = MagicMock()
        event.text = None

        async def mock_stream(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            yield event

        mock_run = MagicMock()
        mock_run.stream = mock_stream
        import exo.runner  # pyright: ignore[reportMissingImports]

        monkeypatch.setattr(exo.runner, "run", mock_run)

        ex = LocalExecutor(agent=agent, console=_silent_console())
        chunks: list[str] = []
        async for chunk in ex.stream("test"):
            chunks.append(chunk)

        assert chunks == []

    async def test_stream_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = _mock_agent()
        mock_run = MagicMock(spec=[])  # no .stream attribute
        import exo.runner  # pyright: ignore[reportMissingImports]

        monkeypatch.setattr(exo.runner, "run", mock_run)

        ex = LocalExecutor(agent=agent, console=_silent_console())
        with pytest.raises(ExecutorError, match="Streaming not available"):
            async for _ in ex.stream("test"):
                pass

    async def test_stream_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = _mock_agent()

        async def broken_stream(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            raise RuntimeError("connection lost")
            yield  # make it a generator  # pragma: no cover

        mock_run = MagicMock()
        mock_run.stream = broken_stream
        import exo.runner  # pyright: ignore[reportMissingImports]

        monkeypatch.setattr(exo.runner, "run", mock_run)

        ex = LocalExecutor(agent=agent, console=_silent_console())
        with pytest.raises(ExecutorError, match="Streaming failed"):
            async for _ in ex.stream("test"):
                pass

    async def test_stream_with_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = _mock_agent()
        provider = MagicMock()
        captured_kwargs: dict[str, Any] = {}

        async def mock_stream(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            captured_kwargs.update(kwargs)
            return
            yield  # pragma: no cover

        mock_run = MagicMock()
        mock_run.stream = mock_stream
        import exo.runner  # pyright: ignore[reportMissingImports]

        monkeypatch.setattr(exo.runner, "run", mock_run)

        ex = LocalExecutor(agent=agent, provider=provider, console=_silent_console())
        async for _ in ex.stream("test"):
            pass

        assert captured_kwargs.get("provider") is provider


# ---------------------------------------------------------------------------
# LocalExecutor — display helpers
# ---------------------------------------------------------------------------


class TestLocalExecutorDisplay:
    def test_print_result(self) -> None:
        agent = _mock_agent("helper")
        console = _silent_console()
        ex = LocalExecutor(agent=agent, console=console)

        result = ExecutionResult(output="Hi there", steps=1)
        ex.print_result(result)
        assert console.file.write.called  # type: ignore[union-attr]

    def test_print_result_verbose(self) -> None:
        agent = _mock_agent("helper")
        console = _silent_console()
        ex = LocalExecutor(agent=agent, console=console, verbose=True)

        result = ExecutionResult(output="Hi", steps=2, elapsed=1.0)
        ex.print_result(result)
        assert console.file.write.called  # type: ignore[union-attr]

    def test_print_error(self) -> None:
        agent = _mock_agent()
        console = _silent_console()
        ex = LocalExecutor(agent=agent, console=console)

        ex.print_error(ExecutorError("something broke"))
        assert console.file.write.called  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# LocalExecutor — default console goes to stdout (Finding #1)
# ---------------------------------------------------------------------------


class TestLocalExecutorDefaultConsole:
    def test_default_console_is_stdout(self) -> None:
        """LocalExecutor without explicit console defaults to stdout (not stderr)."""
        import sys

        from rich.console import Console as RichConsole

        agent = _mock_agent()
        ex = LocalExecutor(agent=agent)
        # Access the private console — it should target stdout (file=None means stdout in Rich)
        console_obj = ex._console  # type: ignore[attr-defined]
        assert isinstance(console_obj, RichConsole)
        # Rich Console defaults to stdout (sys.stdout) when stderr=False
        assert console_obj.file is sys.stdout or console_obj._file is None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# LocalExecutor — streaming timeout via asyncio.timeout (Finding #2)
# ---------------------------------------------------------------------------


class TestLocalExecutorStreamTimeout:
    async def test_stream_timeout_on_stall(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A stalled stream (no chunks) is cancelled by asyncio.timeout, not per-chunk polling."""
        agent = _mock_agent()

        async def stalled_stream(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            # Simulate a stall: wait forever without yielding
            await asyncio.sleep(60)
            yield MagicMock(text="never")  # pragma: no cover

        mock_run = MagicMock()
        mock_run.stream = stalled_stream
        import exo.runner  # pyright: ignore[reportMissingImports]

        monkeypatch.setattr(exo.runner, "run", mock_run)

        ex = LocalExecutor(agent=agent, timeout=0.05, console=_silent_console())
        with pytest.raises(ExecutorError, match="timed out"):
            async for _ in ex.stream("test"):
                pass  # pragma: no cover
