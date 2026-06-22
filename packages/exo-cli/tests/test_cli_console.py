"""Tests for exo_cli.console — interactive REPL."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console as RichConsole

from exo_cli.console import (
    ConsoleError,
    InteractiveConsole,
    format_agents_table,
    parse_command,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_agent(name: str = "test", model: str = "openai:gpt-4o") -> Any:
    """Create a minimal mock agent."""
    agent = MagicMock()
    agent.name = name
    agent.model = model
    agent.describe.return_value = {"name": name, "instructions": "Test agent"}
    agent.run = AsyncMock()
    return agent


def _make_result(output: str = "Hello!") -> Any:
    """Create a mock run result."""
    result = MagicMock()
    result.output = output
    return result


def _make_console(**kwargs: Any) -> InteractiveConsole:
    """Create an InteractiveConsole with mock agents."""
    agents = kwargs.pop("agents", {"alpha": _make_agent("alpha"), "beta": _make_agent("beta")})
    run_fn = kwargs.pop("run_fn", AsyncMock(return_value=_make_result()))
    return InteractiveConsole(
        agents=agents,
        run_fn=run_fn,
        console=RichConsole(file=MagicMock(), no_color=True),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# parse_command
# ---------------------------------------------------------------------------


class TestParseCommand:
    def test_slash_command(self) -> None:
        cmd, arg = parse_command("/help")
        assert cmd == "help"
        assert arg == ""

    def test_slash_command_with_arg(self) -> None:
        cmd, arg = parse_command("/switch alpha")
        assert cmd == "switch"
        assert arg == "alpha"

    def test_non_command(self) -> None:
        cmd, arg = parse_command("hello world")
        assert cmd == ""
        assert arg == "hello world"

    def test_empty(self) -> None:
        cmd, arg = parse_command("")
        assert cmd == ""
        assert arg == ""

    def test_whitespace(self) -> None:
        cmd, arg = parse_command("  /exit  ")
        assert cmd == "exit"
        assert arg == ""

    def test_case_insensitive(self) -> None:
        cmd, arg = parse_command("/EXIT")
        assert cmd == "exit"
        assert arg == ""

    def test_multi_word_arg(self) -> None:
        cmd, arg = parse_command("/switch my agent name")
        assert cmd == "switch"
        assert arg == "my agent name"


# ---------------------------------------------------------------------------
# format_agents_table
# ---------------------------------------------------------------------------


class TestFormatAgentsTable:
    def test_empty(self) -> None:
        table = format_agents_table({})
        assert table.row_count == 0

    def test_single_agent(self) -> None:
        agent = _make_agent("alpha", "openai:gpt-4o")
        table = format_agents_table({"alpha": agent})
        assert table.row_count == 1

    def test_multiple_agents(self) -> None:
        agents = {
            "a": _make_agent("a"),
            "b": _make_agent("b"),
            "c": _make_agent("c"),
        }
        table = format_agents_table(agents)
        assert table.row_count == 3

    def test_agent_without_describe(self) -> None:
        agent = MagicMock(spec=[])
        agent.model = "test"
        table = format_agents_table({"x": agent})
        assert table.row_count == 1


# ---------------------------------------------------------------------------
# InteractiveConsole — init
# ---------------------------------------------------------------------------


class TestConsoleInit:
    def test_requires_agents(self) -> None:
        with pytest.raises(ConsoleError, match="At least one agent"):
            InteractiveConsole(agents={}, run_fn=AsyncMock())

    def test_defaults(self) -> None:
        ic = _make_console()
        assert ic.current_agent_name == "alpha"
        assert "alpha" in ic.agents
        assert "beta" in ic.agents

    def test_streaming_without_stream_fn(self) -> None:
        ic = _make_console(streaming=True, stream_fn=None)
        # Should fallback to non-streaming since no stream_fn provided
        assert ic._streaming is False

    def test_streaming_with_stream_fn(self) -> None:
        stream_fn = AsyncMock()
        ic = _make_console(streaming=True, stream_fn=stream_fn)
        assert ic._streaming is True


# ---------------------------------------------------------------------------
# InteractiveConsole — commands
# ---------------------------------------------------------------------------


class TestConsoleCommands:
    def test_help(self) -> None:
        ic = _make_console()
        ic._handle_help()
        # Should not raise

    def test_agents(self) -> None:
        ic = _make_console()
        ic._handle_agents()

    def test_switch_success(self) -> None:
        ic = _make_console()
        assert ic.current_agent_name == "alpha"
        result = ic._handle_switch("beta")
        assert result is True
        assert ic.current_agent_name == "beta"

    def test_switch_unknown(self) -> None:
        ic = _make_console()
        result = ic._handle_switch("nonexistent")
        assert result is False
        assert ic.current_agent_name == "alpha"

    def test_switch_empty(self) -> None:
        ic = _make_console()
        result = ic._handle_switch("")
        assert result is False

    def test_info(self) -> None:
        ic = _make_console()
        ic._handle_info()

    def test_clear(self) -> None:
        ic = _make_console()
        ic._handle_clear()


# ---------------------------------------------------------------------------
# InteractiveConsole — execution
# ---------------------------------------------------------------------------


class TestConsoleExecution:
    async def test_execute_normal(self) -> None:
        run_fn = AsyncMock(return_value=_make_result("world"))
        ic = _make_console(run_fn=run_fn)
        await ic._execute("hello")
        run_fn.assert_awaited_once_with(ic.current_agent, "hello")

    async def test_execute_error(self) -> None:
        run_fn = AsyncMock(side_effect=RuntimeError("boom"))
        ic = _make_console(run_fn=run_fn)
        # Should not raise — error is printed
        await ic._execute("hello")

    async def test_execute_streaming(self) -> None:
        event = MagicMock()
        event.text = "chunk"

        async def mock_stream(agent: Any, inp: str) -> Any:
            yield event

        ic = _make_console(streaming=True, stream_fn=mock_stream)
        await ic._execute("hello")

    async def test_execute_result_no_output_attr(self) -> None:
        """Result without .output attribute falls back to str()."""
        run_fn = AsyncMock(return_value="plain string")
        ic = _make_console(run_fn=run_fn)
        await ic._execute("hello")


# ---------------------------------------------------------------------------
# InteractiveConsole — start loop
# ---------------------------------------------------------------------------


class TestConsoleLoop:
    async def test_exit_command(self) -> None:
        ic = _make_console()
        with patch.object(InteractiveConsole, "_read_input", side_effect=["/exit"]):
            await ic.start()

    async def test_quit_command(self) -> None:
        ic = _make_console()
        with patch.object(InteractiveConsole, "_read_input", side_effect=["/quit"]):
            await ic.start()

    async def test_eof(self) -> None:
        ic = _make_console()
        with patch.object(InteractiveConsole, "_read_input", side_effect=[None]):
            await ic.start()

    async def test_empty_input_skipped(self) -> None:
        ic = _make_console()
        with patch.object(InteractiveConsole, "_read_input", side_effect=["", "/exit"]):
            await ic.start()

    async def test_unknown_command(self) -> None:
        ic = _make_console()
        with patch.object(InteractiveConsole, "_read_input", side_effect=["/foo", "/exit"]):
            await ic.start()

    async def test_help_then_exit(self) -> None:
        ic = _make_console()
        with patch.object(InteractiveConsole, "_read_input", side_effect=["/help", "/exit"]):
            await ic.start()

    async def test_agents_then_exit(self) -> None:
        ic = _make_console()
        with patch.object(InteractiveConsole, "_read_input", side_effect=["/agents", "/exit"]):
            await ic.start()

    async def test_switch_then_exit(self) -> None:
        ic = _make_console()
        with patch.object(InteractiveConsole, "_read_input", side_effect=["/switch beta", "/exit"]):
            await ic.start()
            assert ic.current_agent_name == "beta"

    async def test_info_then_exit(self) -> None:
        ic = _make_console()
        with patch.object(InteractiveConsole, "_read_input", side_effect=["/info", "/exit"]):
            await ic.start()

    async def test_clear_then_exit(self) -> None:
        ic = _make_console()
        with patch.object(InteractiveConsole, "_read_input", side_effect=["/clear", "/exit"]):
            await ic.start()

    async def test_user_message_executes(self) -> None:
        run_fn = AsyncMock(return_value=_make_result("hi"))
        ic = _make_console(run_fn=run_fn)
        with patch.object(InteractiveConsole, "_read_input", side_effect=["hello world", "/exit"]):
            await ic.start()
        run_fn.assert_awaited_once()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestConsoleProperties:
    def test_current_agent(self) -> None:
        ic = _make_console()
        assert ic.current_agent is ic.agents["alpha"]

    def test_agents_returns_copy(self) -> None:
        ic = _make_console()
        a = ic.agents
        a["new"] = MagicMock()
        assert "new" not in ic.agents


# ---------------------------------------------------------------------------
# isatty caching
# ---------------------------------------------------------------------------


class TestIsttyCaching:
    def test_is_tty_cached_at_init(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_is_tty is set once at __init__, not re-checked per _read_input call."""
        import sys as _sys

        mock_isatty = MagicMock(return_value=True)
        monkeypatch.setattr(_sys.stdin, "isatty", mock_isatty)
        ic = _make_console()
        # isatty should have been called exactly once (at init)
        assert ic._is_tty is True
        init_calls = mock_isatty.call_count
        assert init_calls == 1
        # Accessing the cached value does NOT call isatty again
        _ = ic._is_tty
        _ = ic._is_tty
        assert mock_isatty.call_count == 1  # still exactly one call from init
