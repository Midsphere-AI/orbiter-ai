"""Tests for exo_mcp_cli.commands.tool — tool list and call CLI commands."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from exo_mcp_cli.main import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path, servers: dict | None = None) -> Path:
    """Write an mcp.json and return its path."""
    if servers is None:
        servers = {
            "test-srv": {
                "transport": "stdio",
                "command": "echo",
                "args": ["hi"],
            }
        }
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": servers}))
    return cfg


def _make_mock_tool(name: str, description: str, schema: dict | None = None) -> MagicMock:
    """Create a mock MCP tool object."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.inputSchema = schema or {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    return tool


def _make_mock_result(text: str, is_error: bool = False) -> MagicMock:
    """Create a mock CallToolResult."""
    content_item = MagicMock()
    content_item.text = text
    result = MagicMock()
    result.isError = is_error
    result.content = [content_item]
    return result


def _mock_session(
    tools: list | None = None,
    call_result: MagicMock | None = None,
) -> MagicMock:
    """Build a mock ClientSession with list_tools and call_tool."""
    session = MagicMock()
    tools_response = MagicMock()
    tools_response.tools = tools or []
    session.list_tools = AsyncMock(return_value=tools_response)
    session.call_tool = AsyncMock(return_value=call_result or _make_mock_result("ok"))
    return session


@asynccontextmanager
async def _mock_connect_factory(
    session: MagicMock,
) -> AsyncIterator:
    """Return an async context manager factory that always yields *session*."""
    yield session


# ---------------------------------------------------------------------------
# tool list
# ---------------------------------------------------------------------------


class TestToolList:
    def test_list_shows_tools_table(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        tools = [
            _make_mock_tool("search", "Search the web"),
            _make_mock_tool("read_file", "Read a file from disk"),
        ]
        session = _mock_session(tools=tools)

        with patch(
            "exo_mcp_cli.commands.tool.connect_to_server",
            side_effect=lambda *a, **kw: _mock_connect_factory(session),
        ):
            result = runner.invoke(
                app,
                [
                    "--config",
                    str(cfg),
                    "tool",
                    "list",
                    "test-srv",
                ],
            )

        assert result.exit_code == 0
        assert "search" in result.output
        assert "read_file" in result.output
        assert "Search the web" in result.output

    def test_list_json_output(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        tools = [
            _make_mock_tool(
                "greet",
                "Say hello",
                {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            ),
        ]
        session = _mock_session(tools=tools)

        with patch(
            "exo_mcp_cli.commands.tool.connect_to_server",
            side_effect=lambda *a, **kw: _mock_connect_factory(session),
        ):
            result = runner.invoke(
                app,
                [
                    "--config",
                    str(cfg),
                    "tool",
                    "list",
                    "test-srv",
                    "--json",
                ],
            )

        assert result.exit_code == 0
        assert "greet" in result.output
        assert "Say hello" in result.output

    def test_list_unknown_server_exits_1(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        result = runner.invoke(
            app,
            [
                "--config",
                str(cfg),
                "tool",
                "list",
                "nonexistent-server",
            ],
        )
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_list_no_tools_available(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        session = _mock_session(tools=[])

        with patch(
            "exo_mcp_cli.commands.tool.connect_to_server",
            side_effect=lambda *a, **kw: _mock_connect_factory(session),
        ):
            result = runner.invoke(
                app,
                [
                    "--config",
                    str(cfg),
                    "tool",
                    "list",
                    "test-srv",
                ],
            )

        assert result.exit_code == 0
        assert "No tools available" in result.output


# ---------------------------------------------------------------------------
# tool call
# ---------------------------------------------------------------------------


class TestToolCall:
    def test_call_with_arg(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        call_result = _make_mock_result("Found 3 results")
        session = _mock_session(call_result=call_result)

        with patch(
            "exo_mcp_cli.commands.tool.connect_to_server",
            side_effect=lambda *a, **kw: _mock_connect_factory(session),
        ):
            result = runner.invoke(
                app,
                [
                    "--config",
                    str(cfg),
                    "tool",
                    "call",
                    "test-srv",
                    "search",
                    "--arg",
                    "query=hello world",
                ],
            )

        assert result.exit_code == 0
        assert "Found 3 results" in result.output
        session.call_tool.assert_called_once_with("search", {"query": "hello world"})

    def test_call_with_json_args(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        call_result = _make_mock_result("Done")
        session = _mock_session(call_result=call_result)

        with patch(
            "exo_mcp_cli.commands.tool.connect_to_server",
            side_effect=lambda *a, **kw: _mock_connect_factory(session),
        ):
            result = runner.invoke(
                app,
                [
                    "--config",
                    str(cfg),
                    "tool",
                    "call",
                    "test-srv",
                    "multi_arg",
                    "--json",
                    '{"key": "value", "count": 5}',
                ],
            )

        assert result.exit_code == 0
        session.call_tool.assert_called_once_with("multi_arg", {"key": "value", "count": 5})

    def test_call_invalid_json_exits_1(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        result = runner.invoke(
            app,
            [
                "--config",
                str(cfg),
                "tool",
                "call",
                "test-srv",
                "mytool",
                "--json",
                "{not valid json",
            ],
        )
        assert result.exit_code == 1
        assert "invalid json" in result.output.lower() or "Error" in result.output

    def test_call_json_must_be_object(self, tmp_path: Path) -> None:
        """--json with a non-object (e.g., list) exits 1."""
        cfg = _make_config(tmp_path)
        result = runner.invoke(
            app,
            [
                "--config",
                str(cfg),
                "tool",
                "call",
                "test-srv",
                "mytool",
                "--json",
                '["not", "an", "object"]',
            ],
        )
        assert result.exit_code == 1
        assert "object" in result.output.lower() or "Error" in result.output

    def test_call_raw_outputs_json(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        call_result = _make_mock_result("raw output text")
        session = _mock_session(call_result=call_result)

        with patch(
            "exo_mcp_cli.commands.tool.connect_to_server",
            side_effect=lambda *a, **kw: _mock_connect_factory(session),
        ):
            result = runner.invoke(
                app,
                [
                    "--config",
                    str(cfg),
                    "tool",
                    "call",
                    "test-srv",
                    "search",
                    "--arg",
                    "q=test",
                    "--raw",
                ],
            )

        assert result.exit_code == 0
        # --raw should produce JSON with isError and content keys
        assert "isError" in result.output
        assert "content" in result.output

    def test_call_error_result_shown(self, tmp_path: Path) -> None:
        """When the tool returns isError=True, the error text is printed."""
        cfg = _make_config(tmp_path)
        call_result = _make_mock_result("Something went wrong", is_error=True)
        session = _mock_session(call_result=call_result)

        with patch(
            "exo_mcp_cli.commands.tool.connect_to_server",
            side_effect=lambda *a, **kw: _mock_connect_factory(session),
        ):
            result = runner.invoke(
                app,
                [
                    "--config",
                    str(cfg),
                    "tool",
                    "call",
                    "test-srv",
                    "broken",
                ],
            )

        assert result.exit_code == 0
        assert "Something went wrong" in result.output

    def test_call_unknown_server_exits_1(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        result = runner.invoke(
            app,
            [
                "--config",
                str(cfg),
                "tool",
                "call",
                "nope",
                "sometool",
            ],
        )
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_call_multiple_args(self, tmp_path: Path) -> None:
        """Multiple --arg flags are combined into one arguments dict."""
        cfg = _make_config(tmp_path)
        session = _mock_session()

        with patch(
            "exo_mcp_cli.commands.tool.connect_to_server",
            side_effect=lambda *a, **kw: _mock_connect_factory(session),
        ):
            result = runner.invoke(
                app,
                [
                    "--config",
                    str(cfg),
                    "tool",
                    "call",
                    "test-srv",
                    "multi",
                    "--arg",
                    "a=1",
                    "--arg",
                    "b=2",
                ],
            )

        assert result.exit_code == 0
        session.call_tool.assert_called_once_with("multi", {"a": "1", "b": "2"})

    def test_call_invalid_arg_format_exits_1(self, tmp_path: Path) -> None:
        """--arg without '=' exits 1."""
        cfg = _make_config(tmp_path)
        result = runner.invoke(
            app,
            [
                "--config",
                str(cfg),
                "tool",
                "call",
                "test-srv",
                "mytool",
                "--arg",
                "no-equals-sign",
            ],
        )
        assert result.exit_code == 1
        assert "KEY=VALUE" in result.output or "Error" in result.output


# ---------------------------------------------------------------------------
# tool call — injected arguments
# ---------------------------------------------------------------------------


class TestToolCallInject:
    def test_inject_flag_adds_args(self, tmp_path: Path) -> None:
        """--inject key=value is merged into the arguments dict."""
        cfg = _make_config(tmp_path)
        session = _mock_session()

        with patch(
            "exo_mcp_cli.commands.tool.connect_to_server",
            side_effect=lambda *a, **kw: _mock_connect_factory(session),
        ):
            result = runner.invoke(
                app,
                [
                    "--config",
                    str(cfg),
                    "tool",
                    "call",
                    "test-srv",
                    "search",
                    "--arg",
                    "query=hello",
                    "--inject",
                    "trace_id=abc123",
                ],
            )

        assert result.exit_code == 0
        session.call_tool.assert_called_once_with(
            "search", {"query": "hello", "trace_id": "abc123"}
        )

    def test_inject_env_var(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """EXO_MCP_TOOL_INJECT env var provides baseline injected args."""

        cfg = _make_config(tmp_path)
        session = _mock_session()
        monkeypatch.setenv("EXO_MCP_TOOL_INJECT", '{"user_id": "u42"}')

        with patch(
            "exo_mcp_cli.commands.tool.connect_to_server",
            side_effect=lambda *a, **kw: _mock_connect_factory(session),
        ):
            result = runner.invoke(
                app,
                [
                    "--config",
                    str(cfg),
                    "tool",
                    "call",
                    "test-srv",
                    "fetch",
                    "--arg",
                    "url=http://example.com",
                ],
            )

        assert result.exit_code == 0
        call_args = session.call_tool.call_args[0]
        assert call_args[1]["user_id"] == "u42"
        assert call_args[1]["url"] == "http://example.com"

    def test_inject_flag_overrides_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--inject overrides overlapping keys from EXO_MCP_TOOL_INJECT."""

        cfg = _make_config(tmp_path)
        session = _mock_session()
        monkeypatch.setenv("EXO_MCP_TOOL_INJECT", '{"user_id": "env-val"}')

        with patch(
            "exo_mcp_cli.commands.tool.connect_to_server",
            side_effect=lambda *a, **kw: _mock_connect_factory(session),
        ):
            result = runner.invoke(
                app,
                [
                    "--config",
                    str(cfg),
                    "tool",
                    "call",
                    "test-srv",
                    "fetch",
                    "--inject",
                    "user_id=flag-val",
                ],
            )

        assert result.exit_code == 0
        call_args = session.call_tool.call_args[0]
        assert call_args[1]["user_id"] == "flag-val"

    def test_arg_overrides_inject(self, tmp_path: Path) -> None:
        """--arg has higher precedence than --inject."""
        cfg = _make_config(tmp_path)
        session = _mock_session()

        with patch(
            "exo_mcp_cli.commands.tool.connect_to_server",
            side_effect=lambda *a, **kw: _mock_connect_factory(session),
        ):
            result = runner.invoke(
                app,
                [
                    "--config",
                    str(cfg),
                    "tool",
                    "call",
                    "test-srv",
                    "t",
                    "--inject",
                    "k=injected",
                    "--arg",
                    "k=explicit",
                ],
            )

        assert result.exit_code == 0
        call_args = session.call_tool.call_args[0]
        assert call_args[1]["k"] == "explicit"

    def test_json_overrides_inject(self, tmp_path: Path) -> None:
        """--json has higher precedence than --inject."""
        cfg = _make_config(tmp_path)
        session = _mock_session()

        with patch(
            "exo_mcp_cli.commands.tool.connect_to_server",
            side_effect=lambda *a, **kw: _mock_connect_factory(session),
        ):
            result = runner.invoke(
                app,
                [
                    "--config",
                    str(cfg),
                    "tool",
                    "call",
                    "test-srv",
                    "t",
                    "--inject",
                    "k=low",
                    "--json",
                    '{"k": "high"}',
                ],
            )

        assert result.exit_code == 0
        call_args = session.call_tool.call_args[0]
        assert call_args[1]["k"] == "high"

    def test_inject_invalid_format_exits_1(self, tmp_path: Path) -> None:
        """--inject without '=' exits 1."""
        cfg = _make_config(tmp_path)
        result = runner.invoke(
            app,
            [
                "--config",
                str(cfg),
                "tool",
                "call",
                "test-srv",
                "t",
                "--inject",
                "no-equals",
            ],
        )
        assert result.exit_code == 1
        assert "KEY=VALUE" in result.output or "Error" in result.output

    def test_malformed_env_inject_warns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Malformed EXO_MCP_TOOL_INJECT prints a yellow warning and is ignored."""

        cfg = _make_config(tmp_path)
        session = _mock_session()
        monkeypatch.setenv("EXO_MCP_TOOL_INJECT", "not-json{{{")

        with patch(
            "exo_mcp_cli.commands.tool.connect_to_server",
            side_effect=lambda *a, **kw: _mock_connect_factory(session),
        ):
            result = runner.invoke(
                app,
                [
                    "--config",
                    str(cfg),
                    "tool",
                    "call",
                    "test-srv",
                    "t",
                    "--arg",
                    "x=1",
                ],
            )

        assert result.exit_code == 0
        assert "Warning" in result.output and "EXO_MCP_TOOL_INJECT" in result.output
        call_args = session.call_tool.call_args[0]
        assert call_args[1] == {"x": "1"}
