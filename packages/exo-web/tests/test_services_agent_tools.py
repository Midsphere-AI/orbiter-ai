"""Tests for tool execution inside exo_web.services.agent_runtime._resolve_tools.

Covers audit findings C-7 / L-15:
- 'function'-type tool with code executes for real via the sandbox and returns
  the actual stdout result (not a stub or fake success).
- A tool with no executable code (non-'function' type, or empty code) returns
  a structured not_implemented JSON payload — never a fake success.
- Sandbox execution failures are surfaced as an error JSON payload.
- Tool schema stored in DB is applied to the FunctionTool.parameters field.
- tools_json edge cases: empty list, invalid JSON, empty code string.

NOTE: This file name was chosen to avoid colliding with the existing
test_agent_runtime.py which tests AgentService.build_agent / run_agent / stream_agent.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, patch

from exo_web.services.agent_runtime import _resolve_tools

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class FakeRow:
    """Minimal aiosqlite.Row stand-in."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def keys(self) -> Any:
        return self._data.keys()

    def __iter__(self) -> Any:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


def _make_mock_db(rows: list[dict[str, Any]]) -> AsyncMock:
    db = AsyncMock()
    cursor = AsyncMock()
    fake_rows = [FakeRow(r) for r in rows]
    cursor.fetchone = AsyncMock(return_value=fake_rows[0] if fake_rows else None)
    cursor.fetchall = AsyncMock(return_value=fake_rows)
    db.execute = AsyncMock(return_value=cursor)
    db.commit = AsyncMock()
    return db


@contextmanager
def _db_ctx(rows: list[dict[str, Any]]):  # type: ignore[return]
    """Patch get_db so _resolve_tools sees the given rows."""
    mock_db = _make_mock_db(rows)
    with patch("exo_web.services.agent_runtime.get_db") as mock_get_db:
        mock_get_db.return_value.__aenter__ = AsyncMock(return_value=mock_db)
        mock_get_db.return_value.__aexit__ = AsyncMock(return_value=False)
        yield mock_db


def _make_tool_row(
    *,
    name: str = "my_tool",
    description: str = "A tool",
    tool_type: str = "function",
    code: str = "",
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": "tool-001",
        "name": name,
        "description": description,
        "category": "test",
        "schema_json": json.dumps(schema or {}),
        "code": code,
        "tool_type": tool_type,
        "usage_count": 0,
        "project_id": "proj-001",
        "user_id": "user-001",
        "created_at": "2025-01-01T00:00:00",
    }


async def _resolve_with_row(row: dict[str, Any]) -> list[Any]:
    """Convenience: resolve a single tool row and return the tool list."""
    with _db_ctx([row]):
        return await _resolve_tools('["tool-001"]', "proj-001", "user-001")


# ---------------------------------------------------------------------------
# _resolve_tools — basic structure
# ---------------------------------------------------------------------------


class TestResolveToolsStructure:
    async def test_empty_tools_json_returns_empty_list(self) -> None:
        tools = await _resolve_tools("[]", "proj-001", "user-001")
        assert tools == []

    async def test_invalid_json_returns_empty_list(self) -> None:
        tools = await _resolve_tools("not-json", "proj-001", "user-001")
        assert tools == []

    async def test_null_tools_json_returns_empty_list(self) -> None:
        tools = await _resolve_tools("null", "proj-001", "user-001")
        assert tools == []

    async def test_tool_name_and_description_set_correctly(self) -> None:
        row = _make_tool_row(name="add_numbers", description="Adds two numbers")
        tools = await _resolve_with_row(row)
        assert len(tools) == 1
        assert tools[0].name == "add_numbers"
        assert tools[0].description == "Adds two numbers"

    async def test_schema_applied_to_function_tool(self) -> None:
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        row = _make_tool_row(
            name="search",
            code="print('result')",
            tool_type="function",
            schema=schema,
        )
        tools = await _resolve_with_row(row)
        assert len(tools) == 1
        assert tools[0].parameters == schema

    async def test_empty_schema_json_does_not_crash(self) -> None:
        """When schema_json is '{}', FunctionTool is still created without error."""
        row = _make_tool_row(name="noop", code="print('hi')", tool_type="function", schema={})
        tools = await _resolve_with_row(row)
        assert tools[0].name == "noop"


# ---------------------------------------------------------------------------
# _resolve_tools — 'function' type with code: real sandbox execution (C-7/L-15)
# ---------------------------------------------------------------------------


class TestFunctionToolSandboxExecution:
    async def test_function_tool_executes_code_and_returns_stdout(self) -> None:
        """A function tool with code executes for real in the sandbox."""
        code = "print('sandbox result')"
        row = _make_tool_row(name="echo_tool", code=code, tool_type="function")
        tools = await _resolve_with_row(row)
        assert len(tools) == 1
        result = await tools[0].execute()
        assert result == "sandbox result"

    async def test_function_tool_injects_kwargs_as_variables(self) -> None:
        """kwargs passed to the tool function are bound as variables in the sandbox."""
        code = "print(x + y)"
        row = _make_tool_row(name="adder", code=code, tool_type="function")
        tools = await _resolve_with_row(row)
        result = await tools[0].execute(x=3, y=4)
        assert result == "7"

    async def test_function_tool_with_string_kwarg(self) -> None:
        """String kwargs are injected with repr() so they are valid Python literals."""
        code = "print(greeting.upper())"
        row = _make_tool_row(name="shout", code=code, tool_type="function")
        tools = await _resolve_with_row(row)
        result = await tools[0].execute(greeting="hello")
        assert result == "HELLO"

    async def test_function_tool_execution_failure_returns_error_json(self) -> None:
        """When sandbox execution fails, an error JSON is returned (not raised)."""
        code = "raise RuntimeError('deliberate failure')"
        row = _make_tool_row(name="broken_tool", code=code, tool_type="function")
        tools = await _resolve_with_row(row)
        raw = await tools[0].execute()
        payload = json.loads(raw)
        assert payload["status"] == "error"
        assert payload["tool"] == "broken_tool"
        assert "error" in payload

    async def test_function_tool_no_stdout_returns_ok_json(self) -> None:
        """Code that produces no stdout returns a structured ok JSON."""
        code = "x = 1 + 1  # silent"
        row = _make_tool_row(name="silent_tool", code=code, tool_type="function")
        tools = await _resolve_with_row(row)
        raw = await tools[0].execute()
        payload = json.loads(raw)
        assert payload["status"] == "ok"
        assert payload["tool"] == "silent_tool"

    async def test_function_tool_multiline_code(self) -> None:
        """Multi-line tool code executes correctly."""
        code = "total = 0\nfor i in range(5):\n    total += i\nprint(total)"
        row = _make_tool_row(name="summer", code=code, tool_type="function")
        tools = await _resolve_with_row(row)
        result = await tools[0].execute()
        assert result == "10"

    async def test_function_tool_kwarg_with_special_chars(self) -> None:
        """Kwargs with quotes/backslashes are serialised safely into preamble."""
        code = "print(len(text))"
        row = _make_tool_row(name="len_tool", code=code, tool_type="function")
        tools = await _resolve_with_row(row)
        # Single quotes in the value — repr() must handle them
        result = await tools[0].execute(text='it\'s "tricky"')
        assert result == "13"


# ---------------------------------------------------------------------------
# _resolve_tools — unsupported/no-code tool types return not_implemented (C-7/L-15)
# ---------------------------------------------------------------------------


class TestUnsupportedToolTypeReturnsNotImplemented:
    async def test_http_type_with_no_code_returns_not_implemented(self) -> None:
        """An 'http' tool with no code returns not_implemented status."""
        row = _make_tool_row(name="http_tool", tool_type="http", code="")
        tools = await _resolve_with_row(row)
        raw = await tools[0].execute(some_param="value")
        payload = json.loads(raw)
        assert payload["status"] == "not_implemented"
        assert payload["tool"] == "http_tool"
        assert payload["tool_type"] == "http"

    async def test_mcp_type_returns_not_implemented(self) -> None:
        row = _make_tool_row(name="mcp_tool", tool_type="mcp", code="")
        tools = await _resolve_with_row(row)
        raw = await tools[0].execute()
        payload = json.loads(raw)
        assert payload["status"] == "not_implemented"
        assert payload["tool_type"] == "mcp"

    async def test_schema_type_returns_not_implemented(self) -> None:
        row = _make_tool_row(name="schema_tool", tool_type="schema", code="")
        tools = await _resolve_with_row(row)
        raw = await tools[0].execute()
        payload = json.loads(raw)
        assert payload["status"] == "not_implemented"

    async def test_function_type_with_empty_code_returns_not_implemented(self) -> None:
        """A function-type tool with an empty code string is treated as unimplemented."""
        row = _make_tool_row(name="empty_fn", tool_type="function", code="")
        tools = await _resolve_with_row(row)
        raw = await tools[0].execute()
        payload = json.loads(raw)
        assert payload["status"] == "not_implemented"

    async def test_not_implemented_includes_args(self) -> None:
        """not_implemented response includes the kwargs passed to the tool."""
        row = _make_tool_row(name="stub_tool", tool_type="http", code="")
        tools = await _resolve_with_row(row)
        raw = await tools[0].execute(query="test query", limit=5)
        payload = json.loads(raw)
        assert payload["status"] == "not_implemented"
        assert "args" in payload
        assert payload["args"]["query"] == "test query"
        assert payload["args"]["limit"] == 5

    async def test_not_implemented_is_not_fake_success(self) -> None:
        """The stub must return 'not_implemented', never 'ok' or 'success'."""
        row = _make_tool_row(name="unsupported", tool_type="webhook", code="")
        tools = await _resolve_with_row(row)
        raw = await tools[0].execute()
        payload = json.loads(raw)
        assert payload["status"] not in {"ok", "success"}
        assert payload["status"] == "not_implemented"


# ---------------------------------------------------------------------------
# _resolve_tools — default description fallback
# ---------------------------------------------------------------------------


class TestResolveToolsDescriptionFallback:
    async def test_empty_description_uses_tool_name_fallback(self) -> None:
        row = _make_tool_row(name="my_fn", description="", code="print('hi')", tool_type="function")
        tools = await _resolve_with_row(row)
        assert "my_fn" in tools[0].description
