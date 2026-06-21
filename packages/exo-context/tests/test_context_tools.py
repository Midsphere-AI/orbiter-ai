"""Tests for context tools: planning, knowledge, and file tools."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

from exo.context._internal.knowledge import (  # pyright: ignore[reportMissingImports]
    KnowledgeStore,
)
from exo.context.context import Context  # pyright: ignore[reportMissingImports]
from exo.context.tools import (  # pyright: ignore[reportMissingImports]
    _ContextTool,
    get_context_tools,
    get_file_tools,
    get_knowledge_tools,
    get_planning_tools,
)
from exo.context.workspace import Workspace  # pyright: ignore[reportMissingImports]
from exo.tool import ToolError

# ── Helpers ──────────────────────────────────────────────────────────


def _ctx(state: dict[str, Any] | None = None) -> Context:
    """Create a minimal context with optional initial state."""
    ctx = Context("test-task")
    if state:
        ctx.state.update(state)
    return ctx


def _fresh_planning_tools() -> dict[str, _ContextTool]:
    """Return a fresh dict of planning tools keyed by name."""
    tools = get_planning_tools()
    return {t.name: t for t in tools}  # type: ignore[return-value]


def _fresh_knowledge_tools() -> dict[str, _ContextTool]:
    """Return a fresh dict of knowledge tools keyed by name."""
    tools = get_knowledge_tools()
    return {t.name: t for t in tools}  # type: ignore[return-value]


def _fresh_file_tools() -> dict[str, _ContextTool]:
    """Return a fresh dict of file tools keyed by name."""
    tools = get_file_tools()
    return {t.name: t for t in tools}  # type: ignore[return-value]


# ═══════════════════════════════════════════════════════════════════
# Planning tools
# ═══════════════════════════════════════════════════════════════════


class TestPlanningToolSchema:
    """Planning tools have correct schemas (no ctx param)."""

    def test_add_todo_schema(self) -> None:
        tool = _fresh_planning_tools()["add_todo"]
        schema = tool.parameters
        assert "item" in schema["properties"]
        assert "ctx" not in schema["properties"]
        assert "item" in schema.get("required", [])

    def test_get_todo_schema(self) -> None:
        tool = _fresh_planning_tools()["get_todo"]
        schema = tool.parameters
        assert "ctx" not in schema.get("properties", {})
        assert "required" not in schema  # no required params

    def test_complete_todo_schema(self) -> None:
        tool = _fresh_planning_tools()["complete_todo"]
        schema = tool.parameters
        assert "index" in schema["properties"]
        assert "ctx" not in schema["properties"]

    def test_tool_names(self) -> None:
        tools = _fresh_planning_tools()
        assert "add_todo" in tools
        assert "get_todo" in tools
        assert "complete_todo" in tools


class TestAddTodo:
    """add_todo mutates context state."""

    async def test_add_first_todo(self) -> None:
        ctx = _ctx()
        tools = _fresh_planning_tools()
        add_todo = tools["add_todo"]
        add_todo.bind(ctx)
        result = await add_todo.execute(item="Write tests")
        assert "Write tests" in result
        todos = ctx.state.get("todos")
        assert len(todos) == 1
        assert todos[0]["item"] == "Write tests"
        assert todos[0]["done"] is False

    async def test_add_multiple_todos(self) -> None:
        ctx = _ctx()
        tools = _fresh_planning_tools()
        add_todo = tools["add_todo"]
        add_todo.bind(ctx)
        await add_todo.execute(item="First")
        await add_todo.execute(item="Second")
        todos = ctx.state.get("todos")
        assert len(todos) == 2
        assert todos[0]["item"] == "First"
        assert todos[1]["item"] == "Second"


class TestCompleteTodo:
    """complete_todo marks items as done."""

    async def test_complete_existing(self) -> None:
        ctx = _ctx({"todos": [{"item": "Task A", "done": False}]})
        tools = _fresh_planning_tools()
        complete_todo = tools["complete_todo"]
        complete_todo.bind(ctx)
        result = await complete_todo.execute(index=0)
        assert "done" in result
        assert ctx.state.get("todos")[0]["done"] is True

    async def test_complete_invalid_index(self) -> None:
        ctx = _ctx({"todos": [{"item": "Task A", "done": False}]})
        tools = _fresh_planning_tools()
        complete_todo = tools["complete_todo"]
        complete_todo.bind(ctx)
        result = await complete_todo.execute(index=5)
        assert "Invalid index" in result

    async def test_complete_empty_list(self) -> None:
        ctx = _ctx()
        tools = _fresh_planning_tools()
        complete_todo = tools["complete_todo"]
        complete_todo.bind(ctx)
        result = await complete_todo.execute(index=0)
        assert "No todos" in result


class TestGetTodo:
    """get_todo returns formatted checklist."""

    async def test_empty_todos(self) -> None:
        ctx = _ctx()
        tools = _fresh_planning_tools()
        get_todo = tools["get_todo"]
        get_todo.bind(ctx)
        result = await get_todo.execute()
        assert result == "No todos."

    async def test_with_todos(self) -> None:
        ctx = _ctx(
            {
                "todos": [
                    {"item": "Task A", "done": False},
                    {"item": "Task B", "done": True},
                ]
            }
        )
        tools = _fresh_planning_tools()
        get_todo = tools["get_todo"]
        get_todo.bind(ctx)
        result = await get_todo.execute()
        assert "[ ] Task A" in result
        assert "[x] Task B" in result


# ═══════════════════════════════════════════════════════════════════
# Knowledge tools
# ═══════════════════════════════════════════════════════════════════


class TestKnowledgeToolSchema:
    """Knowledge tools have correct schemas."""

    def test_get_knowledge_schema(self) -> None:
        tool = _fresh_knowledge_tools()["get_knowledge"]
        schema = tool.parameters
        assert "name" in schema["properties"]
        assert "ctx" not in schema["properties"]

    def test_grep_knowledge_schema(self) -> None:
        tool = _fresh_knowledge_tools()["grep_knowledge"]
        schema = tool.parameters
        assert "name" in schema["properties"]
        assert "pattern" in schema["properties"]
        assert "ctx" not in schema["properties"]

    def test_search_knowledge_schema(self) -> None:
        tool = _fresh_knowledge_tools()["search_knowledge"]
        schema = tool.parameters
        assert "query" in schema["properties"]
        assert "ctx" not in schema["properties"]


class TestGetKnowledge:
    """get_knowledge retrieves artifacts from workspace."""

    async def test_no_workspace(self) -> None:
        ctx = _ctx()
        tool = _fresh_knowledge_tools()["get_knowledge"]
        tool.bind(ctx)
        result = await tool.execute(name="doc")
        assert "No workspace" in result

    async def test_artifact_not_found(self) -> None:
        ws = Workspace("test-ws")
        ctx = _ctx({"workspace": ws})
        tool = _fresh_knowledge_tools()["get_knowledge"]
        tool.bind(ctx)
        result = await tool.execute(name="missing")
        assert "not found" in result

    async def test_artifact_found(self) -> None:
        ws = Workspace("test-ws")
        await ws.write("doc", "Hello world content")
        ctx = _ctx({"workspace": ws})
        tool = _fresh_knowledge_tools()["get_knowledge"]
        tool.bind(ctx)
        result = await tool.execute(name="doc")
        assert result == "Hello world content"


class TestGrepKnowledge:
    """grep_knowledge searches artifact lines by regex."""

    async def test_no_workspace(self) -> None:
        ctx = _ctx()
        tool = _fresh_knowledge_tools()["grep_knowledge"]
        tool.bind(ctx)
        result = await tool.execute(name="doc", pattern="test")
        assert "No workspace" in result

    async def test_artifact_not_found(self) -> None:
        ws = Workspace("test-ws")
        ctx = _ctx({"workspace": ws})
        tool = _fresh_knowledge_tools()["grep_knowledge"]
        tool.bind(ctx)
        result = await tool.execute(name="missing", pattern="test")
        assert "not found" in result

    async def test_matching_lines(self) -> None:
        ws = Workspace("test-ws")
        await ws.write("doc", "line one\nline two test\nline three\nline four test")
        ctx = _ctx({"workspace": ws})
        tool = _fresh_knowledge_tools()["grep_knowledge"]
        tool.bind(ctx)
        result = await tool.execute(name="doc", pattern="test")
        assert "2: line two test" in result
        assert "4: line four test" in result
        assert "line one" not in result

    async def test_no_matches(self) -> None:
        ws = Workspace("test-ws")
        await ws.write("doc", "hello\nworld")
        ctx = _ctx({"workspace": ws})
        tool = _fresh_knowledge_tools()["grep_knowledge"]
        tool.bind(ctx)
        result = await tool.execute(name="doc", pattern="xyz")
        assert "No matches" in result

    async def test_invalid_regex(self) -> None:
        ws = Workspace("test-ws")
        await ws.write("doc", "hello")
        ctx = _ctx({"workspace": ws})
        tool = _fresh_knowledge_tools()["grep_knowledge"]
        tool.bind(ctx)
        result = await tool.execute(name="doc", pattern="[invalid")
        assert "Invalid regex" in result


class TestSearchKnowledge:
    """search_knowledge uses KnowledgeStore for keyword search."""

    async def test_no_knowledge_store(self) -> None:
        ctx = _ctx()
        tool = _fresh_knowledge_tools()["search_knowledge"]
        tool.bind(ctx)
        result = await tool.execute(query="test")
        assert "No knowledge store" in result

    async def test_no_results(self) -> None:
        ks = KnowledgeStore()
        ks.add("doc", "hello world")
        ctx = _ctx({"knowledge_store": ks})
        tool = _fresh_knowledge_tools()["search_knowledge"]
        tool.bind(ctx)
        result = await tool.execute(query="xyznotfound")
        assert "No results" in result

    async def test_with_results(self) -> None:
        ks = KnowledgeStore()
        ks.add("doc", "Python programming language guide")
        ctx = _ctx({"knowledge_store": ks})
        tool = _fresh_knowledge_tools()["search_knowledge"]
        tool.bind(ctx)
        result = await tool.execute(query="python")
        assert "doc#" in result
        assert "score=" in result


# ═══════════════════════════════════════════════════════════════════
# File tools
# ═══════════════════════════════════════════════════════════════════


class TestFileToolSchema:
    """File tool has correct schema."""

    def test_read_file_schema(self) -> None:
        tool = _fresh_file_tools()["read_file"]
        schema = tool.parameters
        assert "path" in schema["properties"]
        assert "ctx" not in schema["properties"]
        assert "path" in schema.get("required", [])


class TestReadFile:
    """read_file reads from working directory."""

    async def test_no_working_dir(self) -> None:
        ctx = _ctx()
        tool = _fresh_file_tools()["read_file"]
        tool.bind(ctx)
        result = await tool.execute(path="test.txt")
        assert "No working directory" in result

    async def test_file_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = _ctx({"working_dir": tmpdir})
            tool = _fresh_file_tools()["read_file"]
            tool.bind(ctx)
            result = await tool.execute(path="nonexistent.txt")
            assert "not found" in result

    async def test_read_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "hello.txt"
            p.write_text("Hello from file!", encoding="utf-8")
            ctx = _ctx({"working_dir": tmpdir})
            tool = _fresh_file_tools()["read_file"]
            tool.bind(ctx)
            result = await tool.execute(path="hello.txt")
            assert result == "Hello from file!"

    async def test_path_traversal_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = _ctx({"working_dir": tmpdir})
            tool = _fresh_file_tools()["read_file"]
            tool.bind(ctx)
            result = await tool.execute(path="../../../etc/passwd")
            assert "Access denied" in result

    async def test_subdirectory_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sub = Path(tmpdir) / "subdir"
            sub.mkdir()
            f = sub / "data.txt"
            f.write_text("nested content", encoding="utf-8")
            ctx = _ctx({"working_dir": tmpdir})
            tool = _fresh_file_tools()["read_file"]
            tool.bind(ctx)
            result = await tool.execute(path="subdir/data.txt")
            assert result == "nested content"


# ═══════════════════════════════════════════════════════════════════
# Context tool binding
# ═══════════════════════════════════════════════════════════════════


class TestContextToolBinding:
    """_ContextTool binding and error handling."""

    async def test_unbound_raises(self) -> None:
        # Create a fresh tool instance to ensure it's unbound
        async def dummy(ctx: Any) -> str:
            return "ok"

        t = _ContextTool(dummy, name="dummy")
        with pytest.raises(ToolError, match="requires a bound context"):
            await t.execute()

    def test_bind_returns_self(self) -> None:
        ctx = _ctx()
        tool = _fresh_planning_tools()["add_todo"]
        result = tool.bind(ctx)
        assert result is tool

    def test_to_schema(self) -> None:
        tool = _fresh_planning_tools()["add_todo"]
        schema = tool.to_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "add_todo"
        assert "ctx" not in schema["function"]["parameters"].get("properties", {})


# ═══════════════════════════════════════════════════════════════════
# Factory functions
# ═══════════════════════════════════════════════════════════════════


class TestFactoryFunctions:
    """get_*_tools() return the expected tool lists."""

    def test_get_planning_tools(self) -> None:
        tools = get_planning_tools()
        assert len(tools) == 3
        names = {t.name for t in tools}
        assert names == {"add_todo", "complete_todo", "get_todo"}

    def test_get_knowledge_tools(self) -> None:
        tools = get_knowledge_tools()
        assert len(tools) == 3
        names = {t.name for t in tools}
        assert names == {"get_knowledge", "grep_knowledge", "search_knowledge"}

    def test_get_file_tools(self) -> None:
        tools = get_file_tools()
        assert len(tools) == 1
        assert tools[0].name == "read_file"

    def test_get_context_tools(self) -> None:
        tools = get_context_tools()
        assert len(tools) == 7
        names = {t.name for t in tools}
        assert "add_todo" in names
        assert "get_knowledge" in names
        assert "read_file" in names
