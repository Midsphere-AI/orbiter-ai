"""Context tools — planning, knowledge, and file tools for agent self-management.

Provides ``@tool``-decorated functions that let agents manipulate their own
context during execution: manage a todo checklist, search knowledge artifacts,
and read files from a working directory.

All tools operate on a :class:`Context` instance passed via a ``ctx`` keyword
argument at call time.  The ``ctx`` parameter is excluded from the generated
JSON schema so that LLMs only see the user-facing parameters.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

from exo.tool import Tool, ToolError, _extract_description, _generate_schema
from exo.tool_result import tool_error, tool_ok

logger = logging.getLogger(__name__)

# ── Helper: build a tool that receives ctx at call time ─────────────


class _ContextTool(Tool):
    """A tool whose ``execute`` injects a ``ctx`` from bound state.

    The ``ctx`` parameter is NOT part of the JSON schema — it is injected
    by the caller (runner / agent) at execution time.
    """

    __slots__ = ("_ctx", "_fn", "_is_context_tool", "description", "name", "parameters")

    def __init__(
        self,
        fn: Any,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        self._fn = fn
        self.name = name or fn.__name__
        self.description = description or _extract_description(fn)
        # Generate schema but strip out 'ctx' — it's injected, not user-visible
        schema = _generate_schema(fn)
        props = schema.get("properties", {})
        props.pop("ctx", None)
        req = schema.get("required", [])
        schema["required"] = [r for r in req if r != "ctx"]
        if not schema["required"]:
            del schema["required"]
        schema["properties"] = props
        self.parameters = schema
        self._ctx: Any = None
        self._is_context_tool: bool = True

    def bind(self, ctx: Any) -> _ContextTool:
        """Bind a :class:`Context` instance for subsequent calls."""
        self._ctx = ctx
        return self

    async def execute(self, **kwargs: Any) -> str | dict[str, Any]:
        if self._ctx is None:
            logger.error("tool %r executed without bound context", self.name)
            raise ToolError(f"Tool '{self.name}' requires a bound context (call .bind(ctx) first)")
        logger.debug("executing context tool %r", self.name)
        return await self._fn(ctx=self._ctx, **kwargs)


# ── Planning tool ───────────────────────────────────────────────────


async def _add_todo(ctx: Any, item: str) -> str:
    """Add a todo item to the planning checklist.

    Args:
        item: The todo item text to add.
    """
    try:
        todos: list[dict[str, Any]] = ctx.state.get("todos", [])
        todos = list(todos)  # defensive copy
        todos.append({"item": item, "done": False})
        ctx.state.set("todos", todos)
        return tool_ok(f"Added todo: {item}")
    except Exception as exc:
        return tool_error(
            f"Failed to add todo: {exc}",
            hint="Retry the add_todo call.",
        )


async def _complete_todo(ctx: Any, index: int) -> str:
    """Mark a todo item as completed by its index (0-based).

    Args:
        index: The 0-based index of the todo item to mark done.
    """
    try:
        todos: list[dict[str, Any]] = ctx.state.get("todos", [])
        if not todos:
            return tool_error(
                "No todos to complete",
                hint="Add todos first using add_todo before completing them.",
            )
        todos = [dict(t) for t in todos]  # defensive copy
        if index < 0 or index >= len(todos):
            return tool_error(
                f"Invalid index {index}",
                hint=(
                    "Use get_todo to see current items, then call complete_todo with a valid index."
                ),
                valid_range=f"0-{len(todos) - 1}",
            )
        todos[index]["done"] = True
        ctx.state.set("todos", todos)
        return tool_ok(f"Marked todo #{index} as done: {todos[index]['item']}")
    except Exception as exc:
        return tool_error(
            f"Failed to complete todo: {exc}",
            hint="Retry the complete_todo call.",
        )


async def _get_todo(ctx: Any) -> str:
    """Retrieve the current todo checklist.

    Returns the full checklist in markdown format.
    """
    try:
        todos: list[dict[str, Any]] = ctx.state.get("todos", [])
        if not todos:
            return "No todos."
        lines: list[str] = []
        for i, t in enumerate(todos):
            mark = "x" if t.get("done") else " "
            lines.append(f"{i}. [{mark}] {t.get('item', '')}")
        return "\n".join(lines)
    except Exception as exc:
        return tool_error(
            f"Failed to retrieve todos: {exc}",
            hint="Retry the get_todo call.",
        )


def get_planning_tools() -> list[Tool]:
    """Return fresh planning tool instances (safe for per-agent binding)."""
    return [
        _ContextTool(
            _add_todo, name="add_todo", description="Add a todo item to the planning checklist."
        ),
        _ContextTool(
            _complete_todo,
            name="complete_todo",
            description="Mark a todo item as completed by index.",
        ),
        _ContextTool(
            _get_todo, name="get_todo", description="Retrieve the current todo checklist."
        ),
    ]


# ── Knowledge tool ──────────────────────────────────────────────────


async def _get_knowledge(ctx: Any, name: str) -> str:
    """Retrieve a knowledge artifact by name.

    Args:
        name: The artifact name to retrieve.
    """
    try:
        workspace = ctx.state.get("workspace")
        if workspace is None:
            return tool_error(
                "No workspace attached to context",
                hint="The agent has no workspace. Knowledge artifacts are not available.",
            )
        content = workspace.read(name)
        if content is None:
            return tool_error(
                f"Artifact '{name}' not found",
                hint="Use search_knowledge to find available artifacts.",
            )
        return content
    except Exception as exc:
        return tool_error(
            f"Failed to retrieve artifact: {exc}",
            hint="Retry get_knowledge or try search_knowledge to find artifacts.",
        )


async def _grep_knowledge(ctx: Any, name: str, pattern: str) -> str:
    """Search a knowledge artifact for lines matching a regex pattern.

    Args:
        name: The artifact name to search in.
        pattern: A regex pattern to match against each line.
    """
    try:
        workspace = ctx.state.get("workspace")
        if workspace is None:
            return tool_error(
                "No workspace attached to context",
                hint="The agent has no workspace. Knowledge artifacts are not available.",
            )
        content = workspace.read(name)
        if content is None:
            return tool_error(
                f"Artifact '{name}' not found",
                hint="Use search_knowledge to find available artifacts.",
            )
        # SECURITY: overly long patterns can cause catastrophic backtracking in
        # some regex engines.  Cap pattern length as a cheap guard.
        if len(pattern) > 500:
            return tool_error(
                "Pattern too long (max 500 characters)",
                hint="Use a shorter, more targeted regex pattern.",
            )
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return tool_error(
                f"Invalid regex pattern: {e}",
                hint=(
                    "Fix the regex pattern syntax. Common issues: unmatched "
                    "brackets [], unescaped special characters."
                ),
            )
        lines = content.splitlines()
        matches: list[str] = []
        for i, line in enumerate(lines):
            if compiled.search(line):
                matches.append(f"{i + 1}: {line}")
        if not matches:
            return tool_error(
                f"No matches for pattern '{pattern}' in '{name}'",
                hint=(
                    "Try a broader pattern or use get_knowledge to see the full artifact content."
                ),
            )
        return "\n".join(matches)
    except Exception as exc:
        return tool_error(
            f"Failed to search artifact: {exc}",
            hint="Retry grep_knowledge or try search_knowledge.",
        )


async def _search_knowledge(ctx: Any, query: str, top_k: int = 5) -> str:
    """Search across all knowledge artifacts using keyword matching.

    Args:
        query: The search query string.
        top_k: Maximum number of results to return.
    """
    try:
        knowledge_store = ctx.state.get("knowledge_store")
        if knowledge_store is None:
            return tool_error(
                "No knowledge store attached to context",
                hint="The agent has no knowledge store configured.",
            )
        results = knowledge_store.search(query, top_k=top_k)
        if not results:
            return tool_error(
                f"No results for query '{query}'",
                hint=(
                    "Try different search terms or use get_knowledge with a specific artifact name."
                ),
            )
        lines: list[str] = []
        for r in results:
            lines.append(
                f"[{r.chunk.artifact_name}#{r.chunk.index}] (score={r.score:.2f})\n{r.chunk.content[:200]}"
            )
        return "\n---\n".join(lines)
    except Exception as exc:
        return tool_error(
            f"Failed to search knowledge: {exc}",
            hint="Retry search_knowledge with different terms.",
        )


def get_knowledge_tools() -> list[Tool]:
    """Return fresh knowledge tool instances (safe for per-agent binding)."""
    return [
        _ContextTool(
            _get_knowledge,
            name="get_knowledge",
            description="Retrieve a knowledge artifact by name.",
        ),
        _ContextTool(
            _grep_knowledge,
            name="grep_knowledge",
            description="Search a knowledge artifact for regex matches.",
        ),
        _ContextTool(
            _search_knowledge,
            name="search_knowledge",
            description="Search across all knowledge artifacts.",
        ),
    ]


# ── File tool ───────────────────────────────────────────────────────


async def _read_file(ctx: Any, path: str) -> str:
    """Read a file from the working directory.

    Args:
        path: Relative path to the file within the working directory.
    """
    try:
        working_dir = ctx.state.get("working_dir")
        if working_dir is None:
            return tool_error(
                "No working directory set in context",
                hint="The agent has no working directory configured.",
            )
        base = Path(working_dir).resolve()
        target = (base / path).resolve()
        # Prevent path traversal outside working directory
        if not target.is_relative_to(base):
            logger.warning("path traversal blocked: %r resolved outside working dir %s", path, base)
            return tool_error(
                f"Access denied: '{path}' is outside the working directory",
                hint="Use a relative path within the working directory.",
            )
        if not target.is_file():
            return tool_error(
                f"File not found: '{path}'",
                hint=("Check the file path. Use a relative path from the working directory root."),
            )
        try:
            return await asyncio.to_thread(target.read_text, encoding="utf-8")
        except UnicodeDecodeError:
            return tool_error(
                f"Cannot read '{path}' as text",
                hint="The file may be binary. Try a different file.",
            )
        except OSError as e:
            return tool_error(
                f"Error reading '{path}': {e}",
                hint="Check file permissions and retry.",
            )
    except Exception as exc:
        return tool_error(
            f"read_file failed: {exc}",
            hint="Retry with a different file path.",
        )


def get_file_tools() -> list[Tool]:
    """Return fresh file tool instances (safe for per-agent binding)."""
    return [
        _ContextTool(
            _read_file, name="read_file", description="Read a file from the working directory."
        ),
    ]


# ── All context tools ──────────────────────────────────────────────


def get_context_tools() -> list[Tool]:
    """Return all context tools (planning + knowledge + file)."""
    return get_planning_tools() + get_knowledge_tools() + get_file_tools()
