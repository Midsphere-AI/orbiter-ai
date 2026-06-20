"""Tests for ShortTermMemory."""

from __future__ import annotations

import pytest

from exo.memory.base import (  # pyright: ignore[reportMissingImports]
    AIMemory,
    ExoMemoryError,
    HumanMemory,
    MemoryMetadata,
    MemoryStatus,
    MemoryStore,
    SystemMemory,
    ToolMemory,
)
from exo.memory.short_term import ShortTermMemory  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _meta(**kw: str) -> MemoryMetadata:
    return MemoryMetadata(**kw)


def _sys(content: str = "system", meta: MemoryMetadata | None = None) -> SystemMemory:
    return SystemMemory(content=content, metadata=meta or MemoryMetadata())


def _human(content: str = "hello", meta: MemoryMetadata | None = None) -> HumanMemory:
    return HumanMemory(content=content, metadata=meta or MemoryMetadata())


def _ai(
    content: str = "hi",
    meta: MemoryMetadata | None = None,
    *,
    tool_calls: list[dict[str, str]] | None = None,
) -> AIMemory:
    return AIMemory(content=content, tool_calls=tool_calls or [], metadata=meta or MemoryMetadata())


def _tool(
    content: str = "result",
    meta: MemoryMetadata | None = None,
    *,
    tool_call_id: str = "tc1",
) -> ToolMemory:
    return ToolMemory(
        content=content,
        tool_call_id=tool_call_id,
        tool_name="test_tool",
        metadata=meta or MemoryMetadata(),
    )


# ===========================================================================
# Construction
# ===========================================================================


class TestShortTermMemoryInit:
    def test_defaults(self) -> None:
        mem = ShortTermMemory()
        assert mem.scope == "task"
        assert mem.max_rounds == 0
        assert len(mem) == 0

    def test_custom(self) -> None:
        mem = ShortTermMemory(scope="user", max_rounds=5)
        assert mem.scope == "user"
        assert mem.max_rounds == 5

    def test_invalid_scope(self) -> None:
        with pytest.raises(ExoMemoryError, match="Invalid scope"):
            ShortTermMemory(scope="invalid")

    def test_repr(self) -> None:
        mem = ShortTermMemory(scope="session", max_rounds=3)
        assert "session" in repr(mem)
        assert "max_rounds=3" in repr(mem)

    def test_protocol_conformance(self) -> None:
        assert isinstance(ShortTermMemory(), MemoryStore)


# ===========================================================================
# Add + Get
# ===========================================================================


class TestAddGet:
    async def test_add_and_get(self) -> None:
        mem = ShortTermMemory()
        item = _human("hello")
        await mem.add(item)
        assert len(mem) == 1
        retrieved = await mem.get(item.id)
        assert retrieved is item

    async def test_get_missing(self) -> None:
        mem = ShortTermMemory()
        assert await mem.get("nonexistent") is None

    async def test_add_multiple(self) -> None:
        mem = ShortTermMemory()
        await mem.add(_sys())
        await mem.add(_human())
        await mem.add(_ai())
        assert len(mem) == 3


# ===========================================================================
# Search — basic
# ===========================================================================


class TestSearchBasic:
    async def test_search_all(self) -> None:
        mem = ShortTermMemory()
        await mem.add(_human("hello world"))
        await mem.add(_ai("goodbye world"))
        results = await mem.search()
        assert len(results) == 2

    async def test_search_by_query(self) -> None:
        mem = ShortTermMemory()
        await mem.add(_human("hello world"))
        await mem.add(_ai("goodbye"))
        results = await mem.search(query="hello")
        assert len(results) == 1
        assert results[0].content == "hello world"

    async def test_search_by_type(self) -> None:
        mem = ShortTermMemory()
        await mem.add(_sys())
        await mem.add(_human())
        await mem.add(_ai())
        results = await mem.search(memory_type="human")
        assert len(results) == 1

    async def test_search_by_status(self) -> None:
        mem = ShortTermMemory()
        item = _human()
        item.status = MemoryStatus.DRAFT
        await mem.add(item)
        await mem.add(_human())  # default ACCEPTED
        results = await mem.search(status=MemoryStatus.DRAFT)
        assert len(results) == 1

    async def test_search_limit(self) -> None:
        mem = ShortTermMemory()
        for i in range(20):
            await mem.add(_human(f"msg {i}"))
        results = await mem.search(limit=5)
        assert len(results) == 5


# ===========================================================================
# Search — scope filtering
# ===========================================================================


class TestScopeFiltering:
    async def test_task_scope(self) -> None:
        mem = ShortTermMemory(scope="task")
        m1 = _meta(task_id="t1", session_id="s1", user_id="u1")
        m2 = _meta(task_id="t2", session_id="s1", user_id="u1")
        await mem.add(_human("t1", m1))
        await mem.add(_human("t2", m2))
        results = await mem.search(
            metadata=MemoryMetadata(task_id="t1", session_id="s1", user_id="u1")
        )
        assert len(results) == 1
        assert results[0].content == "t1"

    async def test_session_scope(self) -> None:
        mem = ShortTermMemory(scope="session")
        await mem.add(_human("t1", _meta(task_id="t1", session_id="s1")))
        await mem.add(_human("t2", _meta(task_id="t2", session_id="s1")))
        await mem.add(_human("t3", _meta(task_id="t3", session_id="s2")))
        # Session scope: task_id filter is ignored, only session_id matters
        results = await mem.search(metadata=MemoryMetadata(session_id="s1"))
        assert len(results) == 2

    async def test_user_scope(self) -> None:
        mem = ShortTermMemory(scope="user")
        await mem.add(_human("u1s1", _meta(session_id="s1", user_id="u1")))
        await mem.add(_human("u1s2", _meta(session_id="s2", user_id="u1")))
        await mem.add(_human("u2s1", _meta(session_id="s1", user_id="u2")))
        # User scope: session_id filter is ignored
        results = await mem.search(metadata=MemoryMetadata(user_id="u1"))
        assert len(results) == 2

    async def test_agent_id_filter(self) -> None:
        mem = ShortTermMemory()
        await mem.add(_human("a1", _meta(agent_id="agent-1")))
        await mem.add(_human("a2", _meta(agent_id="agent-2")))
        results = await mem.search(metadata=MemoryMetadata(agent_id="agent-1"))
        assert len(results) == 1


# ===========================================================================
# Windowing
# ===========================================================================


class TestWindowing:
    async def test_no_windowing(self) -> None:
        mem = ShortTermMemory(max_rounds=0)
        for i in range(10):
            await mem.add(_human(f"h{i}"))
            await mem.add(_ai(f"a{i}"))
        results = await mem.search(limit=100)
        assert len(results) == 20

    async def test_window_rounds(self) -> None:
        mem = ShortTermMemory(max_rounds=2)
        for i in range(5):
            await mem.add(_human(f"h{i}"))
            await mem.add(_ai(f"a{i}"))
        results = await mem.search(limit=100)
        # Last 2 rounds = 4 messages
        assert len(results) == 4
        assert results[0].content == "h3"
        assert results[-1].content == "a4"

    async def test_system_messages_preserved(self) -> None:
        mem = ShortTermMemory(max_rounds=1)
        await mem.add(_sys("system prompt"))
        for i in range(5):
            await mem.add(_human(f"h{i}"))
            await mem.add(_ai(f"a{i}"))
        results = await mem.search(limit=100)
        # System + last 1 round (2 messages) = 3
        assert len(results) == 3
        assert results[0].content == "system prompt"
        assert results[1].content == "h4"
        assert results[2].content == "a4"

    async def test_window_fewer_than_max(self) -> None:
        mem = ShortTermMemory(max_rounds=10)
        await mem.add(_human("h0"))
        await mem.add(_ai("a0"))
        results = await mem.search(limit=100)
        assert len(results) == 2


# ===========================================================================
# Tool call integrity
# ===========================================================================


class TestToolCallIntegrity:
    async def test_complete_pair_preserved(self) -> None:
        mem = ShortTermMemory()
        await mem.add(_human("do something"))
        await mem.add(_ai("", tool_calls=[{"id": "tc1", "name": "test"}]))
        await mem.add(_tool("result", tool_call_id="tc1"))
        await mem.add(_ai("done"))
        results = await mem.search(limit=100)
        assert len(results) == 4

    async def test_trailing_ai_with_tool_calls_removed(self) -> None:
        mem = ShortTermMemory()
        await mem.add(_human("do something"))
        await mem.add(_ai("", tool_calls=[{"id": "tc1", "name": "test"}]))
        # No tool result follows — this trailing AI message should be removed
        results = await mem.search(limit=100)
        assert len(results) == 1
        assert results[0].memory_type == "human"

    async def test_orphan_tool_result_removed(self) -> None:
        mem = ShortTermMemory()
        await mem.add(_human("hello"))
        await mem.add(_tool("orphan result", tool_call_id="missing"))
        # Trailing orphan tool result with no matching AI call should be removed
        results = await mem.search(limit=100)
        assert len(results) == 1
        assert results[0].memory_type == "human"

    async def test_complete_chain_with_multiple_tools(self) -> None:
        mem = ShortTermMemory()
        await mem.add(_human("do two things"))
        await mem.add(
            _ai(
                "",
                tool_calls=[
                    {"id": "tc1", "name": "tool_a"},
                    {"id": "tc2", "name": "tool_b"},
                ],
            )
        )
        await mem.add(_tool("r1", tool_call_id="tc1"))
        await mem.add(_tool("r2", tool_call_id="tc2"))
        await mem.add(_ai("all done"))
        results = await mem.search(limit=100)
        assert len(results) == 5

    async def test_ai_without_tool_calls_preserved(self) -> None:
        mem = ShortTermMemory()
        await mem.add(_human("hello"))
        await mem.add(_ai("world"))
        results = await mem.search(limit=100)
        assert len(results) == 2


# ===========================================================================
# Clear
# ===========================================================================


class TestClear:
    async def test_clear_all(self) -> None:
        mem = ShortTermMemory()
        await mem.add(_human("a"))
        await mem.add(_human("b"))
        count = await mem.clear()
        assert count == 2
        assert len(mem) == 0

    async def test_clear_with_metadata(self) -> None:
        mem = ShortTermMemory(scope="task")
        await mem.add(_human("t1", _meta(task_id="t1")))
        await mem.add(_human("t2", _meta(task_id="t2")))
        count = await mem.clear(metadata=MemoryMetadata(task_id="t1"))
        assert count == 1
        assert len(mem) == 1

    async def test_clear_empty(self) -> None:
        mem = ShortTermMemory()
        count = await mem.clear()
        assert count == 0


# ===========================================================================
# Integration
# ===========================================================================


class TestIntegration:
    async def test_scope_and_windowing_combined(self) -> None:
        mem = ShortTermMemory(scope="task", max_rounds=2)
        m = _meta(task_id="t1")
        m2 = _meta(task_id="t2")
        await mem.add(_sys("sys", m))
        for i in range(5):
            await mem.add(_human(f"h{i}", m))
            await mem.add(_ai(f"a{i}", m))
        # Add items from another task
        await mem.add(_human("other", m2))

        results = await mem.search(
            metadata=MemoryMetadata(task_id="t1"),
            limit=100,
        )
        # System + last 2 rounds (4 msgs) = 5
        assert len(results) == 5
        assert results[0].content == "sys"

    async def test_scope_windowing_and_integrity(self) -> None:
        mem = ShortTermMemory(scope="task", max_rounds=3)
        m = _meta(task_id="t1")
        await mem.add(_human("q1", m))
        await mem.add(_ai("a1", m))
        await mem.add(_human("q2", m))
        await mem.add(_ai("", m, tool_calls=[{"id": "tc1", "name": "x"}]))
        await mem.add(_tool("r1", m, tool_call_id="tc1"))
        await mem.add(_ai("final", m))
        await mem.add(_human("q3", m))
        await mem.add(_ai("", m, tool_calls=[{"id": "tc2", "name": "y"}]))
        # Trailing AI with tool calls and no result → removed

        results = await mem.search(
            metadata=MemoryMetadata(task_id="t1"),
            limit=100,
        )
        # After windowing (3 rounds) and integrity filtering
        # The trailing AI with pending tool calls is removed
        assert all(
            not (isinstance(r, AIMemory) and r.tool_calls and r is results[-1]) for r in results
        )
        # Last item should not be an AI with unmatched tool calls
        last = results[-1]
        if isinstance(last, AIMemory):
            assert not last.tool_calls

    async def test_full_conversation_lifecycle(self) -> None:
        mem = ShortTermMemory(scope="task", max_rounds=0)
        m = _meta(task_id="task-1", session_id="sess-1", user_id="user-1")

        # System init
        await mem.add(_sys("You are a helpful assistant.", m))
        # Round 1: simple exchange
        await mem.add(_human("What is 2+2?", m))
        await mem.add(_ai("4", m))
        # Round 2: tool use
        await mem.add(_human("Search for Python docs", m))
        await mem.add(_ai("", m, tool_calls=[{"id": "tc1", "name": "search"}]))
        await mem.add(_tool("Python 3.11 docs found", m, tool_call_id="tc1"))
        await mem.add(_ai("I found the Python 3.11 docs.", m))

        results = await mem.search(
            metadata=MemoryMetadata(task_id="task-1", session_id="sess-1", user_id="user-1"),
            limit=100,
        )
        assert len(results) == 7
        assert results[0].memory_type == "system"
        assert results[-1].content == "I found the Python 3.11 docs."
