"""Tests for PostgresMemoryStore backend.

Unit tests use mocked asyncpg connections. Integration tests with a real
Postgres server are skipped unless a ``POSTGRES_DSN`` environment variable
is set (e.g. ``postgresql://localhost/exo_test``).
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

asyncpg = pytest.importorskip("asyncpg")

from exo.memory.backends._common import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    extra_fields as _extra_fields,
)
from exo.memory.backends.postgres import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    PostgresMemoryStore,
    _parse_rowcount,
)
from exo.memory.backends.postgres import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    _row_to_item_postgres as _row_to_item,
)
from exo.memory.base import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    AIMemory,
    ExoMemoryError,
    HumanMemory,
    MemoryItem,
    MemoryMetadata,
    MemoryStatus,
    MemoryStore,
    SystemMemory,
    ToolMemory,
)

# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------

POSTGRES_DSN = os.environ.get("POSTGRES_DSN", "")
requires_postgres = pytest.mark.skipif(
    not POSTGRES_DSN,
    reason="Set POSTGRES_DSN to run Postgres integration tests",
)


def _make_meta(**kw: str) -> MemoryMetadata:
    return MemoryMetadata(**kw)  # type: ignore[arg-type]


def _make_row(
    *,
    id: str = "r1",
    content: str = "hello",
    memory_type: str = "human",
    status: str = "accepted",
    metadata: dict[str, Any] | None = None,
    extra_json: dict[str, Any] | None = None,
    created_at: str = "2024-01-01T00:00:00",
    updated_at: str = "2024-01-01T00:00:00",
    deleted: int = 0,
    version: int = 1,
) -> dict[str, Any]:
    """Create a row-like dict simulating an asyncpg Record."""
    return {
        "id": id,
        "content": content,
        "memory_type": memory_type,
        "status": status,
        "metadata": metadata or {},
        "extra_json": extra_json or {},
        "created_at": created_at,
        "updated_at": updated_at,
        "deleted": deleted,
        "version": version,
    }


class FakeRecord(dict[str, Any]):
    """dict that also supports attribute-style access like asyncpg.Record."""

    def __getitem__(self, key: str) -> Any:
        return super().__getitem__(key)


def _record(**kw: Any) -> FakeRecord:
    return FakeRecord(_make_row(**kw))


def _mock_pool() -> tuple[MagicMock, AsyncMock]:
    """Create a mock asyncpg pool with context manager support.

    Returns (pool, conn) where conn is the shared connection mock.
    The pool.acquire() returns an async context manager that yields conn.
    """
    conn = AsyncMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=None)

    pool = MagicMock()
    pool.acquire.return_value = ctx
    pool.close = AsyncMock()
    return pool, conn


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_isinstance_check(self) -> None:
        store = PostgresMemoryStore()
        assert isinstance(store, MemoryStore)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    @patch("exo.memory.backends.postgres.asyncpg")
    async def test_init_and_close(self, mock_asyncpg: MagicMock) -> None:
        pool, _conn = _mock_pool()
        mock_asyncpg.create_pool = AsyncMock(return_value=pool)

        store = PostgresMemoryStore("postgresql://localhost/test")
        await store.init()
        assert store._initialized
        mock_asyncpg.create_pool.assert_awaited_once_with("postgresql://localhost/test")

        await store.close()
        assert not store._initialized
        pool.close.assert_awaited_once()

    @patch("exo.memory.backends.postgres.asyncpg")
    async def test_double_init_is_idempotent(self, mock_asyncpg: MagicMock) -> None:
        pool, _conn = _mock_pool()
        mock_asyncpg.create_pool = AsyncMock(return_value=pool)

        store = PostgresMemoryStore()
        await store.init()
        await store.init()  # should not call create_pool again
        assert mock_asyncpg.create_pool.await_count == 1
        await store.close()

    @patch("exo.memory.backends.postgres.asyncpg")
    async def test_context_manager(self, mock_asyncpg: MagicMock) -> None:
        pool, _conn = _mock_pool()
        mock_asyncpg.create_pool = AsyncMock(return_value=pool)

        async with PostgresMemoryStore() as store:
            assert store._initialized
        assert not store._initialized

    async def test_operations_before_init_raise(self) -> None:
        store = PostgresMemoryStore()
        with pytest.raises(ExoMemoryError, match="not initialized"):
            await store.add(HumanMemory(content="hello"))

    def test_repr(self) -> None:
        store = PostgresMemoryStore("postgresql://myhost/mydb")
        assert "myhost" in repr(store)
        assert "mydb" in repr(store)


# ---------------------------------------------------------------------------
# Add + Get (mocked)
# ---------------------------------------------------------------------------


class TestAddGet:
    @patch("exo.memory.backends.postgres.asyncpg")
    async def test_add_and_get(self, mock_asyncpg: MagicMock) -> None:
        pool, conn = _mock_pool()
        mock_asyncpg.create_pool = AsyncMock(return_value=pool)

        item = HumanMemory(content="hello world")
        conn.fetchrow = AsyncMock(return_value=_record(id=item.id, content="hello world"))

        async with PostgresMemoryStore() as store:
            await store.add(item)
            conn.execute.assert_awaited()  # INSERT called

            got = await store.get(item.id)
            assert got is not None
            assert got.content == "hello world"
            assert got.memory_type == "human"

    @patch("exo.memory.backends.postgres.asyncpg")
    async def test_get_nonexistent(self, mock_asyncpg: MagicMock) -> None:
        pool, conn = _mock_pool()
        mock_asyncpg.create_pool = AsyncMock(return_value=pool)
        conn.fetchrow = AsyncMock(return_value=None)

        async with PostgresMemoryStore() as store:
            assert await store.get("nonexistent") is None


# ---------------------------------------------------------------------------
# Search (mocked)
# ---------------------------------------------------------------------------


class TestSearch:
    @patch("exo.memory.backends.postgres.asyncpg")
    async def test_search_all(self, mock_asyncpg: MagicMock) -> None:
        pool, conn = _mock_pool()
        mock_asyncpg.create_pool = AsyncMock(return_value=pool)
        conn.fetch = AsyncMock(
            return_value=[
                _record(id="1", content="first"),
                _record(id="2", content="second"),
            ]
        )

        async with PostgresMemoryStore() as store:
            results = await store.search()
        assert len(results) == 2

    @patch("exo.memory.backends.postgres.asyncpg")
    async def test_search_by_query(self, mock_asyncpg: MagicMock) -> None:
        pool, conn = _mock_pool()
        mock_asyncpg.create_pool = AsyncMock(return_value=pool)
        conn.fetch = AsyncMock(return_value=[_record(content="apple pie")])

        async with PostgresMemoryStore() as store:
            results = await store.search(query="apple")
        assert len(results) == 1

        # Verify ILIKE used (case-insensitive)
        call_args = conn.fetch.call_args
        assert "ILIKE" in call_args[0][0]

    @patch("exo.memory.backends.postgres.asyncpg")
    async def test_search_by_memory_type(self, mock_asyncpg: MagicMock) -> None:
        pool, conn = _mock_pool()
        mock_asyncpg.create_pool = AsyncMock(return_value=pool)
        conn.fetch = AsyncMock(return_value=[_record(content="user msg")])

        async with PostgresMemoryStore() as store:
            results = await store.search(memory_type="human")
        assert len(results) == 1

    @patch("exo.memory.backends.postgres.asyncpg")
    async def test_search_by_status(self, mock_asyncpg: MagicMock) -> None:
        pool, conn = _mock_pool()
        mock_asyncpg.create_pool = AsyncMock(return_value=pool)
        conn.fetch = AsyncMock(
            return_value=[
                _record(content="draft item", status="draft"),
            ]
        )

        async with PostgresMemoryStore() as store:
            results = await store.search(status=MemoryStatus.DRAFT)
        assert len(results) == 1

    @patch("exo.memory.backends.postgres.asyncpg")
    async def test_search_by_metadata(self, mock_asyncpg: MagicMock) -> None:
        pool, conn = _mock_pool()
        mock_asyncpg.create_pool = AsyncMock(return_value=pool)
        meta_dict = {"user_id": "u1", "session_id": "s1"}
        conn.fetch = AsyncMock(
            return_value=[
                _record(content="match", metadata=meta_dict),
            ]
        )

        async with PostgresMemoryStore() as store:
            results = await store.search(metadata=_make_meta(user_id="u1"))
        assert len(results) == 1

        # Verify JSONB operator used
        call_args = conn.fetch.call_args
        assert "metadata->>'user_id'" in call_args[0][0]

    @patch("exo.memory.backends.postgres.asyncpg")
    async def test_search_limit(self, mock_asyncpg: MagicMock) -> None:
        pool, conn = _mock_pool()
        mock_asyncpg.create_pool = AsyncMock(return_value=pool)
        conn.fetch = AsyncMock(return_value=[_record(content=f"item {i}") for i in range(3)])

        async with PostgresMemoryStore() as store:
            results = await store.search(limit=3)
        assert len(results) == 3

        # Verify LIMIT is passed
        call_args = conn.fetch.call_args
        assert 3 in call_args[0]


# ---------------------------------------------------------------------------
# Clear (mocked)
# ---------------------------------------------------------------------------


class TestClear:
    @patch("exo.memory.backends.postgres.asyncpg")
    async def test_clear_all(self, mock_asyncpg: MagicMock) -> None:
        pool, conn = _mock_pool()
        mock_asyncpg.create_pool = AsyncMock(return_value=pool)
        conn.execute = AsyncMock(return_value="UPDATE 2")

        async with PostgresMemoryStore() as store:
            count = await store.clear()
        assert count == 2

    @patch("exo.memory.backends.postgres.asyncpg")
    async def test_clear_with_metadata(self, mock_asyncpg: MagicMock) -> None:
        pool, conn = _mock_pool()
        mock_asyncpg.create_pool = AsyncMock(return_value=pool)
        conn.execute = AsyncMock(return_value="UPDATE 1")

        async with PostgresMemoryStore() as store:
            count = await store.clear(metadata=_make_meta(user_id="u2"))
        assert count == 1

        # Verify metadata filter in query
        call_args = conn.execute.call_args
        assert "metadata->>'user_id'" in call_args[0][0]


# ---------------------------------------------------------------------------
# Count (mocked)
# ---------------------------------------------------------------------------


class TestCount:
    @patch("exo.memory.backends.postgres.asyncpg")
    async def test_count_empty(self, mock_asyncpg: MagicMock) -> None:
        pool, conn = _mock_pool()
        mock_asyncpg.create_pool = AsyncMock(return_value=pool)
        conn.fetchval = AsyncMock(return_value=0)

        async with PostgresMemoryStore() as store:
            assert await store.count() == 0

    @patch("exo.memory.backends.postgres.asyncpg")
    async def test_count_active(self, mock_asyncpg: MagicMock) -> None:
        pool, conn = _mock_pool()
        mock_asyncpg.create_pool = AsyncMock(return_value=pool)
        conn.fetchval = AsyncMock(return_value=5)

        async with PostgresMemoryStore() as store:
            assert await store.count() == 5

    @patch("exo.memory.backends.postgres.asyncpg")
    async def test_count_includes_deleted(self, mock_asyncpg: MagicMock) -> None:
        pool, conn = _mock_pool()
        mock_asyncpg.create_pool = AsyncMock(return_value=pool)
        conn.fetchval = AsyncMock(return_value=10)

        async with PostgresMemoryStore() as store:
            result = await store.count(include_deleted=True)
        assert result == 10

        # Verify no WHERE clause when including deleted
        call_args = conn.fetchval.call_args
        assert "WHERE" not in call_args[0][0]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_parse_rowcount_update(self) -> None:
        assert _parse_rowcount("UPDATE 5") == 5

    def test_parse_rowcount_zero(self) -> None:
        assert _parse_rowcount("UPDATE 0") == 0

    def test_parse_rowcount_invalid(self) -> None:
        assert _parse_rowcount("UNKNOWN") == 0

    def test_parse_rowcount_empty(self) -> None:
        assert _parse_rowcount("") == 0

    def test_extra_fields_human(self) -> None:
        item = HumanMemory(content="hi")
        assert _extra_fields(item) == {}

    def test_extra_fields_ai_with_tools(self) -> None:
        item = AIMemory(content="ok", tool_calls=[{"id": "t1"}])
        extra = _extra_fields(item)
        assert extra["tool_calls"] == [{"id": "t1"}]

    def test_extra_fields_tool(self) -> None:
        item = ToolMemory(content="result", tool_call_id="tc1", tool_name="search", is_error=True)
        extra = _extra_fields(item)
        assert extra["tool_call_id"] == "tc1"
        assert extra["tool_name"] == "search"
        assert extra["is_error"] is True


# ---------------------------------------------------------------------------
# Row-to-item reconstruction
# ---------------------------------------------------------------------------


class TestRowToItem:
    def test_human_memory(self) -> None:
        row = _record(memory_type="human", content="hello")
        item = _row_to_item(row)
        assert isinstance(item, HumanMemory)
        assert item.content == "hello"

    def test_system_memory(self) -> None:
        row = _record(memory_type="system", content="system prompt")
        item = _row_to_item(row)
        assert isinstance(item, SystemMemory)

    def test_ai_memory_with_tool_calls(self) -> None:
        row = _record(
            memory_type="ai",
            content="I'll search",
            extra_json={"tool_calls": [{"id": "tc1", "name": "search"}]},
        )
        item = _row_to_item(row)
        assert isinstance(item, AIMemory)
        assert item.tool_calls == [{"id": "tc1", "name": "search"}]

    def test_tool_memory(self) -> None:
        row = _record(
            memory_type="tool",
            content="search result",
            extra_json={
                "tool_call_id": "tc1",
                "tool_name": "search",
                "is_error": False,
            },
        )
        item = _row_to_item(row)
        assert isinstance(item, ToolMemory)
        assert item.tool_call_id == "tc1"
        assert item.tool_name == "search"
        assert item.is_error is False

    def test_unknown_type(self) -> None:
        row = _record(memory_type="custom", content="custom data")
        item = _row_to_item(row)
        assert isinstance(item, MemoryItem)
        assert item.memory_type == "custom"

    def test_metadata_from_dict(self) -> None:
        row = _record(metadata={"user_id": "alice", "session_id": "s1"})
        item = _row_to_item(row)
        assert item.metadata.user_id == "alice"
        assert item.metadata.session_id == "s1"

    def test_metadata_from_json_string(self) -> None:
        row = _record(metadata=json.dumps({"user_id": "bob"}))
        item = _row_to_item(row)
        assert item.metadata.user_id == "bob"


# ---------------------------------------------------------------------------
# GetRecent (mocked)
# ---------------------------------------------------------------------------


class TestGetRecent:
    @patch("exo.memory.backends.postgres.asyncpg")
    async def test_get_recent_returns_items(self, mock_asyncpg: MagicMock) -> None:
        pool, conn = _mock_pool()
        mock_asyncpg.create_pool = AsyncMock(return_value=pool)
        conn.fetch = AsyncMock(
            return_value=[
                _record(id="1", content="newest", created_at="2024-01-03T00:00:00"),
                _record(id="2", content="older", created_at="2024-01-01T00:00:00"),
            ]
        )

        async with PostgresMemoryStore() as store:
            results = await store.get_recent(n=2)
        assert len(results) == 2
        assert results[0].content == "newest"

        # Verify ORDER BY and LIMIT in SQL
        call_args = conn.fetch.call_args
        assert "ORDER BY created_at DESC" in call_args[0][0]
        assert 2 in call_args[0]

    @patch("exo.memory.backends.postgres.asyncpg")
    async def test_get_recent_with_metadata(self, mock_asyncpg: MagicMock) -> None:
        pool, conn = _mock_pool()
        mock_asyncpg.create_pool = AsyncMock(return_value=pool)
        conn.fetch = AsyncMock(return_value=[_record(content="scoped")])

        async with PostgresMemoryStore() as store:
            results = await store.get_recent(n=5, metadata=_make_meta(agent_id="agent1"))
        assert len(results) == 1

        # Verify metadata filter in query
        call_args = conn.fetch.call_args
        assert "metadata->>'agent_id'" in call_args[0][0]

    @patch("exo.memory.backends.postgres.asyncpg")
    async def test_get_recent_empty(self, mock_asyncpg: MagicMock) -> None:
        pool, conn = _mock_pool()
        mock_asyncpg.create_pool = AsyncMock(return_value=pool)
        conn.fetch = AsyncMock(return_value=[])

        async with PostgresMemoryStore() as store:
            results = await store.get_recent()
        assert results == []

    @patch("exo.memory.backends.postgres.asyncpg")
    async def test_get_recent_default_n(self, mock_asyncpg: MagicMock) -> None:
        pool, conn = _mock_pool()
        mock_asyncpg.create_pool = AsyncMock(return_value=pool)
        conn.fetch = AsyncMock(return_value=[])

        async with PostgresMemoryStore() as store:
            await store.get_recent()
        # Default n=10
        call_args = conn.fetch.call_args
        assert 10 in call_args[0]


# ---------------------------------------------------------------------------
# Agent end-to-end integration (mocked asyncpg)
# ---------------------------------------------------------------------------


class TestAgentIntegration:
    """Verify Agent works end-to-end with AgentMemory(long_term=PostgresMemoryStore)."""

    @patch("exo.memory.backends.postgres.asyncpg")
    async def test_agent_with_postgres_long_term_memory(self, mock_asyncpg: MagicMock) -> None:
        """Agent can be configured with PostgresMemoryStore as long_term backend."""
        pool, conn = _mock_pool()
        mock_asyncpg.create_pool = AsyncMock(return_value=pool)
        conn.execute = AsyncMock(return_value="INSERT 0 1")
        conn.fetch = AsyncMock(return_value=[])
        conn.fetchrow = AsyncMock(return_value=None)

        from exo.agent import Agent  # pyright: ignore[reportMissingImports]
        from exo.memory.base import AgentMemory
        from exo.memory.short_term import ShortTermMemory

        async with PostgresMemoryStore("postgresql://localhost/test") as pg_store:
            agent = Agent(
                name="pg-agent",
                memory=AgentMemory(
                    short_term=ShortTermMemory(),
                    long_term=pg_store,
                ),
            )
            assert agent.memory is not None
            assert isinstance(agent.memory, AgentMemory)
            assert agent.memory.long_term is pg_store
            assert pg_store._initialized

    @patch("exo.memory.backends.postgres.asyncpg")
    async def test_postgres_store_add_and_search(self, mock_asyncpg: MagicMock) -> None:
        """Verify add() and search() work end-to-end with mocked asyncpg."""
        pool, conn = _mock_pool()
        mock_asyncpg.create_pool = AsyncMock(return_value=pool)
        conn.execute = AsyncMock(return_value="INSERT 0 1")
        conn.fetch = AsyncMock(return_value=[_record(id="1", content="persisted message")])

        async with PostgresMemoryStore("postgresql://localhost/test") as store:
            item = HumanMemory(content="persisted message")
            await store.add(item)
            conn.execute.assert_awaited()

            results = await store.search(query="persisted")
            assert len(results) == 1
            assert results[0].content == "persisted message"


# ---------------------------------------------------------------------------
# Integration tests (require real Postgres)
# ---------------------------------------------------------------------------


@requires_postgres
class TestPostgresIntegration:
    """Tests against a real Postgres instance.

    Set POSTGRES_DSN env var to run, e.g.:
        POSTGRES_DSN=postgresql://localhost/exo_test uv run pytest -k postgres
    """

    @pytest.fixture
    async def store(self):
        async with PostgresMemoryStore(POSTGRES_DSN) as s:
            # Clean table before each test
            pool = s._ensure_init()
            async with pool.acquire() as conn:
                await conn.execute("DELETE FROM memory_items")
            yield s

    async def test_add_and_get(self, store: PostgresMemoryStore) -> None:
        item = HumanMemory(content="integration test")
        await store.add(item)
        got = await store.get(item.id)
        assert got is not None
        assert got.content == "integration test"
        assert isinstance(got, HumanMemory)

    async def test_search_by_query(self, store: PostgresMemoryStore) -> None:
        await store.add(HumanMemory(content="apple pie"))
        await store.add(HumanMemory(content="banana split"))
        results = await store.search(query="apple")
        assert len(results) == 1

    async def test_clear_and_count(self, store: PostgresMemoryStore) -> None:
        await store.add(HumanMemory(content="a"))
        await store.add(HumanMemory(content="b"))
        assert await store.count() == 2
        count = await store.clear()
        assert count == 2
        assert await store.count() == 0
        assert await store.count(include_deleted=True) == 2

    async def test_upsert(self, store: PostgresMemoryStore) -> None:
        item = HumanMemory(content="v1")
        await store.add(item)
        item.content = "v2"
        await store.add(item)
        got = await store.get(item.id)
        assert got is not None
        assert got.content == "v2"

    async def test_metadata_search(self, store: PostgresMemoryStore) -> None:
        meta = _make_meta(user_id="alice", session_id="s1")
        await store.add(HumanMemory(content="match", metadata=meta))
        await store.add(HumanMemory(content="no", metadata=_make_meta(user_id="bob")))
        results = await store.search(metadata=_make_meta(user_id="alice"))
        assert len(results) == 1
        assert results[0].content == "match"

    async def test_subtype_reconstruction(self, store: PostgresMemoryStore) -> None:
        await store.add(SystemMemory(content="system"))
        await store.add(AIMemory(content="ai", tool_calls=[{"id": "t1"}]))
        await store.add(ToolMemory(content="tool", tool_call_id="t1", tool_name="search"))

        results = await store.search(memory_type="system")
        assert isinstance(results[0], SystemMemory)

        results = await store.search(memory_type="ai")
        assert isinstance(results[0], AIMemory)
        assert results[0].tool_calls == [{"id": "t1"}]

        results = await store.search(memory_type="tool")
        assert isinstance(results[0], ToolMemory)
        assert results[0].tool_call_id == "t1"
