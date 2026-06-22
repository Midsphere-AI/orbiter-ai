"""Tests for SQLiteMemoryStore backend."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

aiosqlite = pytest.importorskip("aiosqlite")

from exo.memory.backends.sqlite import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    SQLiteMemoryStore,
    _default_db_path,
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
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def store():
    """Create a fresh in-memory SQLite store for each test."""
    async with SQLiteMemoryStore(":memory:") as s:
        yield s


def _make_meta(**kw: str) -> MemoryMetadata:
    return MemoryMetadata(**kw)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_isinstance_check(self) -> None:
        store = SQLiteMemoryStore()
        assert isinstance(store, MemoryStore)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    async def test_init_and_close(self) -> None:
        store = SQLiteMemoryStore(":memory:")
        await store.init()
        assert store._initialized
        await store.close()
        assert not store._initialized

    async def test_double_init_is_idempotent(self) -> None:
        store = SQLiteMemoryStore(":memory:")
        await store.init()
        await store.init()  # should not raise
        await store.close()

    async def test_context_manager(self) -> None:
        async with SQLiteMemoryStore(":memory:") as store:
            assert store._initialized
        assert not store._initialized

    async def test_operations_before_init_raise(self) -> None:
        store = SQLiteMemoryStore(":memory:")
        with pytest.raises(ExoMemoryError, match="not initialized"):
            await store.add(HumanMemory(content="hello"))

    def test_repr(self) -> None:
        store = SQLiteMemoryStore("/tmp/test.db")
        assert "test.db" in repr(store)


# ---------------------------------------------------------------------------
# Add + Get
# ---------------------------------------------------------------------------


class TestAddGet:
    async def test_add_and_get(self, store: SQLiteMemoryStore) -> None:
        item = HumanMemory(content="hello world")
        await store.add(item)
        got = await store.get(item.id)
        assert got is not None
        assert got.content == "hello world"
        assert got.memory_type == "human"
        assert isinstance(got, HumanMemory)

    async def test_get_nonexistent(self, store: SQLiteMemoryStore) -> None:
        assert await store.get("nonexistent") is None

    async def test_upsert_updates_existing(self, store: SQLiteMemoryStore) -> None:
        item = HumanMemory(content="original")
        await store.add(item)
        item.content = "updated"
        await store.add(item)
        got = await store.get(item.id)
        assert got is not None
        assert got.content == "updated"

    async def test_add_system_memory(self, store: SQLiteMemoryStore) -> None:
        item = SystemMemory(content="You are a helpful assistant")
        await store.add(item)
        got = await store.get(item.id)
        assert got is not None
        assert isinstance(got, SystemMemory)
        assert got.memory_type == "system"

    async def test_add_ai_memory_with_tool_calls(self, store: SQLiteMemoryStore) -> None:
        item = AIMemory(
            content="I'll search for that",
            tool_calls=[{"id": "tc1", "name": "search", "arguments": "{}"}],
        )
        await store.add(item)
        got = await store.get(item.id)
        assert got is not None
        assert isinstance(got, AIMemory)
        assert got.tool_calls == [{"id": "tc1", "name": "search", "arguments": "{}"}]

    async def test_add_tool_memory(self, store: SQLiteMemoryStore) -> None:
        item = ToolMemory(
            content="search result",
            tool_call_id="tc1",
            tool_name="search",
            is_error=False,
        )
        await store.add(item)
        got = await store.get(item.id)
        assert got is not None
        assert isinstance(got, ToolMemory)
        assert got.tool_call_id == "tc1"
        assert got.tool_name == "search"
        assert got.is_error is False


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestSearch:
    async def test_search_all(self, store: SQLiteMemoryStore) -> None:
        await store.add(HumanMemory(content="first"))
        await store.add(HumanMemory(content="second"))
        results = await store.search()
        assert len(results) == 2

    async def test_search_by_query(self, store: SQLiteMemoryStore) -> None:
        await store.add(HumanMemory(content="apple pie"))
        await store.add(HumanMemory(content="banana split"))
        results = await store.search(query="apple")
        assert len(results) == 1
        assert results[0].content == "apple pie"

    async def test_search_by_memory_type(self, store: SQLiteMemoryStore) -> None:
        await store.add(HumanMemory(content="user msg"))
        await store.add(AIMemory(content="ai response"))
        results = await store.search(memory_type="human")
        assert len(results) == 1
        assert results[0].content == "user msg"

    async def test_search_by_status(self, store: SQLiteMemoryStore) -> None:
        item = HumanMemory(content="draft item", status=MemoryStatus.DRAFT)
        await store.add(item)
        await store.add(HumanMemory(content="accepted item"))
        results = await store.search(status=MemoryStatus.DRAFT)
        assert len(results) == 1
        assert results[0].content == "draft item"

    async def test_search_by_metadata(self, store: SQLiteMemoryStore) -> None:
        meta = _make_meta(user_id="u1", session_id="s1")
        await store.add(HumanMemory(content="match", metadata=meta))
        await store.add(HumanMemory(content="no match", metadata=_make_meta(user_id="u2")))
        results = await store.search(metadata=_make_meta(user_id="u1"))
        assert len(results) == 1
        assert results[0].content == "match"

    async def test_search_limit(self, store: SQLiteMemoryStore) -> None:
        for i in range(5):
            await store.add(HumanMemory(content=f"item {i}"))
        results = await store.search(limit=3)
        assert len(results) == 3

    async def test_search_ordered_newest_first(self, store: SQLiteMemoryStore) -> None:
        await store.add(HumanMemory(content="old", created_at="2024-01-01T00:00:00"))
        await store.add(HumanMemory(content="new", created_at="2024-12-01T00:00:00"))
        results = await store.search()
        assert results[0].content == "new"
        assert results[1].content == "old"

    async def test_search_metadata_session_and_task(self, store: SQLiteMemoryStore) -> None:
        meta = _make_meta(session_id="s1", task_id="t1")
        await store.add(HumanMemory(content="match", metadata=meta))
        await store.add(
            HumanMemory(content="no", metadata=_make_meta(session_id="s1", task_id="t2"))
        )
        results = await store.search(metadata=_make_meta(session_id="s1", task_id="t1"))
        assert len(results) == 1

    async def test_search_metadata_agent_id(self, store: SQLiteMemoryStore) -> None:
        await store.add(HumanMemory(content="a1", metadata=_make_meta(agent_id="agent-1")))
        await store.add(HumanMemory(content="a2", metadata=_make_meta(agent_id="agent-2")))
        results = await store.search(metadata=_make_meta(agent_id="agent-1"))
        assert len(results) == 1
        assert results[0].content == "a1"


# ---------------------------------------------------------------------------
# Soft delete + clear
# ---------------------------------------------------------------------------


class TestClear:
    async def test_clear_all(self, store: SQLiteMemoryStore) -> None:
        await store.add(HumanMemory(content="a"))
        await store.add(HumanMemory(content="b"))
        count = await store.clear()
        assert count == 2
        assert await store.search() == []

    async def test_clear_with_metadata(self, store: SQLiteMemoryStore) -> None:
        await store.add(HumanMemory(content="keep", metadata=_make_meta(user_id="u1")))
        await store.add(HumanMemory(content="remove", metadata=_make_meta(user_id="u2")))
        count = await store.clear(metadata=_make_meta(user_id="u2"))
        assert count == 1
        results = await store.search()
        assert len(results) == 1
        assert results[0].content == "keep"

    async def test_soft_delete_hides_from_get(self, store: SQLiteMemoryStore) -> None:
        item = HumanMemory(content="to delete")
        await store.add(item)
        await store.clear()
        assert await store.get(item.id) is None

    async def test_soft_delete_preserved_in_db(self, store: SQLiteMemoryStore) -> None:
        item = HumanMemory(content="soft deleted")
        await store.add(item)
        await store.clear()
        # Still exists in DB but hidden from normal ops
        total = await store.count(include_deleted=True)
        active = await store.count(include_deleted=False)
        assert total == 1
        assert active == 0

    async def test_re_add_after_soft_delete(self, store: SQLiteMemoryStore) -> None:
        item = HumanMemory(content="revived")
        await store.add(item)
        await store.clear()
        assert await store.get(item.id) is None
        await store.add(item)
        got = await store.get(item.id)
        assert got is not None
        assert got.content == "revived"


# ---------------------------------------------------------------------------
# Version tracking
# ---------------------------------------------------------------------------


class TestVersion:
    async def test_initial_version_is_1(self, store: SQLiteMemoryStore) -> None:
        item = HumanMemory(content="v1")
        await store.add(item)
        db = store._ensure_init()
        cursor = await db.execute("SELECT version FROM memory_items WHERE id = ?", (item.id,))
        row = await cursor.fetchone()
        assert row["version"] == 1

    async def test_version_increments_on_upsert(self, store: SQLiteMemoryStore) -> None:
        item = HumanMemory(content="v1")
        await store.add(item)
        item.content = "v2"
        await store.add(item)
        db = store._ensure_init()
        cursor = await db.execute("SELECT version FROM memory_items WHERE id = ?", (item.id,))
        row = await cursor.fetchone()
        assert row["version"] == 2


# ---------------------------------------------------------------------------
# Count helper
# ---------------------------------------------------------------------------


class TestCount:
    async def test_count_empty(self, store: SQLiteMemoryStore) -> None:
        assert await store.count() == 0

    async def test_count_active(self, store: SQLiteMemoryStore) -> None:
        await store.add(HumanMemory(content="a"))
        await store.add(HumanMemory(content="b"))
        assert await store.count() == 2

    async def test_count_with_deleted(self, store: SQLiteMemoryStore) -> None:
        await store.add(HumanMemory(content="a"))
        await store.clear()
        assert await store.count() == 0
        assert await store.count(include_deleted=True) == 1


# ---------------------------------------------------------------------------
# File persistence
# ---------------------------------------------------------------------------


class TestFilePersistence:
    async def test_data_persists_across_connections(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "test.db")

        # Write
        async with SQLiteMemoryStore(db_path) as store:
            await store.add(HumanMemory(content="persistent", id="p1"))

        # Read with new connection
        async with SQLiteMemoryStore(db_path) as store:
            got = await store.get("p1")
            assert got is not None
            assert got.content == "persistent"

    async def test_metadata_persists(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "meta.db")
        meta = _make_meta(user_id="alice", session_id="s42")

        async with SQLiteMemoryStore(db_path) as store:
            await store.add(HumanMemory(content="msg", metadata=meta, id="m1"))

        async with SQLiteMemoryStore(db_path) as store:
            got = await store.get("m1")
            assert got is not None
            assert got.metadata.user_id == "alice"
            assert got.metadata.session_id == "s42"


# ---------------------------------------------------------------------------
# Custom memory types
# ---------------------------------------------------------------------------


class TestCustomMemoryType:
    async def test_unknown_type_returns_base_item(self, store: SQLiteMemoryStore) -> None:
        item = MemoryItem(content="custom", memory_type="custom_type")
        await store.add(item)
        got = await store.get(item.id)
        assert got is not None
        assert got.memory_type == "custom_type"
        assert got.content == "custom"


# ---------------------------------------------------------------------------
# Default path — EXO_MEMORY_PATH env var
# ---------------------------------------------------------------------------


class TestDefaultPath:
    def test_default_path_is_home_exo(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("EXO_MEMORY_PATH", raising=False)
        path = _default_db_path()
        home = os.path.expanduser("~")
        assert path == os.path.join(home, ".exo", "memory.db")

    def test_exo_memory_path_env_var_overrides_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        custom_path = str(tmp_path / "custom.db")
        monkeypatch.setenv("EXO_MEMORY_PATH", custom_path)
        path = _default_db_path()
        assert path == custom_path

    def test_exo_memory_path_expands_tilde(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXO_MEMORY_PATH", "~/myapp/memory.db")
        path = _default_db_path()
        home = os.path.expanduser("~")
        assert path == os.path.join(home, "myapp", "memory.db")

    def test_sqlite_store_no_args_uses_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXO_MEMORY_PATH", "/tmp/env_test.db")
        store = SQLiteMemoryStore()
        assert store.db_path == "/tmp/env_test.db"

    def test_sqlite_store_explicit_path_overrides_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXO_MEMORY_PATH", "/tmp/should_not_use.db")
        store = SQLiteMemoryStore(":memory:")
        assert store.db_path == ":memory:"


# ---------------------------------------------------------------------------
# Error boundary — ExoMemoryError taxonomy
# ---------------------------------------------------------------------------


class TestErrorBoundary:
    """Backend failures must surface as ExoMemoryError, not raw library exceptions."""

    async def test_not_initialized_raises_exo_memory_error(self) -> None:
        """_ensure_init() must raise ExoMemoryError (not RuntimeError)."""
        store = SQLiteMemoryStore(":memory:")
        with pytest.raises(ExoMemoryError, match="not initialized"):
            await store.add(HumanMemory(content="never persisted"))

    async def test_not_initialized_error_has_hint(self) -> None:
        """The ExoMemoryError from _ensure_init() must carry an actionable hint."""
        store = SQLiteMemoryStore(":memory:")
        try:
            await store.add(HumanMemory(content="trigger"))
        except ExoMemoryError as exc:
            assert exc.hint is not None, "ExoMemoryError from _ensure_init() must have a hint"
            assert "init()" in exc.hint or "async with" in exc.hint
        else:
            pytest.fail("Expected ExoMemoryError")

    async def test_add_failure_raises_exo_memory_error(self, store: SQLiteMemoryStore) -> None:
        """DB-level failure during add() must raise ExoMemoryError with context."""
        from unittest.mock import AsyncMock, patch

        broken_db = AsyncMock()
        broken_db.execute = AsyncMock(side_effect=Exception("disk I/O error"))

        with patch.object(store, "_db", broken_db), pytest.raises(ExoMemoryError) as exc_info:
            await store.add(HumanMemory(content="oops"))

        err = exc_info.value
        assert err.hint is not None, "ExoMemoryError from add() must have a hint"
        assert "disk space" in err.hint or "permissions" in err.hint
        # Original cause must be chained
        assert err.__cause__ is not None

    async def test_search_failure_raises_exo_memory_error(self, store: SQLiteMemoryStore) -> None:
        """DB-level failure during search() must raise ExoMemoryError with context."""
        from unittest.mock import AsyncMock, MagicMock, patch

        # search() now uses `async with db.execute(...) as cursor:`.
        # We need a mock whose __aenter__ raises so the exception propagates.
        error = Exception("schema mismatch")
        ctx_mock = MagicMock()
        ctx_mock.__aenter__ = AsyncMock(side_effect=error)
        ctx_mock.__aexit__ = AsyncMock(return_value=False)

        broken_db = MagicMock()
        broken_db.execute = MagicMock(return_value=ctx_mock)

        with patch.object(store, "_db", broken_db), pytest.raises(ExoMemoryError) as exc_info:
            await store.search(query="anything")

        err = exc_info.value
        assert err.__cause__ is not None
        assert "schema mismatch" in str(err.__cause__)
