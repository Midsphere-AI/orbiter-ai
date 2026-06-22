"""Tests for PgVectorStore with mocked asyncpg (no real database)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exo.retrieval.types import Chunk, RetrievalError, RetrievalResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(
    document_id: str = "doc-1",
    index: int = 0,
    content: str = "hello",
    metadata: dict[str, Any] | None = None,
) -> Chunk:
    """Build a Chunk with sensible defaults."""
    return Chunk(
        document_id=document_id,
        index=index,
        content=content,
        start=0,
        end=len(content),
        metadata=metadata or {},
    )


def _make_mock_pool() -> tuple[MagicMock, AsyncMock]:
    """Create a mock asyncpg pool with connection context manager.

    asyncpg's ``pool.acquire()`` returns an async context manager directly
    (not a coroutine), so we use a ``MagicMock`` for the pool and configure
    ``acquire()`` to return a context-manager object synchronously.
    """
    mock_conn = AsyncMock()

    class _AcquireCtx:
        async def __aenter__(self) -> AsyncMock:
            return mock_conn

        async def __aexit__(self, *args: object) -> None:
            pass

    mock_pool = MagicMock()
    mock_pool.acquire.return_value = _AcquireCtx()
    mock_pool.close = AsyncMock()

    return mock_pool, mock_conn


# ---------------------------------------------------------------------------
# PgVectorStore — initialization
# ---------------------------------------------------------------------------


class TestPgVectorStoreInit:
    def test_create_with_dsn(self) -> None:
        with patch.dict("sys.modules", {"asyncpg": MagicMock()}):
            from exo.retrieval.backends.pgvector import PgVectorStore

            store = PgVectorStore(dsn="postgresql://localhost/test")
            assert store._dsn == "postgresql://localhost/test"
            assert store._table == "exo_vectors"
            assert store._dimensions == 1536
            assert store._pool is None
            assert store._owns_pool is True

    def test_create_with_pool(self) -> None:
        mock_pool = MagicMock()
        with patch.dict("sys.modules", {"asyncpg": MagicMock()}):
            from exo.retrieval.backends.pgvector import PgVectorStore

            store = PgVectorStore(pool=mock_pool)
            assert store._pool is mock_pool
            assert store._owns_pool is False

    def test_custom_table_and_dimensions(self) -> None:
        with patch.dict("sys.modules", {"asyncpg": MagicMock()}):
            from exo.retrieval.backends.pgvector import PgVectorStore

            store = PgVectorStore(
                dsn="postgresql://localhost/test",
                table="custom_vectors",
                dimensions=768,
            )
            assert store._table == "custom_vectors"
            assert store._dimensions == 768

    def test_invalid_table_name_raises(self) -> None:
        """SQL injection: table names with special characters must be rejected at construction."""
        with patch.dict("sys.modules", {"asyncpg": MagicMock()}):
            from exo.retrieval.backends.pgvector import PgVectorStore

            bad_names = [
                "'; DROP TABLE exo_vectors; --",
                "table name with spaces",
                "my-table",
                "table.dot.dot",
                "123starts_with_digit",
                "",
            ]
            for bad in bad_names:
                with pytest.raises(RetrievalError, match="Invalid table name"):
                    PgVectorStore(dsn="postgresql://localhost/test", table=bad)

    def test_valid_table_names_accepted(self) -> None:
        """Valid plain identifiers and schema-qualified names must be accepted."""
        with patch.dict("sys.modules", {"asyncpg": MagicMock()}):
            from exo.retrieval.backends.pgvector import PgVectorStore

            valid_names = [
                "exo_vectors",
                "public.exo_vectors",
                "MyTable",
                "_private",
                "schema1.table_name",
            ]
            for name in valid_names:
                store = PgVectorStore(dsn="postgresql://localhost/test", table=name)
                assert store._table == name


# ---------------------------------------------------------------------------
# PgVectorStore — initialize (create table)
# ---------------------------------------------------------------------------


class TestPgVectorStoreInitialize:
    @pytest.mark.asyncio
    async def test_initialize_creates_extension_and_table(self) -> None:
        mock_pool, mock_conn = _make_mock_pool()
        with patch.dict("sys.modules", {"asyncpg": MagicMock()}):
            from exo.retrieval.backends.pgvector import PgVectorStore

            store = PgVectorStore(pool=mock_pool, dimensions=384)
            await store.initialize()

            calls = mock_conn.execute.call_args_list
            assert len(calls) == 3

            # Extension creation
            assert "CREATE EXTENSION IF NOT EXISTS vector" in calls[0].args[0]

            # Table creation with correct dimensions
            create_table_sql = calls[1].args[0]
            assert "CREATE TABLE IF NOT EXISTS exo_vectors" in create_table_sql
            assert "vector(384)" in create_table_sql
            assert "document_id TEXT NOT NULL" in create_table_sql
            assert "embedding vector(384) NOT NULL" in create_table_sql

            # Index creation
            assert "CREATE INDEX IF NOT EXISTS" in calls[2].args[0]


# ---------------------------------------------------------------------------
# PgVectorStore — add
# ---------------------------------------------------------------------------


class TestPgVectorStoreAdd:
    @pytest.mark.asyncio
    async def test_add_chunks(self) -> None:
        mock_pool, mock_conn = _make_mock_pool()
        with patch.dict("sys.modules", {"asyncpg": MagicMock()}):
            from exo.retrieval.backends.pgvector import PgVectorStore

            store = PgVectorStore(pool=mock_pool)
            chunks = [_chunk(content="a"), _chunk(content="b", index=1)]
            embeddings = [[1.0, 0.0], [0.0, 1.0]]

            await store.add(chunks, embeddings)

            mock_conn.executemany.assert_called_once()
            call_args = mock_conn.executemany.call_args
            sql = call_args.args[0]
            params = call_args.args[1]

            assert "INSERT INTO exo_vectors" in sql
            assert len(params) == 2
            # Check first row params
            assert params[0][0] == "doc-1"  # document_id
            assert params[0][1] == 0  # chunk_index
            assert params[0][2] == "a"  # content

    @pytest.mark.asyncio
    async def test_add_mismatched_lengths_raises(self) -> None:
        mock_pool, _ = _make_mock_pool()
        with patch.dict("sys.modules", {"asyncpg": MagicMock()}):
            from exo.retrieval.backends.pgvector import PgVectorStore

            store = PgVectorStore(pool=mock_pool)
            with pytest.raises(RetrievalError, match="must match"):
                await store.add([_chunk()], [[1.0], [2.0]])

    @pytest.mark.asyncio
    async def test_add_empty_is_noop(self) -> None:
        mock_pool, mock_conn = _make_mock_pool()
        with patch.dict("sys.modules", {"asyncpg": MagicMock()}):
            from exo.retrieval.backends.pgvector import PgVectorStore

            store = PgVectorStore(pool=mock_pool)
            await store.add([], [])
            mock_conn.executemany.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_serializes_metadata(self) -> None:
        mock_pool, mock_conn = _make_mock_pool()
        with patch.dict("sys.modules", {"asyncpg": MagicMock()}):
            from exo.retrieval.backends.pgvector import PgVectorStore

            store = PgVectorStore(pool=mock_pool)
            meta = {"source": "web", "page": 3}
            await store.add([_chunk(metadata=meta)], [[1.0, 0.0]])

            params = mock_conn.executemany.call_args.args[1]
            assert json.loads(params[0][5]) == meta


# ---------------------------------------------------------------------------
# PgVectorStore — search
# ---------------------------------------------------------------------------


class TestPgVectorStoreSearch:
    @pytest.mark.asyncio
    async def test_search_uses_cosine_distance(self) -> None:
        mock_pool, mock_conn = _make_mock_pool()
        mock_conn.fetch = AsyncMock(return_value=[])

        with patch.dict("sys.modules", {"asyncpg": MagicMock()}):
            from exo.retrieval.backends.pgvector import PgVectorStore

            store = PgVectorStore(pool=mock_pool)
            await store.search([1.0, 0.0], top_k=5)

            sql = mock_conn.fetch.call_args.args[0]
            assert "<=>" in sql
            assert "1 - (embedding <=> $1::vector)" in sql
            assert "ORDER BY embedding <=> $1::vector" in sql
            assert "LIMIT $2" in sql

    @pytest.mark.asyncio
    async def test_search_returns_retrieval_results(self) -> None:
        mock_pool, mock_conn = _make_mock_pool()
        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "document_id": "doc-1",
                    "chunk_index": 0,
                    "content": "hello world",
                    "start_offset": 0,
                    "end_offset": 11,
                    "metadata": {"source": "test"},
                    "score": 0.95,
                }
            ]
        )

        with patch.dict("sys.modules", {"asyncpg": MagicMock()}):
            from exo.retrieval.backends.pgvector import PgVectorStore

            store = PgVectorStore(pool=mock_pool)
            results = await store.search([1.0, 0.0], top_k=5)

            assert len(results) == 1
            assert isinstance(results[0], RetrievalResult)
            assert results[0].chunk.document_id == "doc-1"
            assert results[0].chunk.content == "hello world"
            assert results[0].score == 0.95

    @pytest.mark.asyncio
    async def test_search_with_metadata_filter(self) -> None:
        mock_pool, mock_conn = _make_mock_pool()
        mock_conn.fetch = AsyncMock(return_value=[])

        with patch.dict("sys.modules", {"asyncpg": MagicMock()}):
            from exo.retrieval.backends.pgvector import PgVectorStore

            store = PgVectorStore(pool=mock_pool)
            await store.search([1.0, 0.0], top_k=5, filter={"source": "web"})

            sql = mock_conn.fetch.call_args.args[0]
            assert "metadata->>'source' = $3" in sql
            params = mock_conn.fetch.call_args.args[1:]
            assert params[-1] == "web"

    @pytest.mark.asyncio
    async def test_search_multiple_filters(self) -> None:
        mock_pool, mock_conn = _make_mock_pool()
        mock_conn.fetch = AsyncMock(return_value=[])

        with patch.dict("sys.modules", {"asyncpg": MagicMock()}):
            from exo.retrieval.backends.pgvector import PgVectorStore

            store = PgVectorStore(pool=mock_pool)
            await store.search([1.0, 0.0], top_k=5, filter={"source": "web", "lang": "en"})

            sql = mock_conn.fetch.call_args.args[0]
            assert "WHERE" in sql
            assert "metadata->>'source' = $3" in sql
            assert "metadata->>'lang' = $4" in sql

    @pytest.mark.asyncio
    async def test_search_empty_result(self) -> None:
        mock_pool, mock_conn = _make_mock_pool()
        mock_conn.fetch = AsyncMock(return_value=[])

        with patch.dict("sys.modules", {"asyncpg": MagicMock()}):
            from exo.retrieval.backends.pgvector import PgVectorStore

            store = PgVectorStore(pool=mock_pool)
            results = await store.search([1.0, 0.0])
            assert results == []

    @pytest.mark.asyncio
    async def test_search_parses_json_string_metadata(self) -> None:
        """Metadata may come back as a JSON string from some drivers."""
        mock_pool, mock_conn = _make_mock_pool()
        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "document_id": "doc-1",
                    "chunk_index": 0,
                    "content": "test",
                    "start_offset": 0,
                    "end_offset": 4,
                    "metadata": '{"key": "value"}',
                    "score": 0.9,
                }
            ]
        )

        with patch.dict("sys.modules", {"asyncpg": MagicMock()}):
            from exo.retrieval.backends.pgvector import PgVectorStore

            store = PgVectorStore(pool=mock_pool)
            results = await store.search([1.0, 0.0])
            assert results[0].chunk.metadata == {"key": "value"}

    @pytest.mark.asyncio
    async def test_search_passes_vector_literal(self) -> None:
        mock_pool, mock_conn = _make_mock_pool()
        mock_conn.fetch = AsyncMock(return_value=[])

        with patch.dict("sys.modules", {"asyncpg": MagicMock()}):
            from exo.retrieval.backends.pgvector import PgVectorStore

            store = PgVectorStore(pool=mock_pool)
            await store.search([1.5, 2.5, 3.5], top_k=3)

            params = mock_conn.fetch.call_args.args
            assert params[1] == "[1.5,2.5,3.5]"
            assert params[2] == 3

    @pytest.mark.asyncio
    async def test_search_rejects_invalid_filter_key(self) -> None:
        """SQL injection: filter keys with non-alphanumeric chars must be rejected."""
        mock_pool, mock_conn = _make_mock_pool()
        mock_conn.fetch = AsyncMock(return_value=[])

        with patch.dict("sys.modules", {"asyncpg": MagicMock()}):
            from exo.retrieval.backends.pgvector import PgVectorStore

            store = PgVectorStore(pool=mock_pool)

            # These keys contain characters that could allow SQL injection
            bad_keys = [
                "'; DROP TABLE exo_vectors; --",
                "key WITH INJECTION",
                "key'quote",
                "key-with-hyphens",
                "key.with.dots",
                "key with spaces",
            ]
            for bad_key in bad_keys:
                with pytest.raises(RetrievalError, match="Invalid metadata filter key"):
                    await store.search([1.0, 0.0], filter={bad_key: "value"})

    @pytest.mark.asyncio
    async def test_search_accepts_valid_filter_keys(self) -> None:
        """Valid alphanumeric/underscore filter keys pass through without error."""
        mock_pool, mock_conn = _make_mock_pool()
        mock_conn.fetch = AsyncMock(return_value=[])

        with patch.dict("sys.modules", {"asyncpg": MagicMock()}):
            from exo.retrieval.backends.pgvector import PgVectorStore

            store = PgVectorStore(pool=mock_pool)

            valid_keys = ["source", "lang", "chunk_type", "Page123", "snake_case_key"]
            for key in valid_keys:
                await store.search([1.0, 0.0], filter={key: "value"})
                # Should reach the DB without raising
                assert mock_conn.fetch.called


# ---------------------------------------------------------------------------
# PgVectorStore — delete
# ---------------------------------------------------------------------------


class TestPgVectorStoreDelete:
    @pytest.mark.asyncio
    async def test_delete_by_document_id(self) -> None:
        mock_pool, mock_conn = _make_mock_pool()
        with patch.dict("sys.modules", {"asyncpg": MagicMock()}):
            from exo.retrieval.backends.pgvector import PgVectorStore

            store = PgVectorStore(pool=mock_pool)
            await store.delete("doc-42")

            mock_conn.execute.assert_called_once()
            sql = mock_conn.execute.call_args.args[0]
            assert "DELETE FROM exo_vectors" in sql
            assert "document_id = $1" in sql
            assert mock_conn.execute.call_args.args[1] == "doc-42"


# ---------------------------------------------------------------------------
# PgVectorStore — clear
# ---------------------------------------------------------------------------


class TestPgVectorStoreClear:
    @pytest.mark.asyncio
    async def test_clear_truncates_table(self) -> None:
        mock_pool, mock_conn = _make_mock_pool()
        with patch.dict("sys.modules", {"asyncpg": MagicMock()}):
            from exo.retrieval.backends.pgvector import PgVectorStore

            store = PgVectorStore(pool=mock_pool)
            await store.clear()

            mock_conn.execute.assert_called_once()
            sql = mock_conn.execute.call_args.args[0]
            assert "TRUNCATE exo_vectors" in sql


# ---------------------------------------------------------------------------
# PgVectorStore — close
# ---------------------------------------------------------------------------


class TestPgVectorStoreClose:
    @pytest.mark.asyncio
    async def test_close_closes_owned_pool(self) -> None:
        mock_pool, _ = _make_mock_pool()
        with patch.dict("sys.modules", {"asyncpg": MagicMock()}):
            from exo.retrieval.backends.pgvector import PgVectorStore

            store = PgVectorStore(dsn="postgresql://localhost/test")
            store._pool = mock_pool
            store._owns_pool = True

            await store.close()

            mock_pool.close.assert_called_once()
            assert store._pool is None

    @pytest.mark.asyncio
    async def test_close_does_not_close_external_pool(self) -> None:
        mock_pool, _ = _make_mock_pool()
        with patch.dict("sys.modules", {"asyncpg": MagicMock()}):
            from exo.retrieval.backends.pgvector import PgVectorStore

            store = PgVectorStore(pool=mock_pool)
            await store.close()

            mock_pool.close.assert_not_called()


# ---------------------------------------------------------------------------
# _vector_literal helper
# ---------------------------------------------------------------------------


class TestVectorLiteral:
    def test_vector_literal(self) -> None:
        with patch.dict("sys.modules", {"asyncpg": MagicMock()}):
            from exo.retrieval.backends.pgvector import _vector_literal

            assert _vector_literal([1.0, 2.5, 3.0]) == "[1.0,2.5,3.0]"

    def test_vector_literal_empty(self) -> None:
        with patch.dict("sys.modules", {"asyncpg": MagicMock()}):
            from exo.retrieval.backends.pgvector import _vector_literal

            assert _vector_literal([]) == "[]"

    def test_vector_literal_single(self) -> None:
        with patch.dict("sys.modules", {"asyncpg": MagicMock()}):
            from exo.retrieval.backends.pgvector import _vector_literal

            assert _vector_literal([42.0]) == "[42.0]"
