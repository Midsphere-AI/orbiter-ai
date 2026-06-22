"""Tests for ChromaVectorStore with mocked ChromaDB (no real database)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

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


def _make_mock_chromadb() -> tuple[MagicMock, MagicMock, MagicMock]:
    """Create a mock chromadb module, client, and collection.

    Returns:
        (mock_module, mock_client, mock_collection)
    """
    mock_collection = MagicMock()
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_client.delete_collection = MagicMock()

    mock_module = MagicMock()
    mock_module.EphemeralClient.return_value = mock_client
    mock_module.PersistentClient.return_value = mock_client
    # Make ClientAPI available for type annotation
    mock_module.ClientAPI = type(mock_client)

    return mock_module, mock_client, mock_collection


# ---------------------------------------------------------------------------
# ChromaVectorStore — initialization
# ---------------------------------------------------------------------------


class TestChromaVectorStoreInit:
    def test_create_ephemeral_default(self) -> None:
        mock_module, mock_client, mock_collection = _make_mock_chromadb()
        with patch.dict("sys.modules", {"chromadb": mock_module}):
            from exo.retrieval.backends.chroma import ChromaVectorStore

            store = ChromaVectorStore()
            mock_module.EphemeralClient.assert_called_once()
            mock_client.get_or_create_collection.assert_called_once_with(
                name="exo_vectors",
                metadata={"hnsw:space": "cosine"},
            )
            assert store._collection_name == "exo_vectors"
            assert store._path is None

    def test_create_persistent_with_path(self) -> None:
        mock_module, mock_client, mock_collection = _make_mock_chromadb()
        with patch.dict("sys.modules", {"chromadb": mock_module}):
            from exo.retrieval.backends.chroma import ChromaVectorStore

            store = ChromaVectorStore(path="/tmp/chroma_test")
            mock_module.PersistentClient.assert_called_once_with(path="/tmp/chroma_test")
            assert store._path == "/tmp/chroma_test"

    def test_create_with_custom_collection_name(self) -> None:
        mock_module, mock_client, mock_collection = _make_mock_chromadb()
        with patch.dict("sys.modules", {"chromadb": mock_module}):
            from exo.retrieval.backends.chroma import ChromaVectorStore

            store = ChromaVectorStore(collection_name="my_vectors")
            mock_client.get_or_create_collection.assert_called_once_with(
                name="my_vectors",
                metadata={"hnsw:space": "cosine"},
            )
            assert store._collection_name == "my_vectors"

    def test_create_with_external_client(self) -> None:
        mock_module, _, mock_collection = _make_mock_chromadb()
        external_client = MagicMock()
        external_client.get_or_create_collection.return_value = mock_collection

        with patch.dict("sys.modules", {"chromadb": mock_module}):
            from exo.retrieval.backends.chroma import ChromaVectorStore

            store = ChromaVectorStore(client=external_client)
            # Should use external client, not create new one
            mock_module.EphemeralClient.assert_not_called()
            mock_module.PersistentClient.assert_not_called()
            external_client.get_or_create_collection.assert_called_once()
            assert store._client is external_client


# ---------------------------------------------------------------------------
# ChromaVectorStore — add
# ---------------------------------------------------------------------------


class TestChromaVectorStoreAdd:
    @pytest.mark.asyncio
    async def test_add_chunks(self) -> None:
        mock_module, mock_client, mock_collection = _make_mock_chromadb()
        with patch.dict("sys.modules", {"chromadb": mock_module}):
            from exo.retrieval.backends.chroma import ChromaVectorStore

            store = ChromaVectorStore(client=mock_client)
            chunks = [_chunk(content="a"), _chunk(content="b", index=1)]
            embeddings = [[1.0, 0.0], [0.0, 1.0]]

            await store.add(chunks, embeddings)

            mock_collection.upsert.assert_called_once()
            call_kwargs = mock_collection.upsert.call_args
            assert call_kwargs.kwargs["ids"] == ["doc-1:0", "doc-1:1"]
            assert call_kwargs.kwargs["embeddings"] == [[1.0, 0.0], [0.0, 1.0]]
            assert call_kwargs.kwargs["documents"] == ["a", "b"]
            assert len(call_kwargs.kwargs["metadatas"]) == 2

    @pytest.mark.asyncio
    async def test_add_mismatched_lengths_raises(self) -> None:
        mock_module, mock_client, mock_collection = _make_mock_chromadb()
        with patch.dict("sys.modules", {"chromadb": mock_module}):
            from exo.retrieval.backends.chroma import ChromaVectorStore

            store = ChromaVectorStore(client=mock_client)
            with pytest.raises(RetrievalError, match="must match"):
                await store.add([_chunk()], [[1.0], [2.0]])

    @pytest.mark.asyncio
    async def test_add_empty_is_noop(self) -> None:
        mock_module, mock_client, mock_collection = _make_mock_chromadb()
        with patch.dict("sys.modules", {"chromadb": mock_module}):
            from exo.retrieval.backends.chroma import ChromaVectorStore

            store = ChromaVectorStore(client=mock_client)
            await store.add([], [])
            mock_collection.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_serializes_metadata(self) -> None:
        mock_module, mock_client, mock_collection = _make_mock_chromadb()
        with patch.dict("sys.modules", {"chromadb": mock_module}):
            from exo.retrieval.backends.chroma import ChromaVectorStore

            store = ChromaVectorStore(client=mock_client)
            meta = {"source": "web", "page": 3}
            await store.add([_chunk(metadata=meta)], [[1.0, 0.0]])

            call_kwargs = mock_collection.upsert.call_args.kwargs
            stored_meta = call_kwargs["metadatas"][0]
            assert stored_meta["document_id"] == "doc-1"
            assert stored_meta["chunk_index"] == 0
            assert json.loads(stored_meta["chunk_metadata"]) == meta

    @pytest.mark.asyncio
    async def test_add_generates_correct_ids(self) -> None:
        mock_module, mock_client, mock_collection = _make_mock_chromadb()
        with patch.dict("sys.modules", {"chromadb": mock_module}):
            from exo.retrieval.backends.chroma import ChromaVectorStore

            store = ChromaVectorStore(client=mock_client)
            chunks = [
                _chunk(document_id="doc-A", index=0),
                _chunk(document_id="doc-A", index=1),
                _chunk(document_id="doc-B", index=0),
            ]
            embeddings = [[1.0], [2.0], [3.0]]
            await store.add(chunks, embeddings)

            call_kwargs = mock_collection.upsert.call_args.kwargs
            assert call_kwargs["ids"] == ["doc-A:0", "doc-A:1", "doc-B:0"]


# ---------------------------------------------------------------------------
# ChromaVectorStore — search
# ---------------------------------------------------------------------------


class TestChromaVectorStoreSearch:
    @pytest.mark.asyncio
    async def test_search_calls_query(self) -> None:
        mock_module, mock_client, mock_collection = _make_mock_chromadb()
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        with patch.dict("sys.modules", {"chromadb": mock_module}):
            from exo.retrieval.backends.chroma import ChromaVectorStore

            store = ChromaVectorStore(client=mock_client)
            await store.search([1.0, 0.0], top_k=5)

            mock_collection.query.assert_called_once()
            call_kwargs = mock_collection.query.call_args.kwargs
            assert call_kwargs["query_embeddings"] == [[1.0, 0.0]]
            assert call_kwargs["n_results"] == 5
            assert call_kwargs["where"] is None

    @pytest.mark.asyncio
    async def test_search_returns_retrieval_results(self) -> None:
        mock_module, mock_client, mock_collection = _make_mock_chromadb()
        mock_collection.query.return_value = {
            "ids": [["doc-1:0"]],
            "documents": [["hello world"]],
            "metadatas": [
                [
                    {
                        "document_id": "doc-1",
                        "chunk_index": 0,
                        "start_offset": 0,
                        "end_offset": 11,
                        "chunk_metadata": '{"source": "test"}',
                    }
                ]
            ],
            "distances": [[0.05]],  # cosine distance
        }

        with patch.dict("sys.modules", {"chromadb": mock_module}):
            from exo.retrieval.backends.chroma import ChromaVectorStore

            store = ChromaVectorStore(client=mock_client)
            results = await store.search([1.0, 0.0], top_k=5)

            assert len(results) == 1
            assert isinstance(results[0], RetrievalResult)
            assert results[0].chunk.document_id == "doc-1"
            assert results[0].chunk.content == "hello world"
            assert results[0].chunk.metadata == {"source": "test"}
            assert results[0].score == pytest.approx(0.95)

    @pytest.mark.asyncio
    async def test_search_converts_distance_to_similarity(self) -> None:
        mock_module, mock_client, mock_collection = _make_mock_chromadb()
        mock_collection.query.return_value = {
            "ids": [["doc-1:0", "doc-1:1"]],
            "documents": [["close", "far"]],
            "metadatas": [
                [
                    {
                        "document_id": "doc-1",
                        "chunk_index": 0,
                        "start_offset": 0,
                        "end_offset": 5,
                        "chunk_metadata": "{}",
                    },
                    {
                        "document_id": "doc-1",
                        "chunk_index": 1,
                        "start_offset": 0,
                        "end_offset": 3,
                        "chunk_metadata": "{}",
                    },
                ]
            ],
            "distances": [[0.1, 0.9]],
        }

        with patch.dict("sys.modules", {"chromadb": mock_module}):
            from exo.retrieval.backends.chroma import ChromaVectorStore

            store = ChromaVectorStore(client=mock_client)
            results = await store.search([1.0, 0.0])

            assert results[0].score == pytest.approx(0.9)
            assert results[1].score == pytest.approx(0.1)

    @pytest.mark.asyncio
    async def test_search_with_single_flat_filter(self) -> None:
        """Filtering on a flat scalar field uses $eq (not $contains)."""
        mock_module, mock_client, mock_collection = _make_mock_chromadb()
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        with patch.dict("sys.modules", {"chromadb": mock_module}):
            from exo.retrieval.backends.chroma import ChromaVectorStore

            store = ChromaVectorStore(client=mock_client)
            await store.search([1.0, 0.0], filter={"document_id": "doc-1"})

            call_kwargs = mock_collection.query.call_args.kwargs
            where = call_kwargs["where"]
            assert where is not None
            assert where == {"document_id": {"$eq": "doc-1"}}

    @pytest.mark.asyncio
    async def test_search_with_multiple_flat_filters(self) -> None:
        """Multiple flat-field filters are combined via $and."""
        mock_module, mock_client, mock_collection = _make_mock_chromadb()
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        with patch.dict("sys.modules", {"chromadb": mock_module}):
            from exo.retrieval.backends.chroma import ChromaVectorStore

            store = ChromaVectorStore(client=mock_client)
            await store.search([1.0, 0.0], filter={"document_id": "doc-1", "chunk_index": 0})

            call_kwargs = mock_collection.query.call_args.kwargs
            where = call_kwargs["where"]
            assert "$and" in where
            assert len(where["$and"]) == 2

    @pytest.mark.asyncio
    async def test_search_unknown_filter_key_is_ignored_with_warning(self) -> None:
        """Keys that are not flat metadata fields are skipped with a warning."""
        import logging

        mock_module, mock_client, mock_collection = _make_mock_chromadb()
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        with patch.dict("sys.modules", {"chromadb": mock_module}):
            from exo.retrieval.backends.chroma import ChromaVectorStore

            store = ChromaVectorStore(client=mock_client)

            records: list[logging.LogRecord] = []

            class _CapHandler(logging.Handler):
                def emit(self, record: logging.LogRecord) -> None:
                    records.append(record)

            handler = _CapHandler()
            logging.getLogger("exo.retrieval.backends.chroma").addHandler(handler)
            try:
                await store.search([1.0, 0.0], filter={"source": "web"})
            finally:
                logging.getLogger("exo.retrieval.backends.chroma").removeHandler(handler)

            call_kwargs = mock_collection.query.call_args.kwargs
            # "source" is not a flat field — no valid conditions remain, so where=None
            assert call_kwargs["where"] is None
            assert any("source" in r.getMessage() for r in records)

    @pytest.mark.asyncio
    async def test_search_empty_result(self) -> None:
        mock_module, mock_client, mock_collection = _make_mock_chromadb()
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        with patch.dict("sys.modules", {"chromadb": mock_module}):
            from exo.retrieval.backends.chroma import ChromaVectorStore

            store = ChromaVectorStore(client=mock_client)
            results = await store.search([1.0, 0.0])
            assert results == []


# ---------------------------------------------------------------------------
# ChromaVectorStore — delete
# ---------------------------------------------------------------------------


class TestChromaVectorStoreDelete:
    @pytest.mark.asyncio
    async def test_delete_by_document_id(self) -> None:
        mock_module, mock_client, mock_collection = _make_mock_chromadb()
        with patch.dict("sys.modules", {"chromadb": mock_module}):
            from exo.retrieval.backends.chroma import ChromaVectorStore

            store = ChromaVectorStore(client=mock_client)
            await store.delete("doc-42")

            mock_collection.delete.assert_called_once_with(
                where={"document_id": "doc-42"},
            )


# ---------------------------------------------------------------------------
# ChromaVectorStore — clear
# ---------------------------------------------------------------------------


class TestChromaVectorStoreClear:
    @pytest.mark.asyncio
    async def test_clear_recreates_collection(self) -> None:
        mock_module, mock_client, mock_collection = _make_mock_chromadb()
        with patch.dict("sys.modules", {"chromadb": mock_module}):
            from exo.retrieval.backends.chroma import ChromaVectorStore

            store = ChromaVectorStore(client=mock_client)
            # Reset the call count from init
            mock_client.get_or_create_collection.reset_mock()

            await store.clear()

            mock_client.delete_collection.assert_called_once_with("exo_vectors")
            mock_client.get_or_create_collection.assert_called_once_with(
                name="exo_vectors",
                metadata={"hnsw:space": "cosine"},
            )


# ---------------------------------------------------------------------------
# ChromaVectorStore — close
# ---------------------------------------------------------------------------


class TestChromaVectorStoreClose:
    @pytest.mark.asyncio
    async def test_close_is_noop(self) -> None:
        """close() exists and completes without error (no-op for Chroma)."""
        mock_module, mock_client, _ = _make_mock_chromadb()
        with patch.dict("sys.modules", {"chromadb": mock_module}):
            from exo.retrieval.backends.chroma import ChromaVectorStore

            store = ChromaVectorStore(client=mock_client)
            # Should complete without raising
            await store.close()


# ---------------------------------------------------------------------------
# Import error
# ---------------------------------------------------------------------------


class TestChromaImportError:
    def test_import_error_without_chromadb(self) -> None:
        """Importing the module without chromadb installed raises ImportError."""
        import importlib
        import sys

        # Remove cached module if present
        mod_name = "exo.retrieval.backends.chroma"
        sys.modules.pop(mod_name, None)

        with patch.dict("sys.modules", {"chromadb": None}):
            with pytest.raises(ImportError, match="chromadb is required"):
                importlib.import_module(mod_name)
