"""Tests for VectorStore ABC and InMemoryVectorStore."""

from __future__ import annotations

import math
from typing import Any

import pytest

from exo.models.embeddings import cosine_similarity as _cosine_similarity
from exo.retrieval.types import Chunk, RetrievalError, RetrievalResult
from exo.retrieval.vector_store import (
    InMemoryVectorStore,
    VectorStore,
)

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


# ---------------------------------------------------------------------------
# VectorStore ABC
# ---------------------------------------------------------------------------


class TestVectorStoreABC:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            VectorStore()  # type: ignore[abstract]

    def test_concrete_subclass(self) -> None:
        class Dummy(VectorStore):
            async def add(
                self,
                chunks: list[Chunk],
                embeddings: list[list[float]],
            ) -> None:
                pass

            async def search(
                self,
                query_embedding: list[float],
                top_k: int = 5,
                filter: dict[str, Any] | None = None,
            ) -> list[RetrievalResult]:
                return []

            async def delete(self, document_id: str) -> None:
                pass

            async def clear(self) -> None:
                pass

        d = Dummy()
        assert isinstance(d, VectorStore)


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self) -> None:
        assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_similar_vectors(self) -> None:
        a = [1.0, 1.0]
        b = [1.0, 0.0]
        expected = 1.0 / math.sqrt(2)
        assert _cosine_similarity(a, b) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# InMemoryVectorStore — add
# ---------------------------------------------------------------------------


class TestInMemoryVectorStoreAdd:
    @pytest.mark.asyncio
    async def test_add_chunks(self) -> None:
        store = InMemoryVectorStore()
        chunks = [_chunk(content="a"), _chunk(content="b")]
        embeddings = [[1.0, 0.0], [0.0, 1.0]]
        await store.add(chunks, embeddings)
        assert len(store._chunks) == 2
        assert len(store._embeddings) == 2

    @pytest.mark.asyncio
    async def test_add_mismatched_lengths_raises(self) -> None:
        store = InMemoryVectorStore()
        with pytest.raises(RetrievalError, match="must match"):
            await store.add([_chunk()], [[1.0], [2.0]])

    @pytest.mark.asyncio
    async def test_add_empty(self) -> None:
        store = InMemoryVectorStore()
        await store.add([], [])
        assert len(store._chunks) == 0

    @pytest.mark.asyncio
    async def test_add_multiple_batches(self) -> None:
        store = InMemoryVectorStore()
        await store.add([_chunk(content="a")], [[1.0, 0.0]])
        await store.add([_chunk(content="b")], [[0.0, 1.0]])
        assert len(store._chunks) == 2


# ---------------------------------------------------------------------------
# InMemoryVectorStore — search
# ---------------------------------------------------------------------------


class TestInMemoryVectorStoreSearch:
    @pytest.mark.asyncio
    async def test_search_returns_ranked_results(self) -> None:
        store = InMemoryVectorStore()
        chunks = [
            _chunk(content="far", index=0),
            _chunk(content="close", index=1),
            _chunk(content="closest", index=2),
        ]
        embeddings = [
            [0.0, 1.0],  # orthogonal to query
            [0.7, 0.7],  # somewhat similar
            [1.0, 0.0],  # identical direction to query
        ]
        await store.add(chunks, embeddings)

        results = await store.search([1.0, 0.0], top_k=3)

        assert len(results) == 3
        assert results[0].chunk.content == "closest"
        assert results[0].score == pytest.approx(1.0)
        assert results[1].chunk.content == "close"
        assert results[2].chunk.content == "far"
        # Scores should be in descending order
        assert results[0].score >= results[1].score >= results[2].score

    @pytest.mark.asyncio
    async def test_search_top_k_limits_results(self) -> None:
        store = InMemoryVectorStore()
        chunks = [_chunk(content=f"c{i}", index=i) for i in range(10)]
        embeddings = [[float(i), 1.0] for i in range(10)]
        await store.add(chunks, embeddings)

        results = await store.search([1.0, 0.0], top_k=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_search_empty_store(self) -> None:
        store = InMemoryVectorStore()
        results = await store.search([1.0, 0.0], top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_search_with_metadata_filter(self) -> None:
        store = InMemoryVectorStore()
        chunks = [
            _chunk(content="a", index=0, metadata={"source": "web"}),
            _chunk(content="b", index=1, metadata={"source": "pdf"}),
            _chunk(content="c", index=2, metadata={"source": "web"}),
        ]
        embeddings = [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.5],
        ]
        await store.add(chunks, embeddings)

        results = await store.search([1.0, 0.0], top_k=10, filter={"source": "web"})
        assert len(results) == 2
        assert all(r.chunk.metadata["source"] == "web" for r in results)

    @pytest.mark.asyncio
    async def test_search_filter_no_match(self) -> None:
        store = InMemoryVectorStore()
        await store.add(
            [_chunk(metadata={"lang": "en"})],
            [[1.0, 0.0]],
        )
        results = await store.search([1.0, 0.0], filter={"lang": "fr"})
        assert results == []

    @pytest.mark.asyncio
    async def test_search_filter_multiple_keys(self) -> None:
        store = InMemoryVectorStore()
        chunks = [
            _chunk(content="a", index=0, metadata={"lang": "en", "source": "web"}),
            _chunk(content="b", index=1, metadata={"lang": "en", "source": "pdf"}),
        ]
        await store.add(chunks, [[1.0, 0.0], [1.0, 0.0]])

        results = await store.search([1.0, 0.0], filter={"lang": "en", "source": "pdf"})
        assert len(results) == 1
        assert results[0].chunk.content == "b"

    @pytest.mark.asyncio
    async def test_search_result_is_retrieval_result(self) -> None:
        store = InMemoryVectorStore()
        await store.add([_chunk()], [[1.0, 0.0]])
        results = await store.search([1.0, 0.0])
        assert len(results) == 1
        assert isinstance(results[0], RetrievalResult)
        assert isinstance(results[0].chunk, Chunk)
        assert isinstance(results[0].score, float)


# ---------------------------------------------------------------------------
# InMemoryVectorStore — delete
# ---------------------------------------------------------------------------


class TestInMemoryVectorStoreDelete:
    @pytest.mark.asyncio
    async def test_delete_removes_document_chunks(self) -> None:
        store = InMemoryVectorStore()
        chunks = [
            _chunk(document_id="doc-1", content="a", index=0),
            _chunk(document_id="doc-1", content="b", index=1),
            _chunk(document_id="doc-2", content="c", index=0),
        ]
        embeddings = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        await store.add(chunks, embeddings)

        await store.delete("doc-1")

        assert len(store._chunks) == 1
        remaining = list(store._chunks.values())[0]
        assert remaining.document_id == "doc-2"

    @pytest.mark.asyncio
    async def test_delete_nonexistent_document_is_noop(self) -> None:
        store = InMemoryVectorStore()
        await store.add([_chunk(document_id="doc-1")], [[1.0]])
        await store.delete("doc-999")
        assert len(store._chunks) == 1

    @pytest.mark.asyncio
    async def test_delete_then_search(self) -> None:
        store = InMemoryVectorStore()
        await store.add(
            [_chunk(document_id="doc-1"), _chunk(document_id="doc-2", index=1)],
            [[1.0, 0.0], [0.0, 1.0]],
        )
        await store.delete("doc-1")
        results = await store.search([1.0, 0.0], top_k=10)
        assert len(results) == 1
        assert results[0].chunk.document_id == "doc-2"


# ---------------------------------------------------------------------------
# InMemoryVectorStore — clear
# ---------------------------------------------------------------------------


class TestInMemoryVectorStoreClear:
    @pytest.mark.asyncio
    async def test_clear_empties_store(self) -> None:
        store = InMemoryVectorStore()
        await store.add(
            [_chunk(content="a"), _chunk(content="b")],
            [[1.0, 0.0], [0.0, 1.0]],
        )
        await store.clear()
        assert len(store._chunks) == 0
        assert len(store._embeddings) == 0

    @pytest.mark.asyncio
    async def test_clear_then_add(self) -> None:
        store = InMemoryVectorStore()
        await store.add([_chunk()], [[1.0]])
        await store.clear()
        await store.add([_chunk(content="new")], [[0.5]])
        assert len(store._chunks) == 1
        assert list(store._chunks.values())[0].content == "new"

    @pytest.mark.asyncio
    async def test_clear_empty_store(self) -> None:
        store = InMemoryVectorStore()
        await store.clear()
        assert len(store._chunks) == 0
