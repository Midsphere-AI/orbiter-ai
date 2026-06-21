"""Tests for InMemoryVectorStore and VectorStoreBase ABC.

Covers:
- ``VectorStoreBase`` ABC contract (cannot instantiate, concrete subclass works)
- ``InMemoryVectorStore`` add / search / delete / clear
- Metadata filtering in search
- Cosine scoring correctness
- ``VectorSearchResult`` attributes
"""

from __future__ import annotations

from typing import Any

import pytest

from exo.models.vector import InMemoryVectorStore, VectorSearchResult, VectorStoreBase

# ---------------------------------------------------------------------------
# VectorStoreBase ABC
# ---------------------------------------------------------------------------


class TestVectorStoreBaseABC:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            VectorStoreBase()  # type: ignore[abstract]

    def test_concrete_subclass(self) -> None:
        class Dummy(VectorStoreBase):
            async def add(
                self,
                id: str,
                text: str,
                embedding: list[float],
                metadata: dict[str, Any] | None = None,
            ) -> None:
                pass

            async def search(
                self,
                query_embedding: list[float],
                top_k: int = 5,
                filter: dict[str, Any] | None = None,
            ) -> list[VectorSearchResult]:
                return []

            async def delete(self, id: str) -> None:
                pass

            async def clear(self) -> None:
                pass

            def __len__(self) -> int:
                return 0

        d = Dummy()
        assert isinstance(d, VectorStoreBase)
        assert len(d) == 0


# ---------------------------------------------------------------------------
# VectorSearchResult
# ---------------------------------------------------------------------------


class TestVectorSearchResult:
    def test_attributes(self) -> None:
        r = VectorSearchResult(id="x", text="hello", score=0.9, metadata={"k": "v"})
        assert r.id == "x"
        assert r.text == "hello"
        assert r.score == 0.9
        assert r.metadata == {"k": "v"}

    def test_repr(self) -> None:
        r = VectorSearchResult(id="abc", text="t", score=0.75, metadata={})
        assert "abc" in repr(r)
        assert "0.7500" in repr(r)


# ---------------------------------------------------------------------------
# InMemoryVectorStore — add / len
# ---------------------------------------------------------------------------


class TestInMemoryVectorStoreAdd:
    async def test_add_single_item(self) -> None:
        store = InMemoryVectorStore()
        await store.add("id-1", "hello world", [1.0, 0.0])
        assert len(store) == 1

    async def test_add_multiple_items(self) -> None:
        store = InMemoryVectorStore()
        await store.add("id-1", "first", [1.0, 0.0])
        await store.add("id-2", "second", [0.0, 1.0])
        assert len(store) == 2

    async def test_add_with_metadata(self) -> None:
        store = InMemoryVectorStore()
        await store.add("id-1", "doc", [1.0], metadata={"source": "web"})
        results = await store.search([1.0], top_k=1)
        assert results[0].metadata["source"] == "web"

    async def test_add_overwrites_existing_id(self) -> None:
        store = InMemoryVectorStore()
        await store.add("id-1", "original", [1.0, 0.0])
        await store.add("id-1", "updated", [0.5, 0.5])
        assert len(store) == 1
        results = await store.search([0.5, 0.5], top_k=1)
        assert results[0].text == "updated"

    async def test_add_empty_metadata_default(self) -> None:
        store = InMemoryVectorStore()
        await store.add("id-1", "text", [1.0])
        results = await store.search([1.0], top_k=1)
        assert results[0].metadata == {}

    async def test_repr(self) -> None:
        store = InMemoryVectorStore()
        await store.add("id-1", "text", [1.0])
        assert "InMemoryVectorStore" in repr(store)
        assert "1" in repr(store)


# ---------------------------------------------------------------------------
# InMemoryVectorStore — search / scoring
# ---------------------------------------------------------------------------


class TestInMemoryVectorStoreSearch:
    async def test_search_returns_ranked_results(self) -> None:
        store = InMemoryVectorStore()
        await store.add("far", "far item", [0.0, 1.0])      # orthogonal to query
        await store.add("close", "close item", [0.7, 0.7])  # somewhat similar
        await store.add("closest", "closest item", [1.0, 0.0])  # identical direction

        results = await store.search([1.0, 0.0], top_k=3)

        assert len(results) == 3
        assert results[0].id == "closest"
        assert results[0].score == pytest.approx(1.0)
        assert results[1].id == "close"
        assert results[2].id == "far"

    async def test_search_scores_descending(self) -> None:
        store = InMemoryVectorStore()
        for i in range(5):
            await store.add(f"item-{i}", f"content {i}", [float(i), 1.0])

        results = await store.search([1.0, 0.0], top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    async def test_search_top_k_limits(self) -> None:
        store = InMemoryVectorStore()
        for i in range(10):
            await store.add(f"id-{i}", f"item {i}", [float(i), 1.0])

        results = await store.search([1.0, 0.0], top_k=3)
        assert len(results) == 3

    async def test_search_empty_store_returns_empty(self) -> None:
        store = InMemoryVectorStore()
        results = await store.search([1.0, 0.0])
        assert results == []

    async def test_search_result_has_correct_text(self) -> None:
        store = InMemoryVectorStore()
        await store.add("id-1", "my text", [1.0, 0.0])
        results = await store.search([1.0, 0.0])
        assert results[0].text == "my text"

    async def test_search_result_is_vector_search_result(self) -> None:
        store = InMemoryVectorStore()
        await store.add("id-1", "doc", [1.0, 0.0])
        results = await store.search([1.0, 0.0])
        assert isinstance(results[0], VectorSearchResult)

    async def test_search_zero_magnitude_query_returns_zero_scores(self) -> None:
        store = InMemoryVectorStore()
        await store.add("id-1", "doc", [1.0, 0.0])
        results = await store.search([0.0, 0.0])
        assert results[0].score == 0.0


# ---------------------------------------------------------------------------
# InMemoryVectorStore — metadata filter
# ---------------------------------------------------------------------------


class TestInMemoryVectorStoreFilter:
    async def test_filter_single_key(self) -> None:
        store = InMemoryVectorStore()
        await store.add("a", "web doc", [1.0, 0.0], metadata={"source": "web"})
        await store.add("b", "pdf doc", [1.0, 0.0], metadata={"source": "pdf"})
        await store.add("c", "web doc 2", [0.5, 0.5], metadata={"source": "web"})

        results = await store.search([1.0, 0.0], top_k=10, filter={"source": "web"})
        assert len(results) == 2
        assert all(r.metadata["source"] == "web" for r in results)

    async def test_filter_multiple_keys(self) -> None:
        store = InMemoryVectorStore()
        await store.add("a", "en web", [1.0], metadata={"lang": "en", "source": "web"})
        await store.add("b", "en pdf", [1.0], metadata={"lang": "en", "source": "pdf"})
        await store.add("c", "fr web", [1.0], metadata={"lang": "fr", "source": "web"})

        results = await store.search([1.0], filter={"lang": "en", "source": "pdf"})
        assert len(results) == 1
        assert results[0].id == "b"

    async def test_filter_no_match_returns_empty(self) -> None:
        store = InMemoryVectorStore()
        await store.add("a", "doc", [1.0], metadata={"lang": "en"})
        results = await store.search([1.0], filter={"lang": "fr"})
        assert results == []

    async def test_no_filter_returns_all(self) -> None:
        store = InMemoryVectorStore()
        await store.add("a", "doc a", [1.0, 0.0], metadata={"x": 1})
        await store.add("b", "doc b", [0.0, 1.0], metadata={"x": 2})
        results = await store.search([1.0, 0.0], top_k=10)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# InMemoryVectorStore — delete
# ---------------------------------------------------------------------------


class TestInMemoryVectorStoreDelete:
    async def test_delete_removes_item(self) -> None:
        store = InMemoryVectorStore()
        await store.add("id-1", "hello", [1.0])
        await store.add("id-2", "world", [0.5])
        await store.delete("id-1")
        assert len(store) == 1
        results = await store.search([1.0])
        assert results[0].id == "id-2"

    async def test_delete_nonexistent_is_noop(self) -> None:
        store = InMemoryVectorStore()
        await store.add("id-1", "doc", [1.0])
        await store.delete("does-not-exist")
        assert len(store) == 1

    async def test_delete_empty_store_is_noop(self) -> None:
        store = InMemoryVectorStore()
        await store.delete("anything")
        assert len(store) == 0

    async def test_delete_then_search(self) -> None:
        store = InMemoryVectorStore()
        await store.add("id-1", "first", [1.0, 0.0])
        await store.add("id-2", "second", [0.0, 1.0])
        await store.delete("id-1")
        results = await store.search([1.0, 0.0], top_k=10)
        assert len(results) == 1
        assert results[0].id == "id-2"


# ---------------------------------------------------------------------------
# InMemoryVectorStore — clear
# ---------------------------------------------------------------------------


class TestInMemoryVectorStoreClear:
    async def test_clear_empties_store(self) -> None:
        store = InMemoryVectorStore()
        await store.add("a", "text a", [1.0, 0.0])
        await store.add("b", "text b", [0.0, 1.0])
        await store.clear()
        assert len(store) == 0

    async def test_clear_then_search_empty(self) -> None:
        store = InMemoryVectorStore()
        await store.add("a", "text", [1.0])
        await store.clear()
        results = await store.search([1.0])
        assert results == []

    async def test_clear_then_add_works(self) -> None:
        store = InMemoryVectorStore()
        await store.add("old", "old text", [1.0])
        await store.clear()
        await store.add("new", "new text", [0.5])
        assert len(store) == 1
        results = await store.search([0.5])
        assert results[0].id == "new"

    async def test_clear_empty_store_is_noop(self) -> None:
        store = InMemoryVectorStore()
        await store.clear()
        assert len(store) == 0
