"""Lightweight in-memory vector store for Exo.

``InMemoryVectorStore`` is a pure-Python cosine-similarity store with no heavy
dependencies (no chromadb, no asyncpg, no numpy).  It is the canonical
lightweight store that ``exo-memory`` and ``exo-retrieval`` will use as their
foundation after the C1 consolidation wave.

The store operates on plain ``(document_id, text, metadata)`` tuples rather
than the ``Chunk`` / ``RetrievalResult`` types from ``exo-retrieval`` so that
``exo-memory`` (which sits *below* ``exo-retrieval`` in the dependency graph)
can import it without a circular dep.

For retrieval-layer usage (``Chunk``/``RetrievalResult``), ``exo-retrieval``'s
own ``InMemoryVectorStore`` remains available.
"""

from __future__ import annotations

import abc
import logging
from typing import Any

from exo.models.embeddings import cosine_similarity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class VectorSearchResult:
    """A single result from an ``InMemoryVectorStore`` search.

    Attributes:
        id: The document/item identifier.
        text: The text content that was stored.
        score: Cosine similarity score (higher is more similar).
        metadata: Arbitrary metadata stored alongside the item.
    """

    __slots__ = ("id", "metadata", "score", "text")

    def __init__(
        self,
        *,
        id: str,
        text: str,
        score: float,
        metadata: dict[str, Any],
    ) -> None:
        self.id = id
        self.text = text
        self.score = score
        self.metadata = metadata

    def __repr__(self) -> str:
        return f"VectorSearchResult(id={self.id!r}, score={self.score:.4f})"


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------


class VectorStoreBase(abc.ABC):
    """Abstract base for lightweight vector stores.

    Provides a minimal CRUD interface over text + embedding pairs.
    """

    @abc.abstractmethod
    async def add(
        self,
        id: str,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add (or overwrite) a single item.

        Args:
            id: Unique identifier for this item.
            text: Raw text content.
            embedding: Pre-computed embedding vector.
            metadata: Optional key-value metadata.
        """

    @abc.abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Search for the most similar items to a query embedding.

        Args:
            query_embedding: The query vector to compare against.
            top_k: Maximum number of results to return.
            filter: Optional metadata filter (exact match on each key).

        Returns:
            A list of ``VectorSearchResult`` objects ranked by similarity
            (highest score first).
        """

    @abc.abstractmethod
    async def delete(self, id: str) -> None:
        """Delete an item by ID.

        Args:
            id: The ID of the item to remove.
        """

    @abc.abstractmethod
    async def clear(self) -> None:
        """Remove all stored items."""

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the number of stored items."""


# ---------------------------------------------------------------------------
# InMemoryVectorStore
# ---------------------------------------------------------------------------


class InMemoryVectorStore(VectorStoreBase):
    """Pure-Python in-memory vector store using cosine similarity.

    Stores items in plain Python dicts.  Suitable for development, testing,
    and small datasets (up to ~10k items before linear scan becomes a
    bottleneck).  No external dependencies required.

    Example::

        store = InMemoryVectorStore()
        await store.add("doc-1", "hello world", [0.1, 0.2, 0.9])
        results = await store.search([0.1, 0.2, 0.9], top_k=3)
        print(results[0].id, results[0].score)
    """

    def __init__(self) -> None:
        self._texts: dict[str, str] = {}
        self._embeddings: dict[str, list[float]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    async def add(
        self,
        id: str,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add (or overwrite) a single item.

        Args:
            id: Unique identifier for this item.
            text: Raw text content.
            embedding: Pre-computed embedding vector.
            metadata: Optional key-value metadata.
        """
        self._texts[id] = text
        self._embeddings[id] = embedding
        self._metadata[id] = dict(metadata) if metadata else {}
        logger.debug("InMemoryVectorStore: added id=%s dim=%d", id, len(embedding))

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar items using cosine similarity.

        Args:
            query_embedding: The query vector to compare against.
            top_k: Maximum number of results to return.
            filter: Optional metadata filter (exact match on each key).

        Returns:
            A list of ``VectorSearchResult`` objects ranked by similarity
            (highest score first).
        """
        scored: list[tuple[float, str]] = []

        for item_id, embedding in self._embeddings.items():
            if filter is not None:
                meta = self._metadata.get(item_id, {})
                if not all(meta.get(k) == v for k, v in filter.items()):
                    continue

            score = cosine_similarity(query_embedding, embedding)
            scored.append((score, item_id))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, item_id in scored[:top_k]:
            results.append(
                VectorSearchResult(
                    id=item_id,
                    text=self._texts[item_id],
                    score=score,
                    metadata=self._metadata.get(item_id, {}),
                )
            )

        logger.debug(
            "InMemoryVectorStore: search top_k=%d candidates=%d results=%d",
            top_k,
            len(scored),
            len(results),
        )
        return results

    async def delete(self, id: str) -> None:
        """Delete an item by ID.

        Args:
            id: The ID of the item to remove.

        Note:
            Silently does nothing if the ID does not exist.
        """
        self._texts.pop(id, None)
        self._embeddings.pop(id, None)
        self._metadata.pop(id, None)
        logger.debug("InMemoryVectorStore: deleted id=%s", id)

    async def clear(self) -> None:
        """Remove all stored items."""
        count = len(self._texts)
        self._texts.clear()
        self._embeddings.clear()
        self._metadata.clear()
        logger.debug("InMemoryVectorStore: cleared %d items", count)

    def __len__(self) -> int:
        return len(self._texts)

    def __repr__(self) -> str:
        return f"InMemoryVectorStore(items={len(self._texts)})"
