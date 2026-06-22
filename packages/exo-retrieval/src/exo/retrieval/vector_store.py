"""Abstract base class for vector stores and in-memory implementation.

A ``VectorStore`` persists document chunks alongside their embedding
vectors and supports similarity search.
"""

from __future__ import annotations

import abc
from typing import Any

from exo.models.embeddings import (
    cosine_similarity as _cosine_similarity,  # pyright: ignore[reportMissingImports]
)
from exo.retrieval.types import (  # pyright: ignore[reportMissingImports]
    Chunk,
    RetrievalError,
    RetrievalResult,
)


class VectorStore(abc.ABC):
    """Abstract base class for vector stores.

    Subclasses must implement ``add``, ``search``, ``delete``, and ``clear``.
    """

    @abc.abstractmethod
    async def add(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Add chunks with their embedding vectors.

        Args:
            chunks: The document chunks to store.
            embeddings: Corresponding embedding vectors (one per chunk).

        Raises:
            ValueError: If the number of chunks and embeddings differ.
        """

    @abc.abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Search for the most similar chunks to a query embedding.

        Args:
            query_embedding: The query vector to compare against.
            top_k: Maximum number of results to return.
            filter: Optional metadata filter (exact match on each key).

        Returns:
            A list of ``RetrievalResult`` objects ranked by similarity
            (highest score first).
        """

    @abc.abstractmethod
    async def delete(self, document_id: str) -> None:
        """Delete all chunks belonging to a document.

        Args:
            document_id: The ID of the document whose chunks to remove.
        """

    @abc.abstractmethod
    async def clear(self) -> None:
        """Remove all stored chunks and embeddings."""

    async def close(self) -> None:
        """Release any resources held by this store.

        The default implementation is a no-op.  Backends that manage
        connection pools or file handles (e.g. ``PgVectorStore``) override
        this method.
        """
        return  # default no-op; override in backends that hold resources.


class InMemoryVectorStore(VectorStore):
    """In-memory vector store using cosine similarity.

    Stores chunks and embeddings in plain Python dicts.  Suitable for
    development, testing, and small datasets.

    Args:
        max_size: Optional upper bound on the number of stored chunks.  When
            set, the oldest entries (by insertion order) are evicted to keep
            the store within the limit.  Defaults to ``None`` (unlimited),
            preserving backward-compatible behaviour.
    """

    def __init__(self, max_size: int | None = None) -> None:
        self._chunks: dict[int, Chunk] = {}
        self._embeddings: dict[int, list[float]] = {}
        self._next_id: int = 0
        self._max_size = max_size

    async def add(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Add chunks with their embedding vectors.

        When *max_size* is set and adding the new chunks would exceed it, the
        oldest stored entries are evicted first (insertion-order FIFO).
        """
        if len(chunks) != len(embeddings):
            msg = f"Number of chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must match"
            raise RetrievalError(
                msg,
                hint="Ensure embed_batch() is called on the same chunk list before calling add().",
            )

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            self._chunks[self._next_id] = chunk
            self._embeddings[self._next_id] = embedding
            self._next_id += 1

        # Evict oldest entries when max_size is set and the store exceeds it.
        if self._max_size is not None:
            while len(self._chunks) > self._max_size:
                oldest_key = next(iter(self._chunks))
                del self._chunks[oldest_key]
                del self._embeddings[oldest_key]

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Search for similar chunks using cosine similarity."""
        scored: list[tuple[float, Chunk]] = []

        for idx, chunk in self._chunks.items():
            if filter is not None:
                if not all(chunk.metadata.get(k) == v for k, v in filter.items()):
                    continue

            score = _cosine_similarity(query_embedding, self._embeddings[idx])
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [RetrievalResult(chunk=chunk, score=score) for score, chunk in scored[:top_k]]

    async def delete(self, document_id: str) -> None:
        """Delete all chunks belonging to a document."""
        ids_to_remove = [
            idx for idx, chunk in self._chunks.items() if chunk.document_id == document_id
        ]
        for idx in ids_to_remove:
            del self._chunks[idx]
            del self._embeddings[idx]

    async def clear(self) -> None:
        """Remove all stored chunks and embeddings."""
        self._chunks.clear()
        self._embeddings.clear()
        self._next_id = 0
