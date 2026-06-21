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


class InMemoryVectorStore(VectorStore):
    """In-memory vector store using cosine similarity.

    Stores chunks and embeddings in plain Python dicts.  Suitable for
    development, testing, and small datasets.
    """

    def __init__(self) -> None:
        self._chunks: dict[int, Chunk] = {}
        self._embeddings: dict[int, list[float]] = {}
        self._next_id: int = 0

    async def add(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Add chunks with their embedding vectors."""
        if len(chunks) != len(embeddings):
            msg = f"Number of chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must match"
            raise ValueError(msg)

        for chunk, embedding in zip(chunks, embeddings):
            self._chunks[self._next_id] = chunk
            self._embeddings[self._next_id] = embedding
            self._next_id += 1

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
