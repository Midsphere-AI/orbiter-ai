"""Abstract base class for retrievers and vector retriever implementation.

A ``Retriever`` accepts a text query and returns ranked
``RetrievalResult`` objects.  ``VectorRetriever`` combines an
``Embeddings`` provider with a ``VectorStore`` for dense semantic search.
"""

from __future__ import annotations

import abc
from typing import Any

from exo.models.embeddings import Embeddings  # pyright: ignore[reportMissingImports]
from exo.retrieval.types import (  # pyright: ignore[reportMissingImports]
    RetrievalError,
    RetrievalResult,
)
from exo.retrieval.vector_store import VectorStore  # pyright: ignore[reportMissingImports]


class Retriever(abc.ABC):
    """Abstract base class for retrievers.

    Subclasses must implement ``retrieve`` to return scored chunks
    for a given text query.
    """

    @abc.abstractmethod
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        """Retrieve relevant chunks for a query.

        Args:
            query: The search query text.
            top_k: Maximum number of results to return.
            **kwargs: Additional retriever-specific parameters.

        Returns:
            A list of ``RetrievalResult`` objects ranked by relevance
            (highest score first).
        """


class VectorRetriever(Retriever):
    """Dense vector retriever using embeddings and a vector store.

    Embeds the query text, searches the vector store for similar chunks,
    and optionally filters results below a score threshold.

    Args:
        embeddings: The embedding provider for vectorising queries.
        store: The vector store to search against.
        score_threshold: Optional minimum score; results below this are
            excluded.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        store: VectorStore,
        *,
        score_threshold: float | None = None,
    ) -> None:
        self.embeddings = embeddings
        self.store = store
        self.score_threshold = score_threshold

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        """Retrieve chunks by embedding the query and searching the store.

        Args:
            query: The search query text.
            top_k: Maximum number of results to return.
            **kwargs: Passed through to ``VectorStore.search`` (e.g. ``filter``).

        Returns:
            A list of ``RetrievalResult`` objects ranked by similarity.
        """
        try:
            query_embedding = await self.embeddings.embed(query)
        except Exception as exc:
            raise RetrievalError(f"Embedding failed: {exc}") from exc

        try:
            results = await self.store.search(query_embedding, top_k=top_k, **kwargs)
        except Exception as exc:
            raise RetrievalError(f"Vector store search failed: {exc}") from exc

        if self.score_threshold is not None:
            results = [r for r in results if r.score >= self.score_threshold]

        return results
