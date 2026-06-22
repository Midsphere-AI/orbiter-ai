"""Hybrid retriever combining dense and sparse results via Reciprocal Rank Fusion.

``HybridRetriever`` delegates to a ``VectorRetriever`` (dense) and a
``SparseRetriever`` (sparse), then merges the two ranked lists using
`RRF <https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf>`_.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from exo.retrieval.retriever import Retriever  # pyright: ignore[reportMissingImports]
from exo.retrieval.types import (  # pyright: ignore[reportMissingImports]
    RetrievalError,
    RetrievalResult,
)

logger = logging.getLogger(__name__)

# Default RRF constant (controls how much low-ranked results are penalised).
_DEFAULT_K = 60


class HybridRetriever(Retriever):
    """Hybrid retriever that fuses dense and sparse results via RRF.

    Calls both retrievers concurrently, then combines their ranked lists
    using weighted Reciprocal Rank Fusion.

    Args:
        vector_retriever: Dense (embedding-based) retriever.
        sparse_retriever: Sparse (keyword-based) retriever.
        k: RRF constant — higher values flatten the rank curve (default 60).
        vector_weight: Weight for the vector retriever's contribution
            (0.0–1.0).  The sparse retriever receives ``1 - vector_weight``.
    """

    def __init__(
        self,
        vector_retriever: Retriever,
        sparse_retriever: Retriever,
        *,
        k: int = _DEFAULT_K,
        vector_weight: float = 0.5,
    ) -> None:
        self.vector_retriever = vector_retriever
        self.sparse_retriever = sparse_retriever
        self.k = k
        self.vector_weight = vector_weight

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        """Retrieve chunks by fusing dense and sparse results with RRF.

        Args:
            query: The search query text.
            top_k: Maximum number of results to return.
            **kwargs: Passed through to both underlying retrievers.

        Returns:
            A list of ``RetrievalResult`` objects ranked by fused RRF score
            (highest first).
        """
        raw = await asyncio.gather(
            self.vector_retriever.retrieve(query, top_k=top_k, **kwargs),
            self.sparse_retriever.retrieve(query, top_k=top_k, **kwargs),
            return_exceptions=True,
        )
        vector_raw, sparse_raw = raw[0], raw[1]

        vector_failed = isinstance(vector_raw, BaseException)
        sparse_failed = isinstance(sparse_raw, BaseException)

        if vector_failed and sparse_failed:
            raise RetrievalError(
                "Both vector and sparse retrieval failed.",
                context={
                    "vector_error": str(vector_raw),
                    "sparse_error": str(sparse_raw),
                },
                hint="Check both retriever configurations and that the vector store is reachable.",
            )

        if vector_failed:
            logger.warning(
                "HybridRetriever: vector retrieval failed (%s); continuing with sparse results only.",
                vector_raw,
            )
            vector_results: list[RetrievalResult] = []
        else:
            vector_results = vector_raw  # type: ignore[assignment]

        if sparse_failed:
            logger.warning(
                "HybridRetriever: sparse retrieval failed (%s); continuing with vector results only.",
                sparse_raw,
            )
            sparse_results: list[RetrievalResult] = []
        else:
            sparse_results = sparse_raw  # type: ignore[assignment]

        # Build a mapping from chunk identity to fused score.
        # Use (document_id, index) as the dedup key since Chunk is frozen.
        scores: dict[tuple[str, int], float] = {}
        chunks: dict[tuple[str, int], RetrievalResult] = {}

        sparse_weight = 1.0 - self.vector_weight

        for rank, result in enumerate(vector_results, start=1):
            key = (result.chunk.document_id, result.chunk.index)
            rrf = self.vector_weight * (1.0 / (self.k + rank))
            scores[key] = scores.get(key, 0.0) + rrf
            chunks[key] = result

        for rank, result in enumerate(sparse_results, start=1):
            key = (result.chunk.document_id, result.chunk.index)
            rrf = sparse_weight * (1.0 / (self.k + rank))
            scores[key] = scores.get(key, 0.0) + rrf
            if key not in chunks:
                chunks[key] = result

        # Sort by fused score descending; break ties deterministically by doc key.
        ranked = sorted(scores.items(), key=lambda x: (x[1], x[0][0], x[0][1]), reverse=True)

        return [
            RetrievalResult(
                chunk=chunks[key].chunk,
                score=score,
                metadata=chunks[key].metadata,
            )
            for key, score in ranked[:top_k]
        ]
