"""Semantic reranking using the canonical exo.models embeddings layer.

Reranks search results by computing cosine similarity between query
and result embeddings.  Supports OpenAI, Gemini (via HTTPEmbeddings REST
wrapper), and Vertex AI embedding providers, all sourced from
``exo.models``.  Falls back to keyword overlap scoring when no API key
is configured.

Provider auto-detection order:
    1. Gemini  — if ``GEMINI_API_KEY`` is set
       Implemented via ``exo.models.HTTPEmbeddings`` pointed at the Gemini
       batchEmbedContents REST endpoint.  The Gemini REST response wraps
       vectors under ``embeddings[*].values``; HTTPEmbeddings' configurable
       ``output_field`` / ``vector_field`` parameters handle that shape.
       Note: Gemini batchEmbedContents uses ``?key=`` query-param auth and
       a ``{"requests": [...]}`` body that does NOT match HTTPEmbeddings'
       simple array input.  We therefore use a thin ``_GeminiEmbeddings``
       subclass (still httpx-based, still implements the ``Embeddings`` ABC)
       rather than introducing a 4th bespoke urllib impl.
    2. Vertex AI — if ``GOOGLE_CLOUD_PROJECT`` and ``GOOGLE_CLOUD_ACCESS_TOKEN`` are set
    3. OpenAI  — if ``OPENAI_API_KEY`` is set
    4. Keyword overlap fallback

Usage:
    from exo.search.tools.embeddings import rerank_search_results
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from exo.models.embeddings import (
    EmbeddingError,
    Embeddings,
    OpenAIEmbeddings,
    VertexEmbeddings,
    cosine_similarity,
)

logger = logging.getLogger(__name__)

_GEMINI_DEFAULT_MODEL = "gemini-embedding-2-preview"
_GEMINI_DIMENSION = 3072
_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

_OPENAI_DEFAULT_MODEL = "text-embedding-3-small"
_OPENAI_DIMENSION = 1536


# ---------------------------------------------------------------------------
# Gemini embeddings — thin httpx-based Embeddings subclass
# ---------------------------------------------------------------------------


class _GeminiEmbeddings(Embeddings):
    """Embeddings via the Gemini batchEmbedContents REST API.

    Implements the canonical ``Embeddings`` ABC from ``exo.models`` using
    ``httpx`` — no ``google-genai`` SDK required.

    Args:
        api_key: Gemini API key.
        model: Embedding model name. Defaults to ``"gemini-embedding-2-preview"``.
        dimension: Vector dimensionality. Defaults to 3072.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = _GEMINI_DEFAULT_MODEL,
        dimension: int = _GEMINI_DIMENSION,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """The dimensionality of the embedding vectors."""
        return self._dimension

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: The text to embed.
        """
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts via the Gemini batchEmbedContents API.

        Args:
            texts: A list of texts to embed.

        Returns:
            A list of dense vectors, one per input text.

        Raises:
            EmbeddingError: If the API call fails.
        """
        if not texts:
            return []

        url = f"{_GEMINI_BASE_URL}/{self._model}:batchEmbedContents"
        requests_list: list[dict[str, Any]] = [
            {
                "model": f"models/{self._model}",
                "content": {"parts": [{"text": t}]},
                "taskType": "SEMANTIC_SIMILARITY",
                "outputDimensionality": self._dimension,
            }
            for t in texts
        ]
        payload: dict[str, Any] = {"requests": requests_list}
        params = {"key": self._api_key}

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    url,
                    json=payload,
                    params=params,
                    headers={"Content-Type": "application/json"},
                    timeout=60.0,
                )
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise EmbeddingError(
                    f"Gemini embeddings API error: {exc.response.status_code}",
                    operation="embed",
                    details={"status": exc.response.status_code, "body": exc.response.text},
                ) from exc
            except httpx.HTTPError as exc:
                raise EmbeddingError(
                    f"Gemini embeddings request failed: {exc}",
                    operation="embed",
                ) from exc

        body = resp.json()
        try:
            return [emb["values"] for emb in body["embeddings"]]
        except (KeyError, TypeError) as exc:
            raise EmbeddingError(
                f"Failed to extract Gemini embeddings from response: {exc}",
                operation="embed",
            ) from exc


# ---------------------------------------------------------------------------
# Provider auto-detection
# ---------------------------------------------------------------------------


def _build_embedder() -> Embeddings | None:
    """Auto-detect embedding provider and return a canonical Embeddings instance.

    Returns ``None`` when no provider credentials are available.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    gcp_project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    gcp_token = os.environ.get("GOOGLE_CLOUD_ACCESS_TOKEN", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")

    if gemini_key:
        model = os.environ.get("EXO_SEARCH_EMBEDDING_MODEL", _GEMINI_DEFAULT_MODEL)
        logger.debug("embedding provider=gemini model=%s", model)
        return _GeminiEmbeddings(api_key=gemini_key, model=model)

    if gcp_project and gcp_token:
        model = os.environ.get("EXO_SEARCH_EMBEDDING_MODEL", "text-embedding-005")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        logger.debug("embedding provider=vertex model=%s", model)
        return VertexEmbeddings(
            api_key=gcp_token,
            project=gcp_project,
            model=model,
            location=location,
        )

    if openai_key:
        model = os.environ.get("EXO_SEARCH_EMBEDDING_MODEL", _OPENAI_DEFAULT_MODEL)
        logger.debug("embedding provider=openai model=%s", model)
        return OpenAIEmbeddings(
            api_key=openai_key,
            model=model,
            dimension=_OPENAI_DIMENSION,
        )

    return None


# ---------------------------------------------------------------------------
# Keyword fallback scorer
# ---------------------------------------------------------------------------


def _keyword_score(query_words: list[str], text: str) -> float:
    """Compute keyword overlap score between query words and text.

    Args:
        query_words: List of lowercase query words.
        text: Text to score against.
    """
    text_lower = text.lower()
    return sum(1 for w in query_words if w in text_lower) / max(len(query_words), 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def rerank_search_results(
    query: str,
    results: list,
    top_k: int | None = None,
) -> list:
    """Rerank SearchResult objects by semantic relevance to the query.

    Uses embedding cosine similarity when an API key is available,
    falls back to keyword overlap scoring otherwise.

    Args:
        query: The original search query.
        results: List of SearchResult objects to rerank.
        top_k: Number of top results to return. Defaults to all results.
    """
    if not results:
        return results
    if top_k is None:
        top_k = len(results)

    texts = [query] + [f"{r.title} {r.content[:500]}" for r in results]

    scored: list[tuple[float, object]] = []
    embedder = _build_embedder()
    if embedder is not None:
        try:
            embeddings = await embedder.embed_batch(texts)
            query_emb = embeddings[0]
            for i, r in enumerate(results):
                sim = cosine_similarity(query_emb, embeddings[i + 1])
                scored.append((sim, r))
        except Exception as exc:
            logger.warning("embedding failed, falling back to keyword scoring: %s", exc)
            embedder = None

    if embedder is None:
        query_words = query.lower().split()
        for r in results:
            text = f"{r.title} {r.content[:500]}"
            score = _keyword_score(query_words, text)
            scored.append((score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    logger.debug("reranked %d results", len(scored))
    return [r for _, r in scored[:top_k]]
