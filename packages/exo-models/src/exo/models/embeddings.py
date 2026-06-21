"""Canonical embeddings layer for Exo.

Provides the ``Embeddings`` ABC and three concrete providers — OpenAI, Vertex AI,
and a generic HTTP endpoint — plus a ``cosine_similarity`` helper.

All providers use ``httpx`` for async HTTP so the heavy provider SDKs (openai,
google-genai) are **not** required here.  This module is intentionally kept
light so that ``exo-memory`` and ``exo-search`` can import it without pulling
in SDK-heavy extras.

Design decisions versus the three duplicate impls being consolidated:

- Interface: adopts ``exo-retrieval``'s async-first design (``embed`` / ``embed_batch``
  on the ABC) rather than ``exo-memory``'s sync+async pair (``embed`` / ``aembed``).
  The retrieval design is cleaner and the concrete impls already do the right thing.
- ``cosine_similarity``: public (not prefixed ``_``), uses ``strict=False`` in zip
  so it tolerates vectors of equal length without raising on mismatched input
  (follows ``exo-memory``'s looser convention; the exo-search version uses
  ``strict=True`` which can raise ValueError on mismatched lengths).
- Error type: defines ``EmbeddingError`` locally (subclass of ``Exception``) so
  this module has zero deps on ``exo-retrieval`` or ``exo-core``.  A follow-up
  wave may unify error hierarchies.
- VertexEmbeddings: uses the Vertex AI Prediction REST API (same as the httpx-based
  exo-retrieval impl) rather than the ``google-genai`` SDK used by ``exo-memory``.
  The REST approach avoids SDK deps and is already well-tested.  The ``service_account_base64``
  kwarg from exo-memory's version is **not** carried over here; Vertex REST auth via
  a Bearer token covers the common cases (pass an access token or integrate with
  google-auth externally).  A follow-up may add service-account support.
"""

from __future__ import annotations

import abc
import logging
import math
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
_DEFAULT_OPENAI_DIMENSION = 1536
_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"

_DEFAULT_VERTEX_MODEL = "text-embedding-005"
_DEFAULT_VERTEX_DIMENSION = 768
_DEFAULT_VERTEX_LOCATION = "us-central1"


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class EmbeddingError(Exception):
    """Raised when an embedding operation fails.

    Attributes:
        operation: The operation that failed (e.g. ``"embed"``).
        details: Optional extra context about the failure.
    """

    def __init__(
        self,
        message: str,
        *,
        operation: str = "embed",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.operation = operation
        self.details = details or {}


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------


class Embeddings(abc.ABC):
    """Abstract base class for text embedding providers.

    Subclasses must implement ``embed``, ``embed_batch``, and the
    ``dimension`` property.

    Example::

        vecs = await embedder.embed_batch(["hello", "world"])
        assert len(vecs[0]) == embedder.dimension
    """

    @abc.abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: The text to embed.

        Returns:
            A dense vector of length ``dimension``.
        """

    @abc.abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single call.

        Args:
            texts: A list of texts to embed.

        Returns:
            A list of dense vectors, one per input text.
        """

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """The dimensionality of the embedding vectors."""


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two dense vectors.

    Returns 0.0 when either vector has zero magnitude.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [-1, 1], or 0.0 for zero vectors.
    """
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# OpenAIEmbeddings
# ---------------------------------------------------------------------------


class OpenAIEmbeddings(Embeddings):
    """Embedding provider backed by the OpenAI embeddings API.

    Uses ``httpx`` for async HTTP — the ``openai`` SDK is **not** required.
    Compatible with any API that implements the OpenAI embeddings endpoint
    (OpenAI, Azure OpenAI, vLLM, Ollama OpenAI-compat, etc.).

    Args:
        api_key: OpenAI API key.
        model: Embedding model name. Defaults to ``"text-embedding-3-small"``.
        dimension: Vector dimensionality. Defaults to 1536.
        base_url: API base URL. Defaults to ``https://api.openai.com/v1``.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = _DEFAULT_OPENAI_MODEL,
        dimension: int = _DEFAULT_OPENAI_DIMENSION,
        base_url: str = _DEFAULT_OPENAI_BASE_URL,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._dimension = dimension
        self._base_url = base_url.rstrip("/")

    @property
    def dimension(self) -> int:
        """The dimensionality of the embedding vectors."""
        return self._dimension

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string via the OpenAI API.

        Args:
            text: The text to embed.

        Returns:
            A dense vector of length ``dimension``.
        """
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single OpenAI API call.

        Args:
            texts: A list of texts to embed.

        Returns:
            A list of dense vectors, one per input text.

        Raises:
            EmbeddingError: If the API call fails.
        """
        if not texts:
            return []

        payload: dict[str, Any] = {
            "input": texts,
            "model": self._model,
        }
        if self._dimension:
            payload["dimensions"] = self._dimension

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    f"{self._base_url}/embeddings",
                    json=payload,
                    headers=headers,
                    timeout=60.0,
                )
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise EmbeddingError(
                    f"OpenAI embeddings API error: {exc.response.status_code}",
                    operation="embed",
                    details={"status": exc.response.status_code, "body": exc.response.text},
                ) from exc
            except httpx.HTTPError as exc:
                raise EmbeddingError(
                    f"OpenAI embeddings request failed: {exc}",
                    operation="embed",
                ) from exc

        body = resp.json()
        # Sort by index to guarantee order matches input order.
        data = sorted(body["data"], key=lambda d: d["index"])
        logger.debug("OpenAIEmbeddings: embedded %d texts model=%s", len(texts), self._model)
        return [item["embedding"] for item in data]


# ---------------------------------------------------------------------------
# VertexEmbeddings
# ---------------------------------------------------------------------------


class VertexEmbeddings(Embeddings):
    """Embedding provider backed by Google Vertex AI.

    Uses the Vertex AI Prediction REST API via ``httpx`` — the
    ``google-cloud-aiplatform`` SDK is **not** required.

    Args:
        api_key: Bearer access token for Vertex AI authentication.
        project: Google Cloud project ID.
        model: Embedding model name. Defaults to ``"text-embedding-005"``.
        dimension: Vector dimensionality. Defaults to 768.
        location: Google Cloud region. Defaults to ``"us-central1"``.
    """

    def __init__(
        self,
        *,
        api_key: str,
        project: str,
        model: str = _DEFAULT_VERTEX_MODEL,
        dimension: int = _DEFAULT_VERTEX_DIMENSION,
        location: str = _DEFAULT_VERTEX_LOCATION,
    ) -> None:
        self._api_key = api_key
        self._project = project
        self._model = model
        self._dimension = dimension
        self._location = location

    @property
    def dimension(self) -> int:
        """The dimensionality of the embedding vectors."""
        return self._dimension

    def _endpoint_url(self) -> str:
        """Build the Vertex AI prediction endpoint URL."""
        return (
            f"https://{self._location}-aiplatform.googleapis.com/v1/"
            f"projects/{self._project}/locations/{self._location}/"
            f"publishers/google/models/{self._model}:predict"
        )

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string via the Vertex AI API.

        Args:
            text: The text to embed.

        Returns:
            A dense vector of length ``dimension``.
        """
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single Vertex AI API call.

        Args:
            texts: A list of texts to embed.

        Returns:
            A list of dense vectors, one per input text.

        Raises:
            EmbeddingError: If the API call fails.
        """
        if not texts:
            return []

        payload: dict[str, Any] = {
            "instances": [{"content": t} for t in texts],
            "parameters": {"outputDimensionality": self._dimension},
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    self._endpoint_url(),
                    json=payload,
                    headers=headers,
                    timeout=60.0,
                )
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise EmbeddingError(
                    f"Vertex AI embeddings API error: {exc.response.status_code}",
                    operation="embed",
                    details={"status": exc.response.status_code, "body": exc.response.text},
                ) from exc
            except httpx.HTTPError as exc:
                raise EmbeddingError(
                    f"Vertex AI embeddings request failed: {exc}",
                    operation="embed",
                ) from exc

        body = resp.json()
        predictions = body["predictions"]
        logger.debug("VertexEmbeddings: embedded %d texts model=%s", len(texts), self._model)
        return [p["embeddings"]["values"] for p in predictions]


# ---------------------------------------------------------------------------
# HTTPEmbeddings
# ---------------------------------------------------------------------------


def _get_nested(obj: Any, path: str) -> Any:
    """Traverse a nested dict/list using a dot-separated path.

    Supports integer keys for list indexing (e.g. ``"data.0.embedding"``).
    """
    for key in path.split("."):
        obj = obj[int(key)] if isinstance(obj, list) else obj[key]
    return obj


class HTTPEmbeddings(Embeddings):
    """Embedding provider that calls any HTTP endpoint.

    Sends a POST request to any embedding endpoint and extracts vectors from
    the response using configurable field paths.  Useful for self-hosted
    models (Ollama, TEI, vLLM) or custom embedding APIs.

    Args:
        url: The embedding endpoint URL.
        dimension: Vector dimensionality.
        headers: Optional HTTP headers (e.g. for authentication).
        input_field: Dot path in the request body for the input texts.
            Defaults to ``"input"``.
        output_field: Dot path in the response body to the list of
            embedding objects. Defaults to ``"data"``.
        vector_field: Dot path within each embedding object to the vector.
            Defaults to ``"embedding"``.
        timeout: Request timeout in seconds. Defaults to 60.
    """

    def __init__(
        self,
        *,
        url: str,
        dimension: int,
        headers: dict[str, str] | None = None,
        input_field: str = "input",
        output_field: str = "data",
        vector_field: str = "embedding",
        timeout: float = 60.0,
    ) -> None:
        self._url = url
        self._dimension = dimension
        self._headers = dict(headers) if headers else {}
        self._input_field = input_field
        self._output_field = output_field
        self._vector_field = vector_field
        self._timeout = timeout

    @property
    def dimension(self) -> int:
        """The dimensionality of the embedding vectors."""
        return self._dimension

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: The text to embed.

        Returns:
            A dense vector of length ``dimension``.
        """
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single HTTP call.

        Args:
            texts: A list of texts to embed.

        Returns:
            A list of dense vectors, one per input text.

        Raises:
            EmbeddingError: If the HTTP call or response parsing fails.
        """
        if not texts:
            return []

        payload: dict[str, Any] = {self._input_field: texts}
        req_headers = {"Content-Type": "application/json", **self._headers}

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    self._url,
                    json=payload,
                    headers=req_headers,
                    timeout=self._timeout,
                )
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise EmbeddingError(
                    f"HTTP embeddings API error: {exc.response.status_code}",
                    operation="embed",
                    details={"status": exc.response.status_code, "body": exc.response.text},
                ) from exc
            except httpx.HTTPError as exc:
                raise EmbeddingError(
                    f"HTTP embeddings request failed: {exc}",
                    operation="embed",
                ) from exc

        body = resp.json()
        try:
            items = _get_nested(body, self._output_field)
            return [_get_nested(item, self._vector_field) for item in items]
        except (KeyError, IndexError, TypeError) as exc:
            raise EmbeddingError(
                f"Failed to extract embeddings from response: {exc}",
                operation="embed",
                details={
                    "output_field": self._output_field,
                    "vector_field": self._vector_field,
                },
            ) from exc
