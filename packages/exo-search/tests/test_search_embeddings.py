"""Tests for exo-search semantic reranking via the canonical embeddings layer.

Covers:
- ``_GeminiEmbeddings`` httpx-based implementation (happy path + error)
- ``_build_embedder`` provider auto-detection logic
- ``rerank_search_results`` end-to-end (embedding path + keyword fallback)
- ``_keyword_score`` helper
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from exo.models.embeddings import EmbeddingError, OpenAIEmbeddings, VertexEmbeddings
from exo.search.tools.embeddings import (
    _build_embedder,
    _GeminiEmbeddings,
    _keyword_score,
    rerank_search_results,
)

# ---------------------------------------------------------------------------
# _keyword_score
# ---------------------------------------------------------------------------


class TestKeywordScore:
    def test_full_match(self) -> None:
        assert _keyword_score(["python", "async"], "python is great for async code") == 1.0

    def test_partial_match(self) -> None:
        score = _keyword_score(["python", "java"], "python is great")
        assert score == pytest.approx(0.5)

    def test_no_match(self) -> None:
        assert _keyword_score(["golang"], "python is great") == 0.0

    def test_empty_query_words(self) -> None:
        # max(0, 1) denominator prevents division by zero
        assert _keyword_score([], "any text") == 0.0

    def test_case_insensitive(self) -> None:
        assert _keyword_score(["python"], "PYTHON is great") == 1.0


# ---------------------------------------------------------------------------
# _GeminiEmbeddings
# ---------------------------------------------------------------------------


def _make_gemini_response(embeddings: list[list[float]], status: int = 200) -> httpx.Response:
    body = {"embeddings": [{"values": vec} for vec in embeddings]}
    return httpx.Response(
        status_code=status,
        json=body,
        request=httpx.Request("POST", "https://generativelanguage.googleapis.com/..."),
    )


def _mock_httpx_client(response: httpx.Response) -> AsyncMock:
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


class TestGeminiEmbeddings:
    def test_dimension_default(self) -> None:
        e = _GeminiEmbeddings(api_key="test-key")
        assert e.dimension == 3072

    def test_dimension_custom(self) -> None:
        e = _GeminiEmbeddings(api_key="test-key", dimension=768)
        assert e.dimension == 768

    async def test_embed_single(self) -> None:
        vec = [0.1, 0.2, 0.3]
        mock_client = _mock_httpx_client(_make_gemini_response([vec]))

        with patch("exo.search.tools.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = _GeminiEmbeddings(api_key="test-key")
            result = await e.embed("hello")

        assert result == vec
        call_kwargs = mock_client.post.call_args
        # Verify batchEmbedContents request shape
        body = call_kwargs[1]["json"]
        assert "requests" in body
        assert len(body["requests"]) == 1
        assert body["requests"][0]["content"]["parts"][0]["text"] == "hello"
        # Verify API key is in query params
        params = call_kwargs[1]["params"]
        assert params["key"] == "test-key"

    async def test_embed_batch(self) -> None:
        vecs = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        mock_client = _mock_httpx_client(_make_gemini_response(vecs))

        with patch("exo.search.tools.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = _GeminiEmbeddings(api_key="test-key", dimension=2)
            result = await e.embed_batch(["a", "b", "c"])

        assert result == vecs
        body = mock_client.post.call_args[1]["json"]
        assert len(body["requests"]) == 3

    async def test_embed_batch_empty(self) -> None:
        e = _GeminiEmbeddings(api_key="test-key")
        result = await e.embed_batch([])
        assert result == []

    async def test_http_status_error_raises_embedding_error(self) -> None:
        error_resp = httpx.Response(
            status_code=400,
            text="Bad Request",
            request=httpx.Request("POST", "https://generativelanguage.googleapis.com/..."),
        )
        mock_client = _mock_httpx_client(error_resp)

        with patch("exo.search.tools.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = _GeminiEmbeddings(api_key="bad-key")
            with pytest.raises(EmbeddingError, match="400"):
                await e.embed("hello")

    async def test_connection_error_raises_embedding_error(self) -> None:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("exo.search.tools.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = _GeminiEmbeddings(api_key="test-key")
            with pytest.raises(EmbeddingError, match="request failed"):
                await e.embed("hello")

    async def test_malformed_response_raises_embedding_error(self) -> None:
        bad_resp = httpx.Response(
            status_code=200,
            json={"unexpected": "shape"},
            request=httpx.Request("POST", "https://generativelanguage.googleapis.com/..."),
        )
        mock_client = _mock_httpx_client(bad_resp)

        with patch("exo.search.tools.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = _GeminiEmbeddings(api_key="test-key")
            with pytest.raises(EmbeddingError, match="extract"):
                await e.embed("hello")

    async def test_task_type_and_dimensionality_in_request(self) -> None:
        mock_client = _mock_httpx_client(_make_gemini_response([[0.1]]))

        with patch("exo.search.tools.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = _GeminiEmbeddings(api_key="test-key", dimension=512)
            await e.embed("test")

        req = mock_client.post.call_args[1]["json"]["requests"][0]
        assert req["taskType"] == "SEMANTIC_SIMILARITY"
        assert req["outputDimensionality"] == 512


# ---------------------------------------------------------------------------
# _build_embedder — provider auto-detection
# ---------------------------------------------------------------------------


class TestBuildEmbedder:
    def test_no_env_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert _build_embedder() is None

    def test_gemini_key_returns_gemini_embedder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "gkey")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        embedder = _build_embedder()
        assert isinstance(embedder, _GeminiEmbeddings)

    def test_openai_key_returns_openai_embedder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "okey")
        embedder = _build_embedder()
        assert isinstance(embedder, OpenAIEmbeddings)

    def test_vertex_requires_both_project_and_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-project")
        # Without access token → no provider
        monkeypatch.delenv("GOOGLE_CLOUD_ACCESS_TOKEN", raising=False)
        assert _build_embedder() is None

    def test_vertex_with_both_project_and_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-project")
        monkeypatch.setenv("GOOGLE_CLOUD_ACCESS_TOKEN", "token123")
        embedder = _build_embedder()
        assert isinstance(embedder, VertexEmbeddings)

    def test_gemini_takes_priority_over_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "gkey")
        monkeypatch.setenv("OPENAI_API_KEY", "okey")
        embedder = _build_embedder()
        assert isinstance(embedder, _GeminiEmbeddings)

    def test_custom_model_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "gkey")
        monkeypatch.setenv("EXO_SEARCH_EMBEDDING_MODEL", "my-custom-model")
        embedder = _build_embedder()
        assert isinstance(embedder, _GeminiEmbeddings)
        assert embedder._model == "my-custom-model"


# ---------------------------------------------------------------------------
# rerank_search_results
# ---------------------------------------------------------------------------


def _make_result(title: str, content: str):
    """Create a minimal SearchResult-like object."""
    return SimpleNamespace(title=title, content=content)


class TestRerankSearchResults:
    async def test_empty_results_returned_unchanged(self) -> None:
        result = await rerank_search_results("query", [])
        assert result == []

    async def test_keyword_fallback_when_no_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        r1 = _make_result("Python docs", "Python is a programming language")
        r2 = _make_result("Java docs", "Java is a programming language")
        r3 = _make_result("Cooking recipes", "How to bake bread")

        results = await rerank_search_results("python programming", [r1, r2, r3])
        # Python result should rank first (contains "python")
        assert results[0].title == "Python docs"

    async def test_embedding_path_reranks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        r1 = _make_result("Unrelated", "nothing here")
        r2 = _make_result("Target", "exactly what we want")

        # Simulate embeddings: query is most similar to r2
        query_vec = [1.0, 0.0]
        r1_vec = [0.0, 1.0]  # orthogonal — low similarity
        r2_vec = [0.9, 0.1]  # close to query — high similarity

        mock_embedder = AsyncMock()
        mock_embedder.embed_batch = AsyncMock(return_value=[query_vec, r1_vec, r2_vec])

        with patch("exo.search.tools.embeddings._build_embedder", return_value=mock_embedder):
            results = await rerank_search_results("query", [r1, r2])

        assert results[0].title == "Target"
        assert results[1].title == "Unrelated"

    async def test_top_k_limits_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        results = [_make_result(f"Result {i}", f"content {i}") for i in range(5)]
        reranked = await rerank_search_results("query", results, top_k=3)
        assert len(reranked) == 3

    async def test_embedding_error_falls_back_to_keyword(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        r1 = _make_result("Python guide", "python tutorial")
        r2 = _make_result("Java guide", "java tutorial")

        mock_embedder = AsyncMock()
        mock_embedder.embed_batch = AsyncMock(side_effect=EmbeddingError("API error"))

        with patch("exo.search.tools.embeddings._build_embedder", return_value=mock_embedder):
            results = await rerank_search_results("python", [r1, r2])

        # Fell back to keyword — python result should be first
        assert results[0].title == "Python guide"
