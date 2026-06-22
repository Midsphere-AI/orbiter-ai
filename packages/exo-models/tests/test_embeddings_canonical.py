"""Tests for the canonical Embeddings ABC and provider implementations.

Covers:
- ``Embeddings`` ABC contract (cannot instantiate, concrete subclass works)
- ``cosine_similarity`` math
- ``EmbeddingError`` fields
- ``OpenAIEmbeddings`` via httpx mocking
- ``VertexEmbeddings`` via httpx mocking
- ``HTTPEmbeddings`` via httpx mocking
"""

from __future__ import annotations

import math
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from exo.models.embeddings import (
    EmbeddingError,
    Embeddings,
    HTTPEmbeddings,
    OpenAIEmbeddings,
    VertexEmbeddings,
    _get_nested,
    cosine_similarity,
)

# ---------------------------------------------------------------------------
# Embeddings ABC
# ---------------------------------------------------------------------------


class TestEmbeddingsABC:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            Embeddings()  # type: ignore[abstract]

    def test_concrete_subclass(self) -> None:
        class Dummy(Embeddings):
            async def embed(self, text: str) -> list[float]:
                return [0.0]

            async def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return [[0.0] for _ in texts]

            @property
            def dimension(self) -> int:
                return 1

        d = Dummy()
        assert d.dimension == 1

    async def test_concrete_embed_batch(self) -> None:
        class Dummy(Embeddings):
            async def embed(self, text: str) -> list[float]:
                return [1.0]

            async def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return [[1.0] for _ in texts]

            @property
            def dimension(self) -> int:
                return 1

        d = Dummy()
        result = await d.embed_batch(["a", "b", "c"])
        assert len(result) == 3
        assert all(v == [1.0] for v in result)


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector_a_returns_zero(self) -> None:
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_zero_vector_b_returns_zero(self) -> None:
        assert cosine_similarity([1.0, 1.0], [0.0, 0.0]) == 0.0

    def test_similar_vectors(self) -> None:
        a = [1.0, 1.0]
        b = [1.0, 0.0]
        expected = 1.0 / math.sqrt(2)
        assert cosine_similarity(a, b) == pytest.approx(expected)

    def test_three_dimensions(self) -> None:
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        dot = 1 * 4 + 2 * 5 + 3 * 6  # 32
        mag_a = math.sqrt(1 + 4 + 9)  # sqrt(14)
        mag_b = math.sqrt(16 + 25 + 36)  # sqrt(77)
        expected = dot / (mag_a * mag_b)
        assert cosine_similarity(a, b) == pytest.approx(expected)

    def test_symmetry(self) -> None:
        a = [0.3, 0.7, 0.1]
        b = [0.9, 0.2, 0.5]
        assert cosine_similarity(a, b) == pytest.approx(cosine_similarity(b, a))


# ---------------------------------------------------------------------------
# EmbeddingError
# ---------------------------------------------------------------------------


class TestEmbeddingError:
    def test_message(self) -> None:
        err = EmbeddingError("something went wrong")
        assert "something went wrong" in str(err)

    def test_default_operation(self) -> None:
        err = EmbeddingError("fail")
        assert err.operation == "embed"

    def test_custom_operation(self) -> None:
        err = EmbeddingError("fail", operation="batch")
        assert err.operation == "batch"

    def test_details(self) -> None:
        err = EmbeddingError("fail", details={"status": 401})
        assert err.details == {"status": 401}

    def test_default_details_empty(self) -> None:
        err = EmbeddingError("fail")
        assert err.details == {}

    def test_is_exception(self) -> None:
        with pytest.raises(EmbeddingError):
            raise EmbeddingError("test")


# ---------------------------------------------------------------------------
# OpenAIEmbeddings helpers
# ---------------------------------------------------------------------------


def _make_openai_response(embeddings: list[list[float]], status: int = 200) -> httpx.Response:
    body = {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": vec}
            for i, vec in enumerate(embeddings)
        ],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 10, "total_tokens": 10},
    }
    return httpx.Response(
        status_code=status,
        json=body,
        request=httpx.Request("POST", "https://api.openai.com/v1/embeddings"),
    )


def _mock_openai_client(response: httpx.Response) -> AsyncMock:
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


# ---------------------------------------------------------------------------
# OpenAIEmbeddings
# ---------------------------------------------------------------------------


class TestOpenAIEmbeddings:
    def test_dimension_default(self) -> None:
        e = OpenAIEmbeddings(api_key="test-key")
        assert e.dimension == 1536

    def test_dimension_custom(self) -> None:
        e = OpenAIEmbeddings(api_key="test-key", dimension=768)
        assert e.dimension == 768

    async def test_embed_single(self) -> None:
        vec = [0.1, 0.2, 0.3]
        mock_client = _mock_openai_client(_make_openai_response([vec]))

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = OpenAIEmbeddings(api_key="test-key")
            result = await e.embed("hello")

        assert result == vec
        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        assert call_kwargs[1]["json"]["input"] == ["hello"]
        assert call_kwargs[1]["json"]["model"] == "text-embedding-3-small"
        assert "Bearer test-key" in call_kwargs[1]["headers"]["Authorization"]

    async def test_embed_batch(self) -> None:
        vecs = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        mock_client = _mock_openai_client(_make_openai_response(vecs))

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = OpenAIEmbeddings(api_key="test-key", dimension=2)
            result = await e.embed_batch(["a", "b", "c"])

        assert result == vecs
        call_kwargs = mock_client.post.call_args
        assert call_kwargs[1]["json"]["input"] == ["a", "b", "c"]

    async def test_embed_batch_empty(self) -> None:
        e = OpenAIEmbeddings(api_key="test-key")
        result = await e.embed_batch([])
        assert result == []

    async def test_embed_batch_preserves_order(self) -> None:
        """API may return embeddings out of order; we sort by index."""
        body = {
            "object": "list",
            "data": [
                {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
                {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
            ],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        }
        mock_resp = httpx.Response(
            status_code=200,
            json=body,
            request=httpx.Request("POST", "https://api.openai.com/v1/embeddings"),
        )
        mock_client = _mock_openai_client(mock_resp)

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = OpenAIEmbeddings(api_key="test-key", dimension=2)
            result = await e.embed_batch(["first", "second"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]

    async def test_http_status_error_raises_embedding_error(self) -> None:
        error_resp = httpx.Response(
            status_code=401,
            text="Unauthorized",
            request=httpx.Request("POST", "https://api.openai.com/v1/embeddings"),
        )
        mock_client = _mock_openai_client(error_resp)

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = OpenAIEmbeddings(api_key="bad-key")
            with pytest.raises(EmbeddingError, match="401"):
                await e.embed("hello")

    async def test_connection_error_raises_embedding_error(self) -> None:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = OpenAIEmbeddings(api_key="test-key")
            with pytest.raises(EmbeddingError, match="request failed"):
                await e.embed("hello")

    async def test_custom_base_url(self) -> None:
        mock_client = _mock_openai_client(_make_openai_response([[0.1]]))

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = OpenAIEmbeddings(
                api_key="test-key",
                base_url="https://custom.api.com/v1/",
                dimension=1,
            )
            await e.embed("test")

        url = mock_client.post.call_args[0][0]
        assert url == "https://custom.api.com/v1/embeddings"

    async def test_custom_model(self) -> None:
        mock_client = _mock_openai_client(_make_openai_response([[0.1]]))

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = OpenAIEmbeddings(
                api_key="test-key",
                model="text-embedding-3-large",
                dimension=1,
            )
            await e.embed("test")

        call_kwargs = mock_client.post.call_args
        assert call_kwargs[1]["json"]["model"] == "text-embedding-3-large"

    async def test_dimensions_sent_in_payload(self) -> None:
        mock_client = _mock_openai_client(_make_openai_response([[0.1, 0.2]]))

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = OpenAIEmbeddings(api_key="test-key", dimension=2)
            await e.embed("test")

        payload = mock_client.post.call_args[1]["json"]
        assert payload["dimensions"] == 2


# ---------------------------------------------------------------------------
# VertexEmbeddings helpers
# ---------------------------------------------------------------------------


def _make_vertex_response(embeddings: list[list[float]], status: int = 200) -> httpx.Response:
    body = {
        "predictions": [
            {"embeddings": {"values": vec, "statistics": {"truncated": False}}}
            for vec in embeddings
        ],
    }
    return httpx.Response(
        status_code=status,
        json=body,
        request=httpx.Request("POST", "https://us-central1-aiplatform.googleapis.com/v1/..."),
    )


def _mock_vertex_client(response: httpx.Response) -> AsyncMock:
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


# ---------------------------------------------------------------------------
# VertexEmbeddings
# ---------------------------------------------------------------------------


class TestVertexEmbeddings:
    def test_dimension_default(self) -> None:
        e = VertexEmbeddings(api_key="token", project="my-project")
        assert e.dimension == 768

    def test_dimension_custom(self) -> None:
        e = VertexEmbeddings(api_key="token", project="my-project", dimension=256)
        assert e.dimension == 256

    async def test_embed_single(self) -> None:
        vec = [0.1, 0.2, 0.3]
        mock_client = _mock_vertex_client(_make_vertex_response([vec]))

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = VertexEmbeddings(api_key="token", project="my-project")
            result = await e.embed("hello")

        assert result == vec
        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        assert call_kwargs[1]["json"]["instances"] == [{"content": "hello"}]
        assert call_kwargs[1]["json"]["parameters"]["outputDimensionality"] == 768
        assert "Bearer token" in call_kwargs[1]["headers"]["Authorization"]

    async def test_embed_batch(self) -> None:
        vecs = [[0.1, 0.2], [0.3, 0.4]]
        mock_client = _mock_vertex_client(_make_vertex_response(vecs))

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = VertexEmbeddings(api_key="token", project="my-project", dimension=2)
            result = await e.embed_batch(["a", "b"])

        assert result == vecs
        call_kwargs = mock_client.post.call_args
        assert call_kwargs[1]["json"]["instances"] == [{"content": "a"}, {"content": "b"}]

    async def test_embed_batch_empty(self) -> None:
        e = VertexEmbeddings(api_key="token", project="my-project")
        result = await e.embed_batch([])
        assert result == []

    async def test_endpoint_url_default_region(self) -> None:
        mock_client = _mock_vertex_client(_make_vertex_response([[0.1]]))

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = VertexEmbeddings(api_key="token", project="my-project", dimension=1)
            await e.embed("test")

        url = mock_client.post.call_args[0][0]
        assert "us-central1-aiplatform.googleapis.com" in url
        assert "projects/my-project" in url
        assert "locations/us-central1" in url

    async def test_endpoint_url_custom_location(self) -> None:
        mock_client = _mock_vertex_client(_make_vertex_response([[0.1]]))

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = VertexEmbeddings(
                api_key="token",
                project="my-project",
                location="europe-west1",
                model="text-embedding-004",
                dimension=1,
            )
            await e.embed("test")

        url = mock_client.post.call_args[0][0]
        assert "europe-west1-aiplatform.googleapis.com" in url
        assert "projects/my-project" in url
        assert "locations/europe-west1" in url
        assert "models/text-embedding-004:predict" in url

    async def test_http_status_error_raises_embedding_error(self) -> None:
        error_resp = httpx.Response(
            status_code=403,
            text="Forbidden",
            request=httpx.Request("POST", "https://example.com"),
        )
        mock_client = _mock_vertex_client(error_resp)

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = VertexEmbeddings(api_key="bad-token", project="my-project")
            with pytest.raises(EmbeddingError, match="403"):
                await e.embed("hello")

    async def test_connection_error_raises_embedding_error(self) -> None:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = VertexEmbeddings(api_key="token", project="my-project")
            with pytest.raises(EmbeddingError, match="request failed"):
                await e.embed("hello")


# ---------------------------------------------------------------------------
# HTTPEmbeddings helpers
# ---------------------------------------------------------------------------


def _make_generic_response(
    embeddings: list[list[float]],
    *,
    output_key: str = "data",
    vector_key: str = "embedding",
    status: int = 200,
) -> httpx.Response:
    body = {output_key: [{vector_key: vec} for vec in embeddings]}
    return httpx.Response(
        status_code=status,
        json=body,
        request=httpx.Request("POST", "https://embeddings.example.com/embed"),
    )


def _mock_http_client(response: httpx.Response) -> AsyncMock:
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


# ---------------------------------------------------------------------------
# HTTPEmbeddings
# ---------------------------------------------------------------------------


class TestHTTPEmbeddings:
    def test_dimension(self) -> None:
        e = HTTPEmbeddings(url="https://example.com/embed", dimension=512)
        assert e.dimension == 512

    async def test_embed_single(self) -> None:
        vec = [0.1, 0.2, 0.3]
        mock_client = _mock_http_client(_make_generic_response([vec]))

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = HTTPEmbeddings(url="https://example.com/embed", dimension=3)
            result = await e.embed("hello")

        assert result == vec
        call_kwargs = mock_client.post.call_args
        assert call_kwargs[1]["json"]["input"] == ["hello"]

    async def test_embed_batch(self) -> None:
        vecs = [[0.1, 0.2], [0.3, 0.4]]
        mock_client = _mock_http_client(_make_generic_response(vecs))

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = HTTPEmbeddings(url="https://example.com/embed", dimension=2)
            result = await e.embed_batch(["a", "b"])

        assert result == vecs

    async def test_embed_batch_empty(self) -> None:
        e = HTTPEmbeddings(url="https://example.com/embed", dimension=2)
        result = await e.embed_batch([])
        assert result == []

    async def test_custom_field_paths(self) -> None:
        body = {
            "results": [
                {"vector": [0.1, 0.2]},
                {"vector": [0.3, 0.4]},
            ]
        }
        mock_resp = httpx.Response(
            status_code=200,
            json=body,
            request=httpx.Request("POST", "https://example.com/embed"),
        )
        mock_client = _mock_http_client(mock_resp)

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = HTTPEmbeddings(
                url="https://example.com/embed",
                dimension=2,
                input_field="texts",
                output_field="results",
                vector_field="vector",
            )
            result = await e.embed_batch(["a", "b"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]
        call_kwargs = mock_client.post.call_args
        assert call_kwargs[1]["json"]["texts"] == ["a", "b"]

    async def test_custom_headers(self) -> None:
        mock_client = _mock_http_client(_make_generic_response([[0.1]]))

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = HTTPEmbeddings(
                url="https://example.com/embed",
                dimension=1,
                headers={"X-Api-Key": "secret123"},
            )
            await e.embed("test")

        call_kwargs = mock_client.post.call_args
        assert call_kwargs[1]["headers"]["X-Api-Key"] == "secret123"

    async def test_nested_field_paths(self) -> None:
        body = {
            "response": {
                "embeddings": [
                    {"values": {"dense": [0.5, 0.6]}},
                ]
            }
        }
        mock_resp = httpx.Response(
            status_code=200,
            json=body,
            request=httpx.Request("POST", "https://example.com/embed"),
        )
        mock_client = _mock_http_client(mock_resp)

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = HTTPEmbeddings(
                url="https://example.com/embed",
                dimension=2,
                output_field="response.embeddings",
                vector_field="values.dense",
            )
            result = await e.embed("test")

        assert result == [0.5, 0.6]

    async def test_http_status_error_raises_embedding_error(self) -> None:
        error_resp = httpx.Response(
            status_code=500,
            text="Internal Server Error",
            request=httpx.Request("POST", "https://example.com/embed"),
        )
        mock_client = _mock_http_client(error_resp)

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = HTTPEmbeddings(url="https://example.com/embed", dimension=2)
            with pytest.raises(EmbeddingError, match="500"):
                await e.embed("hello")

    async def test_connection_error_raises_embedding_error(self) -> None:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = HTTPEmbeddings(url="https://example.com/embed", dimension=2)
            with pytest.raises(EmbeddingError, match="request failed"):
                await e.embed("hello")

    async def test_bad_response_shape_raises_embedding_error(self) -> None:
        bad_resp = httpx.Response(
            status_code=200,
            json={"unexpected": "format"},
            request=httpx.Request("POST", "https://example.com/embed"),
        )
        mock_client = _mock_http_client(bad_resp)

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = HTTPEmbeddings(url="https://example.com/embed", dimension=2)
            with pytest.raises(EmbeddingError, match="extract embeddings"):
                await e.embed("hello")

    async def test_timeout_forwarded(self) -> None:
        mock_client = _mock_http_client(_make_generic_response([[0.1]]))

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = HTTPEmbeddings(url="https://example.com/embed", dimension=1, timeout=30.0)
            await e.embed("test")

        call_kwargs = mock_client.post.call_args
        assert call_kwargs[1]["timeout"] == 30.0


# ---------------------------------------------------------------------------
# _get_nested helper
# ---------------------------------------------------------------------------


class TestGetNested:
    def test_flat_key(self) -> None:
        assert _get_nested({"a": 1}, "a") == 1

    def test_nested_dict(self) -> None:
        data = {"a": {"b": {"c": 42}}}
        assert _get_nested(data, "a.b.c") == 42

    def test_list_index(self) -> None:
        data = {"a": {"b": [{"c": 42}]}}
        assert _get_nested(data, "a.b.0.c") == 42

    def test_top_level_list(self) -> None:
        assert _get_nested([10, 20, 30], "1") == 20


# ---------------------------------------------------------------------------
# Persistent HTTP client / aclose / async context manager (finding #3)
# ---------------------------------------------------------------------------


class TestPersistentClientOpenAI:
    def test_client_is_none_before_first_call(self) -> None:
        e = OpenAIEmbeddings(api_key="test")
        assert e._client is None

    def test_get_client_creates_client(self) -> None:
        e = OpenAIEmbeddings(api_key="test")
        c = e._get_client()
        assert c is not None
        assert isinstance(c, httpx.AsyncClient)

    def test_get_client_reuses_same_instance(self) -> None:
        e = OpenAIEmbeddings(api_key="test")
        c1 = e._get_client()
        c2 = e._get_client()
        assert c1 is c2

    async def test_aclose_closes_client(self) -> None:
        e = OpenAIEmbeddings(api_key="test")
        _ = e._get_client()
        assert not e._client.is_closed  # type: ignore[union-attr]
        await e.aclose()
        assert e._client.is_closed  # type: ignore[union-attr]

    async def test_aclose_noop_when_no_client(self) -> None:
        e = OpenAIEmbeddings(api_key="test")
        # Should not raise
        await e.aclose()

    async def test_async_context_manager(self) -> None:
        async with OpenAIEmbeddings(api_key="test") as e:
            _ = e._get_client()
            assert not e._client.is_closed  # type: ignore[union-attr]
        assert e._client.is_closed  # type: ignore[union-attr]


class TestPersistentClientVertex:
    def test_client_is_none_before_first_call(self) -> None:
        e = VertexEmbeddings(api_key="tok", project="proj")
        assert e._client is None

    async def test_aclose_noop_when_no_client(self) -> None:
        e = VertexEmbeddings(api_key="tok", project="proj")
        await e.aclose()  # should not raise

    async def test_async_context_manager(self) -> None:
        async with VertexEmbeddings(api_key="tok", project="proj") as e:
            _ = e._get_client()
        assert e._client.is_closed  # type: ignore[union-attr]


class TestPersistentClientHTTP:
    def test_client_is_none_before_first_call(self) -> None:
        e = HTTPEmbeddings(url="http://example.com", dimension=128)
        assert e._client is None

    async def test_aclose_noop_when_no_client(self) -> None:
        e = HTTPEmbeddings(url="http://example.com", dimension=128)
        await e.aclose()  # should not raise

    async def test_async_context_manager(self) -> None:
        async with HTTPEmbeddings(url="http://example.com", dimension=128) as e:
            _ = e._get_client()
        assert e._client.is_closed  # type: ignore[union-attr]
