"""Tests for Embeddings ABC and embedding providers."""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# OpenAIEmbeddings
# ---------------------------------------------------------------------------


def _make_response(embeddings: list[list[float]], status: int = 200) -> httpx.Response:
    """Build a mock httpx.Response with embedding data."""
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


def _make_error_response(status: int, body: str = "error") -> httpx.Response:
    """Build a mock httpx.Response for an error."""
    return httpx.Response(
        status_code=status,
        text=body,
        request=httpx.Request("POST", "https://api.openai.com/v1/embeddings"),
    )


class TestOpenAIEmbeddings:
    def test_dimension(self) -> None:
        e = OpenAIEmbeddings(api_key="test-key", dimension=768)
        assert e.dimension == 768

    def test_default_dimension(self) -> None:
        e = OpenAIEmbeddings(api_key="test-key")
        assert e.dimension == 1536

    @pytest.mark.asyncio
    async def test_embed_single(self) -> None:
        vec = [0.1, 0.2, 0.3]
        mock_resp = _make_response([vec])

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = OpenAIEmbeddings(api_key="test-key")
            result = await e.embed("hello")

        assert result == vec
        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        assert call_kwargs[1]["json"]["input"] == ["hello"]
        assert call_kwargs[1]["json"]["model"] == "text-embedding-3-small"
        assert "Bearer test-key" in call_kwargs[1]["headers"]["Authorization"]

    @pytest.mark.asyncio
    async def test_embed_batch(self) -> None:
        vecs = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        mock_resp = _make_response(vecs)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = OpenAIEmbeddings(api_key="test-key", dimension=2)
            result = await e.embed_batch(["a", "b", "c"])

        assert result == vecs
        call_kwargs = mock_client.post.call_args
        assert call_kwargs[1]["json"]["input"] == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self) -> None:
        e = OpenAIEmbeddings(api_key="test-key")
        result = await e.embed_batch([])
        assert result == []

    @pytest.mark.asyncio
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

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = OpenAIEmbeddings(api_key="test-key", dimension=2)
            result = await e.embed_batch(["first", "second"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]

    @pytest.mark.asyncio
    async def test_embed_http_error(self) -> None:
        mock_resp = _make_error_response(401, "Unauthorized")

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = OpenAIEmbeddings(api_key="bad-key")
            with pytest.raises(EmbeddingError, match="401"):
                await e.embed("hello")

    @pytest.mark.asyncio
    async def test_embed_connection_error(self) -> None:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = OpenAIEmbeddings(api_key="test-key")
            with pytest.raises(EmbeddingError, match="request failed"):
                await e.embed("hello")

    @pytest.mark.asyncio
    async def test_custom_base_url(self) -> None:
        vec = [0.1]
        mock_resp = _make_response([vec])

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = OpenAIEmbeddings(
                api_key="test-key",
                base_url="https://custom.api.com/v1/",
                dimension=1,
            )
            await e.embed("test")

        url = mock_client.post.call_args[0][0]
        assert url == "https://custom.api.com/v1/embeddings"

    @pytest.mark.asyncio
    async def test_custom_model(self) -> None:
        vec = [0.1]
        mock_resp = _make_response([vec])

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = OpenAIEmbeddings(
                api_key="test-key",
                model="text-embedding-3-large",
                dimension=1,
            )
            await e.embed("test")

        call_kwargs = mock_client.post.call_args
        assert call_kwargs[1]["json"]["model"] == "text-embedding-3-large"

    def test_embedding_error_fields(self) -> None:
        err = EmbeddingError("fail", operation="embed", details={"code": 401})
        assert err.operation == "embed"
        assert err.details == {"code": 401}
        assert "fail" in str(err)


# ---------------------------------------------------------------------------
# VertexEmbeddings
# ---------------------------------------------------------------------------


def _make_vertex_response(embeddings: list[list[float]], status: int = 200) -> httpx.Response:
    """Build a mock httpx.Response with Vertex AI embedding data."""
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


class TestVertexEmbeddings:
    def test_dimension(self) -> None:
        e = VertexEmbeddings(api_key="token", project="my-project", dimension=256)
        assert e.dimension == 256

    def test_default_dimension(self) -> None:
        e = VertexEmbeddings(api_key="token", project="my-project")
        assert e.dimension == 768

    @pytest.mark.asyncio
    async def test_embed_single(self) -> None:
        vec = [0.1, 0.2, 0.3]
        mock_resp = _make_vertex_response([vec])

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = VertexEmbeddings(api_key="token", project="my-project")
            result = await e.embed("hello")

        assert result == vec
        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        assert call_kwargs[1]["json"]["instances"] == [{"content": "hello"}]
        assert call_kwargs[1]["json"]["parameters"]["outputDimensionality"] == 768
        assert "Bearer token" in call_kwargs[1]["headers"]["Authorization"]

    @pytest.mark.asyncio
    async def test_embed_batch(self) -> None:
        vecs = [[0.1, 0.2], [0.3, 0.4]]
        mock_resp = _make_vertex_response(vecs)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = VertexEmbeddings(api_key="token", project="my-project", dimension=2)
            result = await e.embed_batch(["a", "b"])

        assert result == vecs
        call_kwargs = mock_client.post.call_args
        assert call_kwargs[1]["json"]["instances"] == [{"content": "a"}, {"content": "b"}]

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self) -> None:
        e = VertexEmbeddings(api_key="token", project="my-project")
        result = await e.embed_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_endpoint_url(self) -> None:
        vec = [0.1]
        mock_resp = _make_vertex_response([vec])

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

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

    @pytest.mark.asyncio
    async def test_embed_http_error(self) -> None:
        mock_resp = httpx.Response(
            status_code=403,
            text="Forbidden",
            request=httpx.Request("POST", "https://example.com"),
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = VertexEmbeddings(api_key="bad-token", project="my-project")
            with pytest.raises(EmbeddingError, match="403"):
                await e.embed("hello")

    @pytest.mark.asyncio
    async def test_embed_connection_error(self) -> None:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = VertexEmbeddings(api_key="token", project="my-project")
            with pytest.raises(EmbeddingError, match="request failed"):
                await e.embed("hello")


# ---------------------------------------------------------------------------
# HTTPEmbeddings
# ---------------------------------------------------------------------------


def _make_generic_response(
    embeddings: list[list[float]],
    *,
    output_key: str = "data",
    vector_key: str = "embedding",
    status: int = 200,
) -> httpx.Response:
    """Build a mock httpx.Response with generic embedding data."""
    body = {output_key: [{vector_key: vec} for vec in embeddings]}
    return httpx.Response(
        status_code=status,
        json=body,
        request=httpx.Request("POST", "https://embeddings.example.com/embed"),
    )


class TestHTTPEmbeddings:
    def test_dimension(self) -> None:
        e = HTTPEmbeddings(url="https://example.com/embed", dimension=512)
        assert e.dimension == 512

    @pytest.mark.asyncio
    async def test_embed_single(self) -> None:
        vec = [0.1, 0.2, 0.3]
        mock_resp = _make_generic_response([vec])

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = HTTPEmbeddings(url="https://example.com/embed", dimension=3)
            result = await e.embed("hello")

        assert result == vec
        call_kwargs = mock_client.post.call_args
        assert call_kwargs[1]["json"]["input"] == ["hello"]

    @pytest.mark.asyncio
    async def test_embed_batch(self) -> None:
        vecs = [[0.1, 0.2], [0.3, 0.4]]
        mock_resp = _make_generic_response(vecs)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = HTTPEmbeddings(url="https://example.com/embed", dimension=2)
            result = await e.embed_batch(["a", "b"])

        assert result == vecs

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self) -> None:
        e = HTTPEmbeddings(url="https://example.com/embed", dimension=2)
        result = await e.embed_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_custom_field_paths(self) -> None:
        """Test custom input/output/vector field paths."""
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

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

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

    @pytest.mark.asyncio
    async def test_custom_headers(self) -> None:
        vec = [0.1]
        mock_resp = _make_generic_response([vec])

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = HTTPEmbeddings(
                url="https://example.com/embed",
                dimension=1,
                headers={"X-Api-Key": "secret123"},
            )
            await e.embed("test")

        call_kwargs = mock_client.post.call_args
        assert call_kwargs[1]["headers"]["X-Api-Key"] == "secret123"

    @pytest.mark.asyncio
    async def test_nested_field_paths(self) -> None:
        """Test deeply nested output/vector paths."""
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

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = HTTPEmbeddings(
                url="https://example.com/embed",
                dimension=2,
                output_field="response.embeddings",
                vector_field="values.dense",
            )
            result = await e.embed("test")

        assert result == [0.5, 0.6]

    @pytest.mark.asyncio
    async def test_embed_http_error(self) -> None:
        mock_resp = httpx.Response(
            status_code=500,
            text="Internal Server Error",
            request=httpx.Request("POST", "https://example.com/embed"),
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = HTTPEmbeddings(url="https://example.com/embed", dimension=2)
            with pytest.raises(EmbeddingError, match="500"):
                await e.embed("hello")

    @pytest.mark.asyncio
    async def test_embed_connection_error(self) -> None:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = HTTPEmbeddings(url="https://example.com/embed", dimension=2)
            with pytest.raises(EmbeddingError, match="request failed"):
                await e.embed("hello")

    @pytest.mark.asyncio
    async def test_bad_response_shape(self) -> None:
        """Bad response shape raises EmbeddingError."""
        mock_resp = httpx.Response(
            status_code=200,
            json={"unexpected": "format"},
            request=httpx.Request("POST", "https://example.com/embed"),
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("exo.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            e = HTTPEmbeddings(url="https://example.com/embed", dimension=2)
            with pytest.raises(EmbeddingError, match="extract embeddings"):
                await e.embed("hello")

    def test_get_nested_helper(self) -> None:
        data = {"a": {"b": [{"c": 42}]}}
        assert _get_nested(data, "a.b.0.c") == 42
        assert _get_nested(data, "a.b") == [{"c": 42}]
