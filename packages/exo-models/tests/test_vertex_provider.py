"""Tests for the Google Vertex AI LLM provider."""

from __future__ import annotations

import base64
import json
import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import errors as genai_errors

from exo.config import ModelConfig  # pyright: ignore[reportMissingImports]
from exo.models.provider import model_registry  # pyright: ignore[reportMissingImports]
from exo.models.types import ModelError  # pyright: ignore[reportMissingImports]
from exo.models.vertex import (  # pyright: ignore[reportMissingImports]
    VertexProvider,
    _build_config,
    _convert_tools,
    _credentials_from_base64,
    _map_finish_reason,
    _parse_response,
    _parse_stream_chunk,
    _to_google_contents,
)
from exo.types import (  # pyright: ignore[reportMissingImports]
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolResult,
    UserMessage,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides: Any) -> ModelConfig:
    # Default to Vertex mode with a test project so tests exercise the Vertex auth path.
    defaults: dict[str, Any] = {
        "provider": "vertex",
        "model_name": "gemini-2.0-flash",
        "google_project": "test-project",
    }
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _make_part(
    *,
    text: str | None = None,
    function_call: Any | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(text=text, function_call=function_call)


def _make_function_call(
    name: str = "search",
    args: dict[str, Any] | None = None,
    id: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(name=name, args=args or {}, id=id)


def _make_response(
    *,
    content: str = "hello",
    parts: list[Any] | None = None,
    finish_reason: str = "STOP",
    prompt_token_count: int = 10,
    candidates_token_count: int = 5,
    total_token_count: int = 15,
) -> SimpleNamespace:
    if parts is None:
        parts = [_make_part(text=content)]
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=parts),
        finish_reason=finish_reason,
    )
    usage_metadata = SimpleNamespace(
        prompt_token_count=prompt_token_count,
        candidates_token_count=candidates_token_count,
        total_token_count=total_token_count,
    )
    return SimpleNamespace(
        candidates=[candidate],
        usage_metadata=usage_metadata,
    )


def _make_stream_chunk(
    *,
    parts: list[Any] | None = None,
    finish_reason: str | None = None,
    usage_metadata: Any = None,
) -> SimpleNamespace:
    if parts is None:
        parts = []
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=parts),
        finish_reason=finish_reason,
    )
    return SimpleNamespace(
        candidates=[candidate],
        usage_metadata=usage_metadata,
    )


def _make_usage_metadata(
    *, prompt_token_count: int = 10, candidates_token_count: int = 5, total_token_count: int = 15
) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_token_count=prompt_token_count,
        candidates_token_count=candidates_token_count,
        total_token_count=total_token_count,
    )


# ---------------------------------------------------------------------------
# Finish reason mapping
# ---------------------------------------------------------------------------


class TestFinishReasonMapping:
    def test_stop(self) -> None:
        assert _map_finish_reason("STOP") == "stop"

    def test_max_tokens(self) -> None:
        assert _map_finish_reason("MAX_TOKENS") == "length"

    def test_safety(self) -> None:
        assert _map_finish_reason("SAFETY") == "content_filter"

    def test_recitation(self) -> None:
        assert _map_finish_reason("RECITATION") == "content_filter"

    def test_blocklist(self) -> None:
        assert _map_finish_reason("BLOCKLIST") == "content_filter"

    def test_malformed_function_call(self) -> None:
        assert _map_finish_reason("MALFORMED_FUNCTION_CALL") == "stop"

    def test_other(self) -> None:
        assert _map_finish_reason("OTHER") == "stop"

    def test_none_defaults_to_stop(self) -> None:
        assert _map_finish_reason(None) == "stop"

    def test_unknown_defaults_to_stop(self) -> None:
        assert _map_finish_reason("unknown_reason") == "stop"


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


class TestToGoogleContents:
    def test_system_extracted(self) -> None:
        contents, system = _to_google_contents([SystemMessage(content="be helpful")])
        assert system == "be helpful"
        assert contents == []

    def test_multiple_system_joined(self) -> None:
        _, system = _to_google_contents(
            [
                SystemMessage(content="rule 1"),
                SystemMessage(content="rule 2"),
            ]
        )
        assert system == "rule 1\nrule 2"

    def test_user_message(self) -> None:
        contents, _ = _to_google_contents([UserMessage(content="hi")])
        assert contents == [{"role": "user", "parts": [{"text": "hi"}]}]

    def test_assistant_with_text(self) -> None:
        contents, _ = _to_google_contents([AssistantMessage(content="hello")])
        assert contents[0]["role"] == "model"
        assert contents[0]["parts"] == [{"text": "hello"}]

    def test_assistant_with_tool_calls(self) -> None:
        msg = AssistantMessage(
            content="",
            tool_calls=[ToolCall(id="tc1", name="search", arguments='{"q":"x"}')],
        )
        contents, _ = _to_google_contents([msg])
        assert contents[0]["role"] == "model"
        parts = contents[0]["parts"]
        assert len(parts) == 1
        assert parts[0]["function_call"]["name"] == "search"
        assert parts[0]["function_call"]["args"] == {"q": "x"}

    def test_assistant_with_text_and_tool_calls(self) -> None:
        msg = AssistantMessage(
            content="Let me search",
            tool_calls=[ToolCall(id="tc1", name="search", arguments='{"q":"x"}')],
        )
        contents, _ = _to_google_contents([msg])
        parts = contents[0]["parts"]
        assert len(parts) == 2
        assert parts[0] == {"text": "Let me search"}
        assert parts[1]["function_call"]["name"] == "search"

    def test_assistant_empty(self) -> None:
        contents, _ = _to_google_contents([AssistantMessage()])
        assert contents[0]["parts"] == [{"text": ""}]

    def test_tool_result(self) -> None:
        contents, _ = _to_google_contents(
            [
                ToolResult(tool_call_id="tc1", tool_name="search", content="found it"),
            ]
        )
        assert contents[0]["role"] == "user"
        fr = contents[0]["parts"][0]["function_response"]
        assert fr["name"] == "search"
        assert fr["response"]["content"] == "found it"

    def test_tool_result_with_error(self) -> None:
        contents, _ = _to_google_contents(
            [
                ToolResult(tool_call_id="tc1", tool_name="search", content="", error="failed"),
            ]
        )
        fr = contents[0]["parts"][0]["function_response"]
        assert fr["response"]["content"] == "failed"


# ---------------------------------------------------------------------------
# Tool conversion
# ---------------------------------------------------------------------------


class TestConvertTools:
    def test_basic_conversion(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                        "required": ["q"],
                    },
                },
            }
        ]
        result = _convert_tools(tools)
        assert len(result) == 1
        decls = result[0]["function_declarations"]
        assert len(decls) == 1
        assert decls[0]["name"] == "search"
        assert decls[0]["description"] == "Search the web"
        assert decls[0]["parameters"]["properties"]["q"]["type"] == "string"

    def test_empty_tools(self) -> None:
        assert _convert_tools([]) == []


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


class TestBuildConfig:
    def test_all_params(self) -> None:
        tools = [{"type": "function", "function": {"name": "t", "parameters": {}}}]
        config = _build_config(tools, 0.5, 100, "be helpful")
        assert config["system_instruction"] == "be helpful"
        assert config["temperature"] == 0.5
        assert config["max_output_tokens"] == 100
        assert "tools" in config

    def test_no_optional_params(self) -> None:
        config = _build_config(None, None, None, "")
        assert config == {}


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_text_response(self) -> None:
        raw = _make_response(content="Hello!")
        resp = _parse_response(raw, "gemini-2.0-flash")
        assert resp.content == "Hello!"
        assert resp.finish_reason == "stop"
        assert resp.usage.input_tokens == 10
        assert resp.usage.output_tokens == 5
        assert resp.usage.total_tokens == 15

    def test_tool_call_response(self) -> None:
        fc = _make_function_call("search", {"q": "test"}, id="call_0")
        parts = [_make_part(function_call=fc)]
        raw = _make_response(parts=parts, finish_reason="STOP")
        resp = _parse_response(raw, "gemini-2.0-flash")
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "search"
        assert resp.tool_calls[0].arguments == '{"q": "test"}'
        assert resp.tool_calls[0].id == "call_0"

    def test_tool_call_no_id_generates_one(self) -> None:
        fc = _make_function_call("search", {"q": "test"})
        parts = [_make_part(function_call=fc)]
        raw = _make_response(parts=parts)
        resp = _parse_response(raw, "gemini-2.0-flash")
        assert resp.tool_calls[0].id == "call_0"

    def test_no_usage(self) -> None:
        raw = _make_response()
        raw.usage_metadata = None
        resp = _parse_response(raw, "gemini-2.0-flash")
        assert resp.usage.input_tokens == 0

    def test_finish_reason_mapping(self) -> None:
        raw = _make_response(finish_reason="MAX_TOKENS")
        resp = _parse_response(raw, "gemini-2.0-flash")
        assert resp.finish_reason == "length"


# ---------------------------------------------------------------------------
# Stream chunk parsing
# ---------------------------------------------------------------------------


class TestParseStreamChunk:
    def test_text_delta(self) -> None:
        chunk = _make_stream_chunk(parts=[_make_part(text="Hi")])
        result = _parse_stream_chunk(chunk)
        assert result.delta == "Hi"
        assert result.finish_reason is None

    def test_finish_reason(self) -> None:
        chunk = _make_stream_chunk(finish_reason="STOP")
        result = _parse_stream_chunk(chunk)
        assert result.finish_reason == "stop"

    def test_tool_call_delta(self) -> None:
        fc = _make_function_call("search", {"q": "test"}, id="call_0")
        chunk = _make_stream_chunk(parts=[_make_part(function_call=fc)])
        result = _parse_stream_chunk(chunk)
        assert len(result.tool_call_deltas) == 1
        assert result.tool_call_deltas[0].id == "call_0"
        assert result.tool_call_deltas[0].name == "search"

    def test_usage_on_chunk(self) -> None:
        chunk = _make_stream_chunk(
            finish_reason="STOP",
            usage_metadata=_make_usage_metadata(
                prompt_token_count=20, candidates_token_count=30, total_token_count=50
            ),
        )
        result = _parse_stream_chunk(chunk)
        assert result.usage.input_tokens == 20
        assert result.usage.output_tokens == 30

    def test_empty_candidates(self) -> None:
        chunk = SimpleNamespace(candidates=[], usage_metadata=None)
        result = _parse_stream_chunk(chunk)
        assert result.delta == ""
        assert result.usage.input_tokens == 0


# ---------------------------------------------------------------------------
# Provider: complete()
# ---------------------------------------------------------------------------


class TestVertexProviderComplete:
    @patch("exo.models.gemini.genai")
    async def test_basic_complete(self, mock_genai: MagicMock) -> None:
        config = _make_config()
        provider = VertexProvider(config)
        raw = _make_response(content="world")
        provider._client = MagicMock()
        provider._client.aio.models.generate_content = AsyncMock(return_value=raw)

        result = await provider.complete([UserMessage(content="hello")])

        assert result.content == "world"
        provider._client.aio.models.generate_content.assert_awaited_once()

    @patch("exo.models.gemini.genai")
    async def test_complete_with_tools(self, mock_genai: MagicMock) -> None:
        config = _make_config()
        provider = VertexProvider(config)
        raw = _make_response()
        provider._client = MagicMock()
        provider._client.aio.models.generate_content = AsyncMock(return_value=raw)

        tools = [
            {"type": "function", "function": {"name": "t", "description": "d", "parameters": {}}}
        ]
        await provider.complete([UserMessage(content="hi")], tools=tools)

        call_kwargs = provider._client.aio.models.generate_content.call_args[1]
        assert "tools" in call_kwargs["config"]

    @patch("exo.models.gemini.genai")
    async def test_complete_with_temperature_and_max_tokens(self, mock_genai: MagicMock) -> None:
        config = _make_config()
        provider = VertexProvider(config)
        raw = _make_response()
        provider._client = MagicMock()
        provider._client.aio.models.generate_content = AsyncMock(return_value=raw)

        await provider.complete([UserMessage(content="hi")], temperature=0.5, max_tokens=100)

        call_kwargs = provider._client.aio.models.generate_content.call_args[1]
        assert call_kwargs["config"]["temperature"] == 0.5
        assert call_kwargs["config"]["max_output_tokens"] == 100

    @patch("exo.models.gemini.genai")
    async def test_complete_omits_none_params(self, mock_genai: MagicMock) -> None:
        config = _make_config()
        provider = VertexProvider(config)
        raw = _make_response()
        provider._client = MagicMock()
        provider._client.aio.models.generate_content = AsyncMock(return_value=raw)

        await provider.complete([UserMessage(content="hi")])

        call_kwargs = provider._client.aio.models.generate_content.call_args[1]
        cfg = call_kwargs["config"]
        assert "tools" not in cfg
        assert "temperature" not in cfg
        assert "max_output_tokens" not in cfg

    @patch("exo.models.gemini.genai")
    async def test_complete_api_error(self, mock_genai: MagicMock) -> None:
        config = _make_config()
        provider = VertexProvider(config)
        provider._client = MagicMock()
        provider._client.aio.models.generate_content = AsyncMock(
            side_effect=genai_errors.ClientError(429, {"error": {"message": "rate limited"}})
        )

        with pytest.raises(ModelError, match="rate limited"):
            await provider.complete([UserMessage(content="hi")])

    @patch("exo.models.gemini.genai")
    async def test_error_message_prefix(self, mock_genai: MagicMock) -> None:
        config = _make_config()
        provider = VertexProvider(config)
        provider._client = MagicMock()
        provider._client.aio.models.generate_content = AsyncMock(
            side_effect=genai_errors.ClientError(400, {"error": {"message": "fail"}})
        )

        with pytest.raises(ModelError) as exc_info:
            await provider.complete([UserMessage(content="hi")])
        assert "vertex:" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Provider: stream()
# ---------------------------------------------------------------------------


class TestVertexProviderStream:
    @patch("exo.models.gemini.genai")
    async def test_basic_stream(self, mock_genai: MagicMock) -> None:
        config = _make_config()
        provider = VertexProvider(config)

        chunks = [
            _make_stream_chunk(parts=[_make_part(text="Hel")]),
            _make_stream_chunk(parts=[_make_part(text="lo")]),
            _make_stream_chunk(
                finish_reason="STOP",
                usage_metadata=_make_usage_metadata(
                    prompt_token_count=5, candidates_token_count=2, total_token_count=7
                ),
            ),
        ]

        async def mock_stream(**kwargs: Any) -> Any:
            async def gen():
                for c in chunks:
                    yield c

            return gen()

        provider._client = MagicMock()
        provider._client.aio.models.generate_content_stream = AsyncMock(side_effect=mock_stream)

        collected = []
        async for chunk in provider.stream([UserMessage(content="hi")]):
            collected.append(chunk)

        assert len(collected) == 3
        assert collected[0].delta == "Hel"
        assert collected[1].delta == "lo"
        assert collected[2].finish_reason == "stop"
        assert collected[2].usage.input_tokens == 5

    @patch("exo.models.gemini.genai")
    async def test_stream_api_error(self, mock_genai: MagicMock) -> None:
        config = _make_config()
        provider = VertexProvider(config)
        provider._client = MagicMock()
        provider._client.aio.models.generate_content_stream = AsyncMock(
            side_effect=genai_errors.ServerError(500, {"error": {"message": "server error"}})
        )

        with pytest.raises(ModelError, match="server error"):
            async for _ in provider.stream([UserMessage(content="hi")]):
                pass

    @patch("exo.models.gemini.genai")
    async def test_stream_error_prefix(self, mock_genai: MagicMock) -> None:
        config = _make_config()
        provider = VertexProvider(config)
        provider._client = MagicMock()
        provider._client.aio.models.generate_content_stream = AsyncMock(
            side_effect=genai_errors.ServerError(500, {"error": {"message": "fail"}})
        )

        with pytest.raises(ModelError) as exc_info:
            async for _ in provider.stream([UserMessage(content="hi")]):
                pass
        assert "vertex:" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestVertexRegistration:
    def test_registered(self) -> None:
        assert model_registry.get("vertex") is VertexProvider

    @patch("exo.models.gemini.genai")
    def test_get_provider_creates_instance(self, mock_genai: MagicMock) -> None:
        from exo.models.provider import get_provider  # pyright: ignore[reportMissingImports]

        provider = get_provider("vertex:gemini-2.0-flash", google_project="test-project")
        assert isinstance(provider, VertexProvider)
        assert provider.config.model_name == "gemini-2.0-flash"


# ---------------------------------------------------------------------------
# Service account credentials
# ---------------------------------------------------------------------------

_FAKE_SA_INFO = {
    "type": "service_account",
    "project_id": "test-project",
    "private_key_id": "key123",
    "private_key": "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA2a2rwplBQLF29amygykEMmYz0+Kcj3bKBp29DiGFiDEFhmm\nNguMjTmd9jETJFnNpHONGRqJmH3dTaFMnVR7MlDqIClUF8LgVgg6BN7gVFMBef9\nZ79kNjuerMOkYJFLWKn2eSMdLRF7q2FGqxyjRHwcJwmBR4PJMqBXHGfcbF+Gl1WB\nmrzbPFldVVeR0YCzrGJGbOafH9hy8qDdVEKLCIFB98fCD5lnSilJkMFfl50PKtt1\nKP6OQFm0g14KNsVMY0GPAFFeDKdfl1Js03WnRfAlKb1mB6bOOvKSmxNq1TxB3eyT\ncItHnEJMc8JCsJAqjF0eFebSMLB6UALRXJ8jUQIDAQABAoIBAA/jR0GH0bJarP8D\nnleQxnSVxKBhsBO+RaGCNyM0iRkS5nZk/BxMrn6pIYrmcHNMx5GzQWlFRQ+dQjdl\nHnN9JSRlMIKAd3AjKiVjMYRWRKua87li6u6IM7JGqhCpX6S7zY0HnWtHanHnM9XE\nNZWvNJuETz0/DOa1U00t+qLJy2C8h1dPz0NmR3M1Gj8wC7wdGFIGQIr6kNajryq\nnmPOL2b9ynFnnVJBpPb/H2TsS4FJB1PGa7kOXNJxbK8GrPNIvYvKT6pwJSQfcnk1\nmg5iXU8ePwnfHhZ1RRfC+q/cjgP3ElPW07HCB6PgOKIi85S5EleZ/J1q4/bMMpJA\naGS6goECgYEA7hYljAIxfMad8NvIl1JNFllWqccjk/G4D+Q6kD/1t1+VAt10c2E+\ni/KVxDkR7wJy3mLi4vyQLwS3pm2G3K4VUMwb3PE2Z7RjVF36eP2BjLj0PJUQ23tX\nwd/GKLedLi5YqyLtKK/jLrm+4DYimJYCgiT0j9kxasThb3JklNbW+IECgYEA6WRC\nG2f2FLMshV1dcsFn43OjWM9EIi6L0BD2fnKh0NTTfXhWMUaCVJWf5iP0e7DZPXY9\nk3+L4u/JCvL7EqKQvlHRCi4Bq05GI5XFVQ/06KTdSPUXjnOA3S63S/P5GiL98mY\nNpOBC6mcYnJIDFnBT2vUfjCbzzFd+GE1/SEJ0iECgYBaZDF3gPddJAu/8/Ib/QDI\nhVPJKUYN5K71/r/dYMZTEJSJLi1MnXmVXw/ki+5g/+U+k2S5fCm1RWMeWbDRUVv0\nwJRNpMKV6JCK3d6rfT6s3O40t9v5sCjSMCStE4D35qMxO0MABQMJ4mS67u2VfKEN\noJvYN+tDCkh/FJJo+g/gAQKBgFWZiCfOIXFEiJYA/1lJsm8s6xAEWWP0xfwdXNkb\nsEAdojKNJpD0IlONRnWGT3hELisSAKN5deNT6eGFKYqlIGqPNjtr38HNm7WPNjiG\njGLvFj6gPBxHQ01p1eYzL+U2f4hjT1LElpPuPH2cGH2M8d/dVGtukBi1io+dNdEn\nb3RhAoGBAM+sLGOVj36cGaLj0fRil1N9F1cv8RMKhHm8mlE3J3FPqO3+BhpZnkBe\n+H0r/2LaIMuP/CpjGo0FPbHKfsVFKb0F8ItERGOzS6NU0EXO6ILhFNxg9AJnVU+r\nhQqz7o+Z0vMp36PFdJz9jMjHghXn2YCxL8lFT/3hlJH6j1MJL+SN\n-----END RSA PRIVATE KEY-----\n",
    "client_email": "test@test-project.iam.gserviceaccount.com",
    "client_id": "123456789",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
}


class TestServiceAccountCredentials:
    def test_credentials_from_base64_decodes_and_builds(self) -> None:
        encoded = base64.b64encode(json.dumps(_FAKE_SA_INFO).encode()).decode()
        mock_creds = MagicMock()
        with patch(
            "google.oauth2.service_account.Credentials.from_service_account_info",
            return_value=mock_creds,
        ) as mock_from_info:
            result = _credentials_from_base64(encoded)
            mock_from_info.assert_called_once()
            call_args = mock_from_info.call_args
            assert call_args[0][0]["project_id"] == "test-project"
            assert call_args[1]["scopes"] == ["https://www.googleapis.com/auth/cloud-platform"]
            assert result is mock_creds

    @patch("exo.models.gemini.genai")
    def test_init_uses_service_account_from_env(self, mock_genai: MagicMock) -> None:
        encoded = base64.b64encode(json.dumps(_FAKE_SA_INFO).encode()).decode()
        mock_creds = MagicMock()
        with (
            patch.dict(os.environ, {"GOOGLE_SERVICE_ACCOUNT_BASE64": encoded}),
            patch("exo.models.gemini._credentials_from_base64", return_value=mock_creds) as mock_fn,
        ):
            config = _make_config()
            VertexProvider(config)
            mock_fn.assert_called_once_with(encoded)
            call_kwargs = mock_genai.Client.call_args[1]
            assert call_kwargs["credentials"] is mock_creds

    @patch("exo.models.gemini.genai")
    def test_init_no_credentials_when_nothing_set(self, mock_genai: MagicMock) -> None:
        env = {k: v for k, v in os.environ.items() if k != "GOOGLE_SERVICE_ACCOUNT_BASE64"}
        with patch.dict(os.environ, env, clear=True):
            config = _make_config()
            VertexProvider(config)
            call_kwargs = mock_genai.Client.call_args[1]
            assert call_kwargs["credentials"] is None


# ---------------------------------------------------------------------------
# Config-based provider parameters
# ---------------------------------------------------------------------------


class TestVertexConfigParams:
    @patch("exo.models.gemini.genai")
    def test_project_from_config(self, mock_genai: MagicMock) -> None:
        config = _make_config(google_project="cfg-project")
        VertexProvider(config)
        call_kwargs = mock_genai.Client.call_args[1]
        assert call_kwargs["project"] == "cfg-project"

    @patch("exo.models.gemini.genai")
    def test_location_from_config(self, mock_genai: MagicMock) -> None:
        config = _make_config(google_location="europe-west1")
        VertexProvider(config)
        call_kwargs = mock_genai.Client.call_args[1]
        assert call_kwargs["location"] == "europe-west1"

    @patch("exo.models.gemini.genai")
    def test_service_account_from_config(self, mock_genai: MagicMock) -> None:
        encoded = base64.b64encode(json.dumps(_FAKE_SA_INFO).encode()).decode()
        mock_creds = MagicMock()
        env = {k: v for k, v in os.environ.items() if k != "GOOGLE_SERVICE_ACCOUNT_BASE64"}
        with (
            patch.dict(os.environ, env, clear=True),
            patch("exo.models.gemini._credentials_from_base64", return_value=mock_creds) as mock_fn,
        ):
            config = _make_config(google_service_account_base64=encoded)
            VertexProvider(config)
            mock_fn.assert_called_once_with(encoded)
            call_kwargs = mock_genai.Client.call_args[1]
            assert call_kwargs["credentials"] is mock_creds

    @patch("exo.models.gemini.genai")
    def test_config_takes_precedence_over_env(self, mock_genai: MagicMock) -> None:
        with patch.dict(
            os.environ,
            {"GOOGLE_CLOUD_PROJECT": "env-project", "GOOGLE_CLOUD_LOCATION": "env-location"},
        ):
            config = _make_config(google_project="cfg-project", google_location="cfg-location")
            VertexProvider(config)
            call_kwargs = mock_genai.Client.call_args[1]
            assert call_kwargs["project"] == "cfg-project"
            assert call_kwargs["location"] == "cfg-location"

    @patch("exo.models.gemini.genai")
    def test_falls_back_to_env_when_config_absent(self, mock_genai: MagicMock) -> None:
        with patch.dict(
            os.environ,
            {"GOOGLE_CLOUD_PROJECT": "env-project", "GOOGLE_CLOUD_LOCATION": "env-location"},
        ):
            # google_project=None so env fallback is exercised.
            config = _make_config(google_project=None)
            VertexProvider(config)
            call_kwargs = mock_genai.Client.call_args[1]
            assert call_kwargs["project"] == "env-project"
            assert call_kwargs["location"] == "env-location"

    @patch("exo.models.gemini.genai")
    def test_defaults_when_only_project_set(self, mock_genai: MagicMock) -> None:
        # When project is provided but location is not, location defaults to "us-central1".
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION")
        }
        with patch.dict(os.environ, env, clear=True):
            # Use config project only (no env fallback) and omit location.
            config = _make_config(google_project="my-project", google_location=None)
            VertexProvider(config)
            call_kwargs = mock_genai.Client.call_args[1]
            assert call_kwargs["project"] == "my-project"
            assert call_kwargs["location"] == "us-central1"

    @patch("exo.models.gemini.genai")
    def test_get_provider_forwards_extras(self, mock_genai: MagicMock) -> None:
        from exo.models.provider import get_provider  # pyright: ignore[reportMissingImports]

        provider = get_provider(
            "vertex:gemini-2.0-flash",
            google_project="my-project",
            google_location="asia-east1",
        )
        assert isinstance(provider, VertexProvider)
        call_kwargs = mock_genai.Client.call_args[1]
        assert call_kwargs["project"] == "my-project"
        assert call_kwargs["location"] == "asia-east1"
