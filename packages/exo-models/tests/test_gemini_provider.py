"""Tests for the Google Gemini LLM provider."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import errors as genai_errors

from exo.config import ModelConfig  # pyright: ignore[reportMissingImports]
from exo.models.gemini import (  # pyright: ignore[reportMissingImports]
    GeminiProvider,
    _build_config,
    _convert_tools,
    _map_finish_reason,
    _parse_response,
    _parse_stream_chunk,
    _to_google_contents,
)
from exo.models.provider import model_registry  # pyright: ignore[reportMissingImports]
from exo.models.types import ModelError  # pyright: ignore[reportMissingImports]
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
    defaults: dict[str, Any] = {
        "provider": "gemini",
        "model_name": "gemini-2.0-flash",
        "api_key": "test-key",
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

    def test_multiple_tools(self) -> None:
        tools = [
            {"type": "function", "function": {"name": "a", "description": "A", "parameters": {}}},
            {"type": "function", "function": {"name": "b", "description": "B", "parameters": {}}},
        ]
        result = _convert_tools(tools)
        assert len(result) == 1
        assert len(result[0]["function_declarations"]) == 2

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

    def test_only_system_instruction(self) -> None:
        config = _build_config(None, None, None, "be helpful")
        assert config == {"system_instruction": "be helpful"}
        assert "tools" not in config
        assert "temperature" not in config
        assert "max_output_tokens" not in config


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

    def test_tool_call_empty_args(self) -> None:
        fc = _make_function_call("ping", args=None)
        parts = [_make_part(function_call=fc)]
        raw = _make_response(parts=parts)
        resp = _parse_response(raw, "gemini-2.0-flash")
        assert resp.tool_calls[0].arguments == "{}"

    def test_mixed_text_and_tool_call(self) -> None:
        fc = _make_function_call("search", {"q": "test"})
        parts = [_make_part(text="Let me search"), _make_part(function_call=fc)]
        raw = _make_response(parts=parts)
        resp = _parse_response(raw, "gemini-2.0-flash")
        assert resp.content == "Let me search"
        assert len(resp.tool_calls) == 1

    def test_no_usage(self) -> None:
        raw = _make_response()
        raw.usage_metadata = None
        resp = _parse_response(raw, "gemini-2.0-flash")
        assert resp.usage.input_tokens == 0

    def test_finish_reason_mapping(self) -> None:
        raw = _make_response(finish_reason="MAX_TOKENS")
        resp = _parse_response(raw, "gemini-2.0-flash")
        assert resp.finish_reason == "length"

    def test_model_name_passthrough(self) -> None:
        raw = _make_response()
        resp = _parse_response(raw, "gemini-2.0-flash")
        assert resp.model == "gemini-2.0-flash"
        assert resp.id == ""


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
        assert result.tool_call_deltas[0].arguments == '{"q": "test"}'

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
        assert result.usage.total_tokens == 50

    def test_empty_candidates(self) -> None:
        chunk = SimpleNamespace(candidates=[], usage_metadata=None)
        result = _parse_stream_chunk(chunk)
        assert result.delta == ""
        assert result.usage.input_tokens == 0

    def test_empty_candidates_with_usage(self) -> None:
        chunk = SimpleNamespace(
            candidates=[],
            usage_metadata=_make_usage_metadata(
                prompt_token_count=10, candidates_token_count=5, total_token_count=15
            ),
        )
        result = _parse_stream_chunk(chunk)
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5


# ---------------------------------------------------------------------------
# Provider: complete()
# ---------------------------------------------------------------------------


class TestGeminiProviderComplete:
    @patch("exo.models.gemini.genai")
    async def test_basic_complete(self, mock_genai: MagicMock) -> None:
        config = _make_config()
        provider = GeminiProvider(config)
        raw = _make_response(content="world")
        provider._client = MagicMock()
        provider._client.aio.models.generate_content = AsyncMock(return_value=raw)

        result = await provider.complete([UserMessage(content="hello")])

        assert result.content == "world"
        provider._client.aio.models.generate_content.assert_awaited_once()

    @patch("exo.models.gemini.genai")
    async def test_complete_with_tools(self, mock_genai: MagicMock) -> None:
        config = _make_config()
        provider = GeminiProvider(config)
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
        provider = GeminiProvider(config)
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
        provider = GeminiProvider(config)
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
    async def test_complete_with_system_message(self, mock_genai: MagicMock) -> None:
        config = _make_config()
        provider = GeminiProvider(config)
        raw = _make_response()
        provider._client = MagicMock()
        provider._client.aio.models.generate_content = AsyncMock(return_value=raw)

        await provider.complete([SystemMessage(content="be helpful"), UserMessage(content="hi")])

        call_kwargs = provider._client.aio.models.generate_content.call_args[1]
        assert call_kwargs["config"]["system_instruction"] == "be helpful"

    @patch("exo.models.gemini.genai")
    async def test_complete_api_error(self, mock_genai: MagicMock) -> None:
        config = _make_config()
        provider = GeminiProvider(config)
        provider._client = MagicMock()
        provider._client.aio.models.generate_content = AsyncMock(
            side_effect=genai_errors.ClientError(429, {"error": {"message": "rate limited"}})
        )

        with pytest.raises(ModelError, match="rate limited"):
            await provider.complete([UserMessage(content="hi")])


# ---------------------------------------------------------------------------
# Provider: stream()
# ---------------------------------------------------------------------------


class TestGeminiProviderStream:
    @patch("exo.models.gemini.genai")
    async def test_basic_stream(self, mock_genai: MagicMock) -> None:
        config = _make_config()
        provider = GeminiProvider(config)

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
    async def test_stream_with_tool_calls(self, mock_genai: MagicMock) -> None:
        config = _make_config()
        provider = GeminiProvider(config)

        fc = _make_function_call("search", {"q": "test"}, id="call_0")
        chunks = [
            _make_stream_chunk(parts=[_make_part(function_call=fc)]),
            _make_stream_chunk(finish_reason="STOP"),
        ]

        async def mock_stream(**kwargs: Any) -> Any:
            async def gen():
                for c in chunks:
                    yield c

            return gen()

        provider._client = MagicMock()
        provider._client.aio.models.generate_content_stream = AsyncMock(side_effect=mock_stream)

        collected = []
        async for chunk in provider.stream([UserMessage(content="search")]):
            collected.append(chunk)

        assert len(collected) == 2
        assert collected[0].tool_call_deltas[0].name == "search"
        assert collected[1].finish_reason == "stop"

    @patch("exo.models.gemini.genai")
    async def test_stream_api_error(self, mock_genai: MagicMock) -> None:
        config = _make_config()
        provider = GeminiProvider(config)
        provider._client = MagicMock()
        provider._client.aio.models.generate_content_stream = AsyncMock(
            side_effect=genai_errors.ServerError(500, {"error": {"message": "server error"}})
        )

        with pytest.raises(ModelError, match="server error"):
            async for _ in provider.stream([UserMessage(content="hi")]):
                pass


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestGeminiRegistration:
    def test_registered(self) -> None:
        assert model_registry.get("gemini") is GeminiProvider

    @patch("exo.models.gemini.genai")
    def test_get_provider_creates_instance(self, mock_genai: MagicMock) -> None:
        from exo.models.provider import get_provider  # pyright: ignore[reportMissingImports]

        provider = get_provider("gemini:gemini-2.0-flash", api_key="test-key")
        assert isinstance(provider, GeminiProvider)
        assert provider.config.model_name == "gemini-2.0-flash"


# ---------------------------------------------------------------------------
# Safety-blocked response guard (findings #1, #2)
# ---------------------------------------------------------------------------


class TestParseResponseSafetyBlocked:
    def test_empty_candidates_returns_content_filter(self) -> None:
        """Safety-blocked responses have candidates=[] — must not raise IndexError."""
        raw = SimpleNamespace(
            candidates=[],
            usage_metadata=SimpleNamespace(
                prompt_token_count=10,
                candidates_token_count=0,
                total_token_count=10,
            ),
        )
        resp = _parse_response(raw, "gemini-2.0-flash")
        assert resp.content == ""
        assert resp.finish_reason == "content_filter"
        assert resp.tool_calls == []
        assert resp.usage.input_tokens == 10

    def test_empty_candidates_no_usage(self) -> None:
        raw = SimpleNamespace(candidates=[], usage_metadata=None)
        resp = _parse_response(raw, "gemini-2.0-flash")
        assert resp.finish_reason == "content_filter"
        assert resp.usage.total_tokens == 0

    def test_null_candidate_content_returns_empty(self) -> None:
        """A candidate with content=None must not raise AttributeError."""
        candidate = SimpleNamespace(content=None, finish_reason="SAFETY")
        raw = SimpleNamespace(
            candidates=[candidate],
            usage_metadata=None,
        )
        resp = _parse_response(raw, "gemini-2.0-flash")
        assert resp.content == ""
        assert resp.finish_reason == "content_filter"


# ---------------------------------------------------------------------------
# New finish reason codes (finding #9)
# ---------------------------------------------------------------------------


class TestFinishReasonNewCodes:
    def test_prohibited_content(self) -> None:
        assert _map_finish_reason("PROHIBITED_CONTENT") == "content_filter"

    def test_spii(self) -> None:
        assert _map_finish_reason("SPII") == "content_filter"

    def test_image_safety(self) -> None:
        assert _map_finish_reason("IMAGE_SAFETY") == "content_filter"

    def test_image_prohibited_content(self) -> None:
        assert _map_finish_reason("IMAGE_PROHIBITED_CONTENT") == "content_filter"


# ---------------------------------------------------------------------------
# Vertex AI missing project validation (finding #5)
# ---------------------------------------------------------------------------


class TestVertexProjectValidation:
    @patch("exo.models.gemini.genai")
    def test_location_without_project_raises(self, mock_genai: MagicMock) -> None:
        """Setting google_location without google_project should raise ModelError early."""
        config = ModelConfig(
            provider="gemini",
            model_name="gemini-2.0-flash",
            api_key="",
            google_location="us-central1",  # type: ignore[call-arg]
        )
        with pytest.raises(ModelError, match="Google Cloud project"):
            GeminiProvider(config)

    @patch("exo.models.gemini.genai")
    def test_project_and_location_ok(self, mock_genai: MagicMock) -> None:
        """Both project and location provided — should not raise."""
        config = ModelConfig(
            provider="gemini",
            model_name="gemini-2.0-flash",
            api_key="",
            google_project="my-project",  # type: ignore[call-arg]
            google_location="us-central1",  # type: ignore[call-arg]
        )
        # Should construct without error
        provider = GeminiProvider(config)
        assert provider._prefix == "vertex"
