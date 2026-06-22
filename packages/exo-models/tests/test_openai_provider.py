"""Tests for the OpenAI LLM provider."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from exo.config import ModelConfig  # pyright: ignore[reportMissingImports]
from exo.models.openai import (  # pyright: ignore[reportMissingImports]
    OpenAIProvider,
    _map_finish_reason,
    _parse_response,
    _parse_stream_chunk,
    _to_openai_messages,
)
from exo.models.provider import model_registry  # pyright: ignore[reportMissingImports]
from exo.models.types import ModelError  # pyright: ignore[reportMissingImports]
from exo.types import (  # pyright: ignore[reportMissingImports]
    AssistantMessage,
    ImageDataBlock,
    SystemMessage,
    TextBlock,
    ToolCall,
    ToolResult,
    UserMessage,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides: Any) -> ModelConfig:
    defaults: dict[str, Any] = {"provider": "openai", "model_name": "gpt-4o", "api_key": "test-key"}
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _make_response(
    *,
    content: str = "hello",
    tool_calls: list[Any] | None = None,
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    total_tokens: int = 15,
    response_id: str = "chatcmpl-abc",
    model: str = "gpt-4o",
    reasoning_content: str | None = None,
) -> SimpleNamespace:
    msg = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        model_extra={"reasoning_content": reasoning_content} if reasoning_content else {},
    )
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )
    return SimpleNamespace(id=response_id, model=model, choices=[choice], usage=usage)


def _make_stream_chunk(
    *,
    content: str | None = None,
    tool_calls: list[Any] | None = None,
    finish_reason: str | None = None,
    usage: Any = None,
) -> SimpleNamespace:
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_usage_chunk(
    *, prompt_tokens: int = 10, completion_tokens: int = 5, total_tokens: int = 15
) -> SimpleNamespace:
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )
    return SimpleNamespace(choices=[], usage=usage)


# ---------------------------------------------------------------------------
# Finish reason mapping
# ---------------------------------------------------------------------------


class TestFinishReasonMapping:
    def test_stop(self) -> None:
        assert _map_finish_reason("stop") == "stop"

    def test_tool_calls(self) -> None:
        assert _map_finish_reason("tool_calls") == "tool_calls"

    def test_length(self) -> None:
        assert _map_finish_reason("length") == "length"

    def test_content_filter(self) -> None:
        assert _map_finish_reason("content_filter") == "content_filter"

    def test_none_defaults_to_stop(self) -> None:
        assert _map_finish_reason(None) == "stop"

    def test_unknown_defaults_to_stop(self) -> None:
        assert _map_finish_reason("unknown_reason") == "stop"


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


class TestToOpenAIMessages:
    def test_system_message(self) -> None:
        result = _to_openai_messages([SystemMessage(content="be helpful")])
        assert result == [{"role": "system", "content": "be helpful"}]

    def test_user_message(self) -> None:
        result = _to_openai_messages([UserMessage(content="hi")])
        assert result == [{"role": "user", "content": "hi"}]

    def test_assistant_with_text(self) -> None:
        result = _to_openai_messages([AssistantMessage(content="hello")])
        assert result == [{"role": "assistant", "content": "hello"}]

    def test_assistant_with_tool_calls(self) -> None:
        msg = AssistantMessage(
            content="",
            tool_calls=[ToolCall(id="tc1", name="search", arguments='{"q":"x"}')],
        )
        result = _to_openai_messages([msg])
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["tool_calls"][0]["id"] == "tc1"
        assert result[0]["tool_calls"][0]["type"] == "function"
        assert result[0]["tool_calls"][0]["function"]["name"] == "search"

    def test_assistant_empty(self) -> None:
        result = _to_openai_messages([AssistantMessage()])
        assert result[0]["content"] == ""

    def test_tool_result(self) -> None:
        result = _to_openai_messages(
            [
                ToolResult(tool_call_id="tc1", tool_name="search", content="found it"),
            ]
        )
        assert result == [{"role": "tool", "tool_call_id": "tc1", "content": "found it"}]

    def test_tool_result_with_error(self) -> None:
        result = _to_openai_messages(
            [
                ToolResult(tool_call_id="tc1", tool_name="search", content="", error="failed"),
            ]
        )
        assert result[0]["content"] == "failed"


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_text_response(self) -> None:
        raw = _make_response(content="Hello!")
        resp = _parse_response(raw, "gpt-4o")
        assert resp.content == "Hello!"
        assert resp.finish_reason == "stop"
        assert resp.usage.input_tokens == 10
        assert resp.usage.output_tokens == 5

    def test_tool_call_response(self) -> None:
        tc = SimpleNamespace(
            id="tc1",
            function=SimpleNamespace(name="search", arguments='{"q":"test"}'),
        )
        raw = _make_response(content="", tool_calls=[tc], finish_reason="tool_calls")
        resp = _parse_response(raw, "gpt-4o")
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "search"
        assert resp.finish_reason == "tool_calls"

    def test_reasoning_content(self) -> None:
        raw = _make_response(content="answer", reasoning_content="I think...")
        resp = _parse_response(raw, "o1")
        assert resp.reasoning_content == "I think..."

    def test_no_usage(self) -> None:
        raw = _make_response()
        raw.usage = None
        resp = _parse_response(raw, "gpt-4o")
        assert resp.usage.input_tokens == 0

    def test_model_id_passthrough(self) -> None:
        raw = _make_response(response_id="id-123", model="gpt-4o-2024")
        resp = _parse_response(raw, "gpt-4o")
        assert resp.id == "id-123"
        assert resp.model == "gpt-4o-2024"


# ---------------------------------------------------------------------------
# Stream chunk parsing
# ---------------------------------------------------------------------------


class TestParseStreamChunk:
    def test_text_delta(self) -> None:
        chunk = _make_stream_chunk(content="Hi")
        result = _parse_stream_chunk(chunk)
        assert result.delta == "Hi"
        assert result.finish_reason is None

    def test_finish_reason(self) -> None:
        chunk = _make_stream_chunk(finish_reason="stop")
        result = _parse_stream_chunk(chunk)
        assert result.finish_reason == "stop"

    def test_tool_call_delta(self) -> None:
        tc_delta = SimpleNamespace(
            index=0,
            id="tc1",
            function=SimpleNamespace(name="search", arguments='{"q":'),
        )
        chunk = _make_stream_chunk(tool_calls=[tc_delta])
        result = _parse_stream_chunk(chunk)
        assert len(result.tool_call_deltas) == 1
        assert result.tool_call_deltas[0].id == "tc1"
        assert result.tool_call_deltas[0].name == "search"
        assert result.tool_call_deltas[0].arguments == '{"q":'

    def test_tool_call_delta_no_function(self) -> None:
        tc_delta = SimpleNamespace(index=0, id=None, function=None)
        chunk = _make_stream_chunk(tool_calls=[tc_delta])
        result = _parse_stream_chunk(chunk)
        assert result.tool_call_deltas[0].name is None
        assert result.tool_call_deltas[0].arguments == ""

    def test_usage_only_chunk(self) -> None:
        chunk = _make_usage_chunk(prompt_tokens=20, completion_tokens=30, total_tokens=50)
        result = _parse_stream_chunk(chunk)
        assert result.usage.input_tokens == 20
        assert result.usage.output_tokens == 30
        assert result.delta == ""

    def test_usage_only_chunk_no_usage(self) -> None:
        chunk = SimpleNamespace(choices=[], usage=None)
        result = _parse_stream_chunk(chunk)
        assert result.usage.input_tokens == 0


# ---------------------------------------------------------------------------
# Provider: complete()
# ---------------------------------------------------------------------------


class TestOpenAIProviderComplete:
    async def test_basic_complete(self) -> None:
        config = _make_config()
        provider = OpenAIProvider(config)
        raw = _make_response(content="world")
        provider._client.chat.completions.create = AsyncMock(return_value=raw)

        result = await provider.complete([UserMessage(content="hello")])

        assert result.content == "world"
        provider._client.chat.completions.create.assert_awaited_once()

    async def test_complete_with_tools(self) -> None:
        config = _make_config()
        provider = OpenAIProvider(config)
        raw = _make_response()
        provider._client.chat.completions.create = AsyncMock(return_value=raw)

        tools = [{"type": "function", "function": {"name": "t", "parameters": {}}}]
        await provider.complete([UserMessage(content="hi")], tools=tools)

        call_kwargs = provider._client.chat.completions.create.call_args[1]
        assert call_kwargs["tools"] == tools

    async def test_complete_with_temperature_and_max_tokens(self) -> None:
        config = _make_config()
        provider = OpenAIProvider(config)
        raw = _make_response()
        provider._client.chat.completions.create = AsyncMock(return_value=raw)

        await provider.complete([UserMessage(content="hi")], temperature=0.5, max_tokens=100)

        call_kwargs = provider._client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100

    async def test_complete_omits_none_params(self) -> None:
        config = _make_config()
        provider = OpenAIProvider(config)
        raw = _make_response()
        provider._client.chat.completions.create = AsyncMock(return_value=raw)

        await provider.complete([UserMessage(content="hi")])

        call_kwargs = provider._client.chat.completions.create.call_args[1]
        assert "tools" not in call_kwargs
        assert "temperature" not in call_kwargs
        assert "max_tokens" not in call_kwargs

    async def test_complete_api_error(self) -> None:
        import openai as openai_mod

        config = _make_config()
        provider = OpenAIProvider(config)
        provider._client.chat.completions.create = AsyncMock(
            side_effect=openai_mod.APIError(
                message="rate limited",
                request=MagicMock(),
                body=None,
            )
        )

        with pytest.raises(ModelError, match="rate limited"):
            await provider.complete([UserMessage(content="hi")])


# ---------------------------------------------------------------------------
# Provider: stream()
# ---------------------------------------------------------------------------


class TestOpenAIProviderStream:
    async def test_basic_stream(self) -> None:
        config = _make_config()
        provider = OpenAIProvider(config)

        chunks = [
            _make_stream_chunk(content="Hel"),
            _make_stream_chunk(content="lo"),
            _make_stream_chunk(finish_reason="stop"),
            _make_usage_chunk(prompt_tokens=5, completion_tokens=2, total_tokens=7),
        ]

        async def mock_stream(**kwargs: Any) -> Any:
            async def gen():
                for c in chunks:
                    yield c

            return gen()

        provider._client.chat.completions.create = AsyncMock(side_effect=mock_stream)

        collected = []
        async for chunk in provider.stream([UserMessage(content="hi")]):
            collected.append(chunk)

        assert len(collected) == 4
        assert collected[0].delta == "Hel"
        assert collected[1].delta == "lo"
        assert collected[2].finish_reason == "stop"
        assert collected[3].usage.input_tokens == 5

    async def test_stream_includes_stream_options(self) -> None:
        config = _make_config()
        provider = OpenAIProvider(config)

        async def mock_stream(**kwargs: Any) -> Any:
            assert kwargs["stream"] is True
            assert kwargs["stream_options"] == {"include_usage": True}

            async def gen():
                return
                yield  # make it an async generator

            return gen()

        provider._client.chat.completions.create = AsyncMock(side_effect=mock_stream)

        async for _ in provider.stream([UserMessage(content="hi")]):
            pass

    async def test_stream_api_error(self) -> None:
        import openai as openai_mod

        config = _make_config()
        provider = OpenAIProvider(config)
        provider._client.chat.completions.create = AsyncMock(
            side_effect=openai_mod.APIError(
                message="server error",
                request=MagicMock(),
                body=None,
            )
        )

        with pytest.raises(ModelError, match="server error"):
            async for _ in provider.stream([UserMessage(content="hi")]):
                pass


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestOpenAIRegistration:
    def test_registered(self) -> None:
        assert model_registry.get("openai") is OpenAIProvider

    def test_get_provider_creates_instance(self) -> None:
        from exo.models.provider import get_provider  # pyright: ignore[reportMissingImports]

        provider = get_provider("openai:gpt-4o", api_key="test-key")
        assert isinstance(provider, OpenAIProvider)
        assert provider.config.model_name == "gpt-4o"


# ---------------------------------------------------------------------------
# ToolResult text extraction from list content (finding #6)
# ---------------------------------------------------------------------------


class TestToolResultTextExtraction:
    def test_text_block_content_extracted(self) -> None:
        """ToolResult with only TextBlocks must use the text, not '[media result]'."""
        result = _to_openai_messages(
            [
                ToolResult(
                    tool_call_id="tc1",
                    tool_name="search",
                    content=[TextBlock(text="the answer")],
                )
            ]
        )
        tool_msg = result[0]
        assert tool_msg["role"] == "tool"
        assert tool_msg["content"] == "the answer"
        # No synthetic user message when no non-text media blocks
        assert len(result) == 1

    def test_multiple_text_blocks_joined(self) -> None:
        result = _to_openai_messages(
            [
                ToolResult(
                    tool_call_id="tc1",
                    tool_name="search",
                    content=[TextBlock(text="part one"), TextBlock(text="part two")],
                )
            ]
        )
        assert result[0]["content"] == "part one part two"

    def test_media_only_content_uses_placeholder(self) -> None:
        """No TextBlocks at all → fall back to '[media result]'."""
        result = _to_openai_messages(
            [
                ToolResult(
                    tool_call_id="tc1",
                    tool_name="search",
                    content=[ImageDataBlock(data="abc", media_type="image/png")],
                )
            ]
        )
        assert result[0]["content"] == "[media result]"
        # Synthetic user message with the image parts
        assert result[1]["role"] == "user"

    def test_error_overrides_text_blocks(self) -> None:
        """Error string takes precedence even when TextBlocks are present."""
        result = _to_openai_messages(
            [
                ToolResult(
                    tool_call_id="tc1",
                    tool_name="search",
                    content=[TextBlock(text="ignored")],
                    error="something went wrong",
                )
            ]
        )
        assert result[0]["content"] == "something went wrong"
