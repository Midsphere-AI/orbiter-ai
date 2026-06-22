"""Tests for ModelProvider ABC, model_registry, and get_provider()."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from exo.config import ModelConfig  # pyright: ignore[reportMissingImports]
from exo.models.provider import (  # pyright: ignore[reportMissingImports]
    ModelProvider,
    get_provider,
    model_registry,
)
from exo.models.types import (  # pyright: ignore[reportMissingImports]
    ModelError,
    ModelResponse,
    StreamChunk,
)
from exo.registry import Registry  # pyright: ignore[reportMissingImports]
from exo.types import Message, UserMessage  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Test double
# ---------------------------------------------------------------------------


class StubProvider(ModelProvider):
    """Concrete stub for testing the ABC contract."""

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ModelResponse:
        return ModelResponse(content="stub", model=self.config.model_name)

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(delta="hello ")
        yield StreamChunk(delta="world", finish_reason="stop")


# ---------------------------------------------------------------------------
# TestModelProvider
# ---------------------------------------------------------------------------


class TestModelProvider:
    """Tests for the ModelProvider ABC itself."""

    def test_abc_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            ModelProvider(ModelConfig(provider="x", model_name="y"))  # type: ignore[abstract]

    def test_stub_instantiates(self) -> None:
        provider = StubProvider(ModelConfig(provider="test", model_name="stub-1"))
        assert isinstance(provider, ModelProvider)

    def test_config_stored(self) -> None:
        cfg = ModelConfig(provider="test", model_name="m")
        provider = StubProvider(cfg)
        assert provider.config is cfg

    async def test_complete_returns_model_response(self) -> None:
        provider = StubProvider(ModelConfig(provider="test", model_name="m"))
        result = await provider.complete([UserMessage(content="hi")])
        assert isinstance(result, ModelResponse)
        assert result.content == "stub"

    async def test_stream_yields_stream_chunks(self) -> None:
        provider = StubProvider(ModelConfig(provider="test", model_name="m"))
        chunks = [c async for c in provider.stream([UserMessage(content="hi")])]
        assert len(chunks) == 2
        assert all(isinstance(c, StreamChunk) for c in chunks)

    async def test_complete_accepts_tools(self) -> None:
        provider = StubProvider(ModelConfig(provider="test", model_name="m"))
        result = await provider.complete(
            [UserMessage(content="hi")],
            tools=[{"type": "function", "function": {"name": "f"}}],
        )
        assert result.content == "stub"

    async def test_complete_with_multiple_messages(self) -> None:
        provider = StubProvider(ModelConfig(provider="test", model_name="m"))
        msgs: list[Message] = [
            UserMessage(content="a"),
            UserMessage(content="b"),
        ]
        result = await provider.complete(msgs)
        assert isinstance(result, ModelResponse)

    async def test_stream_is_async_iterator(self) -> None:
        provider = StubProvider(ModelConfig(provider="test", model_name="m"))
        it = provider.stream([UserMessage(content="hi")])
        assert isinstance(it, AsyncIterator)


# ---------------------------------------------------------------------------
# TestModelRegistry
# ---------------------------------------------------------------------------


class TestModelRegistry:
    """Tests for the global model_registry."""

    def test_is_registry_instance(self) -> None:
        assert isinstance(model_registry, Registry)

    def test_name_contains_model(self) -> None:
        assert "model" in model_registry._name


# ---------------------------------------------------------------------------
# TestGetProvider — uses setup/teardown for registry isolation
# ---------------------------------------------------------------------------


class TestGetProvider:
    """Tests for the get_provider() factory."""

    def setup_method(self) -> None:
        model_registry.register("stub", StubProvider)

    def teardown_method(self) -> None:
        model_registry._items.pop("stub", None)

    def test_basic_factory(self) -> None:
        provider = get_provider("stub:my-model")
        assert isinstance(provider, StubProvider)
        assert provider.config.model_name == "my-model"
        assert provider.config.provider == "stub"

    def test_api_key_forwarded(self) -> None:
        provider = get_provider("stub:m", api_key="sk-test")
        assert provider.config.api_key == "sk-test"

    def test_base_url_forwarded(self) -> None:
        provider = get_provider("stub:m", base_url="https://custom.api")
        assert provider.config.base_url == "https://custom.api"

    def test_kwargs_forwarded(self) -> None:
        provider = get_provider("stub:m", max_retries=5, timeout=60.0)
        assert provider.config.max_retries == 5
        assert provider.config.timeout == 60.0

    def test_unregistered_raises_model_error(self) -> None:
        with pytest.raises(ModelError):
            get_provider("nonexistent:m")

    def test_error_includes_available_list(self) -> None:
        with pytest.raises(ModelError, match="Available:"):
            get_provider("nonexistent:m")

    def test_error_has_model_field(self) -> None:
        with pytest.raises(ModelError) as exc_info:
            get_provider("nonexistent:m")
        assert exc_info.value.model == "nonexistent:m"

    async def test_factory_result_is_functional(self) -> None:
        provider = get_provider("stub:m")
        result = await provider.complete([UserMessage(content="test")])
        assert result.content == "stub"


# ---------------------------------------------------------------------------
# TestIncompleteProvider
# ---------------------------------------------------------------------------


class TestIncompleteProvider:
    """Tests that incomplete subclasses cannot be instantiated."""

    def test_missing_complete_raises_type_error(self) -> None:
        class NoComplete(ModelProvider):
            async def stream(self, messages, **kw):  # type: ignore[override]
                yield StreamChunk()  # pragma: no cover

        with pytest.raises(TypeError):
            NoComplete(ModelConfig())  # type: ignore[abstract]

    def test_missing_stream_raises_type_error(self) -> None:
        class NoStream(ModelProvider):
            async def complete(self, messages, **kw):  # type: ignore[override]
                return ModelResponse()  # pragma: no cover

        with pytest.raises(TypeError):
            NoStream(ModelConfig())  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# TestContextWindows — model name key alignment (finding #10)
# ---------------------------------------------------------------------------


class TestContextWindowLookup:
    """Context window entries must be keyed by the model_name (after the colon)."""

    def setup_method(self) -> None:
        model_registry.register("stub", StubProvider)

    def teardown_method(self) -> None:
        model_registry._items.pop("stub", None)

    def _ctx(self, model_name: str) -> int | None:
        from exo.models.context_windows import MODEL_CONTEXT_WINDOWS

        return MODEL_CONTEXT_WINDOWS.get(model_name)

    def test_gpt4o_resolves(self) -> None:
        assert self._ctx("gpt-4o") == 128_000

    def test_gpt4o_mini_resolves(self) -> None:
        assert self._ctx("gpt-4o-mini") == 128_000

    def test_gpt41_resolves(self) -> None:
        assert self._ctx("gpt-4.1") == 1_047_576

    def test_o1_resolves(self) -> None:
        assert self._ctx("o1") == 200_000

    def test_o3_resolves(self) -> None:
        assert self._ctx("o3") == 200_000

    def test_o4_mini_resolves(self) -> None:
        assert self._ctx("o4-mini") == 200_000

    def test_claude_sonnet_4_5_resolves(self) -> None:
        assert self._ctx("claude-sonnet-4-5-20250929") == 200_000

    def test_claude_sonnet_4_6_resolves(self) -> None:
        assert self._ctx("claude-sonnet-4-6") == 200_000

    def test_claude_opus_4_6_resolves(self) -> None:
        assert self._ctx("claude-opus-4-6") == 200_000

    def test_gemini_2_5_pro_resolves(self) -> None:
        assert self._ctx("gemini-2.5-pro") == 1_048_576

    def test_gemini_2_5_flash_resolves(self) -> None:
        assert self._ctx("gemini-2.5-flash") == 1_048_576

    def test_gemini_2_0_flash_resolves(self) -> None:
        assert self._ctx("gemini-2.0-flash") == 1_048_576

    def test_get_provider_wires_context_window(self) -> None:
        """get_provider() must pass the resolved context window to ModelConfig."""
        provider = get_provider("stub:gpt-4o")
        assert provider.config.context_window_tokens == 128_000

    def test_unknown_model_returns_none(self) -> None:
        assert self._ctx("totally-unknown-model-xyz") is None
