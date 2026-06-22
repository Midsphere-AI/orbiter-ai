"""Integration tests for the exo.models public API.

All imports use the public surface (``exo.models``), not internal modules.
These tests validate that the package __init__.py re-exports are correct,
auto-registration works, and get_provider() produces functional providers.
"""

from __future__ import annotations

import pytest

import exo.models as models  # pyright: ignore[reportMissingImports]
import exo.models.types as model_types  # pyright: ignore[reportMissingImports]
from exo.models import (  # pyright: ignore[reportMissingImports]
    AnthropicProvider,
    GeminiProvider,
    ModelError,
    ModelProvider,
    OpenAIProvider,
    VertexProvider,
    get_provider,
    model_registry,
)

EXPECTED_ALL = [
    "AnthropicProvider",
    "EmbeddingError",
    "Embeddings",
    "FinishReason",
    "GeminiProvider",
    "HTTPEmbeddings",
    "InMemoryVectorStore",
    "MODEL_CONTEXT_WINDOWS",
    "ModelError",
    "ModelProvider",
    "ModelResponse",
    "OpenAIEmbeddings",
    "OpenAIProvider",
    "StreamChunk",
    "ToolCallDelta",
    "VectorSearchResult",
    "VectorStoreBase",
    "VertexEmbeddings",
    "VertexProvider",
    "cosine_similarity",
    "dalle_generate_image",
    "get_provider",
    "imagen_generate_image",
    "model_registry",
    "veo_generate_video",
]

INTERNAL_HELPERS = [
    "_map_finish_reason",
    "_to_openai_messages",
    "_parse_response",
    "_parse_stream_chunk",
    "_map_stop_reason",
    "_build_messages",
    "_convert_tools",
]


class TestPublicAPI:
    """Verify the __all__ exports and re-export identity."""

    def test_all_exports_defined(self) -> None:
        assert sorted(models.__all__) == EXPECTED_ALL

    def test_all_exports_importable(self) -> None:
        for name in models.__all__:
            assert hasattr(models, name), f"{name!r} not importable from exo.models"

    def test_no_private_leakage(self) -> None:
        for name in INTERNAL_HELPERS:
            assert not hasattr(models, name), f"internal {name!r} leaked onto exo.models"

    def test_type_identity(self) -> None:
        assert models.ModelResponse is model_types.ModelResponse
        assert models.ModelError is model_types.ModelError
        assert models.StreamChunk is model_types.StreamChunk
        assert models.ToolCallDelta is model_types.ToolCallDelta


class TestAutoRegistration:
    """Verify providers are auto-registered on import."""

    def test_openai_registered(self) -> None:
        assert "openai" in model_registry
        assert model_registry.get("openai") is OpenAIProvider

    def test_anthropic_registered(self) -> None:
        assert "anthropic" in model_registry
        assert model_registry.get("anthropic") is AnthropicProvider

    def test_gemini_registered(self) -> None:
        assert "gemini" in model_registry
        assert model_registry.get("gemini") is GeminiProvider

    def test_vertex_registered(self) -> None:
        assert "vertex" in model_registry
        assert model_registry.get("vertex") is VertexProvider

    def test_registry_lists_all(self) -> None:
        all_providers = model_registry.list_all()
        assert "openai" in all_providers
        assert "anthropic" in all_providers
        assert "gemini" in all_providers
        assert "vertex" in all_providers


class TestGetProviderEndToEnd:
    """End-to-end tests for get_provider() with real providers."""

    def test_openai_model_string(self) -> None:
        provider = get_provider("openai:gpt-4o", api_key="sk-test")
        assert isinstance(provider, OpenAIProvider)
        assert provider.config.provider == "openai"
        assert provider.config.model_name == "gpt-4o"
        assert provider.config.api_key == "sk-test"

    def test_anthropic_model_string(self) -> None:
        provider = get_provider("anthropic:claude-sonnet-4-5-20250929", api_key="sk-ant")
        assert isinstance(provider, AnthropicProvider)
        assert provider.config.provider == "anthropic"
        assert provider.config.model_name == "claude-sonnet-4-5-20250929"
        assert provider.config.api_key == "sk-ant"

    def test_default_provider_is_openai(self) -> None:
        provider = get_provider("gpt-4o-mini", api_key="sk-test")
        assert isinstance(provider, OpenAIProvider)
        assert provider.config.provider == "openai"
        assert provider.config.model_name == "gpt-4o-mini"

    def test_unknown_provider_error(self) -> None:
        with pytest.raises(ModelError, match="Available:"):
            get_provider("unknown:model")


class TestCrossProviderConsistency:
    """Verify both providers satisfy the same contract."""

    def test_both_are_model_providers(self) -> None:
        openai = get_provider("openai:gpt-4o", api_key="sk-test")
        anthropic = get_provider("anthropic:claude-sonnet-4-5-20250929", api_key="sk-ant")
        assert isinstance(openai, ModelProvider)
        assert isinstance(anthropic, ModelProvider)

    def test_both_have_required_methods(self) -> None:
        openai = get_provider("openai:gpt-4o", api_key="sk-test")
        anthropic = get_provider("anthropic:claude-sonnet-4-5-20250929", api_key="sk-ant")
        for provider in (openai, anthropic):
            assert callable(getattr(provider, "complete", None))
            assert callable(getattr(provider, "stream", None))


class TestModelContextWindows:
    """Tests for MODEL_CONTEXT_WINDOWS registry and get_provider() integration."""

    def test_registry_is_dict(self) -> None:
        from exo.models import MODEL_CONTEXT_WINDOWS  # pyright: ignore[reportMissingImports]

        assert isinstance(MODEL_CONTEXT_WINDOWS, dict)

    def test_well_known_models_present(self) -> None:
        from exo.models import MODEL_CONTEXT_WINDOWS  # pyright: ignore[reportMissingImports]

        for model in (
            "gpt-4o",
            "gpt-4o-mini",
            "o1",
            "claude-sonnet-4-6",
            "claude-opus-4-6",
            "claude-haiku-4-5-20251001",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
        ):
            assert model in MODEL_CONTEXT_WINDOWS, f"{model!r} missing from registry"

    def test_gpt4o_context_window(self) -> None:
        from exo.models import MODEL_CONTEXT_WINDOWS  # pyright: ignore[reportMissingImports]

        assert MODEL_CONTEXT_WINDOWS["gpt-4o"] == 128000

    def test_claude_context_window(self) -> None:
        from exo.models import MODEL_CONTEXT_WINDOWS  # pyright: ignore[reportMissingImports]

        assert MODEL_CONTEXT_WINDOWS["claude-sonnet-4-6"] == 200000
        assert MODEL_CONTEXT_WINDOWS["claude-opus-4-6"] == 200000
        assert MODEL_CONTEXT_WINDOWS["claude-haiku-4-5-20251001"] == 200000

    def test_gemini_context_window(self) -> None:
        from exo.models import MODEL_CONTEXT_WINDOWS  # pyright: ignore[reportMissingImports]

        assert MODEL_CONTEXT_WINDOWS["gemini-2.0-flash"] == 1048576
        assert MODEL_CONTEXT_WINDOWS["gemini-1.5-pro"] == 2097152

    def test_get_provider_populates_context_window_tokens_known_model(self) -> None:
        provider = get_provider("openai:gpt-4o", api_key="sk-test")
        assert provider.config.context_window_tokens == 128000

    def test_get_provider_populates_context_window_tokens_anthropic(self) -> None:
        provider = get_provider("anthropic:claude-sonnet-4-6", api_key="sk-ant")
        assert provider.config.context_window_tokens == 200000

    def test_get_provider_context_window_none_for_unknown_model(self) -> None:
        provider = get_provider("openai:gpt-unknown-xyz", api_key="sk-test")
        assert provider.config.context_window_tokens is None

    def test_get_provider_explicit_override(self) -> None:
        provider = get_provider("openai:gpt-4o", api_key="sk-test", context_window_tokens=999)
        assert provider.config.context_window_tokens == 999


class TestGoogleProviderMerge:
    """After the Gemini/Vertex merge both prefixes must resolve to the same class."""

    def test_gemini_and_vertex_are_same_class(self) -> None:
        from exo.models.gemini import GoogleProvider  # pyright: ignore[reportMissingImports]

        assert model_registry.get("gemini") is GoogleProvider
        assert model_registry.get("vertex") is GoogleProvider
        assert model_registry.get("gemini") is model_registry.get("vertex")

    def test_gemini_prefix_resolves(self) -> None:
        from unittest.mock import patch

        from exo.models.gemini import GoogleProvider  # pyright: ignore[reportMissingImports]

        with patch("exo.models.gemini.genai"):
            provider = get_provider("gemini:gemini-2.0-flash", api_key="key")
        assert isinstance(provider, GoogleProvider)
        assert provider.config.provider == "gemini"
        assert provider.config.model_name == "gemini-2.0-flash"

    def test_vertex_prefix_resolves(self) -> None:
        from unittest.mock import patch

        from exo.models.gemini import GoogleProvider  # pyright: ignore[reportMissingImports]

        with patch("exo.models.gemini.genai"):
            provider = get_provider(
                "vertex:gemini-2.0-flash",
                google_project="my-project",
            )
        assert isinstance(provider, GoogleProvider)
        assert provider.config.provider == "vertex"
        assert provider.config.model_name == "gemini-2.0-flash"

    def test_vertex_alias_backward_compat(self) -> None:
        """VertexProvider is GeminiProvider is GoogleProvider (all the same class)."""
        from exo.models.gemini import (  # pyright: ignore[reportMissingImports]
            GeminiProvider,
            GoogleProvider,
        )
        from exo.models.vertex import VertexProvider  # pyright: ignore[reportMissingImports]

        assert GeminiProvider is GoogleProvider
        assert VertexProvider is GoogleProvider


class TestAnthropicUseCacheOpt:
    """use_cache=False disables prompt-cache breakpoints."""

    def test_cache_on_by_default(self) -> None:
        from exo.config import ModelConfig  # pyright: ignore[reportMissingImports]
        from exo.models.anthropic import AnthropicProvider  # pyright: ignore[reportMissingImports]
        from exo.types import UserMessage  # pyright: ignore[reportMissingImports]

        provider = AnthropicProvider(
            ModelConfig(provider="anthropic", model_name="claude-sonnet-4-6", api_key="key")
        )
        kwargs = provider._build_kwargs([UserMessage(content="hi")])
        # system is absent so the cache breakpoint manifests on user messages
        user_msg = kwargs["messages"][0]
        # Content is converted to a list with cache_control when caching is on
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][-1].get("cache_control") == {"type": "ephemeral"}

    def test_cache_off_skips_breakpoints(self) -> None:
        from exo.config import ModelConfig  # pyright: ignore[reportMissingImports]
        from exo.models.anthropic import AnthropicProvider  # pyright: ignore[reportMissingImports]
        from exo.types import UserMessage  # pyright: ignore[reportMissingImports]

        provider = AnthropicProvider(
            ModelConfig(
                provider="anthropic", model_name="claude-sonnet-4-6", api_key="key", use_cache=False
            )
        )
        kwargs = provider._build_kwargs([UserMessage(content="hi")])
        user_msg = kwargs["messages"][0]
        # With use_cache=False content stays as a plain string, no cache_control injected
        assert user_msg["content"] == "hi"


class TestApiKeyErrors:
    """Missing API keys raise ModelError at construction time, not later."""

    def test_anthropic_no_key_raises(self) -> None:
        import os
        from unittest.mock import patch

        from exo.config import ModelConfig  # pyright: ignore[reportMissingImports]
        from exo.models.anthropic import AnthropicProvider  # pyright: ignore[reportMissingImports]
        from exo.models.types import ModelError  # pyright: ignore[reportMissingImports]

        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with (
            patch.dict(os.environ, env, clear=True),
            pytest.raises(ModelError, match="ANTHROPIC_API_KEY"),
        ):
            AnthropicProvider(ModelConfig(provider="anthropic", model_name="claude-sonnet-4-6"))

    def test_gemini_no_key_raises(self) -> None:
        import os
        from unittest.mock import patch

        from exo.config import ModelConfig  # pyright: ignore[reportMissingImports]
        from exo.models.gemini import GoogleProvider  # pyright: ignore[reportMissingImports]
        from exo.models.types import ModelError  # pyright: ignore[reportMissingImports]

        env = {k: v for k, v in os.environ.items() if k not in ("GOOGLE_API_KEY", "GEMINI_API_KEY")}
        with (
            patch.dict(os.environ, env, clear=True),
            pytest.raises(ModelError, match="GOOGLE_API_KEY"),
        ):
            GoogleProvider(ModelConfig(provider="gemini", model_name="gemini-2.0-flash"))
