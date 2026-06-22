"""Abstract base class for LLM provider implementations.

``ModelProvider`` defines the contract that concrete providers (OpenAI,
Anthropic, etc.) must implement. The ``get_provider()`` factory builds a
provider instance from a ``"provider:model_name"`` string.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from exo.config import ModelConfig, parse_model_string
from exo.models.context_windows import MODEL_CONTEXT_WINDOWS
from exo.models.types import ModelError, ModelResponse, StreamChunk
from exo.registry import Registry, RegistryError
from exo.types import Message

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

model_registry: Registry[type[ModelProvider]] = Registry("model_registry")
"""Global registry mapping provider names to ``ModelProvider`` subclasses."""


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class ModelProvider(ABC):
    """Abstract base class for LLM providers.

    Subclasses implement ``complete()`` for single-shot calls and
    ``stream()`` for incremental token delivery.

    Args:
        config: Provider connection configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ModelResponse:
        """Send a completion request and return the full response.

        Args:
            messages: Conversation history.
            tools: JSON-schema tool definitions for the provider.
            temperature: Sampling temperature override.
            max_tokens: Maximum output tokens override.

        Returns:
            The provider's complete response.
        """

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion request, yielding chunks incrementally.

        Args:
            messages: Conversation history.
            tools: JSON-schema tool definitions for the provider.
            temperature: Sampling temperature override.
            max_tokens: Maximum output tokens override.

        Yields:
            Incremental response chunks.
        """
        # Sentinel yield for async generator typing.
        yield  # type: ignore[misc]  # pragma: no cover


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_provider(
    model: str, *, api_key: str | None = None, base_url: str | None = None, **kwargs: Any
) -> ModelProvider:
    """Build a ``ModelProvider`` from a model string.

    Parses the ``"provider:model_name"`` format, looks up the provider
    class in ``model_registry``, and returns an instance.

    Args:
        model: Model string, e.g. ``"openai:gpt-4o"``.
        api_key: API key for authentication.
        base_url: Custom API base URL.
        **kwargs: Extra fields forwarded to ``ModelConfig``.

    Returns:
        A configured ``ModelProvider`` instance.

    Raises:
        ModelError: If the provider is not registered.
    """
    provider_name, model_name = parse_model_string(model)
    try:
        cls = model_registry.get(provider_name)
    except RegistryError:
        available = model_registry.list_all()
        raise ModelError(
            f"Provider '{provider_name}' not registered. Available: {available}",
            model=model,
            hint=f"Check the model string format 'provider:model'. Registered providers: {available}",
        ) from None  # RegistryError only says "key not found" — ModelError already contains all context.
    ctx_tokens = kwargs.pop("context_window_tokens", MODEL_CONTEXT_WINDOWS.get(model_name))
    config = ModelConfig(
        provider=provider_name,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        context_window_tokens=ctx_tokens,
        **kwargs,
    )
    provider = cls(config)
    logger.debug("Resolved provider '%s' for model '%s'", provider_name, model_name)
    return provider
