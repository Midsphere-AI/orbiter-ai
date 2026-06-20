"""Exo Models: LLM provider abstractions."""

# Import providers to trigger auto-registration with model_registry.
from exo.models.anthropic import AnthropicProvider
from exo.models.context_windows import MODEL_CONTEXT_WINDOWS
from exo.models.gemini import GeminiProvider
from exo.models.media_tools import dalle_generate_image, imagen_generate_image, veo_generate_video
from exo.models.openai import OpenAIProvider
from exo.models.provider import ModelProvider, get_provider, model_registry
from exo.models.types import (
    FinishReason,
    ModelError,
    ModelResponse,
    StreamChunk,
    ToolCallDelta,
)
from exo.models.vertex import VertexProvider

__version__: str = "0.1.0"

__all__: list[str] = [
    "MODEL_CONTEXT_WINDOWS",
    "AnthropicProvider",
    "FinishReason",
    "GeminiProvider",
    "ModelError",
    "ModelProvider",
    "ModelResponse",
    "OpenAIProvider",
    "StreamChunk",
    "ToolCallDelta",
    "VertexProvider",
    "dalle_generate_image",
    "get_provider",
    "imagen_generate_image",
    "model_registry",
    "veo_generate_video",
]
