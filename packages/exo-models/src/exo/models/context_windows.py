"""Context window sizes for well-known LLM models.

Keys are the ``model_name`` portion of the ``"provider:model_name"`` string
(i.e. the part *after* the colon), which is what ``get_provider()`` passes to
``MODEL_CONTEXT_WINDOWS.get(model_name)``.

Values are the *input* context window in tokens.  Output limits are
intentionally omitted — they vary per-request and are controlled by
``max_tokens``.  When a model is absent the framework defaults to no limit.
"""

MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # ------------------------------------------------------------------
    # OpenAI
    # ------------------------------------------------------------------
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4o-mini-2024-07-18": 128_000,
    "gpt-4o-2024-05-13": 128_000,
    "gpt-4o-2024-08-06": 128_000,
    "gpt-4o-2024-11-20": 128_000,
    "gpt-4.1": 1_047_576,
    "gpt-4.1-mini": 1_047_576,
    "gpt-4.1-nano": 1_047_576,
    "o1": 200_000,
    "o1-mini": 128_000,
    "o1-preview": 128_000,
    "o3": 200_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,
    # ------------------------------------------------------------------
    # Anthropic
    # ------------------------------------------------------------------
    # claude-3.5 family
    "claude-3-5-sonnet-20240620": 200_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-5-haiku-20241022": 200_000,
    # claude-3 family
    "claude-3-opus-20240229": 200_000,
    "claude-3-sonnet-20240229": 200_000,
    "claude-3-haiku-20240307": 200_000,
    # claude-4 / claude-4.x family (stable names without date suffix)
    "claude-sonnet-4-5": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-opus-4-6": 200_000,
    # claude-4 with date suffixes used in tests and production configs
    "claude-sonnet-4-5-20250929": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
    # ------------------------------------------------------------------
    # Google — Gemini API and Vertex AI (same model_name for both)
    # ------------------------------------------------------------------
    "gemini-2.5-pro": 1_048_576,
    "gemini-2.5-pro-preview-05-06": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.5-flash-preview-04-17": 1_048_576,
    "gemini-2.0-flash": 1_048_576,
    "gemini-2.0-flash-lite": 1_048_576,
    "gemini-2.0-flash-thinking-exp": 32_767,
    "gemini-1.5-pro": 2_097_152,
    "gemini-1.5-flash": 1_048_576,
    "gemini-1.5-flash-8b": 1_048_576,
    # ------------------------------------------------------------------
    # Misc / community
    # ------------------------------------------------------------------
    "zai-org/glm-5-maas": 128_000,
}
