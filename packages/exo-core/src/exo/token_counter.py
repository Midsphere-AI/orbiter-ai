"""Provider-aware token counting via tiktoken.

Uses tiktoken for accurate token counts with provider-aware encoding
selection.  Falls back to a character-based heuristic when tiktoken is
not installed.

Example::

    from exo.token_counter import TokenCounter, count_tokens

    # Class-based (reusable, caches encoding)
    counter = TokenCounter("anthropic:claude-sonnet-4-6")
    n = counter.count("Hello, world!")

    # Quick one-liner
    n = count_tokens("Hello, world!", model="openai:gpt-4o")
"""

from __future__ import annotations

from typing import Any

from exo.config import parse_model_string

# ---------------------------------------------------------------------------
# Encoding resolution tables
# ---------------------------------------------------------------------------

# OpenAI model prefix → tiktoken encoding.
_OPENAI_ENCODINGS: dict[str, str] = {
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "o1": "o200k_base",
    "o1-mini": "o200k_base",
    "o1-preview": "o200k_base",
    "o3": "o200k_base",
    "o3-mini": "o200k_base",
    "o4-mini": "o200k_base",
    "gpt-4": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4-turbo-preview": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
}

# Provider → default tiktoken encoding (best available approximation).
_PROVIDER_ENCODINGS: dict[str, str] = {
    "openai": "o200k_base",
    "anthropic": "cl100k_base",
    "gemini": "o200k_base",
    "vertex": "o200k_base",
}

_DEFAULT_ENCODING = "o200k_base"

# Per-message token overhead for chat-format messages (role + delimiters).
_MESSAGE_OVERHEAD = 3

# Extra tokens appended after all messages (assistant reply priming).
_REPLY_OVERHEAD = 3

# Empirical chars-per-token ratios by encoding.
_CHARS_PER_TOKEN_BY_ENCODING: dict[str, float] = {
    "cl100k_base": 3.7,
    "o200k_base": 4.2,
}
_CHARS_PER_TOKEN_FALLBACK = 4.0


# ---------------------------------------------------------------------------
# Encoding resolution
# ---------------------------------------------------------------------------


def _resolve_encoding_name(model: str) -> tuple[str, str, str]:
    """Resolve a model string to (provider, model_name, encoding_name).

    Accepts ``"provider:model"`` or bare ``"model"`` (defaults to openai).
    """
    provider, model_name = parse_model_string(model)

    if provider == "openai":
        # Exact match first, then longest-prefix match.
        if model_name in _OPENAI_ENCODINGS:
            return provider, model_name, _OPENAI_ENCODINGS[model_name]
        for prefix, enc in sorted(
            _OPENAI_ENCODINGS.items(), key=lambda kv: len(kv[0]), reverse=True
        ):
            if model_name.startswith(prefix):
                return provider, model_name, enc
        return provider, model_name, _PROVIDER_ENCODINGS.get(provider, _DEFAULT_ENCODING)

    return provider, model_name, _PROVIDER_ENCODINGS.get(provider, _DEFAULT_ENCODING)


# ---------------------------------------------------------------------------
# TokenCounter
# ---------------------------------------------------------------------------


class TokenCounter:
    """Provider-aware token counter with tiktoken backend.

    Accepts model strings in ``"provider:model"`` format and selects the
    best tiktoken encoding for that provider/model combination.

    Parameters
    ----------
    model:
        Model string, e.g. ``"openai:gpt-4o"``, ``"anthropic:claude-sonnet-4-6"``,
        or bare ``"gpt-4o"`` (defaults to openai provider).

    Example::

        counter = TokenCounter("anthropic:claude-sonnet-4-6")
        n = counter.count("Hello, world!")
        total = counter.count_messages([
            {"role": "user", "content": "Hi"},
        ])
    """

    __slots__ = ("_chars_per_token", "_encoding", "_encoding_name", "_has_tiktoken",
                 "_model", "_provider")

    def __init__(self, model: str = "openai:gpt-4o") -> None:
        provider, model_name, encoding_name = _resolve_encoding_name(model)
        self._provider = provider
        self._model = model_name
        self._encoding_name = encoding_name
        self._encoding: Any = None
        self._has_tiktoken = False
        self._chars_per_token = _CHARS_PER_TOKEN_BY_ENCODING.get(
            encoding_name, _CHARS_PER_TOKEN_FALLBACK
        )

        try:
            import tiktoken

            self._encoding = tiktoken.get_encoding(encoding_name)
            self._has_tiktoken = True
        except ImportError:
            pass

    # ── properties ────────────────────────────────────────────────────

    @property
    def model(self) -> str:
        """The model name (without provider prefix)."""
        return self._model

    @property
    def provider(self) -> str:
        """The resolved provider name."""
        return self._provider

    @property
    def encoding_name(self) -> str:
        """The tiktoken encoding used for this model."""
        return self._encoding_name

    @property
    def has_tiktoken(self) -> bool:
        """Whether tiktoken is available for exact counting."""
        return self._has_tiktoken

    # ── counting ──────────────────────────────────────────────────────

    def count(self, text: str) -> int:
        """Count tokens in a plain text string.

        Returns exact count when tiktoken is installed, otherwise a
        character-based estimate using the encoding-specific ratio.
        """
        if self._has_tiktoken:
            return len(self._encoding.encode(text))
        return max(1, round(len(text) / self._chars_per_token))

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        """Count tokens for a list of chat messages.

        Each message is expected to have ``role`` and ``content`` keys.
        Adds per-message overhead (3 tokens) and reply priming overhead
        (3 tokens) following the OpenAI token counting convention.
        """
        total = 0
        for msg in messages:
            total += _MESSAGE_OVERHEAD
            for value in msg.values():
                if isinstance(value, str):
                    total += self.count(value)
        total += _REPLY_OVERHEAD
        return total

    # ── conversion ────────────────────────────────────────────────────

    def tokens_to_chars(self, tokens: int) -> int:
        """Convert a token count to estimated character count.

        Uses an encoding-specific chars-per-token ratio that is more
        accurate than a generic ``* 4`` multiplier.
        """
        return round(tokens * self._chars_per_token)

    def chars_to_tokens(self, chars: int) -> int:
        """Convert a character count to estimated token count.

        Inverse of :meth:`tokens_to_chars`.
        """
        return max(1, round(chars / self._chars_per_token))

    def __repr__(self) -> str:
        backend = "tiktoken" if self._has_tiktoken else "fallback"
        return (
            f"TokenCounter(provider={self._provider!r}, "
            f"model={self._model!r}, "
            f"encoding={self._encoding_name!r}, "
            f"backend={backend!r})"
        )


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_counter_cache: dict[str, TokenCounter] = {}
_COUNTER_CACHE_MAX = 64  # bound the cache to avoid unbounded growth


def count_tokens(text: str, model: str = "openai:gpt-4o") -> int:
    """Count tokens in *text* for the given model.

    Caches :class:`TokenCounter` instances per model string so the
    tiktoken encoding is loaded only once.  The cache is capped at
    ``_COUNTER_CACHE_MAX`` entries; when full, the oldest entry is
    evicted (FIFO).

    Parameters
    ----------
    text:
        The text to count tokens for.
    model:
        Model string, e.g. ``"openai:gpt-4o"`` or ``"anthropic:claude-sonnet-4-6"``.
    """
    counter = _counter_cache.get(model)
    if counter is None:
        counter = TokenCounter(model)
        if len(_counter_cache) >= _COUNTER_CACHE_MAX:
            # Evict the oldest (first inserted) entry
            _counter_cache.pop(next(iter(_counter_cache)))
        _counter_cache[model] = counter
    return counter.count(text)
