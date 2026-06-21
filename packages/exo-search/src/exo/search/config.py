"""Configuration for the Exo Search search engine."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from exo.observability.logging import get_logger  # pyright: ignore[reportMissingImports]

from .types import SearchSource

_log = get_logger(__name__)

_VALID_SOURCES: frozenset[str] = frozenset(s.value for s in SearchSource)


@dataclass
class SearchConfig:
    """Exo Search configuration with environment variable overrides."""

    searxng_url: str = ""
    model: str = ""
    fast_model: str = ""
    embedding_model: str = "gemini-embedding-2-preview"
    max_results: int = 10
    searxng_timeout: int = 15
    research_mode: str = ""  # speed, balanced, quality
    system_instructions: str = ""  # user custom instructions for writer
    sources: list[str] = field(default_factory=lambda: ["web"])

    # Performance tuning
    max_iterations: int | None = None  # override per-mode default (speed=2, balanced=6, quality=25)
    use_reasoning_preamble: bool | None = None  # None=auto-detect thinking models, True/False=force
    max_writer_words: int | None = None  # override quality mode 2000-word target
    max_writer_sources: int = 15  # cap sources passed to writer for citation accuracy
    jina_reader_url: str = ""  # Jina Reader for full-page content extraction
    jina_api_key: str = ""  # Jina Cloud API key (enables r.jina.ai / s.jina.ai)
    serper_api_key: str = ""  # Serper API key for Google Search

    # Deep research settings
    max_deep_research_steps: int = 7  # max steps in sequential research plan
    deep_research_enrich_per_step: int = 5  # pages to enrich per research step
    max_content_chars: int = 10_000  # max chars per page (increased for quality/deep)

    # Context window override — set for custom/fine-tuned models
    context_window_tokens: int | None = None  # None=auto-detect from model name

    # Verification settings
    llm_verification: bool = False  # enable LLM-based claim verification (quality/deep)
    llm_verify_source_chars: int = 4000  # max source chars sent to LLM verifier

    # Writing settings
    claim_first_writing: bool = True  # use claim-first writing (quality/deep)

    # Revision loop settings
    max_revision_rounds: int = 2  # max write-verify-revise rounds
    revision_threshold: float = 0.3  # revise if removed/total > this

    def __post_init__(self) -> None:
        if not self.searxng_url:
            self.searxng_url = os.environ.get("SEARXNG_URL", "http://localhost:8888")
        if not self.model:
            self.model = os.environ.get("EXO_SEARCH_MODEL", "openai:gpt-4o")
        if not self.fast_model:
            self.fast_model = os.environ.get("EXO_SEARCH_FAST_MODEL", "openai:gpt-4o-mini")
        if not self.embedding_model:
            self.embedding_model = os.environ.get(
                "EXO_SEARCH_EMBEDDING_MODEL", "text-embedding-3-small"
            )
        if not self.research_mode:
            self.research_mode = os.environ.get("EXO_SEARCH_RESEARCH_MODE", "balanced")
        if not self.jina_reader_url:
            self.jina_reader_url = os.environ.get("JINA_READER_URL", "http://127.0.0.1:3000")
        if not self.jina_api_key:
            self.jina_api_key = os.environ.get("JINA_API_KEY", "")
        if not self.serper_api_key:
            self.serper_api_key = os.environ.get("SERPER_API_KEY", "")

        # Validate sources — accept strings or SearchSource enum values, but fail
        # loudly on unknown entries so typos (e.g. "web,academic" as one string,
        # or "Discussions" with wrong case) don't silently yield zero results.
        normalized: list[str] = []
        bad: list[str] = []
        for s in self.sources:
            low = str(s).lower()
            if low in _VALID_SOURCES:
                normalized.append(low)
            else:
                bad.append(str(s))
        if bad:
            valid_list = ", ".join(sorted(_VALID_SOURCES))
            raise ValueError(
                f"Unknown search source(s): {bad!r}. "
                f"Valid sources are: {valid_list}. "
                f"Example: SearchConfig(sources=['web', 'academic'])"
            )
        self.sources = normalized

        _log.debug(
            "config model=%s fast=%s search=%s",
            self.model,
            self.fast_model,
            self.searxng_url,
        )


# ---------------------------------------------------------------------------
# Known model context windows (tokens) — used when exo.models lookup fails
# ---------------------------------------------------------------------------

_FALLBACK_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
    "o1": 200_000,
    "o1-mini": 128_000,
    "o1-pro": 200_000,
    "o3": 200_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-3-5-haiku": 200_000,
    "claude-3-opus": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-opus-4": 200_000,
    "gemini-2.0-flash": 1_000_000,
    "gemini-2.5-flash": 1_000_000,
    "gemini-2.5-pro": 1_000_000,
    "gemini-2.0-pro": 1_000_000,
    "gemini-1.5-pro": 2_000_000,
    "gemini-1.5-flash": 1_000_000,
}

_DEFAULT_CONTEXT_TOKENS = 128_000


def _resolve_context_window(model: str, override: int | None = None) -> int:
    """Resolve the context window size in tokens for a model string.

    Tries ``exo.models`` lookup first, then falls back to the built-in table.
    """
    if override is not None:
        return override

    # Parse provider:model format
    model_name = model.split(":")[-1] if ":" in model else model

    # Try exo.models import (may not be available)
    try:
        from exo.models import MODEL_CONTEXT_WINDOWS  # type: ignore[import-not-found]

        ctx = MODEL_CONTEXT_WINDOWS.get(model_name)
        if ctx:
            return ctx
    except (ImportError, AttributeError):
        pass

    # Fallback table — try exact match, then prefix match
    if model_name in _FALLBACK_CONTEXT_WINDOWS:
        return _FALLBACK_CONTEXT_WINDOWS[model_name]
    for key, val in _FALLBACK_CONTEXT_WINDOWS.items():
        if model_name.startswith(key):
            return val

    return _DEFAULT_CONTEXT_TOKENS


def compute_context_budget(
    model: str,
    mode: str,
    chat_history_chars: int = 0,
    narrative_chars: int = 0,
    context_window_override: int | None = None,
) -> tuple[int, int, int]:
    """Compute how much source context the writer can receive.

    Returns:
        ``(max_sources, max_chars_per_source, enrich_cap)``
        - ``max_sources``: how many sources to pass to the writer
        - ``max_chars_per_source``: character budget per source
        - ``enrich_cap``: how many sources should be enriched (0 for speed)
    """
    from exo.token_counter import TokenCounter

    ctx_tokens = _resolve_context_window(model, context_window_override)
    counter = TokenCounter(model)
    ctx_chars = counter.tokens_to_chars(ctx_tokens)

    # Reserve space for response and prompt overhead
    response_reserve = counter.tokens_to_chars(4096)  # 4096 tokens for response
    prompt_overhead = 8_000  # system prompt, instructions, formatting
    history_chars = max(chat_history_chars, 0)
    narr_chars = max(narrative_chars, 0)

    available = ctx_chars - response_reserve - prompt_overhead - history_chars - narr_chars
    available = max(available, 20_000)  # floor of 20K available chars

    # Mode-specific enrichment ratios
    if mode == "speed":
        # Speed: snippets only, no enrichment
        enrich_ratio = 0.0
        chars_per_source_target = 500  # snippet size
    elif mode == "balanced":
        enrich_ratio = 0.8
        chars_per_source_target = min(available // 8, 15_000)
    else:
        # quality / deep
        enrich_ratio = 0.9
        chars_per_source_target = min(available // 6, 50_000)

    # Compute max_sources from available budget
    max_sources = available // chars_per_source_target if chars_per_source_target > 0 else 10

    # Clamp bounds
    max_sources = max(3, min(max_sources, 30))
    chars_per_source = max(2_000, min(chars_per_source_target, 50_000))

    # Enrich cap: proportion of max_sources
    enrich_cap = 0 if mode == "speed" else max(1, int(max_sources * enrich_ratio))

    _log.debug(
        "context_budget model=%s mode=%s ctx_tokens=%d available=%d "
        "-> max_sources=%d chars_per_source=%d enrich_cap=%d",
        model,
        mode,
        ctx_tokens,
        available,
        max_sources,
        chars_per_source,
        enrich_cap,
    )

    return max_sources, chars_per_source, enrich_cap
