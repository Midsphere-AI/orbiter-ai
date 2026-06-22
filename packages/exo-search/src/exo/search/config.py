"""Configuration for the Exo Search search engine."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from exo.observability.logging import get_logger  # pyright: ignore[reportMissingImports]

from .errors import SearchConfigError
from .types import SearchSource

_log = get_logger(__name__)

_VALID_SOURCES: frozenset[str] = frozenset(s.value for s in SearchSource)


@dataclass(frozen=True)
class SearchModels:
    """Model selection namespace for SearchConfig.

    Args:
        model: Main LLM model string (provider:model).
        fast_model: Faster/cheaper model for lightweight tasks.
        embedding_model: Embedding model for semantic operations.
        context_window_tokens: Override context window size in tokens (None=auto-detect).
    """

    model: str = ""
    fast_model: str = ""
    embedding_model: str = "gemini-embedding-2-preview"
    context_window_tokens: int | None = None


@dataclass(frozen=True)
class SearchSources:
    """Source/fetching namespace for SearchConfig.

    Args:
        searxng_url: SearXNG instance URL.
        searxng_timeout: Request timeout in seconds.
        jina_reader_url: Jina Reader URL for full-page content extraction.
        jina_api_key: Jina Cloud API key (enables r.jina.ai / s.jina.ai).
        serper_api_key: Serper API key for Google Search.
        sources: List of search source names (e.g. ["web", "academic"]).
        max_results: Maximum number of search results to fetch.
    """

    searxng_url: str = ""
    searxng_timeout: int = 15
    jina_reader_url: str = ""
    jina_api_key: str = ""
    serper_api_key: str = ""
    sources: list[str] = field(default_factory=lambda: ["web"])
    max_results: int = 10


@dataclass(frozen=True)
class ResearchConfig:
    """Deep research namespace for SearchConfig.

    Args:
        research_mode: Quality tier — "speed", "balanced", or "quality".
        max_iterations: Override per-mode default iteration count.
        max_deep_research_steps: Maximum steps in sequential research plan.
        deep_research_enrich_per_step: Pages to enrich per research step.
        max_content_chars: Maximum characters per fetched page.
    """

    research_mode: str = ""
    max_iterations: int | None = None
    max_deep_research_steps: int = 7
    deep_research_enrich_per_step: int = 5
    max_content_chars: int = 10_000


@dataclass(frozen=True)
class WriterConfig:
    """Writer namespace for SearchConfig.

    Args:
        system_instructions: Custom instructions for the writer agent.
        max_writer_words: Override quality mode word target (None=mode default).
        max_writer_sources: Cap on sources passed to writer for citation accuracy.
        claim_first_writing: Use claim-first writing style (quality/deep).
        use_reasoning_preamble: None=auto-detect thinking models, True/False=force.
    """

    system_instructions: str = ""
    max_writer_words: int | None = None
    max_writer_sources: int = 15
    claim_first_writing: bool = True
    use_reasoning_preamble: bool | None = None


@dataclass(frozen=True)
class VerificationConfig:
    """Verification namespace for SearchConfig.

    Args:
        llm_verification: Enable LLM-based claim verification (quality/deep).
        llm_verify_source_chars: Max source chars sent to LLM verifier.
    """

    llm_verification: bool = False
    llm_verify_source_chars: int = 4000


@dataclass(frozen=True)
class RevisionConfig:
    """Revision loop namespace for SearchConfig.

    Args:
        max_revision_rounds: Maximum write-verify-revise rounds.
        revision_threshold: Revise if removed/total claim ratio exceeds this.
    """

    max_revision_rounds: int = 2
    revision_threshold: float = 0.3


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

    # Grouped namespace configs (Tier-3 DX) — exploded in __post_init__
    models_config: SearchModels | None = field(default=None, repr=False)
    sources_config: SearchSources | None = field(default=None, repr=False)
    research: ResearchConfig | None = field(default=None, repr=False)
    writer: WriterConfig | None = field(default=None, repr=False)
    verification: VerificationConfig | None = field(default=None, repr=False)
    revision: RevisionConfig | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # --- Explode grouped namespace configs into flat fields ---
        if self.models_config is not None:
            cfg = self.models_config
            conflicts = []
            if self.model:
                conflicts.append("model")
            if self.fast_model:
                conflicts.append("fast_model")
            if self.embedding_model != "gemini-embedding-2-preview":
                conflicts.append("embedding_model")
            if self.context_window_tokens is not None:
                conflicts.append("context_window_tokens")
            if conflicts:
                raise SearchConfigError(
                    f"Cannot combine models_config=SearchModels(...) with flat fields: {conflicts}.",
                    context={"conflicting_fields": conflicts},
                    hint="Pass either the grouped models_config=SearchModels(...) or the flat kwargs, not both.",
                )
            self.model = cfg.model
            self.fast_model = cfg.fast_model
            self.embedding_model = cfg.embedding_model
            self.context_window_tokens = cfg.context_window_tokens

        if self.sources_config is not None:
            cfg = self.sources_config
            conflicts = []
            if self.searxng_url:
                conflicts.append("searxng_url")
            if self.searxng_timeout != 15:
                conflicts.append("searxng_timeout")
            if self.jina_reader_url:
                conflicts.append("jina_reader_url")
            if self.jina_api_key:
                conflicts.append("jina_api_key")
            if self.serper_api_key:
                conflicts.append("serper_api_key")
            if self.sources != ["web"]:
                conflicts.append("sources")
            if self.max_results != 10:
                conflicts.append("max_results")
            if conflicts:
                raise SearchConfigError(
                    f"Cannot combine sources_config=SearchSources(...) with flat fields: {conflicts}.",
                    context={"conflicting_fields": conflicts},
                    hint="Pass either the grouped sources_config=SearchSources(...) or the flat kwargs, not both.",
                )
            self.searxng_url = cfg.searxng_url
            self.searxng_timeout = cfg.searxng_timeout
            self.jina_reader_url = cfg.jina_reader_url
            self.jina_api_key = cfg.jina_api_key
            self.serper_api_key = cfg.serper_api_key
            self.sources = list(cfg.sources)
            self.max_results = cfg.max_results

        if self.research is not None:
            cfg = self.research
            conflicts = []
            if self.research_mode:
                conflicts.append("research_mode")
            if self.max_iterations is not None:
                conflicts.append("max_iterations")
            if self.max_deep_research_steps != 7:
                conflicts.append("max_deep_research_steps")
            if self.deep_research_enrich_per_step != 5:
                conflicts.append("deep_research_enrich_per_step")
            if self.max_content_chars != 10_000:
                conflicts.append("max_content_chars")
            if conflicts:
                raise SearchConfigError(
                    f"Cannot combine research=ResearchConfig(...) with flat fields: {conflicts}.",
                    context={"conflicting_fields": conflicts},
                    hint="Pass either the grouped research=ResearchConfig(...) or the flat kwargs, not both.",
                )
            self.research_mode = cfg.research_mode
            self.max_iterations = cfg.max_iterations
            self.max_deep_research_steps = cfg.max_deep_research_steps
            self.deep_research_enrich_per_step = cfg.deep_research_enrich_per_step
            self.max_content_chars = cfg.max_content_chars

        if self.writer is not None:
            cfg = self.writer
            conflicts = []
            if self.system_instructions:
                conflicts.append("system_instructions")
            if self.max_writer_words is not None:
                conflicts.append("max_writer_words")
            if self.max_writer_sources != 15:
                conflicts.append("max_writer_sources")
            if not self.claim_first_writing:
                conflicts.append("claim_first_writing")
            if self.use_reasoning_preamble is not None:
                conflicts.append("use_reasoning_preamble")
            if conflicts:
                raise SearchConfigError(
                    f"Cannot combine writer=WriterConfig(...) with flat fields: {conflicts}.",
                    context={"conflicting_fields": conflicts},
                    hint="Pass either the grouped writer=WriterConfig(...) or the flat kwargs, not both.",
                )
            self.system_instructions = cfg.system_instructions
            self.max_writer_words = cfg.max_writer_words
            self.max_writer_sources = cfg.max_writer_sources
            self.claim_first_writing = cfg.claim_first_writing
            self.use_reasoning_preamble = cfg.use_reasoning_preamble

        if self.verification is not None:
            cfg = self.verification
            conflicts = []
            if self.llm_verification:
                conflicts.append("llm_verification")
            if self.llm_verify_source_chars != 4000:
                conflicts.append("llm_verify_source_chars")
            if conflicts:
                raise SearchConfigError(
                    f"Cannot combine verification=VerificationConfig(...) with flat fields: {conflicts}.",
                    context={"conflicting_fields": conflicts},
                    hint="Pass either the grouped verification=VerificationConfig(...) or the flat kwargs, not both.",
                )
            self.llm_verification = cfg.llm_verification
            self.llm_verify_source_chars = cfg.llm_verify_source_chars

        if self.revision is not None:
            cfg = self.revision
            conflicts = []
            if self.max_revision_rounds != 2:
                conflicts.append("max_revision_rounds")
            if self.revision_threshold != 0.3:
                conflicts.append("revision_threshold")
            if conflicts:
                raise SearchConfigError(
                    f"Cannot combine revision=RevisionConfig(...) with flat fields: {conflicts}.",
                    context={"conflicting_fields": conflicts},
                    hint="Pass either the grouped revision=RevisionConfig(...) or the flat kwargs, not both.",
                )
            self.max_revision_rounds = cfg.max_revision_rounds
            self.revision_threshold = cfg.revision_threshold

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
            raise SearchConfigError(
                f"Unknown search source(s): {bad!r}.",
                context={"unknown_sources": bad, "valid_sources": valid_list},
                hint=f"Use one of the valid source strings: {valid_list}. Example: SearchConfig(sources=['web', 'academic'])",
            )
        self.sources = normalized

        _log.debug(
            "config model=%s fast=%s search=%s",
            self.model,
            self.fast_model,
            self.searxng_url,
        )

    @property
    def model_config(self) -> SearchModels:
        """Read-only view of model settings (avoids shadowing the flat ``model`` field)."""
        return SearchModels(
            model=self.model,
            fast_model=self.fast_model,
            embedding_model=self.embedding_model,
            context_window_tokens=self.context_window_tokens,
        )

    @property
    def source_config(self) -> SearchSources:
        """Read-only view of source/fetching settings."""
        return SearchSources(
            searxng_url=self.searxng_url,
            searxng_timeout=self.searxng_timeout,
            jina_reader_url=self.jina_reader_url,
            jina_api_key=self.jina_api_key,
            serper_api_key=self.serper_api_key,
            sources=list(self.sources),
            max_results=self.max_results,
        )

    @property
    def research_config(self) -> ResearchConfig:
        """Read-only view of research/iteration settings."""
        return ResearchConfig(
            research_mode=self.research_mode,
            max_iterations=self.max_iterations,
            max_deep_research_steps=self.max_deep_research_steps,
            deep_research_enrich_per_step=self.deep_research_enrich_per_step,
            max_content_chars=self.max_content_chars,
        )

    @property
    def writer_config(self) -> WriterConfig:
        """Read-only view of writer settings."""
        return WriterConfig(
            system_instructions=self.system_instructions,
            max_writer_words=self.max_writer_words,
            max_writer_sources=self.max_writer_sources,
            claim_first_writing=self.claim_first_writing,
            use_reasoning_preamble=self.use_reasoning_preamble,
        )

    @property
    def verification_config(self) -> VerificationConfig:
        """Read-only view of verification settings."""
        return VerificationConfig(
            llm_verification=self.llm_verification,
            llm_verify_source_chars=self.llm_verify_source_chars,
        )

    @property
    def revision_config(self) -> RevisionConfig:
        """Read-only view of revision loop settings."""
        return RevisionConfig(
            max_revision_rounds=self.max_revision_rounds,
            revision_threshold=self.revision_threshold,
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
