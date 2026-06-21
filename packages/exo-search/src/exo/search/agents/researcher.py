"""Researcher agent — iterative tool-calling research loop."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from exo import Agent, run
from exo.observability.logging import get_logger  # pyright: ignore[reportMissingImports]
from exo.types import StreamEvent

from .._utils import resolve_provider
from ..config import SearchConfig
from ..prompts.instructions import (
    ACADEMIC_SEARCH_PROMPT,
    DONE_PROMPT,
    REASONING_PREAMBLE_PROMPT,
    SCRAPE_URL_PROMPT,
    SOCIAL_SEARCH_PROMPT,
    get_researcher_prompt,
    get_sub_researcher_prompt,
    get_web_search_prompt,
)
from ..tools.researcher_tools import done, reasoning_preamble
from ..tools.searxng import (
    academic_search,
    clear_collected_results,
    get_collected_results,
    search_and_collect,
    social_search,
    web_search,
)
from ..tools.web_fetcher import scrape_url
from ..types import ClassifierOutput, SearchResult

_log = get_logger(__name__)

_MODE_ITERATIONS = {"speed": 2, "balanced": 6, "quality": 25, "deep": 25}

# Gemini thinking models that have built-in reasoning — reasoning_preamble is redundant.
_THINKING_MODEL_PATTERNS = (
    "gemini-2.5",
    "gemini-3",
)


def _get_max_iterations(mode: str, override: int | None = None) -> int:
    if override is not None:
        return override
    return _MODE_ITERATIONS.get(mode, 25)


def _is_thinking_model(model: str) -> bool:
    """Check if the model has built-in thinking/reasoning capabilities."""
    model_lower = model.lower()
    return any(p in model_lower for p in _THINKING_MODEL_PATTERNS)


def _build_tools_and_action_desc(
    classification: ClassifierOutput,
    sources: list[str],
    mode: str,
    include_reasoning_preamble: bool = True,
) -> tuple[list, str]:
    """Build available tools and action descriptions based on classification."""
    tools = []
    action_lines = []

    # Web search (if web source enabled and search not skipped)
    if "web" in sources and not classification.classification.skip_search:
        tools.append(web_search)
        action_lines.append(f"- web_search: {get_web_search_prompt(mode)}")

    # Academic search
    if (
        "academic" in sources
        and classification.classification.academic_search
        and not classification.classification.skip_search
    ):
        tools.append(academic_search)
        action_lines.append(f"- academic_search: {ACADEMIC_SEARCH_PROMPT}")

    # Social/discussion search
    if (
        "discussions" in sources
        and classification.classification.discussion_search
        and not classification.classification.skip_search
    ):
        tools.append(social_search)
        action_lines.append(f"- social_search: {SOCIAL_SEARCH_PROMPT}")

    # Scrape URL (always available)
    tools.append(scrape_url)
    action_lines.append(f"- scrape_url: {SCRAPE_URL_PROMPT}")

    # Done (always available)
    tools.append(done)
    action_lines.append(f"- done: {DONE_PROMPT}")

    # Reasoning preamble (skip for speed mode and thinking models)
    if include_reasoning_preamble and mode != "speed":
        tools.append(reasoning_preamble)
        action_lines.append(f"- reasoning_preamble: {REASONING_PREAMBLE_PROMPT}")

    return tools, "\n".join(action_lines)


async def research(
    query: str,
    classification: ClassifierOutput,
    chat_history: list[tuple[str, str]],
    mode: str = "balanced",
    config: SearchConfig | None = None,
) -> list[SearchResult]:
    """Run the researcher agent with iterative tool calls.

    Matches Exo Search's Researcher.research() flow.
    Uses a shared result collector in the search tools to gather results,
    since the exo framework doesn't expose tool call history in RunResult.
    """
    from ..config import SearchConfig as Cfg

    cfg = config or Cfg()
    _log.debug("research query=%r mode=%s", query, mode)

    max_iterations = _get_max_iterations(mode, cfg.max_iterations)

    # Determine whether to include reasoning_preamble tool
    if cfg.use_reasoning_preamble is not None:
        include_reasoning = cfg.use_reasoning_preamble
    else:
        # Auto-detect: skip for thinking models that reason internally
        include_reasoning = not _is_thinking_model(cfg.model)

    tools, action_desc = _build_tools_and_action_desc(
        classification, cfg.sources, mode, include_reasoning
    )

    # Use no-reasoning-preamble prompt variant for thinking models
    effective_mode = mode if include_reasoning else f"{mode}_no_reasoning"
    instructions = get_researcher_prompt(
        action_desc=action_desc,
        mode=effective_mode,
        iteration=1,
        max_iteration=max_iterations,
    )

    # Build input with chat history context
    parts = []
    for q, a in chat_history:
        parts.append(f"User: {q}")
        short_a = a[:500] + "..." if len(a) > 500 else a
        parts.append(f"Assistant: {short_a}")
    parts.append(f"User: {query}")
    formatted_input = "\n".join(parts)

    researcher = Agent(
        name="researcher",
        model=cfg.model,
        instructions=instructions,
        tools=tools,
        temperature=0.1,
        max_steps=max_iterations * 3,
    )

    # Clear collector before run, collect results after
    clear_collected_results()

    provider = resolve_provider(cfg.model)
    await run(researcher, formatted_input, provider=provider)

    # Get results from the shared collector (populated by tool side-effects)
    raw_results = get_collected_results()
    _log.info("research complete results=%d", len(raw_results))

    # Deduplicate and convert to SearchResult objects
    seen_urls: set[str] = set()
    results: list[SearchResult] = []
    for r in raw_results:
        url = r.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            results.append(
                SearchResult(
                    title=r.get("title", ""),
                    url=url,
                    content=r.get("content", ""),
                    enriched=r.get("enriched", False),
                )
            )

    return results


async def stream_research(
    query: str,
    classification: ClassifierOutput,
    chat_history: list[tuple[str, str]],
    mode: str = "balanced",
    config: SearchConfig | None = None,
) -> AsyncIterator[SearchResult | StreamEvent]:
    """Stream researcher events, then yield collected SearchResults at the end.

    Yields StreamEvent objects during execution (TextEvent, ToolCallEvent, etc.),
    then yields SearchResult objects once the agent finishes.
    """
    from ..config import SearchConfig as Cfg

    cfg = config or Cfg()

    max_iterations = _get_max_iterations(mode, cfg.max_iterations)

    if cfg.use_reasoning_preamble is not None:
        include_reasoning = cfg.use_reasoning_preamble
    else:
        include_reasoning = not _is_thinking_model(cfg.model)

    tools, action_desc = _build_tools_and_action_desc(
        classification, cfg.sources, mode, include_reasoning
    )

    effective_mode = mode if include_reasoning else f"{mode}_no_reasoning"
    instructions = get_researcher_prompt(
        action_desc=action_desc,
        mode=effective_mode,
        iteration=1,
        max_iteration=max_iterations,
    )

    parts = []
    for q, a in chat_history:
        parts.append(f"User: {q}")
        short_a = a[:500] + "..." if len(a) > 500 else a
        parts.append(f"Assistant: {short_a}")
    parts.append(f"User: {query}")
    formatted_input = "\n".join(parts)

    researcher = Agent(
        name="researcher",
        model=cfg.model,
        instructions=instructions,
        tools=tools,
        temperature=0.1,
        max_steps=max_iterations * 3,
    )

    clear_collected_results()

    provider = resolve_provider(cfg.model)
    async for event in run.stream(researcher, formatted_input, provider=provider, detailed=True):
        yield event

    # Yield collected search results
    raw_results = get_collected_results()
    seen_urls: set[str] = set()
    for r in raw_results:
        url = r.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            yield SearchResult(
                title=r.get("title", ""),
                url=url,
                content=r.get("content", ""),
                enriched=r.get("enriched", False),
            )


# ---------------------------------------------------------------------------
# Direct search — no LLM researcher, fastest path
# ---------------------------------------------------------------------------


def _split_query(query: str) -> list[str]:
    """Split a compound question into focused sub-queries for SearXNG."""
    import re

    parts = re.split(r",\s*and\s+|;\s+|\?\s+", query)
    queries = [p.strip().rstrip("?") for p in parts if len(p.strip()) > 10]
    return queries[:3] if queries else [query]


async def direct_search(query: str) -> list[SearchResult]:
    """Fast-path: query SearXNG directly, no LLM researcher overhead."""
    queries = _split_query(query)
    _log.debug("direct_search queries=%s", queries)
    raw = await search_and_collect(queries)
    seen_urls: set[str] = set()
    results: list[SearchResult] = []
    for r in raw:
        url = r.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            results.append(
                SearchResult(
                    title=r.get("title", ""),
                    url=url,
                    content=r.get("content", ""),
                    enriched=r.get("enriched", False),
                )
            )
    _log.info("direct_search results=%d", len(results))
    return results


# ---------------------------------------------------------------------------
# Parallel research — fan-out across research angles
# ---------------------------------------------------------------------------

_BREADTH_ANGLES = [
    "recent developments, timeline, and current state",
    "expert analysis, specific data points, and named entities",
    "broader context, comparisons, and implications",
]

_MAX_ANGLES = 5

# Minimum iterations each worker needs to be useful per mode.
_ITERS_PER_WORKER = {"speed": 1, "balanced": 2, "quality": 3}


def _derive_research_angles(query: str) -> list[str]:
    """Derive research angles from the question's own parts, not topic depth.

    Splits multi-part questions on conjunctions and question words so each
    sub-researcher covers a distinct PART of the question rather than a
    different depth of the same topic.
    """
    import re

    parts = re.split(
        r"[,;]\s+and\s+|[;?]\s+|,\s+(?=(?:what|how|who|when|where|why)\b)",
        query,
        flags=re.IGNORECASE,
    )
    angles = [p.strip().rstrip("?,. ") for p in parts if len(p.strip()) > 15]

    # Pad with general breadth angles to reach _MAX_ANGLES
    for g in _BREADTH_ANGLES:
        if len(angles) >= _MAX_ANGLES:
            break
        angles.append(g)
    return angles[:_MAX_ANGLES]


async def parallel_research(
    query: str,
    classification: ClassifierOutput,
    chat_history: list[tuple[str, str]],
    mode: str = "balanced",
    config: SearchConfig | None = None,
    sub_questions: list[str] | None = None,
) -> list[SearchResult]:
    """Run multiple sub-researchers in parallel, each covering one angle.

    Worker count is computed dynamically from the iteration budget:
      speed   (2 iters)  -> 2 workers x 1 iter each
      balanced (6 iters) -> 3 workers x 2 iters each
      quality (25 iters) -> 5 workers x 5 iters each
    """
    from ..config import SearchConfig as Cfg

    cfg = config or Cfg()

    # Prefer classifier-provided sub-questions over regex splitting
    if sub_questions and len(sub_questions) > 1:
        angles = sub_questions[:_MAX_ANGLES]
        # Pad with breadth angles if needed
        for g in _BREADTH_ANGLES:
            if len(angles) >= _MAX_ANGLES:
                break
            angles.append(g)
    else:
        angles = _derive_research_angles(query)
    max_iterations = _get_max_iterations(mode, cfg.max_iterations)
    min_iters = _ITERS_PER_WORKER.get(mode, 2)
    num_workers = min(len(angles), max(1, max_iterations // min_iters))
    iters_per_worker = max(1, max_iterations // num_workers)
    _log.info("parallel workers=%d iters_each=%d angles=%s", num_workers, iters_per_worker, angles)

    # Build chat history context once
    parts: list[str] = []
    for q, a in chat_history:
        parts.append(f"User: {q}")
        short_a = a[:500] + "..." if len(a) > 500 else a
        parts.append(f"Assistant: {short_a}")
    parts.append(f"User: {query}")
    formatted_input = "\n".join(parts)

    clear_collected_results()

    # Researchers just pick search queries — fast model suffices for speed/balanced
    research_model = cfg.model if mode == "quality" else cfg.fast_model
    provider = resolve_provider(research_model)

    async def _run_worker(angle: str) -> None:
        # Sub-researchers always use "speed" tool descriptions — the balanced/quality
        # web_search prompts reference reasoning_preamble which sub-researchers lack.
        tools, action_desc = _build_tools_and_action_desc(
            classification,
            cfg.sources,
            "speed",
            include_reasoning_preamble=False,
        )
        instructions = get_sub_researcher_prompt(
            action_desc=action_desc,
            angle=angle,
            max_iteration=iters_per_worker,
        )
        agent = Agent(
            name=f"researcher-{angle.split(',')[0].strip()[:20]}",
            model=research_model,
            instructions=instructions,
            tools=tools,
            temperature=0.1,
            max_steps=iters_per_worker * 3,
        )
        await run(agent, formatted_input, provider=provider)

    active_angles = angles[:num_workers]
    results = await asyncio.gather(
        *[_run_worker(angle) for angle in active_angles],
        return_exceptions=True,
    )

    # Log failures but don't abort — partial results are still valuable
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            _log.warning("sub-researcher '%s' failed: %s", active_angles[i], result)

    # Collect and deduplicate from the shared collector
    raw_results = get_collected_results()
    seen_urls: set[str] = set()
    search_results: list[SearchResult] = []
    for r in raw_results:
        url = r.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            search_results.append(
                SearchResult(
                    title=r.get("title", ""),
                    url=url,
                    content=r.get("content", ""),
                    enriched=r.get("enriched", False),
                )
            )
    _log.info("parallel_research total=%d", len(search_results))
    return search_results
