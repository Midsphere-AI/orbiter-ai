"""Query planner — adaptive structured-output research with hybrid orchestration."""

from __future__ import annotations

import asyncio
import datetime

from exo import Agent, run
from exo.observability.logging import get_logger  # pyright: ignore[reportMissingImports]

from .._utils import resolve_provider
from ..config import SearchConfig
from ..tools.searxng import search_and_collect
from ..types import ClassifierOutput, QueryPlan, SearchResult

_log = get_logger(__name__)


_INITIAL_PROMPT = """\
You are a search query strategist. Given the user's question, generate {num_queries} \
targeted search queries that will find the most relevant, authoritative information.

Today's date: {today}

{sub_questions_block}\
Requirements:
- Each query MUST be under 100 characters — short keyword phrases only
- NO full sentences, NO question marks, NO filler words (the, a, of, in, for)
- CRITICAL: Generate at least one query per sub-question listed above. \
Do NOT cluster all queries on a single sub-question.
- Add year qualifiers for recent topics (e.g., "2025", "2026")
- Include specific entity names, technical terms, and domain keywords

Good: "nuclear fusion breakthroughs 2025 tokamak", "solar energy LCOE cost MWh 2025"
Bad: "What are the latest developments in fusion energy research?" (too long, sentence)

Set "sufficient" to false (you haven't searched yet).
"""

_FOLLOWUP_PROMPT = """\
You are a search query strategist refining a research plan. The user asked:
"{query}"

{sub_questions_block}\
Here are the search results found so far:
{results_summary}

Check each sub-question above against the results. Identify which sub-questions \
have ZERO or WEAK coverage, then generate 3 targeted follow-up queries to fill \
those gaps. Prioritize:
- Sub-questions with no matching results (most critical)
- Claims that need primary sources or specific data
- Recent developments that might have newer coverage

Today's date: {today}

Set "sufficient" to true ONLY if every sub-question has at least 2-3 relevant results.
"""


def _format_sub_questions_block(sub_questions: list[str]) -> str:
    """Format sub-questions as a numbered block for prompts."""
    if not sub_questions:
        return ""
    lines = [
        "The question decomposes into these sub-questions (each needs coverage):",
    ]
    for i, sq in enumerate(sub_questions, 1):
        lines.append(f"  {i}. {sq}")
    lines.append("")
    return "\n".join(lines) + "\n"


def _sub_questions_to_queries(sub_questions: list[str]) -> list[str]:
    """Convert sub-questions to keyword-style search queries (deterministic fallback)."""
    import re

    year = datetime.date.today().year
    queries = []
    for sq in sub_questions:
        kw = re.sub(
            r"^(what|how|who|when|where|why|does|is|are|do|did|has|have|was|were)\s+",
            "",
            sq,
            flags=re.IGNORECASE,
        )
        kw = kw.rstrip("?. ")
        if str(year) not in kw and str(year - 1) not in kw:
            kw = f"{kw} {year}"
        queries.append(kw)
    return queries


def _trim_query(query: str, max_len: int = 200) -> str:
    """Compress a verbose LLM query into keyword-style search terms."""
    import re

    if len(query) <= max_len:
        return query
    # Strip question phrasing and filler
    q = re.sub(
        r"^(what|how|who|when|where|why|does|is|are|do|did|has|have|was|were|"
        r"find|search|look up|can you|please|tell me about)\s+",
        "",
        query,
        flags=re.IGNORECASE,
    )
    q = re.sub(r"\b(the|a|an|of|in|on|for|to|and|or|with|that|this|from)\b", " ", q)
    q = re.sub(r"\s+", " ", q).strip().rstrip("?. ")
    if len(q) > max_len:
        q = " ".join(q[:max_len].rsplit(" ", 1)[0:1])
    return q


async def _generate_query_plan(
    query: str,
    chat_history: list[tuple[str, str]],
    existing_results: list[SearchResult],
    config: SearchConfig,
    sub_questions: list[str] | None = None,
) -> QueryPlan:
    """Generate a query plan — initial queries or gap-filling follow-ups."""
    _log.debug("query_plan round existing=%d", len(existing_results))
    today = datetime.date.today().strftime("%B %d, %Y")
    sq_block = _format_sub_questions_block(sub_questions or [])

    num_queries = max(3, len(sub_questions)) if sub_questions else 3

    if not existing_results:
        prompt = _INITIAL_PROMPT.format(
            today=today,
            sub_questions_block=sq_block,
            num_queries=num_queries,
        )
    else:
        lines = []
        for i, r in enumerate(existing_results[:20], 1):
            snippet = r.content[:150] if r.content else r.title
            lines.append(f"  {i}. {r.title}: {snippet}")
        prompt = _FOLLOWUP_PROMPT.format(
            query=query,
            results_summary="\n".join(lines),
            today=today,
            sub_questions_block=sq_block,
        )

    parts = []
    for q, a in chat_history:
        parts.append(f"User: {q}")
        short_a = a[:500] + "..." if len(a) > 500 else a
        parts.append(f"Assistant: {short_a}")
    parts.append(f"User: {query}")
    formatted_input = "\n".join(parts)

    planner = Agent(
        name="query-planner",
        model=config.fast_model,
        instructions=prompt,
        output_type=QueryPlan,
        temperature=0.3,
        max_steps=1,
    )

    provider = resolve_provider(config.fast_model)
    result = await run(planner, formatted_input, provider=provider)

    try:
        plan = QueryPlan.model_validate_json(result.output)
        _log.debug("query_plan queries=%s sufficient=%s", plan.queries, plan.sufficient)
    except Exception as exc:
        _log.warning("query_plan parse failed: %s — falling back to raw query", exc)
        plan = QueryPlan(queries=[query], sufficient=False)

    # Validate and salvage: trim overly long queries, then fill gaps.
    plan = QueryPlan(
        queries=[_trim_query(q) for q in plan.queries],
        sufficient=plan.sufficient,
    )
    if plan.queries and sub_questions and len(sub_questions) > 1:
        fallback_queries = _sub_questions_to_queries(sub_questions)
        good = [q for q in plan.queries if len(q) <= 200]
        if not good:
            _log.warning("query_plan all queries bad, using sub-question conversion")
            plan = QueryPlan(queries=fallback_queries, sufficient=False)
        elif len(good) < len(sub_questions):
            needed = len(sub_questions) - len(good)
            extras = [q for q in fallback_queries if q not in good][:needed]
            _log.debug(
                "query_plan salvaged %d/%d, added %d fallback",
                len(good),
                len(plan.queries),
                len(extras),
            )
            plan = QueryPlan(queries=good + extras, sufficient=plan.sufficient)
    elif not plan.queries and sub_questions:
        _log.warning("query_plan empty, falling back to sub-question conversion")
        plan = QueryPlan(
            queries=_sub_questions_to_queries(sub_questions),
            sufficient=False,
        )

    return plan


def _merge_results(
    primary: list[SearchResult],
    secondary: list[SearchResult],
) -> list[SearchResult]:
    """Merge two result lists, deduplicating by URL."""
    seen: set[str] = set()
    merged: list[SearchResult] = []
    for r in [*primary, *secondary]:
        if r.url and r.url not in seen:
            seen.add(r.url)
            merged.append(r)
    return merged


async def adaptive_research(
    query: str,
    chat_history: list[tuple[str, str]],
    mode: str,
    config: SearchConfig,
    sub_questions: list[str] | None = None,
    seed_results: list[SearchResult] | None = None,
) -> list[SearchResult]:
    """Adaptive multi-round research using structured output query generation.

    Each round the LLM sees existing results and either generates follow-up
    queries to fill gaps or signals that coverage is sufficient.

    Args:
        seed_results: Pre-fetched results from speculative search to bootstrap
            the first round (avoids redundant initial queries).
    """
    max_rounds = {"balanced": 1, "quality": 3, "deep": 3}.get(mode, 1)
    all_results: list[SearchResult] = list(seed_results) if seed_results else []

    for round_num in range(max_rounds):
        _log.debug("adaptive round=%d/%d existing=%d", round_num + 1, max_rounds, len(all_results))
        plan = await _generate_query_plan(
            query,
            chat_history,
            all_results,
            config,
            sub_questions,
        )

        if plan.sufficient and round_num > 0:
            break
        if not plan.queries:
            break

        query_cap = max(3, len(sub_questions)) if sub_questions else 3
        raw = await search_and_collect(plan.queries[:query_cap])
        new_results = [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                content=r.get("content", ""),
                enriched=r.get("enriched", False),
            )
            for r in raw
        ]
        all_results = _merge_results(all_results, new_results)
        _log.info(
            "adaptive round=%d/%d new=%d total=%d",
            round_num + 1,
            max_rounds,
            len(new_results),
            len(all_results),
        )

    _log.info("adaptive done total=%d", len(all_results))
    return all_results


async def hybrid_research(
    query: str,
    classification: ClassifierOutput,
    chat_history: list[tuple[str, str]],
    mode: str = "balanced",
    config: SearchConfig | None = None,
    sub_questions: list[str] | None = None,
    seed_results: list[SearchResult] | None = None,
) -> list[SearchResult]:
    """Hybrid research: adaptive structured queries + parallel researchers.

    Structured output path is guaranteed to produce results and adapts
    across rounds. Tool-calling researchers run concurrently as best-effort
    bonus depth, time-boxed to avoid stalling the pipeline.

    Args:
        seed_results: Pre-fetched results from speculative search to bootstrap
            the adaptive path (saves one full search round).
    """
    from .researcher import parallel_research

    cfg = config or SearchConfig()

    async def _timed_agents() -> list[SearchResult]:
        timeout = {"balanced": 10, "quality": 20, "deep": 20}.get(mode, 10)
        _log.debug("hybrid timeout=%ds for parallel agents", timeout)
        try:
            return await asyncio.wait_for(
                parallel_research(
                    query,
                    classification,
                    chat_history,
                    mode,
                    cfg,
                    sub_questions,
                ),
                timeout=timeout,
            )
        except Exception as exc:
            _log.debug("timed agents failed/timed out: %s", exc)
            return []

    structured_task = asyncio.ensure_future(
        adaptive_research(
            query,
            chat_history,
            mode,
            cfg,
            sub_questions,
            seed_results,
        )
    )
    agent_task = asyncio.ensure_future(_timed_agents())

    gather_results = await asyncio.gather(structured_task, agent_task, return_exceptions=True)
    structured_r, agent_r = gather_results
    if isinstance(structured_r, BaseException):
        _log.warning("adaptive_research failed: %s", structured_r)
        structured_results: list[SearchResult] = []
    else:
        structured_results = structured_r
    if isinstance(agent_r, BaseException):
        _log.debug("timed agent gather failed: %s", agent_r)
        agent_results: list[SearchResult] = []
    else:
        agent_results = agent_r
    merged = _merge_results(structured_results, agent_results)
    _log.info(
        "hybrid merged=%d (structured=%d agents=%d)",
        len(merged),
        len(structured_results),
        len(agent_results),
    )
    return merged
