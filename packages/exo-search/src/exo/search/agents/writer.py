"""Writer agent — generates final cited answer using Exo Search's 'Vane' prompt."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from exo import Agent, run
from exo.observability.logging import get_logger  # pyright: ignore[reportMissingImports]
from exo.types import StreamEvent

from .._utils import resolve_provider
from ..config import SearchConfig
from ..conversation import format_chat_history
from ..prompts.instructions import (
    get_claim_extraction_prompt,
    get_claim_first_writer_prompt,
    get_revision_prompt,
    get_writer_prompt,
)
from ..types import ExtractedClaim, SearchResult

_log = get_logger(__name__)


def format_results_as_context(search_results: list[SearchResult]) -> str:
    """Format search results as <result> context blocks for the writer prompt."""
    if not search_results:
        return ""
    parts = []
    for i, r in enumerate(search_results, 1):
        content_type = "full_page" if r.enriched else "snippet"
        chars = len(r.content)
        parts.append(
            f'<result index={i} title="{r.title}" url="{r.url}"'
            f' content_type="{content_type}" chars="{chars}">'
            f"\n{r.content}\n</result>"
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Claim extraction (for claim-first writing)
# ---------------------------------------------------------------------------


async def extract_claims(
    query: str,
    search_results: list[SearchResult],
    config: SearchConfig | None = None,
) -> list[ExtractedClaim]:
    """Extract structured claims from source content using fast_model.

    Returns a list of :class:`ExtractedClaim` objects. On failure, returns
    an empty list (caller falls back to standard writing).
    """
    from ..config import SearchConfig as Cfg

    cfg = config or Cfg()
    context = format_results_as_context(search_results)
    prompt = get_claim_extraction_prompt(context, query)

    try:
        agent = Agent(
            name="claim_extractor",
            model=cfg.fast_model,
            instructions="Extract factual claims from sources. Output only valid JSON.",
            temperature=0.0,
            max_steps=1,
        )
        provider = resolve_provider(cfg.fast_model)
        result = await run(agent, prompt, provider=provider)

        # Parse JSON from output — strip markdown fences if present
        raw = result.output.strip()
        if raw.startswith("```"):
            # Remove ```json ... ``` wrapper
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        claims_data = json.loads(raw)
        claims = [ExtractedClaim(**c) for c in claims_data]
        _log.info("extracted %d claims from %d sources", len(claims), len(search_results))
        return claims
    except Exception as exc:
        _log.warning("claim extraction failed (falling back to standard writer): %s", exc)
        return []


def _format_claims_for_writer(claims: list[ExtractedClaim]) -> str:
    """Format extracted claims as text for the writer prompt."""
    parts = []
    for i, c in enumerate(claims, 1):
        quote_part = f' (verbatim: "{c.verbatim_quote}")' if c.verbatim_quote else ""
        parts.append(f"{i}. [Source {c.source_index}] {c.claim}{quote_part}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Standard write
# ---------------------------------------------------------------------------


async def write_answer(
    query: str,
    search_results: list[SearchResult],
    chat_history: list[tuple[str, str]],
    system_instructions: str = "",
    mode: str = "balanced",
    config: SearchConfig | None = None,
    research_narrative: str = "",
) -> str:
    """Generate final answer using Exo Search's writer prompt with context."""
    from ..config import SearchConfig as Cfg

    cfg = config or Cfg()
    _log.debug("writer query=%r sources=%d model=%s", query, len(search_results), cfg.model)

    context = format_results_as_context(search_results)

    # Claim-first path for quality/deep modes
    if mode in ("quality", "deep") and cfg.claim_first_writing:
        claims = await extract_claims(query, search_results, config=cfg)
        if claims:
            _log.info("using claim-first writing with %d claims", len(claims))
            claims_text = _format_claims_for_writer(claims)
            instructions = get_claim_first_writer_prompt(
                claims_text,
                context,
                system_instructions,
                cfg.max_writer_words,
                research_narrative,
            )
        else:
            # Fallback to standard writer
            instructions = get_writer_prompt(
                context, system_instructions, mode, cfg.max_writer_words, research_narrative
            )
    else:
        instructions = get_writer_prompt(
            context, system_instructions, mode, cfg.max_writer_words, research_narrative
        )

    # Build messages with chat history (budget-based truncation)
    history_text = format_chat_history(chat_history)
    formatted_input = f"{history_text}\nUser: {query}" if history_text else f"User: {query}"

    writer = Agent(
        name="writer",
        model=cfg.model,
        instructions=instructions,
        temperature=0.2,
        max_steps=1,
    )

    provider = resolve_provider(cfg.model)
    result = await run(writer, formatted_input, provider=provider)
    _log.info("writer done len=%d", len(result.output))
    return result.output


async def stream_write_answer(
    query: str,
    search_results: list[SearchResult],
    chat_history: list[tuple[str, str]],
    system_instructions: str = "",
    mode: str = "balanced",
    config: SearchConfig | None = None,
    research_narrative: str = "",
) -> AsyncIterator[StreamEvent]:
    """Stream writer text tokens as they are generated.

    For claim-first writing (quality/deep), claim extraction runs before
    streaming begins; only the composition step is streamed.
    """
    from ..config import SearchConfig as Cfg

    cfg = config or Cfg()
    _log.debug("stream_writer query=%r sources=%d model=%s", query, len(search_results), cfg.model)

    context = format_results_as_context(search_results)

    # Claim-first path for quality/deep modes
    if mode in ("quality", "deep") and cfg.claim_first_writing:
        claims = await extract_claims(query, search_results, config=cfg)
        if claims:
            _log.info("streaming claim-first writing with %d claims", len(claims))
            claims_text = _format_claims_for_writer(claims)
            instructions = get_claim_first_writer_prompt(
                claims_text,
                context,
                system_instructions,
                cfg.max_writer_words,
                research_narrative,
            )
        else:
            instructions = get_writer_prompt(
                context, system_instructions, mode, cfg.max_writer_words, research_narrative
            )
    else:
        instructions = get_writer_prompt(
            context, system_instructions, mode, cfg.max_writer_words, research_narrative
        )

    history_text = format_chat_history(chat_history)
    formatted_input = f"{history_text}\nUser: {query}" if history_text else f"User: {query}"

    writer = Agent(
        name="writer",
        model=cfg.model,
        instructions=instructions,
        temperature=0.2,
        max_steps=1,
    )

    provider = resolve_provider(cfg.model)
    async for event in run.stream(writer, formatted_input, provider=provider):
        yield event


# ---------------------------------------------------------------------------
# Revision (for write-verify-revise loop)
# ---------------------------------------------------------------------------


async def revise_answer(
    query: str,
    original_answer: str,
    failed_claims: list[tuple[str, int]],
    search_results: list[SearchResult],
    chat_history: list[tuple[str, str]],
    system_instructions: str = "",
    mode: str = "balanced",
    config: SearchConfig | None = None,
    research_narrative: str = "",
) -> str:
    """Revise an answer after citation verification failures.

    The writer sees which claims failed and rewrites without them.
    """
    from ..config import SearchConfig as Cfg

    cfg = config or Cfg()
    _log.info(
        "revising answer: %d failed claims, original_len=%d",
        len(failed_claims),
        len(original_answer),
    )

    context = format_results_as_context(search_results)
    instructions = get_revision_prompt(
        original_answer=original_answer,
        failed_claims=failed_claims,
        context=context,
        system_instructions=system_instructions,
        max_writer_words=cfg.max_writer_words,
        research_narrative=research_narrative,
    )

    history_text = format_chat_history(chat_history)
    formatted_input = f"{history_text}\nUser: {query}" if history_text else f"User: {query}"

    writer = Agent(
        name="writer_reviser",
        model=cfg.model,
        instructions=instructions,
        temperature=0.2,
        max_steps=1,
    )

    provider = resolve_provider(cfg.model)
    result = await run(writer, formatted_input, provider=provider)
    _log.info("revision done len=%d", len(result.output))
    return result.output
