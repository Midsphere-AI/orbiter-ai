"""Contradiction detection for search answers.

Extracts key factual claims from an answer and cross-checks them against
source content to identify genuine disagreements between sources.  Only
runs in ``quality`` and ``deep`` modes to avoid unnecessary latency.
"""

from __future__ import annotations

import json

from exo import Agent, run
from exo.observability.logging import get_logger  # pyright: ignore[reportMissingImports]

from .._utils import resolve_provider
from ..config import SearchConfig
from ..types import Contradiction, ContradictionReport, FactualClaim, SearchResult

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# 1. Claim extraction
# ---------------------------------------------------------------------------

_CLAIM_EXTRACTION_PROMPT = """\
You are a fact extractor. Given the following answer text, extract up to 10 key \
verifiable factual claims. Focus on specific facts: numbers, dates, names, \
relationships, statistics, and concrete assertions.

Return a JSON array of objects, each with:
- "claim_text": the factual claim (one sentence)
- "cited_sources": array of source numbers (integers) cited for this claim, or []

Return ONLY the JSON array, no other text.

Answer text:
{answer}
"""


async def _extract_claims(answer: str, config: SearchConfig) -> list[FactualClaim]:
    """Extract key factual claims from an answer using a fast LLM."""
    if not answer.strip():
        return []

    provider = resolve_provider(config.fast_model)
    agent = Agent(
        name="claim_extractor",
        model=config.fast_model,
        instructions=_CLAIM_EXTRACTION_PROMPT.format(answer=answer[:8000]),
        temperature=0.0,
        max_steps=1,
    )

    try:
        result = await run(agent, "Extract claims.", provider=provider)
        raw = result.output if hasattr(result, "output") else str(result)

        # Parse JSON from response — handle markdown code fences
        text = raw.strip()
        if text.startswith("```"):
            # Strip ```json ... ``` wrapper
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            text = text.strip()

        claims_data = json.loads(text)
        claims = []
        for item in claims_data[:10]:
            claims.append(
                FactualClaim(
                    claim_text=item.get("claim_text", ""),
                    cited_sources=item.get("cited_sources", []),
                )
            )
        _log.debug("extracted %d claims from answer", len(claims))
        return claims
    except Exception as exc:
        _log.warning("claim extraction failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# 2. Conflict analysis
# ---------------------------------------------------------------------------

_CONFLICT_ANALYSIS_PROMPT = """\
You are a fact-checker. Given a set of factual claims and source documents, \
check if any sources present DIFFERENT or CONTRADICTORY facts for the same claim.

Only flag genuine factual disagreements — not differences in emphasis, phrasing, \
or scope. Minor omissions are NOT contradictions.

Claims to check:
{claims_text}

Source documents:
{sources_text}

Return a JSON array of contradiction objects. Each object has:
- "claim_text": the claim being disputed
- "position_a": what one set of sources says
- "position_b": what another set of sources says
- "source_indices_a": array of 1-based source numbers supporting position A
- "source_indices_b": array of 1-based source numbers supporting position B
- "severity": "minor", "moderate", or "major"

If no contradictions are found, return an empty array: []

Return ONLY the JSON array, no other text.
"""


async def _analyze_conflicts(
    claims: list[FactualClaim],
    sources: list[SearchResult],
    config: SearchConfig,
) -> list[Contradiction]:
    """Check claims against source content for contradictions."""
    if not claims or not sources:
        return []

    # Build claims text
    claims_lines = []
    for i, claim in enumerate(claims, 1):
        src_str = (
            f" (cited: [{', '.join(str(s) for s in claim.cited_sources)}])"
            if claim.cited_sources
            else ""
        )
        claims_lines.append(f"{i}. {claim.claim_text}{src_str}")
    claims_text = "\n".join(claims_lines)

    # Build sources text — truncate each source to 2K chars, max 10 sources
    sources_lines = []
    for i, source in enumerate(sources[:10], 1):
        content = source.content[:2000] if source.content else "(no content)"
        sources_lines.append(f"[Source {i}] {source.title}\n{content}\n")
    sources_text = "\n".join(sources_lines)

    provider = resolve_provider(config.fast_model)
    agent = Agent(
        name="conflict_analyzer",
        model=config.fast_model,
        instructions=_CONFLICT_ANALYSIS_PROMPT.format(
            claims_text=claims_text,
            sources_text=sources_text,
        ),
        temperature=0.0,
        max_steps=1,
    )

    try:
        result = await run(agent, "Analyze for contradictions.", provider=provider)
        raw = result.output if hasattr(result, "output") else str(result)

        # Parse JSON from response
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            text = text.strip()

        conflicts_data = json.loads(text)
        contradictions = []
        for item in conflicts_data:
            contradictions.append(
                Contradiction(
                    claim_text=item.get("claim_text", ""),
                    position_a=item.get("position_a", ""),
                    position_b=item.get("position_b", ""),
                    source_indices_a=item.get("source_indices_a", []),
                    source_indices_b=item.get("source_indices_b", []),
                    severity=item.get("severity", "moderate"),
                )
            )
        _log.debug("found %d contradictions", len(contradictions))
        return contradictions
    except Exception as exc:
        _log.warning("conflict analysis failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# 3. Main entry point
# ---------------------------------------------------------------------------


async def detect_contradictions(
    answer: str,
    sources: list[SearchResult],
    mode: str,
    config: SearchConfig,
) -> ContradictionReport:
    """Detect contradictions between sources referenced in the answer.

    Only runs in ``quality`` and ``deep`` modes.  Returns an empty report
    for ``speed`` and ``balanced`` modes.
    """
    if mode not in ("quality", "deep"):
        return ContradictionReport()

    _log.debug("running contradiction detection (mode=%s)", mode)

    # Step 1: Extract claims
    claims = await _extract_claims(answer, config)
    if not claims:
        _log.debug("no claims extracted, skipping conflict analysis")
        return ContradictionReport(claims_checked=0)

    # Step 2: Analyze conflicts
    contradictions = await _analyze_conflicts(claims, sources, config)

    report = ContradictionReport(
        contradictions=contradictions,
        claims_checked=len(claims),
        has_contradictions=len(contradictions) > 0,
    )

    if report.has_contradictions:
        _log.info(
            "contradiction report: %d contradictions found across %d claims",
            len(contradictions),
            len(claims),
        )
    else:
        _log.debug("no contradictions found across %d claims", len(claims))

    return report
