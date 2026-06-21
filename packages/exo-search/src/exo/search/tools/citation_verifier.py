"""Post-hoc citation verification for writer-generated answers.

Parses ``[N]`` citation markers from the answer text and checks whether
the cited source actually supports the claim.  Unsupported citations are
removed from the returned answer and tallied in a
:class:`CitationVerification` report.

Three-phase verification (quality/deep modes):
1. **Keyword filter** — cheap pass/fail heuristic
2. **LLM spot-check** — keyword-passed citations get semantic verification
3. **LLM second-chance** — keyword-failed citations get LLM review before removal
"""

from __future__ import annotations

import asyncio
import re

from exo.observability.logging import get_logger  # pyright: ignore[reportMissingImports]

from .._utils import resolve_provider
from ..config import SearchConfig
from ..types import CitationVerification, SearchResult

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Common English words excluded from key-term extraction
# ---------------------------------------------------------------------------

_COMMON_WORDS: set[str] = {
    "the",
    "and",
    "was",
    "were",
    "have",
    "been",
    "this",
    "that",
    "with",
    "from",
    "which",
    "their",
    "about",
    "would",
    "could",
    "should",
    "other",
    "after",
    "before",
    "between",
    "through",
    "during",
    "without",
    "within",
    "against",
    "across",
    "along",
    "behind",
    "around",
    "above",
    "below",
    "major",
    "large",
    "small",
    "important",
    "significant",
    "various",
    "several",
    "different",
    "similar",
    "specific",
    "general",
    "current",
    "recent",
    "first",
    "second",
    "third",
    "new",
    "old",
    "many",
    "more",
    "most",
    "some",
    "such",
    "also",
    "very",
    "often",
    "still",
    "already",
    "just",
    "even",
    "only",
    "however",
    "although",
    "because",
    "since",
    "while",
    "where",
    "when",
    "then",
    "than",
    "both",
    "each",
    "every",
    "much",
    "well",
    "long",
    "high",
    "right",
    "back",
    "made",
    "found",
    "known",
    "called",
    "used",
    "based",
    "according",
    "including",
    "following",
    "leading",
    "among",
    "under",
    "over",
    "into",
    "these",
    "those",
    "there",
    "here",
    "what",
    "will",
    "being",
    "does",
    "done",
    "doing",
    "like",
    "same",
    "part",
    "come",
    "came",
    "take",
    "took",
    "given",
    "using",
    "making",
    "going",
    "having",
    "said",
    "says",
    "show",
    "shown",
    "shows",
    "work",
    "works",
    "need",
    "needs",
    "help",
    "helps",
    "keep",
    "keeps",
    "start",
    "started",
    "provide",
    "provided",
    "create",
    "created",
    "allow",
    "allowed",
    "system",
    "point",
    "number",
    "people",
    "world",
    "years",
    "state",
    "become",
    "became",
    "related",
    "common",
    "available",
    "possible",
}

# Regex for ``[N]`` citation markers (N = one or more digits).
_CITE_RE = re.compile(r"\[(\d+)\]")

# Sentence-boundary pattern: period followed by space, or newline.
_SENT_BOUNDARY_RE = re.compile(r"(?:\.\s|\n)")

# Numbers / dates / percentages in text.
_NUMBER_RE = re.compile(r"\b\d[\d,.]*%?\b")

# Capitalised multi-word proper nouns (2+ consecutive capitalised words).
_PROPER_NOUN_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")


# ---------------------------------------------------------------------------
# 1. Extract citations
# ---------------------------------------------------------------------------


def _extract_citations(answer: str) -> list[tuple[int, str]]:
    """Parse ``[N]`` markers and pair each with the preceding claim text.

    Returns a list of ``(source_index_0_based, claim_text)`` tuples.
    Consecutive citations on the same sentence (e.g. ``[1][2]``) share the
    same claim text.
    """
    results: list[tuple[int, str]] = []
    # Find all citation positions.
    matches = list(_CITE_RE.finditer(answer))
    if not matches:
        return results

    # Group consecutive citations that share a claim.  Two citations are
    # "consecutive" when they are adjacent with only optional whitespace
    # between the closing ``]`` and the opening ``[``.
    groups: list[tuple[list[int], int, int]] = []  # (indices_0based, text_end, span_end)
    i = 0
    while i < len(matches):
        m = matches[i]
        cite_indices = [int(m.group(1)) - 1]  # convert to 0-based
        span_end = m.end()
        # Absorb consecutive citations like ``[1][2]``.
        while i + 1 < len(matches):
            gap = answer[span_end : matches[i + 1].start()]
            if gap.strip() == "":
                i += 1
                cite_indices.append(int(matches[i].group(1)) - 1)
                span_end = matches[i].end()
            else:
                break
        groups.append((cite_indices, m.start(), span_end))
        i += 1

    for cite_indices, cite_start, _span_end in groups:
        # Walk backwards from the first ``[`` to find the sentence boundary.
        text_before = answer[:cite_start]
        # Find the last sentence boundary before the citation.
        boundary_matches = list(_SENT_BOUNDARY_RE.finditer(text_before))
        claim_start = boundary_matches[-1].end() if boundary_matches else 0
        claim = text_before[claim_start:].strip()
        # Strip any leading list markers like ``- `` or ``* ``.
        claim = re.sub(r"^[-*]\s+", "", claim)
        for idx in cite_indices:
            results.append((idx, claim))

    _log.debug("extracted %d citation(s) from answer", len(results))
    return results


# ---------------------------------------------------------------------------
# 2. Keyword verification
# ---------------------------------------------------------------------------


def _keyword_verify(claim: str, source_content: str) -> bool:
    """Check whether *source_content* plausibly supports *claim*.

    Uses a lightweight heuristic: extract key terms (numbers, proper nouns,
    technical terms) from the claim and verify that a sufficient fraction
    appear in the source.
    """
    if len(claim) < 20:
        # Too short / generic to meaningfully verify.
        return True

    source_lower = source_content.lower()
    key_terms: list[str] = []
    number_terms: list[str] = []

    # --- Numbers / dates / percentages ---
    for m in _NUMBER_RE.finditer(claim):
        token = m.group()
        number_terms.append(token)
        key_terms.append(token)

    # --- Proper nouns ---
    for m in _PROPER_NOUN_RE.finditer(claim):
        key_terms.append(m.group().lower())

    # --- Technical terms (>6 chars, not common English) ---
    words = re.findall(r"[A-Za-z]+", claim)
    for w in words:
        low = w.lower()
        if len(low) > 6 and low not in _COMMON_WORDS:
            key_terms.append(low)

    if not key_terms:
        # No specific terms to verify — pass by default.
        return True

    # Count how many key terms appear in the source.
    matched = sum(1 for t in key_terms if t in source_lower)
    ratio = matched / len(key_terms)

    # If the claim has specific numbers, at least ONE must be present.
    if number_terms:
        number_hit = any(n in source_lower for n in number_terms)
        if not number_hit:
            _log.debug(
                "keyword_verify FAIL (no number match): claim=%r numbers=%r",
                claim[:80],
                number_terms,
            )
            return False

    if ratio < 0.4:
        _log.debug(
            "keyword_verify FAIL (%.0f%% < 40%%): claim=%r matched=%d/%d",
            ratio * 100,
            claim[:80],
            matched,
            len(key_terms),
        )
        return False

    return True


# ---------------------------------------------------------------------------
# 3. LLM verification
# ---------------------------------------------------------------------------


async def _llm_verify(
    claim: str,
    source_content: str,
    source_title: str,
    model: str,
    provider: object | None,
) -> bool:
    """Use an LLM to verify whether *source_content* supports *claim*.

    Returns ``True`` if the source supports the claim, ``False`` otherwise.
    On any error, returns ``True`` (optimistic fallback).
    """
    try:
        from exo import Agent, run

        truncated = source_content[:4000] if len(source_content) > 4000 else source_content
        prompt = (
            f"Does this source contain information that supports or is consistent with "
            f"this claim? The claim does not need to be a verbatim quote — it just needs "
            f"to be reasonably supported by the source content.\n\n"
            f"CLAIM: {claim}\n\n"
            f"SOURCE TITLE: {source_title}\n"
            f"SOURCE CONTENT:\n{truncated}\n\n"
            f"Answer YES if the source supports this claim, or NO if the source clearly "
            f"contradicts it or contains no relevant information. When in doubt, say YES."
        )
        agent = Agent(
            name="citation_verifier",
            model=model,
            instructions="Answer only YES or NO. When in doubt, say YES.",
            temperature=0.0,
            max_steps=1,
        )
        result = await run(agent, prompt, provider=provider)
        answer = result.output.strip().upper()
        return answer.startswith("YES")
    except Exception as exc:
        _log.warning("LLM verify failed (optimistic fallback): %s", exc)
        return True


async def _llm_verify_batch(
    citations: list[tuple[int, str, str, str]],
    model: str,
    provider: object | None,
    max_source_chars: int = 4000,
) -> dict[int, bool]:
    """Verify a batch of citations in parallel using LLM.

    *citations* is a list of ``(citation_index, claim, source_content, source_title)``.
    Returns a dict mapping citation_index -> supported bool.
    """
    if not citations:
        return {}

    async def _verify_one(idx: int, claim: str, content: str, title: str) -> tuple[int, bool]:
        truncated = content[:max_source_chars] if len(content) > max_source_chars else content
        supported = await _llm_verify(claim, truncated, title, model, provider)
        return idx, supported

    tasks = [_verify_one(idx, claim, content, title) for idx, claim, content, title in citations]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    verified: dict[int, bool] = {}
    for r in results:
        if isinstance(r, Exception):
            _log.warning("LLM batch verify exception (optimistic fallback): %s", r)
            continue
        idx, supported = r
        verified[idx] = supported

    return verified


# ---------------------------------------------------------------------------
# 4. Main entry point
# ---------------------------------------------------------------------------


async def verify_citations(
    answer: str,
    sources: list[SearchResult],
    mode: str = "balanced",
    config: SearchConfig | None = None,
) -> tuple[str, CitationVerification]:
    """Verify citations in *answer* against *sources* and return a cleaned copy.

    Three-phase verification (quality/deep modes with LLM enabled):

    1. **Keyword filter** — cheap heuristic for all modes
    2. **LLM spot-check** — keyword-passed citations get LLM verification to
       catch semantic fabrication
    3. **LLM second-chance** — keyword-failed citations get LLM review before
       removal

    Returns:
        A ``(cleaned_answer, verification_stats)`` tuple.
    """
    from ..config import SearchConfig as Cfg

    cfg = config or Cfg()

    citations = _extract_citations(answer)
    if not citations:
        return answer, CitationVerification()

    stats = CitationVerification(total_citations=len(citations))
    use_llm = mode in ("quality", "deep") and cfg.llm_verification

    # Phase 1: keyword filter
    keyword_passed: list[tuple[int, int, str]] = []  # (cite_idx, source_idx, claim)
    keyword_failed: list[tuple[int, int, str]] = []  # (cite_idx, source_idx, claim)
    invalid_markers: set[str] = set()

    for cite_idx, (src_idx, claim) in enumerate(citations):
        if src_idx < 0 or src_idx >= len(sources):
            _log.debug("citation [%d] out of range (sources=%d)", src_idx + 1, len(sources))
            invalid_markers.add(f"[{src_idx + 1}]")
            stats.removed += 1
            stats.failed_claims.append((claim, src_idx + 1))
            continue

        source = sources[src_idx]
        if _keyword_verify(claim, source.content):
            keyword_passed.append((cite_idx, src_idx, claim))
        else:
            keyword_failed.append((cite_idx, src_idx, claim))

    # For non-LLM modes, keyword result is final
    if not use_llm:
        for _cite_idx, _src_idx, _claim in keyword_passed:
            stats.verified += 1
        for _cite_idx, src_idx, claim in keyword_failed:
            if mode in ("quality", "deep"):
                stats.flagged += 1
                _log.info(
                    "citation [%d] flagged for review (mode=%s): claim=%r",
                    src_idx + 1,
                    mode,
                    claim[:80],
                )
            invalid_markers.add(f"[{src_idx + 1}]")
            stats.removed += 1
            stats.failed_claims.append((claim, src_idx + 1))
    else:
        # Phase 2: LLM spot-check keyword-passed citations
        model = cfg.fast_model
        provider = resolve_provider(model)

        spot_check_batch: list[tuple[int, str, str, str]] = []
        for cite_idx, src_idx, claim in keyword_passed:
            source = sources[src_idx]
            spot_check_batch.append((cite_idx, claim, source.content, source.title))

        spot_results = await _llm_verify_batch(
            spot_check_batch, model, provider, cfg.llm_verify_source_chars
        )

        # Safety valve: if LLM rejects >80% of keyword-passed citations,
        # the LLM is likely miscalibrated — fall back to keyword-only results
        llm_reject_count = sum(1 for c in keyword_passed if not spot_results.get(c[0], True))
        if keyword_passed and llm_reject_count / len(keyword_passed) > 0.8:
            _log.warning(
                "LLM rejected %d/%d keyword-passed citations — miscalibrated, "
                "falling back to keyword-only verification",
                llm_reject_count,
                len(keyword_passed),
            )
            for _cite_idx, _src_idx, _claim in keyword_passed:
                stats.verified += 1
        else:
            for cite_idx, src_idx, claim in keyword_passed:
                llm_ok = spot_results.get(cite_idx, True)  # optimistic fallback
                if llm_ok:
                    stats.verified += 1
                    stats.llm_verified += 1
                else:
                    _log.info(
                        "citation [%d] LLM rejected (keyword passed): claim=%r",
                        src_idx + 1,
                        claim[:80],
                    )
                    invalid_markers.add(f"[{src_idx + 1}]")
                    stats.removed += 1
                    stats.failed_claims.append((claim, src_idx + 1))

        # Phase 3: LLM second-chance for keyword-failed citations
        second_chance_batch: list[tuple[int, str, str, str]] = []
        for cite_idx, src_idx, claim in keyword_failed:
            source = sources[src_idx]
            second_chance_batch.append((cite_idx, claim, source.content, source.title))

        second_results = await _llm_verify_batch(
            second_chance_batch, model, provider, cfg.llm_verify_source_chars
        )

        for cite_idx, src_idx, claim in keyword_failed:
            llm_ok = second_results.get(cite_idx, False)  # pessimistic fallback
            if llm_ok:
                _log.info(
                    "citation [%d] LLM rescued (keyword failed): claim=%r",
                    src_idx + 1,
                    claim[:80],
                )
                stats.verified += 1
                stats.llm_verified += 1
            else:
                invalid_markers.add(f"[{src_idx + 1}]")
                stats.removed += 1
                stats.failed_claims.append((claim, src_idx + 1))

    # Build the cleaned answer by stripping bad markers.
    cleaned = answer
    for marker in invalid_markers:
        cleaned = cleaned.replace(marker, "")

    # Collapse any leftover double-spaces from removed markers.
    cleaned = re.sub(r"  +", " ", cleaned)
    # Remove trailing whitespace on lines.
    cleaned = re.sub(r" +\n", "\n", cleaned)

    _log.info(
        "citation verification: total=%d verified=%d removed=%d flagged=%d llm_verified=%d",
        stats.total_citations,
        stats.verified,
        stats.removed,
        stats.flagged,
        stats.llm_verified,
    )
    return cleaned, stats
