"""Pipeline — programmatic orchestration matching Exo Search's SearchAgent.searchAsync()."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace

from exo._internal.errors import unwrap_exception_group  # pyright: ignore[reportMissingImports]
from exo.observability.logging import get_logger  # pyright: ignore[reportMissingImports]
from exo.types import StreamEvent

from .agents.classifier import classify
from .agents.deep_researcher import deep_research
from .agents.query_planner import hybrid_research
from .agents.researcher import direct_search
from .agents.suggestion_generator import generate_suggestions
from .agents.writer import revise_answer, stream_write_answer, write_answer
from .config import SearchConfig, compute_context_budget
from .errors import SearchError
from .tools.citation_verifier import verify_citations
from .tools.confidence import compute_confidence
from .tools.contradiction_detector import detect_contradictions
from .tools.embeddings import rerank_search_results
from .tools.searxng import configure_search_keys
from .tools.web_fetcher import enrich_results
from .types import (
    CitationVerification,
    ClassifierOutput,
    PipelineEvent,
    ResearchMode,
    SearchResponse,
    SearchResult,
    Source,
)

_VALID_MODES: frozenset[str] = frozenset(m.value for m in ResearchMode)


def _normalize_mode(mode: str) -> str:
    """Normalize and validate a mode string.

    Accepts ``ResearchMode`` enum values or plain strings (case-insensitive).
    Raises ``ValueError`` with a teaching message on unknown modes.
    """
    low = mode.lower()
    if low in _VALID_MODES:
        return low
    valid_list = ", ".join(sorted(_VALID_MODES))
    raise SearchError(
        f"Unknown search mode: {mode!r}.",
        context={"mode": mode, "valid_modes": valid_list},
        hint=f"Use one of the valid mode strings: {valid_list}. Example: search('query', mode='balanced')",
    )


_log = get_logger(__name__)


def _should_use_deep_research(classification: ClassifierOutput, mode: str) -> bool:
    """Determine whether to use deep sequential research based on classification and mode."""
    if mode == "deep":
        return True
    if mode == "speed":
        return False
    if not classification.requires_sequential_research:
        return False
    if mode == "quality":
        return True
    return mode == "balanced" and classification.estimated_complexity == "complex"


async def run_search_pipeline(
    query: str,
    chat_history: list[tuple[str, str]] | None = None,
    mode: str = "balanced",
    system_instructions: str = "",
    config: SearchConfig | None = None,
) -> SearchResponse:
    """Run the full Exo Search search pipeline.

    Matches Exo Search's SearchAgent.searchAsync():
    1. Classify query
    2. If skipSearch, go directly to writer with no context
    3. Otherwise, run researcher (iterative tool calls)
    4. Run writer with context from research
    5. Run suggestion generator
    """
    mode = _normalize_mode(mode)
    cfg = config or SearchConfig()
    history = chat_history or []
    configure_search_keys(cfg.serper_api_key, cfg.jina_api_key, cfg.searxng_url)
    _log.info("pipeline query=%r mode=%s", query, mode)

    # Step 1 + 2: Classify and research
    search_results: list[SearchResult] = []
    research_narrative = ""

    if mode == "speed":
        # Fast path: classifier and direct search in parallel
        classification_r, search_results_r = await asyncio.gather(
            classify(query, history, cfg),
            direct_search(query),
            return_exceptions=True,
        )
        if isinstance(classification_r, BaseException):
            _log.warning("classifier failed in speed mode: %s", classification_r)
            raise SearchError(
                "Classifier failed in speed mode.",
                context={"stage": "classifier", "mode": mode, "query": query},
                hint="Check model credentials and network. Try mode='speed' to use a lighter path.",
            ) from unwrap_exception_group(classification_r)
        classification = classification_r
        if isinstance(search_results_r, BaseException):
            _log.warning("direct_search failed in speed mode: %s", search_results_r)
            search_results = []
        else:
            search_results = search_results_r
    else:
        # Speculative search: start a raw-query search concurrently with
        # classification so the first adaptive round sees existing results
        classification_r, seed_results_r = await asyncio.gather(
            classify(query, history, cfg),
            direct_search(query),
            return_exceptions=True,
        )
        if isinstance(classification_r, BaseException):
            _log.warning("classifier failed: %s", classification_r)
            raise SearchError(
                "Classifier failed.",
                context={"stage": "classifier", "mode": mode, "query": query},
                hint="Check model credentials and network. Try mode='speed' to skip the classifier for fast fallback.",
            ) from unwrap_exception_group(classification_r)
        classification = classification_r
        if isinstance(seed_results_r, BaseException):
            _log.warning("direct_search failed (seed): %s", seed_results_r)
            seed_results = []
        else:
            seed_results = seed_results_r
        effective_query = classification.standalone_follow_up or query

        if not classification.classification.skip_search:
            use_deep = _should_use_deep_research(classification, mode)
            _log.info(
                "routing decision: use_deep=%s sequential=%s complexity=%s mode=%s",
                use_deep,
                classification.requires_sequential_research,
                classification.estimated_complexity,
                mode,
            )
            if use_deep:
                _log.info("using deep sequential research")
                search_results, research_narrative = await deep_research(
                    query=effective_query,
                    classification=classification,
                    chat_history=history,
                    config=cfg,
                    sub_questions=classification.sub_questions,
                    seed_results=seed_results,
                )
            else:
                search_results = await hybrid_research(
                    query=effective_query,
                    classification=classification,
                    chat_history=history,
                    mode=mode,
                    config=cfg,
                    sub_questions=classification.sub_questions,
                    seed_results=seed_results,
                )

    effective_query = classification.standalone_follow_up or query
    _log.info(
        "classifier done skip_search=%s results=%d",
        classification.classification.skip_search,
        len(search_results),
    )

    # Rerank by relevance (skip for speed — not worth the latency)
    if mode != "speed" and search_results:
        _log.debug("reranking %d results", len(search_results))
        search_results = await rerank_search_results(effective_query, search_results)

    # Speed mode uses fast model for writing too
    writer_cfg = replace(cfg, model=cfg.fast_model) if mode == "speed" else cfg

    # Compute context budget based on model context window
    history_text = " ".join(f"{q} {a}" for q, a in history)
    budget_sources, budget_chars, budget_enrich = compute_context_budget(
        model=writer_cfg.model,
        mode=mode,
        chat_history_chars=len(history_text),
        narrative_chars=len(research_narrative),
        context_window_override=cfg.context_window_tokens,
    )

    # Cap sources using computed budget
    writer_cap = min(budget_sources, cfg.max_writer_sources)
    writer_results = search_results[:writer_cap]

    # Start suggestions early (independent of enrichment and writing)
    suggest_task = asyncio.create_task(generate_suggestions([*history, (query, "")], cfg))

    try:
        # Enrich top results with full page content (skip for speed)
        enrich_cap = budget_enrich
        content_chars = budget_chars
        if enrich_cap > 0 and writer_results:
            _log.debug("enriching top %d results (max_chars=%d)", enrich_cap, content_chars)
            writer_results = await enrich_results(
                writer_results,
                cfg.jina_reader_url,
                max_results=enrich_cap,
                jina_api_key=cfg.jina_api_key,
                query=effective_query,
                max_chars=content_chars,
            )

        # Step 3: Write answer
        _log.debug(
            "writing answer sources=%d narrative=%d", len(writer_results), len(research_narrative)
        )
        answer = await write_answer(
            query=effective_query,
            search_results=writer_results,
            chat_history=history,
            system_instructions=system_instructions or cfg.system_instructions,
            mode=mode,
            config=writer_cfg,
            research_narrative=research_narrative,
        )

        # Step 3b: Verify citations and detect contradictions
        if mode in ("quality", "deep"):
            # Run citation verification and contradiction detection in parallel
            verify_r, contradict_r = await asyncio.gather(
                verify_citations(answer, writer_results, mode, config=cfg),
                detect_contradictions(answer, writer_results, mode, cfg),
                return_exceptions=True,
            )
            if isinstance(verify_r, BaseException):
                _log.warning(
                    "citation verification failed, skipping: %s", verify_r, exc_info=verify_r
                )
                verification = CitationVerification()
            else:
                answer, verification = verify_r
            if isinstance(contradict_r, BaseException):
                _log.warning("contradiction detection failed, skipping: %s", contradict_r)
                contradiction_report = None
            else:
                contradiction_report = contradict_r
        else:
            answer, verification = await verify_citations(answer, writer_results, mode, config=cfg)
            contradiction_report = None
        _log.info(
            "citation verification total=%d verified=%d removed=%d",
            verification.total_citations,
            verification.verified,
            verification.removed,
        )

        # Step 3c: Write-verify-revise loop
        revision_count = 0
        while (
            verification.total_citations > 0
            and verification.removed / verification.total_citations > cfg.revision_threshold
            and revision_count < cfg.max_revision_rounds
            and verification.failed_claims
        ):
            revision_count += 1
            _log.info(
                "revision round %d: removed=%d/%d (%.0f%% > %.0f%%)",
                revision_count,
                verification.removed,
                verification.total_citations,
                100 * verification.removed / verification.total_citations,
                100 * cfg.revision_threshold,
            )
            answer = await revise_answer(
                query=effective_query,
                original_answer=answer,
                failed_claims=verification.failed_claims,
                search_results=writer_results,
                chat_history=history,
                system_instructions=system_instructions or cfg.system_instructions,
                mode=mode,
                config=writer_cfg,
                research_narrative=research_narrative,
            )
            answer, verification = await verify_citations(answer, writer_results, mode, config=cfg)
            _log.info(
                "post-revision verification: total=%d verified=%d removed=%d",
                verification.total_citations,
                verification.verified,
                verification.removed,
            )
        verification.revision_count = revision_count

        if contradiction_report and contradiction_report.has_contradictions:
            _log.info("contradictions detected: %d", len(contradiction_report.contradictions))

        # Step 4: Get suggestions (should be done by now)
        suggestions = await suggest_task
        _log.info("writer done answer_len=%d", len(answer))
    except BaseException:
        suggest_task.cancel()
        raise

    # Sources list matches what the writer cited (same order, same indices)
    sources = [Source(title=r.title, url=r.url, content=r.content) for r in writer_results]
    _log.info("pipeline complete sources=%d suggestions=%d", len(sources), len(suggestions))

    # Step 5: Confidence scoring
    confidence_score, confidence_breakdown = compute_confidence(
        verification=verification,
        sources=writer_results,
        sub_questions=classification.sub_questions,
        answer=answer,
    )

    return SearchResponse(
        answer=answer,
        sources=sources,
        suggestions=suggestions,
        query=effective_query,
        mode=mode,
        verification=verification,
        contradictions=contradiction_report,
        confidence=confidence_score,
        confidence_breakdown=confidence_breakdown.model_dump(),
    )


async def stream_search_pipeline(
    query: str,
    chat_history: list[tuple[str, str]] | None = None,
    mode: str = "balanced",
    system_instructions: str = "",
    config: SearchConfig | None = None,
) -> AsyncIterator[PipelineEvent | StreamEvent | SearchResponse]:
    """Stream the full search pipeline.

    Yields:
        PipelineEvent — stage transitions (classifier started/completed, etc.)
        StreamEvent — exo events from researcher (ToolCallEvent, TextEvent, etc.)
                      and writer (TextEvent for each token)
        SearchResponse — final complete response as the last item
    """
    mode = _normalize_mode(mode)
    cfg = config or SearchConfig()
    history = chat_history or []
    configure_search_keys(cfg.serper_api_key, cfg.jina_api_key, cfg.searxng_url)
    _log.info("stream pipeline query=%r mode=%s", query, mode)

    # Step 1 + 2: Classify and research
    search_results: list[SearchResult] = []
    research_narrative = ""

    if mode == "speed":
        yield PipelineEvent(stage="classifier", status="started")
        classification_r, search_results_r = await asyncio.gather(
            classify(query, history, cfg),
            direct_search(query),
            return_exceptions=True,
        )
        if isinstance(classification_r, BaseException):
            _log.warning("classifier failed in speed mode (stream): %s", classification_r)
            raise SearchError(
                "Classifier failed in speed mode.",
                context={"stage": "classifier", "mode": mode, "query": query},
                hint="Check model credentials and network. Try mode='speed' to use a lighter path.",
            ) from unwrap_exception_group(classification_r)
        classification = classification_r
        if isinstance(search_results_r, BaseException):
            _log.warning("direct_search failed in speed mode (stream): %s", search_results_r)
            search_results = []
        else:
            search_results = search_results_r
        yield PipelineEvent(stage="classifier", status="completed")
        yield PipelineEvent(
            stage="researcher",
            status="completed",
            message=f"{len(search_results)} results (direct)",
        )
    else:
        yield PipelineEvent(stage="classifier", status="started")
        classification_r, seed_results_r = await asyncio.gather(
            classify(query, history, cfg),
            direct_search(query),
            return_exceptions=True,
        )
        if isinstance(classification_r, BaseException):
            _log.warning("classifier failed (stream): %s", classification_r)
            raise SearchError(
                "Classifier failed.",
                context={"stage": "classifier", "mode": mode, "query": query},
                hint="Check model credentials and network. Try mode='speed' to skip the classifier for fast fallback.",
            ) from unwrap_exception_group(classification_r)
        classification = classification_r
        if isinstance(seed_results_r, BaseException):
            _log.warning("direct_search failed (stream seed): %s", seed_results_r)
            seed_results = []
        else:
            seed_results = seed_results_r
        effective_query = classification.standalone_follow_up or query
        yield PipelineEvent(
            stage="classifier",
            status="completed",
            message=f"skip_search={classification.classification.skip_search}",
        )
        if not classification.classification.skip_search:
            use_deep = _should_use_deep_research(classification, mode)
            _log.info(
                "routing decision: use_deep=%s sequential=%s complexity=%s mode=%s",
                use_deep,
                classification.requires_sequential_research,
                classification.estimated_complexity,
                mode,
            )
            if use_deep:
                _log.info("using deep sequential research (streaming)")
                yield PipelineEvent(
                    stage="deep_research",
                    status="started",
                    message="Decomposing query into sequential research steps",
                )
                search_results, research_narrative = await deep_research(
                    query=effective_query,
                    classification=classification,
                    chat_history=history,
                    config=cfg,
                    sub_questions=classification.sub_questions,
                    seed_results=seed_results,
                )
                yield PipelineEvent(
                    stage="deep_research",
                    status="completed",
                    message=f"{len(search_results)} results from sequential research",
                )
            else:
                yield PipelineEvent(stage="researcher", status="started")
                search_results = await hybrid_research(
                    query=effective_query,
                    classification=classification,
                    chat_history=history,
                    mode=mode,
                    config=cfg,
                    sub_questions=classification.sub_questions,
                    seed_results=seed_results,
                )
                yield PipelineEvent(
                    stage="researcher",
                    status="completed",
                    message=f"{len(search_results)} results",
                )
    effective_query = classification.standalone_follow_up or query

    # Rerank by relevance (skip for speed)
    if mode != "speed" and search_results:
        _log.debug("reranking %d results", len(search_results))
        search_results = await rerank_search_results(effective_query, search_results)

    # Speed mode uses fast model for writing too
    writer_cfg = replace(cfg, model=cfg.fast_model) if mode == "speed" else cfg

    # Compute context budget based on model context window
    history_text = " ".join(f"{q} {a}" for q, a in history)
    budget_sources, budget_chars, budget_enrich = compute_context_budget(
        model=writer_cfg.model,
        mode=mode,
        chat_history_chars=len(history_text),
        narrative_chars=len(research_narrative),
        context_window_override=cfg.context_window_tokens,
    )

    # Cap sources using computed budget
    writer_cap = min(budget_sources, cfg.max_writer_sources)
    writer_results = search_results[:writer_cap]

    # Start suggestions concurrently (independent of enrichment and writing)
    suggest_task = asyncio.create_task(generate_suggestions([*history, (query, "")], cfg))

    try:
        # Enrich top results with full page content (skip for speed)
        enrich_cap = budget_enrich
        content_chars = budget_chars
        if enrich_cap > 0 and writer_results:
            _log.debug("enriching top %d results (max_chars=%d)", enrich_cap, content_chars)
            yield PipelineEvent(stage="enrichment", status="started")
            writer_results = await enrich_results(
                writer_results,
                cfg.jina_reader_url,
                max_results=enrich_cap,
                jina_api_key=cfg.jina_api_key,
                query=effective_query,
                max_chars=content_chars,
            )
            yield PipelineEvent(
                stage="enrichment",
                status="completed",
                message=f"{min(enrich_cap, len(writer_results))} pages scraped",
            )

        # Step 3: Write answer (streaming text tokens)
        yield PipelineEvent(stage="writer", status="started")
        answer_parts: list[str] = []
        async for event in stream_write_answer(
            query=effective_query,
            search_results=writer_results,
            chat_history=history,
            system_instructions=system_instructions or cfg.system_instructions,
            mode=mode,
            config=writer_cfg,
            research_narrative=research_narrative,
        ):
            from exo.types import TextEvent

            if isinstance(event, TextEvent):
                answer_parts.append(event.text)
            yield event
        answer = "".join(answer_parts)
        yield PipelineEvent(stage="writer", status="completed")

        # Step 3b: Verify citations and detect contradictions
        if mode in ("quality", "deep"):
            # Run citation verification and contradiction detection in parallel
            verify_r, contradict_r = await asyncio.gather(
                verify_citations(answer, writer_results, mode, config=cfg),
                detect_contradictions(answer, writer_results, mode, cfg),
                return_exceptions=True,
            )
            if isinstance(verify_r, BaseException):
                _log.warning(
                    "citation verification failed (stream), skipping: %s",
                    verify_r,
                    exc_info=verify_r,
                )
                verification = CitationVerification()
            else:
                answer, verification = verify_r
            if isinstance(contradict_r, BaseException):
                _log.warning("contradiction detection failed (stream), skipping: %s", contradict_r)
                contradiction_report = None
            else:
                contradiction_report = contradict_r
        else:
            answer, verification = await verify_citations(answer, writer_results, mode, config=cfg)
            contradiction_report = None
        _log.info(
            "citation verification total=%d verified=%d removed=%d",
            verification.total_citations,
            verification.verified,
            verification.removed,
        )

        # Step 3c: Write-verify-revise loop (revisions run silently, not streamed)
        revision_count = 0
        while (
            verification.total_citations > 0
            and verification.removed / verification.total_citations > cfg.revision_threshold
            and revision_count < cfg.max_revision_rounds
            and verification.failed_claims
        ):
            revision_count += 1
            _log.info(
                "revision round %d: removed=%d/%d (%.0f%% > %.0f%%)",
                revision_count,
                verification.removed,
                verification.total_citations,
                100 * verification.removed / verification.total_citations,
                100 * cfg.revision_threshold,
            )
            yield PipelineEvent(
                stage="revision",
                status="started",
                message=f"Revising answer (round {revision_count})",
            )
            answer = await revise_answer(
                query=effective_query,
                original_answer=answer,
                failed_claims=verification.failed_claims,
                search_results=writer_results,
                chat_history=history,
                system_instructions=system_instructions or cfg.system_instructions,
                mode=mode,
                config=writer_cfg,
                research_narrative=research_narrative,
            )
            answer, verification = await verify_citations(answer, writer_results, mode, config=cfg)
            yield PipelineEvent(
                stage="revision",
                status="completed",
                message=(
                    f"Round {revision_count}: {verification.verified}/{verification.total_citations}"
                    f" citations verified"
                ),
            )
            _log.info(
                "post-revision verification: total=%d verified=%d removed=%d",
                verification.total_citations,
                verification.verified,
                verification.removed,
            )
        verification.revision_count = revision_count

        if contradiction_report and contradiction_report.has_contradictions:
            _log.info("contradictions detected: %d", len(contradiction_report.contradictions))

        # Step 4: Suggestions (should be done by now)
        yield PipelineEvent(stage="suggestions", status="started")
        suggestions = await suggest_task
        yield PipelineEvent(stage="suggestions", status="completed")
    except BaseException:
        suggest_task.cancel()
        raise

    _log.info("writer done answer_len=%d", len(answer))

    # Sources list matches what the writer cited (same order, same indices)
    sources = [Source(title=r.title, url=r.url, content=r.content) for r in writer_results]
    _log.info("stream pipeline complete sources=%d suggestions=%d", len(sources), len(suggestions))

    # Step 5: Confidence scoring
    confidence_score, confidence_breakdown = compute_confidence(
        verification=verification,
        sources=writer_results,
        sub_questions=classification.sub_questions,
        answer=answer,
    )

    yield SearchResponse(
        answer=answer,
        sources=sources,
        suggestions=suggestions,
        query=effective_query,
        mode=mode,
        verification=verification,
        contradictions=contradiction_report,
        confidence=confidence_score,
        confidence_breakdown=confidence_breakdown.model_dump(),
    )
