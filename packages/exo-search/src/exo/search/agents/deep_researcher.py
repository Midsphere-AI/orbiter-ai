"""Deep researcher — sequential multi-step research for complex queries.

Handles questions that require finding information in step N before knowing
what to search for in step N+1 (e.g., "Who acquired company X, and what was
their revenue?" requires first identifying the acquirer).

Steps are organized as a DAG with dependencies. Independent steps run in
parallel; dependent steps wait for their prerequisites to complete.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from exo import Agent, run
from exo.observability.logging import get_logger  # pyright: ignore[reportMissingImports]

from .._utils import resolve_provider
from ..config import SearchConfig
from ..tools.searxng import search_and_collect
from ..tools.web_fetcher import enrich_results
from ..types import (
    ClassifierOutput,
    PipelineEvent,
    ResearchPlan,
    ResearchStep,
    SearchResult,
    StepExtraction,
)

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Planner prompt
# ---------------------------------------------------------------------------

_PLANNER_PROMPT = """\
You are a research planner. Given a complex question, decompose it into a sequence of \
research steps where later steps may depend on information found in earlier steps.

Rules:
- Each step should be a single, focused search task
- If step B needs information from step A's results to form its search query, \
list A in depends_on
- Steps without dependencies can run in parallel
- Maximum {max_steps} steps
- Each step_id should be a short label like "find_document", "extract_items", \
"check_status"
- The description should be a specific, searchable query or task
- The extraction_goal should describe EXACTLY what information to pull from results
- For questions about documents: first find the document, then search for specific \
content within it
- For questions requiring cross-referencing: search for each entity separately, \
then combine
- For questions with conditional filtering: first find the full list, then filter

CRITICAL — Extraction goals must preserve the user's EXACT filtering language:
- NEVER paraphrase or simplify filtering criteria. Copy the user's words verbatim \
into the extraction goal.
- "specifically marked as X" is NOT the same as "in the X section". The first \
means the word X appears in the item's own name. The second means everything in \
that section. This distinction matters enormously.
- "contain the whole name" means a literal substring match — spell out which names \
to match against.
- Exclusion criteria ("but not if", "excluding") must be stated explicitly in the \
extraction goal, not left implicit.
- When the question applies multiple filters in sequence, make each filter its own \
step with a precise extraction goal. Do not combine filters into one vague step.

Example for "Who acquired company X, and what was their revenue?":
```json
{{
  "steps": [
    {{"step_id": "find_acquirer", "description": "company X acquisition", \
"depends_on": [], "extraction_goal": "Name of the company that acquired X and \
the acquisition date"}},
    {{"step_id": "find_revenue", "description": "[acquirer name] annual revenue", \
"depends_on": ["find_acquirer"], "extraction_goal": "Annual revenue figures"}}
  ],
  "reasoning": "Need to find acquirer first, then look up their revenue"
}}
```

Chat history (if any):
{chat_history}

You MUST respond with ONLY a JSON object in this exact format (no other text):
```json
{{
  "steps": [
    {{"step_id": "...", "description": "...", "depends_on": [], "extraction_goal": "..."}},
    ...
  ],
  "reasoning": "..."
}}
```

Decompose the following question into research steps:
"""

# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """\
You are a research assistant extracting specific information from search results.

Extraction goal: {extraction_goal}

Search results:
{results_text}

Instructions:
- Extract ONLY information directly stated in the search results above
- Be specific — include exact names, numbers, dates, lists
- If the results contain a document or list, transcribe the relevant parts exactly
- Apply filtering criteria LITERALLY as worded in the extraction goal. \
"In the X section" means everything in that section. "Whose name contains X" \
means only items where X appears in the item name. These are different — read \
the extraction goal carefully and apply exactly what it says.
- When filtering a list, go through EACH item and check whether it matches the \
criteria. List only those that pass. State the total count at the end.
- Say NOT FOUND only if the information is genuinely not present in any result
- When listing items, use exact names as they appear in the source
"""


# ---------------------------------------------------------------------------
# 1. Plan research
# ---------------------------------------------------------------------------


async def _plan_research(
    query: str,
    chat_history: list[tuple[str, str]],
    config: SearchConfig,
) -> ResearchPlan:
    """Decompose a complex query into a DAG of sequential research steps."""
    _log.debug("planning research for query=%r", query)

    history_lines: list[str] = []
    for q, a in chat_history:
        history_lines.append(f"User: {q}")
        short_a = a[:500] + "..." if len(a) > 500 else a
        history_lines.append(f"Assistant: {short_a}")
    chat_block = "\n".join(history_lines) if history_lines else "(none)"

    instructions = _PLANNER_PROMPT.format(
        max_steps=config.max_deep_research_steps,
        chat_history=chat_block,
    )

    # Use main model for planning — plan quality is critical
    planner = Agent(
        name="research-planner",
        model=config.model,
        instructions=instructions,
        temperature=0.2,
        max_steps=1,
    )

    provider = resolve_provider(config.model)
    result = await run(planner, query, provider=provider)

    try:
        # Extract JSON from the response (may be wrapped in markdown code blocks)
        raw = result.output or ""
        import json as _json
        import re as _re

        json_match = _re.search(r"\{[\s\S]*\}", raw)
        if json_match:
            plan_data = _json.loads(json_match.group())
            plan = ResearchPlan.model_validate(plan_data)
        else:
            plan = ResearchPlan.model_validate_json(raw)
        _log.info(
            "research plan: %d steps, reasoning=%r",
            len(plan.steps),
            plan.reasoning[:120] if plan.reasoning else "",
        )
    except Exception as exc:
        _log.warning("plan parse failed (%s), falling back to single step", exc)
        plan = ResearchPlan(
            steps=[
                ResearchStep(
                    step_id="search",
                    description=query,
                    depends_on=[],
                    extraction_goal="Find the answer to the question",
                )
            ],
            reasoning="Fallback: single-step plan due to parse failure",
        )

    # Clamp to max steps
    if len(plan.steps) > config.max_deep_research_steps:
        plan = ResearchPlan(
            steps=plan.steps[: config.max_deep_research_steps],
            reasoning=plan.reasoning,
        )

    return plan


# ---------------------------------------------------------------------------
# 2. Execute a single step
# ---------------------------------------------------------------------------


async def _execute_step(
    step: ResearchStep,
    prior_context: dict[str, str],
    config: SearchConfig,
) -> tuple[list[SearchResult], StepExtraction]:
    """Execute a single research step: search, enrich, and extract.

    Args:
        step: The research step to execute.
        prior_context: Mapping of step_id -> extracted_info from completed deps.
        config: Search configuration.

    Returns:
        Tuple of (search results, extraction with findings).
    """
    _log.debug("executing step=%s deps=%s", step.step_id, step.depends_on)

    # Build search queries incorporating context from dependencies
    queries: list[str] = [step.description]
    if prior_context:
        # Filter out "NOT FOUND" — don't poison queries with failure markers
        context_parts = [
            v
            for dep in step.depends_on
            if (v := prior_context.get(dep)) and "NOT FOUND" not in v.upper()
        ]
        if context_parts:
            context_str = "; ".join(context_parts)
            enriched_query = f"{step.description} {context_str}"
            # Trim if excessively long
            if len(enriched_query) > 200:
                enriched_query = enriched_query[:200].rsplit(" ", 1)[0]
            queries = [enriched_query, step.description]
            # Add a third angle combining extraction goal with context
            goal_query = f"{step.extraction_goal} {context_parts[0]}"
            if len(goal_query) > 200:
                goal_query = goal_query[:200].rsplit(" ", 1)[0]
            queries.append(goal_query)

    queries = queries[:3]
    _log.debug("step=%s queries=%s", step.step_id, queries)

    # Search
    raw = await search_and_collect(queries)
    results = [
        SearchResult(
            title=r.get("title", ""),
            url=r.get("url", ""),
            content=r.get("content", ""),
            enriched=r.get("enriched", False),
        )
        for r in raw
    ]

    # Enrich top results with full page content
    if results and (config.jina_reader_url or config.jina_api_key):
        results = await enrich_results(
            results,
            config.jina_reader_url,
            max_results=config.deep_research_enrich_per_step,
            jina_api_key=config.jina_api_key,
        )

    # Build results text for extraction — prioritize enriched (full page) results
    # and give them more space since they have the actual content
    results_lines: list[str] = []
    total_chars = 0
    max_extraction_chars = 20_000
    # Sort: enriched results first (they have real content)
    sorted_results = sorted(results[:15], key=lambda r: (not r.enriched, -len(r.content)))
    for i, r in enumerate(sorted_results, 1):
        # Give enriched results up to 8K, snippets up to 1K
        char_limit = 8000 if r.enriched else 1000
        content = r.content[:char_limit] if r.content else r.title
        line = f"[{i}] {r.title} | {r.url}\n{content}"
        if total_chars + len(line) > max_extraction_chars:
            break
        results_lines.append(line)
        total_chars += len(line)
    results_text = "\n\n".join(results_lines) if results_lines else "(no results found)"

    # Extract specific information using LLM
    extraction_instructions = _EXTRACTION_PROMPT.format(
        extraction_goal=step.extraction_goal,
        results_text=results_text,
    )

    try:
        extractor = Agent(
            name=f"extractor-{step.step_id}",
            model=config.fast_model,
            instructions=extraction_instructions,
            temperature=0.1,
            max_steps=1,
        )

        provider = resolve_provider(config.fast_model)
        extract_result = await run(extractor, step.extraction_goal, provider=provider)
        extracted_info = str(extract_result.output) if extract_result.output else "NOT FOUND"
    except Exception as exc:
        _log.warning("extraction failed for step %s: %s", step.step_id, exc)
        # Fall back to using the raw results titles as extracted info
        extracted_info = "; ".join(r.title for r in results[:5]) if results else "NOT FOUND"

    found = "NOT FOUND" not in extracted_info.upper()

    extraction = StepExtraction(
        step_id=step.step_id,
        extracted_info=extracted_info,
        found=found,
    )

    _log.info(
        "step=%s results=%d found=%s extracted=%r",
        step.step_id,
        len(results),
        found,
        extracted_info[:100],
    )

    return results, extraction


# ---------------------------------------------------------------------------
# 3. Build research narrative
# ---------------------------------------------------------------------------


def _build_research_narrative(
    plan: ResearchPlan,
    extractions: dict[str, StepExtraction],
) -> str:
    """Build a human-readable narrative from completed research steps."""
    found_count = sum(1 for e in extractions.values() if e.found)
    total = len(plan.steps)
    lines = [
        "## Research Chain",
        f"({found_count}/{total} steps found information)",
    ]
    if found_count == 0:
        lines.append(
            "WARNING: No research steps found the requested information. "
            "The answer below relies entirely on raw search results. "
            "Be explicit about what was NOT found and DO NOT fill gaps with "
            "unsourced information."
        )
    for i, step in enumerate(plan.steps, 1):
        ext = extractions.get(step.step_id)
        if ext:
            status = "Found" if ext.found else "NOT FOUND"
            lines.append(f"Step {i} ({step.step_id}): {status}. {ext.extracted_info}")
        else:
            lines.append(f"Step {i} ({step.step_id}): Skipped or failed.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Topological sort helper
# ---------------------------------------------------------------------------


def _topological_layers(steps: list[ResearchStep]) -> list[list[ResearchStep]]:
    """Sort steps into layers where each layer's deps are met by prior layers.

    Steps within the same layer have no mutual dependencies and can run in
    parallel. Returns layers in execution order.
    """
    completed: set[str] = set()
    remaining = list(steps)
    layers: list[list[ResearchStep]] = []

    while remaining:
        layer: list[ResearchStep] = []
        still_remaining: list[ResearchStep] = []
        for step in remaining:
            if all(dep in completed for dep in step.depends_on):
                layer.append(step)
            else:
                still_remaining.append(step)
        if not layer:
            # Circular dependency or missing deps — force remaining into one layer
            _log.warning(
                "topological sort stuck, forcing %d remaining steps",
                len(still_remaining),
            )
            layer = still_remaining
            still_remaining = []
        completed.update(s.step_id for s in layer)
        layers.append(layer)
        remaining = still_remaining

    return layers


# ---------------------------------------------------------------------------
# 4. Main entry point — deep_research
# ---------------------------------------------------------------------------


async def deep_research(
    query: str,
    classification: ClassifierOutput,
    chat_history: list[tuple[str, str]],
    config: SearchConfig,
    sub_questions: list[str] | None = None,
    seed_results: list[SearchResult] | None = None,
) -> tuple[list[SearchResult], str]:
    """Run sequential multi-step deep research.

    Decomposes the query into a DAG of research steps, executes them in
    dependency order (parallelizing independent steps), and returns merged
    results with a research narrative for the writer.

    Args:
        query: The user's question.
        classification: Classifier output (not directly used but kept for API
            consistency with other research functions).
        chat_history: Prior conversation turns.
        config: Search configuration.
        sub_questions: Optional sub-questions from the classifier.
        seed_results: Pre-fetched results to include in the final set.

    Returns:
        Tuple of (deduplicated search results, research narrative string).
    """
    _log.info("deep_research query=%r", query)

    # Plan
    plan = await _plan_research(query, chat_history, config)

    # Execute in topological order
    layers = _topological_layers(plan.steps)
    all_results: list[SearchResult] = list(seed_results) if seed_results else []
    extractions: dict[str, StepExtraction] = {}
    prior_context: dict[str, str] = {}

    for layer_idx, layer in enumerate(layers):
        _log.debug(
            "deep_research layer=%d/%d steps=%s",
            layer_idx + 1,
            len(layers),
            [s.step_id for s in layer],
        )

        async def _run_step(step: ResearchStep) -> tuple[list[SearchResult], StepExtraction]:
            return await _execute_step(step, prior_context, config)

        step_results = await asyncio.gather(
            *[_run_step(step) for step in layer],
            return_exceptions=True,
        )

        for step, result in zip(layer, step_results, strict=True):
            if isinstance(result, Exception):
                _log.warning("step %s failed: %s", step.step_id, result)
                extraction = StepExtraction(
                    step_id=step.step_id,
                    extracted_info=f"Step failed: {result}",
                    found=False,
                )
                extractions[step.step_id] = extraction
            else:
                results, extraction = result
                all_results.extend(results)
                extractions[step.step_id] = extraction
                prior_context[step.step_id] = extraction.extracted_info

    # Deduplicate by URL
    seen_urls: set[str] = set()
    merged: list[SearchResult] = []
    for r in all_results:
        if r.url and r.url not in seen_urls:
            seen_urls.add(r.url)
            merged.append(r)

    narrative = _build_research_narrative(plan, extractions)
    _log.info("deep_research complete steps=%d results=%d", len(plan.steps), len(merged))

    return merged, narrative


# ---------------------------------------------------------------------------
# 5. Streaming variant
# ---------------------------------------------------------------------------


async def stream_deep_research(
    query: str,
    classification: ClassifierOutput,
    chat_history: list[tuple[str, str]],
    config: SearchConfig,
    sub_questions: list[str] | None = None,
    seed_results: list[SearchResult] | None = None,
) -> AsyncIterator[PipelineEvent | SearchResult]:
    """Stream deep research events and results.

    Yields PipelineEvent objects for each stage transition and SearchResult
    objects for all collected results at the end.

    Args:
        query: The user's question.
        classification: Classifier output.
        chat_history: Prior conversation turns.
        config: Search configuration.
        sub_questions: Optional sub-questions from the classifier.
        seed_results: Pre-fetched results to include in the final set.

    Yields:
        PipelineEvent for planning/step progress, then SearchResult for each
        collected result.
    """
    _log.info("stream_deep_research query=%r", query)

    # Planning phase
    yield PipelineEvent(
        stage="deep_research",
        status="planning",
        message="Decomposing query into research steps",
    )
    plan = await _plan_research(query, chat_history, config)
    yield PipelineEvent(
        stage="deep_research",
        status="planned",
        message=f"{len(plan.steps)} research steps planned",
    )

    # Execute in topological order
    layers = _topological_layers(plan.steps)
    all_results: list[SearchResult] = list(seed_results) if seed_results else []
    extractions: dict[str, StepExtraction] = {}
    prior_context: dict[str, str] = {}
    total_steps = len(plan.steps)
    step_counter = 0

    for _layer_idx, layer in enumerate(layers):

        async def _run_step(step: ResearchStep) -> tuple[list[SearchResult], StepExtraction]:
            return await _execute_step(step, prior_context, config)

        # Yield start events for each step in this layer
        for step in layer:
            step_counter += 1
            yield PipelineEvent(
                stage="deep_research",
                status="step_started",
                message=f"Step {step_counter}/{total_steps}: {step.description}",
            )

        step_results = await asyncio.gather(
            *[_run_step(step) for step in layer],
            return_exceptions=True,
        )

        # Process results and yield completion events
        completion_counter = step_counter - len(layer)
        for step, result in zip(layer, step_results, strict=True):
            completion_counter += 1
            if isinstance(result, Exception):
                _log.warning("step %s failed: %s", step.step_id, result)
                extraction = StepExtraction(
                    step_id=step.step_id,
                    extracted_info=f"Step failed: {result}",
                    found=False,
                )
                extractions[step.step_id] = extraction
                yield PipelineEvent(
                    stage="deep_research",
                    status="step_completed",
                    message=f"Step {completion_counter}: Failed - {result}",
                )
            else:
                results, extraction = result
                all_results.extend(results)
                extractions[step.step_id] = extraction
                prior_context[step.step_id] = extraction.extracted_info
                summary = extraction.extracted_info[:100]
                yield PipelineEvent(
                    stage="deep_research",
                    status="step_completed",
                    message=f"Step {completion_counter}: {summary}",
                )

    # Final completion event
    yield PipelineEvent(
        stage="deep_research",
        status="completed",
        message=f"{total_steps} steps completed",
    )

    # Deduplicate and yield all results
    seen_urls: set[str] = set()
    for r in all_results:
        if r.url and r.url not in seen_urls:
            seen_urls.add(r.url)
            yield r
