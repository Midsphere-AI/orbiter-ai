"""Classifier agent — determines search intent using Exo Search's classification schema."""

from __future__ import annotations

from exo import Agent, run
from exo.observability.logging import get_logger  # pyright: ignore[reportMissingImports]

from .._utils import resolve_provider
from ..config import SearchConfig
from ..prompts.instructions import CLASSIFIER_PROMPT
from ..types import ClassifierOutput

_log = get_logger(__name__)


async def classify(
    query: str, chat_history: list[tuple[str, str]], config: SearchConfig
) -> ClassifierOutput:
    """Classify a query using Exo Search's classifier prompt."""
    _log.debug("classifying query=%r history_len=%d", query, len(chat_history))
    # Format chat history for the classifier
    parts = []
    for q, a in chat_history:
        parts.append(f"User: {q}")
        short_a = a[:500] + "..." if len(a) > 500 else a
        parts.append(f"Assistant: {short_a}")
    parts.append(f"User: {query}")
    formatted_input = "\n".join(parts) if parts else query

    classifier = Agent(
        name="classifier",
        model=config.fast_model,
        instructions=CLASSIFIER_PROMPT,
        output_type=ClassifierOutput,
        temperature=0.2,
        max_steps=1,
    )

    provider = resolve_provider(config.fast_model)
    result = await run(classifier, formatted_input, provider=provider)

    try:
        output = ClassifierOutput.model_validate_json(result.output)
        _log.info(
            "classified skip=%s follow_up=%r subs=%d",
            output.classification.skip_search,
            output.standalone_follow_up,
            len(output.sub_questions),
        )
        return output
    except Exception as exc:
        _log.warning(
            "classifier parse failed (%s), raw=%r, using fallback",
            exc,
            (result.output or "")[:200],
        )
        # Fallback: no skip, just web search
        from ..types import Classification

        return ClassifierOutput(
            classification=Classification(),
            standalone_follow_up=query,
        )
