"""Follow-up suggestion generator agent — structured output."""

from __future__ import annotations

import json
import re

from exo import Agent, run
from exo.observability.logging import get_logger  # pyright: ignore[reportMissingImports]

from .._utils import resolve_provider
from ..config import SearchConfig
from ..prompts.instructions import get_suggestion_prompt
from ..types import SuggestionOutput

_log = get_logger(__name__)


def _extract_suggestions_from_text(text: str) -> list[str]:
    """Best-effort extraction of suggestions from unstructured LLM output."""
    # Try to find a JSON array embedded in the text
    arr_match = re.search(r"\[([^\[\]]+)\]", text)
    if arr_match:
        try:
            items = json.loads(f"[{arr_match.group(1)}]")
            if isinstance(items, list) and all(isinstance(s, str) for s in items):
                return items[:5]
        except (json.JSONDecodeError, TypeError):
            pass

    # Fall back to numbered lines (e.g. "1. How does..." or "- What are...")
    lines = re.findall(r"(?:^|\n)\s*(?:\d+[.):]\s*|[-*]\s+)(.{15,120})", text)
    return [line.strip().rstrip("?").strip() + "?" for line in lines[:5]]


async def generate_suggestions(
    chat_history: list[tuple[str, str]],
    config: SearchConfig | None = None,
) -> list[str]:
    """Generate follow-up suggestions based on conversation history."""
    _log.debug("generating suggestions history_len=%d", len(chat_history))
    from ..config import SearchConfig as Cfg

    cfg = config or Cfg()

    # Format conversation for the suggestion generator
    parts = []
    for q, a in chat_history:
        parts.append(f"User: {q}")
        short_a = a[:500] + "..." if len(a) > 500 else a
        parts.append(f"Assistant: {short_a}")
    formatted_input = "\n".join(parts)

    suggestion_agent = Agent(
        name="suggestion_generator",
        model=cfg.fast_model,
        instructions=get_suggestion_prompt(),
        output_type=SuggestionOutput,
        temperature=0.8,
        max_steps=1,
    )

    provider = resolve_provider(cfg.fast_model)
    result = await run(suggestion_agent, formatted_input, provider=provider)

    try:
        output = SuggestionOutput.model_validate_json(result.output)
        return output.suggestions
    except Exception:
        _log.debug("suggestion structured parse failed, trying raw JSON")

    # Try to parse as raw JSON object
    try:
        data = json.loads(result.output)
        if isinstance(data, dict) and "suggestions" in data:
            return data["suggestions"][:5]
        if isinstance(data, list):
            return [str(s) for s in data[:5]]
    except (json.JSONDecodeError, TypeError):
        pass

    # Last resort: regex extraction from free-form text
    extracted = _extract_suggestions_from_text(result.output)
    if extracted:
        _log.debug("suggestion extracted %d from text", len(extracted))
        return extracted

    _log.warning("suggestion parse failed completely")
    return []
