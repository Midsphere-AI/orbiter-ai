"""Summary trigger logic and multi-template summary generation.

Compresses long conversations while preserving key information via
configurable triggers and LLM-powered summarization templates.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

from exo.memory.base import (  # pyright: ignore[reportMissingImports]
    MemoryItem,
)

# ---------------------------------------------------------------------------
# Summary templates
# ---------------------------------------------------------------------------


class SummaryTemplate(StrEnum):
    """Built-in summarization templates."""

    CONVERSATION = "conversation"
    FACTS = "facts"
    PROFILES = "profiles"


_DEFAULT_PROMPTS: dict[SummaryTemplate, str] = {
    SummaryTemplate.CONVERSATION: (
        "Summarize the following conversation, preserving key decisions, "
        "action items, and important context:\n\n{content}"
    ),
    SummaryTemplate.FACTS: (
        "Extract factual statements and verified information from the "
        "following conversation. Return them as a bullet list:\n\n{content}"
    ),
    SummaryTemplate.PROFILES: (
        "Extract user preferences, personality traits, and background "
        "information mentioned in the following conversation:\n\n{content}"
    ),
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SummaryConfig:
    """Configuration for summary triggers and generation.

    Attributes:
        message_threshold: Trigger when message count exceeds this value.
        token_threshold: Trigger when estimated token count exceeds this value.
        templates: Which summary templates to generate.
        prompts: Custom prompt overrides per template (uses defaults if empty).
        keep_recent: Number of recent messages to preserve after compression.
        token_estimate_ratio: Characters-per-token ratio for estimation.
    """

    message_threshold: int = 20
    token_threshold: int = 4000
    templates: tuple[SummaryTemplate, ...] = (SummaryTemplate.CONVERSATION,)
    prompts: dict[SummaryTemplate, str] = field(default_factory=dict)
    keep_recent: int = 4
    token_estimate_ratio: float = 4.0

    def get_prompt(self, template: SummaryTemplate) -> str:
        """Get the prompt for a template, falling back to defaults."""
        return self.prompts.get(template, _DEFAULT_PROMPTS[template])


# ---------------------------------------------------------------------------
# Trigger detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TriggerResult:
    """Result of a trigger check.

    Attributes:
        triggered: Whether a summary should be generated.
        reason: Description of the trigger condition.
        message_count: Current message count.
        estimated_tokens: Estimated total tokens.
    """

    estimated_tokens: int
    message_count: int
    reason: str
    triggered: bool


def check_trigger(
    items: Sequence[MemoryItem],
    config: SummaryConfig,
) -> TriggerResult:
    """Check whether summary generation should be triggered.

    Triggers when EITHER message count or estimated token count exceeds
    the configured thresholds.
    """
    message_count = len(items)
    estimated_tokens = _estimate_tokens(items, config.token_estimate_ratio)

    if message_count > config.message_threshold:
        logger.info(
            "Summarization triggered: threshold=%d messages=%d",
            config.message_threshold,
            message_count,
        )
        return TriggerResult(
            triggered=True,
            reason=f"Message count {message_count} exceeds threshold {config.message_threshold}",
            message_count=message_count,
            estimated_tokens=estimated_tokens,
        )

    if estimated_tokens > config.token_threshold:
        logger.info(
            "Summarization triggered: threshold=%d messages=%d",
            config.token_threshold,
            message_count,
        )
        return TriggerResult(
            triggered=True,
            reason=f"Estimated tokens {estimated_tokens} exceeds threshold {config.token_threshold}",
            message_count=message_count,
            estimated_tokens=estimated_tokens,
        )

    return TriggerResult(
        triggered=False,
        reason="No threshold exceeded",
        message_count=message_count,
        estimated_tokens=estimated_tokens,
    )


def _estimate_tokens(items: Sequence[MemoryItem], ratio: float) -> int:
    """Estimate token count for a sequence of memory items.

    Attempts to use tiktoken-based counting via ``exo.token_counter``.
    Falls back to character-ratio estimation (``len(content) / ratio``)
    when tiktoken is unavailable — using the caller-supplied *ratio*
    (from :attr:`SummaryConfig.token_estimate_ratio`, default ``4.0``).
    """
    try:
        from exo.token_counter import count_tokens  # pyright: ignore[reportMissingImports]

        return sum(count_tokens(item.content) for item in items)
    except Exception:
        # tiktoken unavailable or count_tokens not importable — fall back
        # to character-ratio heuristic using the configured ratio.
        return int(sum(len(item.content) for item in items) / max(ratio, 0.1))


# ---------------------------------------------------------------------------
# LLM summarizer protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Summarizer(Protocol):
    """Protocol for LLM-powered summarization.

    Callers provide an implementation wrapping their LLM provider.
    """

    async def summarize(self, prompt: str) -> str:
        """Generate a summary from a prompt. Returns the summary text."""
        ...


# ---------------------------------------------------------------------------
# Summary result
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SummaryResult:
    """Result of a summary generation run.

    Attributes:
        summaries: Generated summaries keyed by template name.
        compressed_items: The items to keep after compression (recent tail).
        original_count: Number of items before compression.
    """

    compressed_items: list[MemoryItem]
    original_count: int
    summaries: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------


async def generate_summary(
    items: Sequence[MemoryItem],
    config: SummaryConfig,
    summarizer: Any,
) -> SummaryResult:
    """Generate summaries from memory items using configured templates.

    The *summarizer* must implement the ``Summarizer`` protocol (an object
    with an ``async summarize(prompt: str) -> str`` method).

    Args:
        items: Memory items to summarize.
        config: Summary configuration.
        summarizer: Object implementing the Summarizer protocol.

    Returns:
        SummaryResult with generated summaries and compressed item list.
    """
    original_count = len(items)

    if original_count == 0:
        return SummaryResult(
            summaries={},
            compressed_items=[],
            original_count=0,
        )

    # Split: items to compress vs. recent items to keep
    keep_count = min(config.keep_recent, original_count)
    to_compress = list(items[: original_count - keep_count]) if keep_count < original_count else []
    recent = list(items[original_count - keep_count :])

    # Build conversation text from items to compress
    content = _format_items(to_compress) if to_compress else _format_items(list(items))

    # Generate summaries for each configured template
    summaries: dict[str, str] = {}
    for template in config.templates:
        prompt = config.get_prompt(template).format(content=content)
        logger.debug(
            "generating summary template=%s items=%d",
            template.value,
            len(to_compress) or len(items),
        )
        result = await summarizer.summarize(prompt)
        summaries[template.value] = result

    total_length = sum(len(s) for s in summaries.values())
    logger.debug("Summary generated length=%d", total_length)
    return SummaryResult(
        summaries=summaries,
        compressed_items=recent,
        original_count=original_count,
    )


def _format_items(items: Sequence[MemoryItem]) -> str:
    """Format memory items into a readable conversation string."""
    lines: list[str] = []
    for item in items:
        role = item.memory_type.upper()
        lines.append(f"[{role}]: {item.content}")
    return "\n".join(lines)
