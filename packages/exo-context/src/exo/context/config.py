"""Context configuration — simplified API with internal windowing thresholds."""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)


class OverflowStrategy(StrEnum):
    """What happens when the conversation exceeds ``limit``.

    - summarize: Oldest messages are compressed into a summary, recent kept verbatim.
    - truncate: Oldest messages are dropped, recent kept.
    - none: No context management — grows until model token limit.
    - hook: Context management delegated to ``CONTEXT_WINDOW`` hooks.
    """

    SUMMARIZE = "summarize"
    TRUNCATE = "truncate"
    NONE = "none"
    HOOK = "hook"


class ContextConfig(BaseModel):
    """Immutable configuration for the context engine.

    Public constructor accepts only the simplified API::

        ContextConfig(limit=20, overflow="summarize", cache=True)

    Internal windowing thresholds are derived automatically from these fields
    and exposed as read-only properties (``_history_rounds``, ``_summary_threshold``,
    ``_offload_threshold``, ``_token_budget_trigger``, ``_enable_snapshots``).
    """

    model_config = ConfigDict(frozen=True)

    # ── Public API ───────────────────────────────────────────────────────

    limit: int = Field(
        default=20,
        ge=1,
        description="Max non-system messages to keep in the conversation window",
    )

    overflow: OverflowStrategy = Field(
        default=OverflowStrategy.SUMMARIZE,
        description="Strategy when limit is exceeded: summarize, truncate, none, or hook",
    )

    keep_recent: int = Field(
        default=5,
        ge=1,
        description="Messages to keep verbatim after summarization (overflow='summarize')",
    )

    token_pressure: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Token fill ratio (0-1) that triggers early overflow",
    )

    cache: bool = Field(
        default=False,
        description="Persist processed messages between runs (avoids re-summarizing)",
    )

    # Section/neuron selection — neuron_names is the stored field (backward-compat);
    # section_names is a property alias pointing to the same value.
    neuron_names: tuple[str, ...] = Field(
        default=(),
        description="Names of prompt sections to include in prompt building "
        "(also accessible as section_names)",
    )

    # Extensible metadata
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional configuration for custom processors or sections",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_section_names(cls, data: Any) -> Any:
        """Accept list or tuple for neuron_names/section_names, normalise to tuple.

        ``section_names`` is a new alias for ``neuron_names`` accepted at construction.
        """
        if isinstance(data, dict):
            # New alias: section_names → neuron_names (if only section_names given)
            if "section_names" in data and "neuron_names" not in data:
                data["neuron_names"] = data.pop("section_names")
            elif "section_names" in data:
                # Both provided — prefer section_names, discard duplicate
                data["neuron_names"] = data.pop("section_names")
            # Coerce list → tuple
            if "neuron_names" in data:
                val = data["neuron_names"]
                if isinstance(val, list):
                    data["neuron_names"] = tuple(val)
        return data

    @property
    def section_names(self) -> tuple[str, ...]:
        """Canonical alias for :attr:`neuron_names` — preferred new name."""
        return self.neuron_names

    # ── Internal derived windowing thresholds (read-only properties) ─────

    @property
    def _history_rounds(self) -> int:
        """Max non-system messages — equals ``limit`` for summarize/truncate; 10_000 for none/hook."""
        strategy = str(self.overflow)
        if strategy in ("summarize", "truncate"):
            return self.limit
        return 10_000

    @property
    def _summary_threshold(self) -> int:
        """Message count at which summarization triggers (overflow='summarize')."""
        strategy = str(self.overflow)
        if strategy == "summarize":
            return self.limit
        return 10_000

    @property
    def _offload_threshold(self) -> int:
        """Message count at which emergency offload triggers (overflow='summarize')."""
        strategy = str(self.overflow)
        if strategy == "summarize":
            return max(self.limit, int(self.limit * 2.5))
        return 10_000

    @property
    def _token_budget_trigger(self) -> float:
        """Alias for ``token_pressure`` — legacy name used by internal callers."""
        return self.token_pressure

    @property
    def _enable_snapshots(self) -> bool:
        """Alias for ``cache`` — legacy name used by internal callers."""
        return self.cache
