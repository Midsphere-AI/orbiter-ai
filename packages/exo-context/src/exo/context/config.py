"""Context configuration — automation modes, history, summary, retrieval settings."""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AutomationMode(StrEnum):
    """Automation level for context management.

    - pilot: Minimal automation, user controls context manually.
    - copilot: Basic automation — summarization, offloading, history windowing.
    - navigator: Full automation — all context features enabled.
    """

    PILOT = "pilot"
    COPILOT = "copilot"
    NAVIGATOR = "navigator"


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


# Keys that indicate the caller is using the new simplified API.
_NEW_API_KEYS = frozenset({"limit", "overflow", "keep_recent", "token_pressure", "cache"})


class ContextConfig(BaseModel):
    """Immutable configuration for the context engine.

    **Simple API** (preferred)::

        ContextConfig(limit=20, overflow="summarize", cache=True)

    **Legacy API** (still fully supported)::

        ContextConfig(history_rounds=20, summary_threshold=10, offload_threshold=50)

    Both field sets are always present and kept in sync by an internal
    normalisation validator.
    """

    model_config = ConfigDict(frozen=True)

    mode: AutomationMode = AutomationMode.COPILOT

    # ── Legacy fields (internal plumbing, kept for backward compat) ──────

    # History windowing
    history_rounds: int = Field(default=20, ge=1, description="Max conversation rounds to keep")

    # Summarization
    summary_threshold: int = Field(
        default=10,
        ge=1,
        description="Number of messages before triggering summarization",
    )

    # Context offloading
    offload_threshold: int = Field(
        default=50,
        ge=1,
        description="Number of messages before offloading older context",
    )

    # Retrieval
    enable_retrieval: bool = Field(
        default=False,
        description="Enable RAG retrieval from workspace artifacts",
    )

    # Neuron selection
    neuron_names: tuple[str, ...] = Field(
        default=(),
        description="Names of neurons to include in prompt building",
    )

    # Token budget trigger for early summarization
    token_budget_trigger: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Token fill ratio (0-1) above which summarization triggers early",
    )

    # Context snapshot persistence
    enable_snapshots: bool = Field(
        default=False,
        description="Persist processed msg_list at end of each run and load it "
        "on the next run instead of rebuilding from raw history. "
        "Saves LLM summarization costs and preserves hook mutations.",
    )

    # ── New simplified API ───────────────────────────────────────────────

    limit: int = Field(
        default=20,
        ge=1,
        description="Max non-system messages to keep in the conversation window",
    )

    overflow: OverflowStrategy = Field(
        default=OverflowStrategy.SUMMARIZE,
        description="Strategy when limit is exceeded: summarize, truncate, or none",
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

    # Extensible metadata
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional configuration for custom processors or neurons",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_api_fields(cls, data: Any) -> Any:
        """Keep new and legacy field sets in sync.

        When the caller uses the new API (``limit``, ``overflow``, etc.) the
        legacy fields are derived automatically.  When only legacy fields are
        provided the new fields are back-filled to match.
        """
        if not isinstance(data, dict):
            return data

        has_new = bool(_NEW_API_KEYS & data.keys())

        if has_new:
            # ── New API → derive legacy fields ───────────────────────
            limit = data.get("limit", 20)
            overflow = str(data.get("overflow", "summarize"))

            if overflow == "summarize":
                data.setdefault("history_rounds", limit)
                data.setdefault("limit", limit)
                data.setdefault("summary_threshold", limit)
                data.setdefault("offload_threshold", max(limit, int(limit * 2.5)))
            elif overflow == "truncate":
                data.setdefault("history_rounds", limit)
                data.setdefault("limit", limit)
                data.setdefault("summary_threshold", 10_000)
                data.setdefault("offload_threshold", 10_000)
            elif overflow == "none":
                data.setdefault("history_rounds", 10_000)
                data.setdefault("limit", 10_000)
                data.setdefault("summary_threshold", 10_000)
                data.setdefault("offload_threshold", 10_000)
            elif overflow == "hook":
                # Hook-based: built-in cascade disabled, thresholds set high.
                # limit is preserved as-is for hooks to reference via info.limit.
                data.setdefault("history_rounds", 10_000)
                data.setdefault("summary_threshold", 10_000)
                data.setdefault("offload_threshold", 10_000)

            # Alias pairs
            if "token_pressure" in data:
                data.setdefault("token_budget_trigger", data["token_pressure"])
            if "cache" in data:
                data.setdefault("enable_snapshots", data["cache"])
        else:
            # ── Legacy-only (or bare constructor) → back-fill new fields ─
            data.setdefault("limit", data.get("history_rounds", 20))
            data.setdefault("overflow", "summarize")
            data.setdefault("keep_recent", max(2, data.get("summary_threshold", 10) // 2))
            data.setdefault("token_pressure", data.get("token_budget_trigger", 0.8))
            data.setdefault("cache", data.get("enable_snapshots", False))

        return data

    @model_validator(mode="before")
    @classmethod
    def _coerce_neuron_names(cls, data: Any) -> Any:
        """Accept list or tuple for neuron_names, normalise to tuple."""
        if isinstance(data, dict) and "neuron_names" in data:
            val = data["neuron_names"]
            if isinstance(val, list):
                data["neuron_names"] = tuple(val)
        return data

    @model_validator(mode="after")
    def _validate_thresholds(self) -> ContextConfig:
        """Ensure summary_threshold <= offload_threshold."""
        if self.summary_threshold > self.offload_threshold:
            msg = (
                f"summary_threshold ({self.summary_threshold}) "
                f"must be <= offload_threshold ({self.offload_threshold})"
            )
            raise ValueError(msg)
        return self


def make_config(mode: AutomationMode | str = "copilot", **overrides: Any) -> ContextConfig:
    """Factory for creating ContextConfig at a given automation level.

    Preset defaults per mode:
    - pilot:     history_rounds=100, no summary, no offload, no retrieval
    - copilot:   history_rounds=20, summary at 10, offload at 50
    - navigator: history_rounds=10, summary at 5, offload at 20, retrieval enabled
    """
    mode = AutomationMode(mode)
    presets: dict[AutomationMode, dict[str, Any]] = {
        AutomationMode.PILOT: {
            "history_rounds": 100,
            "summary_threshold": 100,
            "offload_threshold": 100,
            "enable_retrieval": False,
            "enable_snapshots": False,
        },
        AutomationMode.COPILOT: {
            "history_rounds": 20,
            "summary_threshold": 10,
            "offload_threshold": 50,
            "enable_retrieval": False,
            "enable_snapshots": False,
        },
        AutomationMode.NAVIGATOR: {
            "history_rounds": 10,
            "summary_threshold": 5,
            "offload_threshold": 20,
            "enable_retrieval": True,
            "enable_snapshots": True,
        },
    }
    defaults = presets[mode]
    defaults.update(overrides)
    config = ContextConfig(mode=mode, **defaults)
    logger.debug(
        "ContextConfig created mode=%s history_rounds=%d", mode.value, config.history_rounds
    )
    return config
