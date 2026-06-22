"""ACE — Adaptive Context Engine memory evolution strategy.

Scores memories using per-item counters (helpful / harmful / neutral) and
prunes items whose harmful ratio exceeds a threshold.  Supports LLM-based
reflection for counter updates and periodic curation.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from exo.memory.base import ExoMemoryError, MemoryItem  # pyright: ignore[reportMissingImports]
from exo.memory.evolution import MemoryEvolutionStrategy  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class Counters:
    """Per-memory feedback counters."""

    __slots__ = ("harmful", "helpful", "neutral")

    def __init__(self, helpful: int = 0, harmful: int = 0, neutral: int = 0) -> None:
        self.helpful = helpful
        self.harmful = harmful
        self.neutral = neutral

    @property
    def total(self) -> int:
        return self.helpful + self.harmful + self.neutral

    def score(self) -> float:
        """Return a quality score in [0, 1].  Higher is better.

        Score is ``(helpful - harmful) / total``, mapped to [0, 1].
        Returns 0.5 when no feedback has been recorded.
        """
        if self.total == 0:
            return 0.5
        raw = (self.helpful - self.harmful) / self.total
        # Map [-1, 1] → [0, 1]
        return (raw + 1.0) / 2.0

    def to_dict(self) -> dict[str, int]:
        return {
            "helpful": self.helpful,
            "harmful": self.harmful,
            "neutral": self.neutral,
        }

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> Counters:
        return cls(
            helpful=data.get("helpful", 0),
            harmful=data.get("harmful", 0),
            neutral=data.get("neutral", 0),
        )


@runtime_checkable
class ReflectionModel(Protocol):
    """Protocol for the LLM callable used by ``reflect()``.

    Accepts a prompt string and returns a classification string
    (``"helpful"``, ``"harmful"``, or ``"neutral"``).
    """

    async def __call__(self, prompt: str) -> str: ...


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

_REFLECT_PROMPT = (
    "Classify the following memory as helpful, harmful, or neutral "
    "given the user feedback.\n\n"
    "Memory: {content}\n"
    "Feedback: {feedback}\n\n"
    "Respond with exactly one word: helpful, harmful, or neutral."
)

_VALID_LABELS = frozenset({"helpful", "harmful", "neutral"})


class ACEStrategy(MemoryEvolutionStrategy):
    """Adaptive Context Engine — playbook-based memory scoring.

    Maintains per-memory counters (helpful, harmful, neutral) and uses
    ratio-based scoring to prune low-quality memories.

    Args:
        counter_path: Path to the JSON file for persisting counters.
            Created automatically on first save.  Pass ``None`` for
            in-memory-only operation (counters lost on process exit).
        harmful_threshold: Maximum harmful ratio (harmful / total) before
            a memory is pruned during ``evolve()``.  Default ``0.5``.
    """

    name: str = "ace"

    def __init__(
        self,
        counter_path: str | Path | None = None,
        *,
        harmful_threshold: float = 0.5,
    ) -> None:
        self._counter_path: Path | None = Path(counter_path) if counter_path is not None else None
        self._harmful_threshold = harmful_threshold
        self._counters: dict[str, Counters] = {}
        self._load()

    # -- persistence --------------------------------------------------------

    def _load(self) -> None:
        """Load counters from disk (if configured and file exists)."""
        if self._counter_path is None or not self._counter_path.exists():
            return
        try:
            raw = json.loads(self._counter_path.read_text(encoding="utf-8"))
            self._counters = {mid: Counters.from_dict(data) for mid, data in raw.items()}
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("ACE: failed to load counters: %s", exc)

    def _save(self) -> None:
        """Persist counters to disk (if configured).

        When called from an async context, use :meth:`_async_save` instead
        to avoid blocking the event loop on the file write.
        """
        if self._counter_path is None:
            return
        data = {mid: c.to_dict() for mid, c in self._counters.items()}
        self._counter_path.parent.mkdir(parents=True, exist_ok=True)
        self._counter_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    async def _async_save(self) -> None:
        """Persist counters to disk without blocking the event loop."""
        if self._counter_path is None:
            return
        data = {mid: c.to_dict() for mid, c in self._counters.items()}
        await asyncio.to_thread(self._write_data, data)

    def _write_data(self, data: dict[str, Any]) -> None:
        """Blocking helper that writes *data* to disk (called via to_thread)."""
        assert self._counter_path is not None
        self._counter_path.parent.mkdir(parents=True, exist_ok=True)
        self._counter_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # -- public API ---------------------------------------------------------

    def get_counters(self, memory_id: str) -> Counters:
        """Return the counters for a memory item (creates if missing)."""
        if memory_id not in self._counters:
            self._counters[memory_id] = Counters()
        return self._counters[memory_id]

    def record(self, memory_id: str, label: str) -> None:
        """Record a feedback label for a memory item.

        Args:
            memory_id: The memory item's ID.
            label: One of ``"helpful"``, ``"harmful"``, or ``"neutral"``.

        Raises:
            ValueError: If *label* is not a valid classification.
        """
        if label not in _VALID_LABELS:
            raise ExoMemoryError(
                f"Invalid feedback label {label!r}.",
                context={"label": label, "valid": sorted(_VALID_LABELS)},
                hint=f"Pass one of: {sorted(_VALID_LABELS)}.",
            )
        counters = self.get_counters(memory_id)
        setattr(counters, label, getattr(counters, label) + 1)
        # _save() does a blocking write — callers from async code should use
        # _async_save() after updating all counters (e.g. reflect() does this).
        self._save()

    async def evolve(
        self,
        items: list[MemoryItem],
        *,
        context: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Score items and prune those with a high harmful ratio.

        Items with ``harmful / total > harmful_threshold`` are removed.
        Items with no recorded feedback are kept (score defaults to 0.5).
        """
        result: list[MemoryItem] = []
        for item in items:
            counters = self.get_counters(item.id)
            if counters.total == 0:
                result.append(item)
                continue
            harmful_ratio = counters.harmful / counters.total
            if harmful_ratio <= self._harmful_threshold:
                result.append(item)
        return result

    async def reflect(
        self,
        items: list[MemoryItem],
        feedback: str,
        *,
        model: ReflectionModel | None = None,
    ) -> dict[str, str]:
        """Classify each memory via LLM and update counters.

        Args:
            items: Memory items to reflect on.
            feedback: User feedback text used in the prompt.
            model: Async callable ``(prompt) -> label``.  If ``None``,
                all items are classified as ``"neutral"``.

        Returns:
            Mapping of memory ID to assigned label.
        """
        labels: dict[str, str] = {}
        for item in items:
            if model is not None:
                prompt = _REFLECT_PROMPT.format(content=item.content, feedback=feedback)
                raw = await model(prompt)
                label = raw.strip().lower()
                if label not in _VALID_LABELS:
                    label = "neutral"
            else:
                label = "neutral"
            # Update in-memory counters only (skip the blocking _save() per item).
            if label not in _VALID_LABELS:
                raise ExoMemoryError(
                    f"Invalid feedback label {label!r}.",
                    context={"label": label, "valid": sorted(_VALID_LABELS)},
                    hint=f"Pass one of: {sorted(_VALID_LABELS)}.",
                )
            counters = self.get_counters(item.id)
            setattr(counters, label, getattr(counters, label) + 1)
            labels[item.id] = label
        # Flush all counter updates to disk once (non-blocking).
        await self._async_save()
        return labels

    async def curate(
        self,
        items: list[MemoryItem],
        *,
        threshold: float = 0.3,
    ) -> list[MemoryItem]:
        """Remove memories whose quality score falls below *threshold*.

        Args:
            items: Memory items to curate.
            threshold: Minimum score to keep (default ``0.3``).

        Returns:
            Filtered list of memory items.
        """
        result: list[MemoryItem] = []
        for item in items:
            counters = self.get_counters(item.id)
            if counters.score() >= threshold:
                result.append(item)
        return result


__all__ = ["ACEStrategy", "Counters", "ReflectionModel"]
