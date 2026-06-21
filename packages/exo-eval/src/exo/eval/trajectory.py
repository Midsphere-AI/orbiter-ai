"""Canonical trajectory types for agent execution recording.

``TrajectoryItem`` is the single source of truth for a trajectory step across
the Exo monorepo.  ``exo-train`` re-exports from here.
"""

from __future__ import annotations

import csv
import io
import json
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from exo.types import ExoError  # pyright: ignore[reportMissingImports]


class TrajectoryError(ExoError):
    """Error during trajectory operations."""


@dataclass(frozen=True, slots=True)
class TrajectoryItem:
    """A single step in an agent execution trajectory.

    Follows the State-Action-Reward (SAR) pattern.
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    task_id: str = ""
    agent_id: str = ""
    step: int = 0
    timestamp: float = field(default_factory=time.time)
    # State
    input: str = ""
    messages: tuple[dict[str, Any], ...] = ()
    context: dict[str, Any] = field(default_factory=dict)
    # Action
    output: str = ""
    tool_calls: tuple[dict[str, Any], ...] = ()
    # Reward
    score: float | None = None
    status: str = "success"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "step": self.step,
            "timestamp": self.timestamp,
            "input": self.input,
            "messages": list(self.messages),
            "context": dict(self.context),
            "output": self.output,
            "tool_calls": list(self.tool_calls),
            "score": self.score,
            "status": self.status,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrajectoryItem:
        """Create from a plain dict."""
        return cls(
            id=data.get("id", uuid.uuid4().hex[:12]),
            task_id=data.get("task_id", ""),
            agent_id=data.get("agent_id", ""),
            step=data.get("step", 0),
            timestamp=data.get("timestamp", time.time()),
            input=data.get("input", ""),
            messages=tuple(data.get("messages", ())),
            context=data.get("context", {}),
            output=data.get("output", ""),
            tool_calls=tuple(data.get("tool_calls", ())),
            score=data.get("score"),
            status=data.get("status", "success"),
            metadata=data.get("metadata", {}),
        )


class TrajectoryStrategy(ABC):
    """Pluggable strategy for generating trajectory items from messages."""

    @abstractmethod
    def build_item(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        task_id: str = "",
        agent_id: str = "",
        step: int = 0,
        **kwargs: Any,
    ) -> TrajectoryItem:
        """Build a single trajectory item from a message list."""

    def validate(self, items: Sequence[TrajectoryItem]) -> bool:
        """Return True if the trajectory is valid. Override for custom checks."""
        return len(items) > 0


class DefaultStrategy(TrajectoryStrategy):
    """Default strategy: extracts input/output/tool_calls from message dicts."""

    def build_item(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        task_id: str = "",
        agent_id: str = "",
        step: int = 0,
        **kwargs: Any,
    ) -> TrajectoryItem:
        user_input = ""
        output = ""
        tool_calls: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "")
            if role == "user" and not user_input:
                user_input = msg.get("content", "")
            elif role == "assistant":
                output = msg.get("content", "")
                if msg.get("tool_calls"):
                    tool_calls.extend(msg["tool_calls"])

        return TrajectoryItem(
            task_id=task_id,
            agent_id=agent_id,
            step=step,
            input=user_input,
            messages=tuple(dict(m) for m in messages),
            output=output,
            tool_calls=tuple(tool_calls),
            score=kwargs.get("score"),
            status=kwargs.get("status", "success"),
            metadata=kwargs.get("metadata", {}),
        )


class TrajectoryDataset:
    """Dataset of trajectory items with capture, strategy, and export."""

    __slots__ = ("_items", "_step_counters", "_strategy")

    def __init__(self, *, strategy: TrajectoryStrategy | None = None) -> None:
        self._items: list[TrajectoryItem] = []
        self._step_counters: dict[str, int] = {}
        self._strategy = strategy or DefaultStrategy()

    @property
    def items(self) -> list[TrajectoryItem]:
        return list(self._items)

    @property
    def strategy(self) -> TrajectoryStrategy:
        return self._strategy

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return (
            f"TrajectoryDataset(items={len(self._items)}, strategy={type(self._strategy).__name__})"
        )

    # --- Capture ---

    def append_trajectory(self, item: TrajectoryItem) -> None:
        """Append a pre-built trajectory item."""
        self._items.append(item)

    def from_messages(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        task_id: str = "",
        agent_id: str = "",
        **kwargs: Any,
    ) -> TrajectoryItem:
        """Build a trajectory item from messages via strategy and append it."""
        key = f"{task_id}:{agent_id}"
        step = self._step_counters.get(key, 0)
        item = self._strategy.build_item(
            messages, task_id=task_id, agent_id=agent_id, step=step, **kwargs
        )
        self._items.append(item)
        self._step_counters[key] = step + 1
        return item

    def save_task_trajectory(self, task_id: str, items: Sequence[TrajectoryItem]) -> int:
        """Bulk-append items for a given task. Returns count added."""
        count = 0
        for item in items:
            self._items.append(item)
            count += 1
        return count

    def get_task_trajectory(self, task_id: str) -> list[TrajectoryItem]:
        """Return all items matching *task_id*."""
        return [it for it in self._items if it.task_id == task_id]

    def validate(self) -> bool:
        """Delegate validation to the strategy."""
        return self._strategy.validate(self._items)

    # --- Export ---

    def to_json(self) -> str:
        """Export all items as a JSON string."""
        return json.dumps([it.to_dict() for it in self._items], indent=2)

    def to_csv(self) -> str:
        """Export all items as CSV text."""
        if not self._items:
            return ""
        fieldnames = list(self._items[0].to_dict().keys())
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for item in self._items:
            row = item.to_dict()
            # Serialise complex fields to JSON strings for CSV
            for key in ("messages", "tool_calls", "context", "metadata"):
                row[key] = json.dumps(row[key])
            writer.writerow(row)
        return buf.getvalue()

    def clear(self) -> None:
        """Remove all items and reset step counters."""
        self._items.clear()
        self._step_counters.clear()


class TrajectoryExtractor:
    """Extracts complete execution DAGs from agent session traces.

    Given a flat message history (OpenAI chat format), produces a list of
    :class:`TrajectoryItem` objects representing individual execution steps.

    Args:
        dataset: The trajectory dataset to append extracted items to.
    """

    __slots__ = ("_dataset",)

    def __init__(self, dataset: TrajectoryDataset) -> None:
        self._dataset = dataset

    @property
    def dataset(self) -> TrajectoryDataset:
        """The underlying trajectory dataset."""
        return self._dataset

    def extract(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        include_tool_calls: bool = True,
        task_id: str = "",
        agent_id: str = "",
    ) -> list[TrajectoryItem]:
        """Extract trajectory items from a message history."""
        if not messages:
            return []

        segments = self._segment_messages(messages, include_tool_calls=include_tool_calls)

        items: list[TrajectoryItem] = []
        for step, segment in enumerate(segments):
            item = self._dataset.strategy.build_item(
                segment,
                task_id=task_id,
                agent_id=agent_id,
                step=step,
            )
            self._dataset.append_trajectory(item)
            items.append(item)

        return items

    @staticmethod
    def _segment_messages(
        messages: Sequence[dict[str, Any]],
        *,
        include_tool_calls: bool = True,
    ) -> list[list[dict[str, Any]]]:
        """Split a flat message list into logical execution segments."""
        if not messages:
            return []

        turns: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "user" and current:
                turns.append(current)
                current = []
            current.append(msg)
        if current:
            turns.append(current)

        segments: list[list[dict[str, Any]]] = []

        for turn in turns:
            if include_tool_calls:
                asst_indices = [i for i, m in enumerate(turn) if m.get("role") == "assistant"]
                if len(asst_indices) <= 1:
                    segments.append(list(turn))
                else:
                    for idx in asst_indices:
                        segments.append(list(turn[: idx + 1]))
            else:
                filtered: list[dict[str, Any]] = []
                for m in turn:
                    role = m.get("role", "")
                    if role == "tool":
                        continue
                    if role == "assistant" and m.get("tool_calls"):
                        if not m.get("content"):
                            continue
                        filtered.append({k: v for k, v in m.items() if k != "tool_calls"})
                    else:
                        filtered.append(m)
                if filtered:
                    segments.append(filtered)

        return segments
