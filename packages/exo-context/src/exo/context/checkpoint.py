"""Checkpoint — save and restore complete execution state for long-running tasks.

Checkpoints capture the full context state (values, metadata, token usage) at a
point in time.  They are versioned per session so multiple snapshots can be
taken and any prior version can be restored.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from exo.types import ExoError  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)


class CheckpointError(ExoError):
    """Raised for checkpoint lifecycle errors."""


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """Immutable snapshot of context state at a point in time.

    Parameters
    ----------
    task_id:
        The task this checkpoint belongs to.
    version:
        Monotonically increasing version number within a session.
    values:
        Snapshot of context state key-value pairs (deep copy).
    token_usage:
        Snapshot of token usage counters at checkpoint time.
    metadata:
        Optional user-provided metadata (e.g., description, tags).
    created_at:
        UTC timestamp of when the checkpoint was created.
    """

    task_id: str
    version: int
    values: dict[str, Any]
    token_usage: dict[str, int]
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Serialize checkpoint to a dictionary."""
        return {
            "task_id": self.task_id,
            "version": self.version,
            "values": dict(self.values),
            "token_usage": dict(self.token_usage),
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Checkpoint:
        """Deserialize checkpoint from a dictionary."""
        created = data.get("created_at")
        if isinstance(created, str):
            created = datetime.fromisoformat(created)
        return cls(
            task_id=data["task_id"],
            version=data["version"],
            values=dict(data.get("values", {})),
            token_usage=dict(data.get("token_usage", {})),
            metadata=dict(data.get("metadata", {})),
            created_at=created if isinstance(created, datetime) else datetime.now(UTC),
        )

    def __repr__(self) -> str:
        keys = len(self.values)
        return f"Checkpoint(task_id={self.task_id!r}, version={self.version}, keys={keys})"


class CheckpointStore:
    """Per-session checkpoint store with version tracking.

    Manages a sequence of checkpoints for a given task, providing
    snapshot creation, version listing, and retrieval.
    """

    __slots__ = ("_checkpoints", "_task_id")

    def __init__(self, task_id: str) -> None:
        if not task_id:
            msg = "task_id is required and must be non-empty"
            raise CheckpointError(msg)
        self._task_id = task_id
        self._checkpoints: list[Checkpoint] = []

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def version(self) -> int:
        """Current version (number of checkpoints taken)."""
        return len(self._checkpoints)

    def save(
        self,
        values: dict[str, Any],
        token_usage: dict[str, int],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Checkpoint:
        """Create a new checkpoint from the given state.

        Parameters
        ----------
        values:
            Context state key-value pairs to snapshot.
        token_usage:
            Token usage counters to snapshot.
        metadata:
            Optional metadata for this checkpoint.

        Returns
        -------
        The created :class:`Checkpoint`.
        """
        cp = Checkpoint(
            task_id=self._task_id,
            version=self.version + 1,
            values=dict(values),
            token_usage=dict(token_usage),
            metadata=dict(metadata) if metadata else {},
        )
        self._checkpoints.append(cp)
        logger.debug(
            "checkpoint saved: task_id=%r version=%d keys=%d",
            self._task_id,
            cp.version,
            len(cp.values),
        )
        return cp

    def get(self, version: int) -> Checkpoint:
        """Retrieve a checkpoint by version number (1-based).

        Raises
        ------
        CheckpointError
            If the version does not exist.
        """
        if version < 1 or version > len(self._checkpoints):
            msg = f"Checkpoint version {version} not found (available: 1-{len(self._checkpoints)})"
            logger.warning("checkpoint get failed: %s", msg)
            raise CheckpointError(msg)
        return self._checkpoints[version - 1]

    @property
    def latest(self) -> Checkpoint | None:
        """Most recent checkpoint, or None if no checkpoints exist."""
        return self._checkpoints[-1] if self._checkpoints else None

    def list_versions(self) -> list[int]:
        """List all available checkpoint versions."""
        return [cp.version for cp in self._checkpoints]

    def __len__(self) -> int:
        return len(self._checkpoints)

    def __repr__(self) -> str:
        return f"CheckpointStore(task_id={self._task_id!r}, checkpoints={len(self._checkpoints)})"
