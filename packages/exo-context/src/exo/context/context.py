"""Context — core lifecycle with fork/merge for hierarchical task decomposition.

Context holds per-task configuration and hierarchical state.  Forking creates
a child context with inherited state; merging consolidates child state back
into the parent with net token calculation.
"""

from __future__ import annotations

import logging
from typing import Any

from exo.types import ExoError  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)

from exo.context.checkpoint import (  # pyright: ignore[reportMissingImports]
    Checkpoint,
    CheckpointError,
    CheckpointStore,
)
from exo.context.config import ContextConfig  # pyright: ignore[reportMissingImports]
from exo.context.state import ContextState  # pyright: ignore[reportMissingImports]


class ContextError(ExoError):
    """Raised for context lifecycle errors."""


class Context:
    """Per-task context with hierarchical state and fork/merge lifecycle.

    Parameters
    ----------
    task_id:
        Unique identifier for the task this context belongs to.
    config:
        Immutable configuration (automation mode, thresholds, etc.).
        Defaults to ``ContextConfig()`` (copilot mode).
    parent:
        Optional parent context.  Forked contexts automatically set this.
    state:
        Initial state entries.  If omitted an empty state is created.
        For forked contexts the state's *parent* is set to the parent's state.
    """

    __slots__ = (
        "_checkpoint_store",
        "_children",
        "_config",
        "_parent",
        "_state",
        "_task_id",
        "_token_snapshot",
        "_token_usage",
    )

    def __init__(
        self,
        task_id: str,
        *,
        config: ContextConfig | None = None,
        parent: Context | None = None,
        state: ContextState | None = None,
    ) -> None:
        if not task_id:
            raise ContextError(
                "task_id is required and must be non-empty.",
                hint="Pass a non-empty string as task_id, e.g. Context('my-task-123').",
            )

        self._task_id = task_id
        self._config = config or (parent._config if parent else ContextConfig())
        self._parent = parent

        if state is not None:
            self._state = state
        elif parent is not None:
            # Child inherits parent state via ContextState parent chain
            self._state = ContextState(parent=parent._state)
        else:
            self._state = ContextState()

        # Token usage tracking: {key: int}
        # For forked contexts, starts as a copy of parent usage.
        self._token_usage: dict[str, int] = dict(parent._token_usage) if parent else {}
        # Snapshot of parent's usage at fork time for net-delta merge calculation.
        self._token_snapshot: dict[str, int] = dict(parent._token_usage) if parent else {}
        self._children: list[Context] = []
        self._checkpoint_store = CheckpointStore(task_id)

    # ── Properties ────────────────────────────────────────────────────

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def config(self) -> ContextConfig:
        return self._config

    @property
    def parent(self) -> Context | None:
        return self._parent

    @property
    def state(self) -> ContextState:
        return self._state

    @property
    def children(self) -> list[Context]:
        """Direct child contexts created via :meth:`fork`."""
        return list(self._children)

    @property
    def token_usage(self) -> dict[str, int]:
        """Current token usage counters (copy)."""
        return dict(self._token_usage)

    # ── Token tracking ────────────────────────────────────────────────

    def add_tokens(self, usage: dict[str, int]) -> None:
        """Add token counts to this context's usage tracker."""
        for key, value in usage.items():
            self._token_usage[key] = self._token_usage.get(key, 0) + value

    # ── Fork / Merge ─────────────────────────────────────────────────

    def fork(self, task_id: str) -> Context:
        """Create a child context for a sub-task.

        The child inherits:
        - config (shared reference, immutable)
        - state (via ContextState parent chain — reads inherit, writes isolate)
        - token_usage (snapshot for net-delta calculation on merge)

        The child is registered in :attr:`children`.
        """
        child = Context(task_id, parent=self)
        self._children.append(child)
        logger.debug("forked context %r from parent %r", task_id, self._task_id)
        return child

    def merge(self, child: Context) -> None:
        """Consolidate a child context back into this context.

        Merges:
        1. Child's *local* state entries into parent state.
        2. Net token usage (child - parent snapshot at fork time).

        Raises :class:`ContextError` if *child* is not a direct child.
        """
        if child._parent is not self:
            msg = f"Context {child.task_id!r} is not a child of {self.task_id!r}"
            logger.warning("merge rejected: %s", msg)
            raise ContextError(
                msg,
                hint="Only merge a context returned by this context's .fork() method.",
                context={"parent": self._task_id, "child": child.task_id},
            )

        # Guard against double-merge of the same child.
        if child not in self._children:
            msg = f"Context {child.task_id!r} has already been merged into {self.task_id!r}"
            logger.warning("merge rejected: %s", msg)
            raise ContextError(
                msg,
                hint="Each forked child can only be merged once.",
                context={"parent": self._task_id, "child": child.task_id},
            )

        # 1. Merge child's local state into parent
        local = child._state.local_dict()
        if local:
            self._state.update(local)

        # 2. Merge net token usage
        #    child._token_snapshot holds the parent's usage at fork time.
        #    Net delta = child_current - snapshot_at_fork.
        for key, child_value in child._token_usage.items():
            snapshot_value = child._token_snapshot.get(key, 0)
            net = child_value - snapshot_value
            if net > 0:
                self._token_usage[key] = self._token_usage.get(key, 0) + net

        # Remove child from registry so double-merge is impossible.
        self._children.remove(child)

        logger.debug(
            "merged child %r into parent %r: %d state keys, token delta applied",
            child.task_id,
            self._task_id,
            len(local),
        )

    # ── Checkpoint ────────────────────────────────────────────────────

    @property
    def checkpoints(self) -> CheckpointStore:
        """Access the checkpoint store for this context."""
        return self._checkpoint_store

    def snapshot(self, *, metadata: dict[str, Any] | None = None) -> Checkpoint:
        """Save a checkpoint of the current context state.

        Captures a deep copy of state values and token usage.
        Checkpoints are versioned per context session (monotonically increasing).

        Parameters
        ----------
        metadata:
            Optional metadata to attach (e.g., description, step number).

        Returns
        -------
        The created :class:`Checkpoint`.
        """
        cp = self._checkpoint_store.save(
            values=self._state.to_dict(),
            token_usage=dict(self._token_usage),
            metadata=metadata,
        )
        logger.debug("snapshot created: task_id=%r version=%d", self._task_id, cp.version)
        return cp

    @classmethod
    def restore(cls, checkpoint: Checkpoint, *, config: ContextConfig | None = None) -> Context:
        """Restore a context from a checkpoint.

        Creates a new :class:`Context` with state and token usage
        reconstructed from the checkpoint data.

        Parameters
        ----------
        checkpoint:
            The checkpoint to restore from.
        config:
            Optional config override.  If ``None``, uses ``ContextConfig()`` (default).

        Returns
        -------
        A new :class:`Context` with the restored state.

        Raises
        ------
        CheckpointError
            If the checkpoint data is invalid.

        Note
        ----
        **Config is not persisted in checkpoints.**  The original context config is
        not stored in the checkpoint serialization format, so ``restore()`` always
        applies the *config* argument (or the default ``ContextConfig()`` when
        omitted).  If you need to recreate the original config you must supply it
        explicitly via the ``config`` parameter.
        """
        if not isinstance(checkpoint, Checkpoint):
            raise CheckpointError(
                f"Expected Checkpoint, got {type(checkpoint).__name__}.",
                hint="Pass a Checkpoint object returned by .snapshot() or CheckpointStore.get().",
            )

        ctx = cls(checkpoint.task_id, config=config)
        # Restore state from checkpoint values
        ctx._state.update(checkpoint.values)
        # Restore token usage
        ctx._token_usage = dict(checkpoint.token_usage)
        logger.debug(
            "context restored from checkpoint: task_id=%r version=%d",
            checkpoint.task_id,
            checkpoint.version,
        )
        return ctx

    # ── Representation ───────────────────────────────────────────────

    def __repr__(self) -> str:
        children_n = len(self._children)
        parent_info = f", parent={self._parent.task_id!r}" if self._parent else ""
        child_info = f", children={children_n}" if children_n else ""
        return f"Context(task_id={self._task_id!r}{parent_info}{child_info})"
