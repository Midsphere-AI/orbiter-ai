"""OperatorTrainer — training loop for operator-based agents with checkpoint/resume.

Orchestrates operators and optimizers through an epoch-based training loop:
evaluate → backward → step → apply → validate, with checkpoint persistence
after each epoch.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from exo.train.operator.base import (  # pyright: ignore[reportMissingImports]
    Operator,
)
from exo.train.trainer import (  # pyright: ignore[reportMissingImports]
    TrainConfig,
    Trainer,
    TrainerError,
    TrainerState,
    TrainMetrics,
)

# ---------------------------------------------------------------------------
# Checkpoint store
# ---------------------------------------------------------------------------


@runtime_checkable
class CheckpointStore(Protocol):
    """Protocol for checkpoint persistence."""

    def save(self, epoch: int, data: dict[str, Any]) -> None: ...
    def load(self, epoch: int | None = None) -> dict[str, Any] | None: ...
    def latest_epoch(self) -> int | None: ...


class FileCheckpointStore:
    """Persists checkpoint data as JSON files in a directory.

    Each epoch is stored as ``checkpoint_<epoch>.json``.

    Args:
        directory: Path to the checkpoint directory. Created on first save.
            The path is resolved to an absolute path at construction time.
        base_dir: Optional containment root.  When provided, every write
            path is asserted to be a descendant of *base_dir*; paths that
            would escape (e.g. via ``../../``) raise :exc:`TrainerError`.
            When *None* (default) no containment check is performed but the
            directory is still resolved to an absolute path.
    """

    __slots__ = ("_base_dir", "_directory")

    def __init__(self, directory: str | Path, *, base_dir: str | Path | None = None) -> None:
        self._directory = Path(directory).resolve()
        self._base_dir = Path(base_dir).resolve() if base_dir is not None else None
        if self._base_dir is not None:
            self._assert_contained(self._directory)

    def _assert_contained(self, path: Path) -> None:
        """Raise TrainerError if *path* is not under self._base_dir."""
        assert self._base_dir is not None
        try:
            path.relative_to(self._base_dir)
        except ValueError:
            msg = (
                f"Checkpoint path {path!r} escapes the allowed base directory "
                f"{self._base_dir!r}. Refusing to write."
            )
            raise TrainerError(msg) from None

    @property
    def directory(self) -> Path:
        """Checkpoint directory path (resolved absolute path)."""
        return self._directory

    def save(self, epoch: int, data: dict[str, Any]) -> None:
        """Save checkpoint data for the given epoch."""
        self._directory.mkdir(parents=True, exist_ok=True)
        path = (self._directory / f"checkpoint_{epoch}.json").resolve()
        if self._base_dir is not None:
            self._assert_contained(path)
        path.write_text(json.dumps(data, default=str), encoding="utf-8")

    def load(self, epoch: int | None = None) -> dict[str, Any] | None:
        """Load checkpoint data for a specific epoch, or the latest.

        Args:
            epoch: Specific epoch to load. If *None*, loads the latest.

        Returns:
            Checkpoint data dict, or *None* if no checkpoint exists.
        """
        if epoch is None:
            epoch = self.latest_epoch()
        if epoch is None:
            return None
        path = self._directory / f"checkpoint_{epoch}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]

    def latest_epoch(self) -> int | None:
        """Return the highest saved epoch number, or *None*."""
        if not self._directory.exists():
            return None
        epochs: list[int] = []
        for entry in os.listdir(self._directory):
            if entry.startswith("checkpoint_") and entry.endswith(".json"):
                try:
                    epochs.append(int(entry[len("checkpoint_") : -len(".json")]))
                except ValueError:
                    continue
        return max(epochs) if epochs else None


# ---------------------------------------------------------------------------
# Evaluate function protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class EvalFn(Protocol):
    """Async callable that evaluates agent performance on data."""

    async def __call__(
        self,
        agent: Any,
        data: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return evaluated cases with at least ``input``, ``output``, ``score``."""
        ...


# ---------------------------------------------------------------------------
# OperatorTrainer configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OperatorTrainConfig(TrainConfig):
    """Extended training configuration for OperatorTrainer.

    Args:
        checkpoint_dir: Directory for checkpoint files. Empty string disables.
        resume_from: Epoch number to resume from, or *None* to start fresh.
    """

    checkpoint_dir: str = ""
    resume_from: int | None = None


# ---------------------------------------------------------------------------
# OperatorTrainer
# ---------------------------------------------------------------------------


class OperatorTrainer(Trainer):
    """Trains operator-based agents via evaluate → backward → step → apply loop.

    Extends :class:`Trainer` to work with agents that expose :class:`Operator`
    instances. Uses :class:`InstructionOptimizer` (or similar) for backward/step
    phases and persists state via :class:`FileCheckpointStore`.

    Args:
        config: Training configuration.
        optimizer: Object with ``backward(cases)`` and ``step()`` methods.
        eval_fn: Async callable ``(agent, data) -> list[evaluated_case_dicts]``.
        checkpoint_store: Custom checkpoint store. If *None* and
            ``config.checkpoint_dir`` is set, a :class:`FileCheckpointStore`
            is created automatically.
        resume_from: Epoch to resume from. Overrides ``config.resume_from``.
    """

    __slots__ = (
        "_agent",
        "_checkpoint_store",
        "_eval_fn",
        "_optimizer",
        "_resume_from",
        "_test_data",
        "_train_data",
    )

    def __init__(
        self,
        config: OperatorTrainConfig | TrainConfig | None = None,
        *,
        optimizer: Any = None,
        eval_fn: EvalFn | Any = None,
        checkpoint_store: CheckpointStore | None = None,
        resume_from: int | None = None,
    ) -> None:
        if config is None:
            config = OperatorTrainConfig()
        super().__init__(config)
        self._optimizer = optimizer
        self._eval_fn = eval_fn
        self._agent: Any = None
        self._train_data: Sequence[dict[str, Any]] = ()
        self._test_data: Sequence[dict[str, Any]] | None = None

        # Checkpoint store.
        if checkpoint_store is not None:
            self._checkpoint_store: CheckpointStore | None = checkpoint_store
        elif isinstance(config, OperatorTrainConfig) and config.checkpoint_dir:
            self._checkpoint_store = FileCheckpointStore(config.checkpoint_dir)
        else:
            self._checkpoint_store = None

        # Resume epoch.
        if resume_from is not None:
            self._resume_from: int | None = resume_from
        elif isinstance(config, OperatorTrainConfig):
            self._resume_from = config.resume_from
        else:
            self._resume_from = None

    # --- Validation phase ---

    def check_agent(self, agent: Any) -> None:
        """Validate that *agent* has operators.

        Checks for an ``operators`` attribute that is a non-empty sequence
        of :class:`Operator` instances.

        Raises:
            TrainerError: If agent has no operators.
        """
        if agent is None:
            msg = "agent is required"
            raise TrainerError(msg)
        operators = getattr(agent, "operators", None)
        if not operators:
            msg = "agent must have a non-empty 'operators' attribute"
            raise TrainerError(msg)
        for op in operators:
            if not isinstance(op, Operator):
                msg = f"all operators must be Operator instances, got {type(op).__name__}"
                raise TrainerError(msg)
        self._agent = agent

    def check_dataset(
        self,
        train_data: Any,
        test_data: Any | None = None,
    ) -> None:
        """Validate training and optional test data.

        Both must be sequences of dicts with at least an ``input`` key.

        Raises:
            TrainerError: If data is empty or malformed.
        """
        if not train_data:
            msg = "train_data must be a non-empty sequence"
            raise TrainerError(msg)
        if not isinstance(train_data, (list, tuple)):
            msg = "train_data must be a list or tuple"
            raise TrainerError(msg)
        for item in train_data:
            if not isinstance(item, dict) or "input" not in item:
                msg = "each train_data item must be a dict with an 'input' key"
                raise TrainerError(msg)
        self._train_data = list(train_data)

        if test_data is not None:
            if not isinstance(test_data, (list, tuple)):
                msg = "test_data must be a list or tuple"
                raise TrainerError(msg)
            self._test_data = list(test_data)

    def check_reward(self, reward_fn: Any | None = None) -> None:
        """No-op — OperatorTrainer uses eval_fn instead of reward_fn."""

    def check_config(
        self,
        config: TrainConfig | dict[str, Any] | None = None,
    ) -> None:
        """Validate configuration and update self._config if a new config is provided.

        Raises:
            TrainerError: If epochs < 1.
        """
        cfg: TrainConfig
        if isinstance(config, TrainConfig):
            cfg = config
            self._config = cfg
        elif isinstance(config, dict):
            cfg = TrainConfig(**config)
            self._config = cfg
        else:
            cfg = self._config
        if cfg.epochs < 1:
            msg = "epochs must be >= 1"
            raise TrainerError(msg)

    # --- Training phase ---

    async def train(self) -> TrainMetrics:
        """Execute the operator training loop.

        For each epoch:
        1. **Evaluate** — run eval_fn on train_data to get scored cases.
        2. **Backward** — generate textual gradients from evaluated cases.
        3. **Step** — apply gradients to produce improved operator params.
        4. **Validate** — if test_data exists, evaluate on it.
        5. **Checkpoint** — save operator + optimizer states.

        Returns:
            :class:`TrainMetrics` with final loss and accuracy.

        Raises:
            TrainerError: If not validated or missing optimizer/eval_fn.
        """
        self._require_validated()
        if self._optimizer is None:
            msg = "optimizer is required for training"
            raise TrainerError(msg)
        if self._eval_fn is None:
            msg = "eval_fn is required for training"
            raise TrainerError(msg)

        self._state = TrainerState.TRAINING

        start_epoch = 0
        best_accuracy = 0.0
        total_loss = 0.0

        # Resume from checkpoint if requested.
        if self._resume_from is not None and self._checkpoint_store is not None:
            checkpoint = self._checkpoint_store.load(self._resume_from)
            if checkpoint is not None:
                self._restore_checkpoint(checkpoint)
                start_epoch = self._resume_from + 1
                best_accuracy = checkpoint.get("best_accuracy", 0.0)

        try:
            for epoch in range(start_epoch, self._config.epochs):
                # 1. Evaluate on training data.
                eval_cases = await self._eval_fn(self._agent, list(self._train_data))

                # 2. Backward — generate gradients.
                await self._optimizer.backward(eval_cases)

                # 3. Step — apply gradients.
                await self._optimizer.step()

                # 4. Compute epoch metrics from eval cases.
                scores = [c.get("score", 0.0) for c in eval_cases if "score" in c]
                epoch_accuracy = sum(scores) / len(scores) if scores else 0.0
                epoch_loss = 1.0 - epoch_accuracy
                total_loss = epoch_loss
                if epoch_accuracy > best_accuracy:
                    best_accuracy = epoch_accuracy

                # 5. Validate on test data if available.
                if self._test_data:
                    test_cases = await self._eval_fn(self._agent, list(self._test_data))
                    test_scores = [c.get("score", 0.0) for c in test_cases if "score" in c]
                    if test_scores:
                        test_accuracy = sum(test_scores) / len(test_scores)
                        if test_accuracy > best_accuracy:
                            best_accuracy = test_accuracy

                # 6. Checkpoint.
                if self._checkpoint_store is not None:
                    self._save_checkpoint(epoch, best_accuracy)

        except Exception:
            self._state = TrainerState.FAILED
            raise

        self._state = TrainerState.COMPLETED
        return TrainMetrics(
            loss=total_loss,
            accuracy=best_accuracy,
            steps=self._config.epochs - start_epoch,
        )

    async def evaluate(self, test_data: Any | None = None) -> TrainMetrics:
        """Run evaluation on test data.

        Args:
            test_data: Test dataset. Falls back to data from check_dataset.

        Returns:
            Evaluation metrics.

        Raises:
            TrainerError: If no eval_fn or no test data available.
        """
        if self._eval_fn is None:
            msg = "eval_fn is required for evaluation"
            raise TrainerError(msg)
        data = test_data or self._test_data or self._train_data
        if not data:
            msg = "no evaluation data available"
            raise TrainerError(msg)

        cases = await self._eval_fn(self._agent, list(data))
        scores = [c.get("score", 0.0) for c in cases if "score" in c]
        accuracy = sum(scores) / len(scores) if scores else 0.0
        return TrainMetrics(
            loss=1.0 - accuracy,
            accuracy=accuracy,
            steps=len(cases),
        )

    # --- Checkpoint helpers ---

    def _save_checkpoint(self, epoch: int, best_accuracy: float) -> None:
        """Save operator states and metadata to checkpoint store."""
        assert self._checkpoint_store is not None
        operators = getattr(self._agent, "operators", [])
        operator_states = {op.name: op.get_state() for op in operators}
        data = {
            "epoch": epoch,
            "best_accuracy": best_accuracy,
            "operator_states": operator_states,
        }
        self._checkpoint_store.save(epoch, data)

    def _restore_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Restore operator states from checkpoint data."""
        operator_states = checkpoint.get("operator_states", {})
        operators = getattr(self._agent, "operators", [])
        for op in operators:
            if op.name in operator_states:
                op.load_state(operator_states[op.name])
