"""Base Trainer ABC for agent fine-tuning.

Defines the multi-phase lifecycle for training agents:
1. Validation phase — check_agent, check_dataset, check_reward, check_config
2. Training phase  — train()
3. Evaluation phase — evaluate()

Subclasses implement concrete backends (VeRL, TRL, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from exo.types import ExoError  # pyright: ignore[reportMissingImports]


class TrainerError(ExoError):
    """Error during training operations."""


class TrainerState(StrEnum):
    """Trainer lifecycle state."""

    CREATED = "created"
    VALIDATED = "validated"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(slots=True)
class TrainMetrics:
    """Metrics produced by training or evaluation.

    Concrete trainers may extend with framework-specific fields via *extra*.
    """

    loss: float = 0.0
    accuracy: float = 0.0
    steps: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainConfig:
    """Minimal training configuration.

    Subclass or extend via *extra* for backend-specific settings.
    """

    epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 1e-5
    output_dir: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class Trainer(ABC):
    """Abstract base class for agent training frameworks.

    The lifecycle follows a strict phase ordering:
    1. Call ``check_agent``, ``check_dataset``, ``check_reward``, ``check_config``
       (in any order) to validate all inputs.
    2. Call ``mark_validated`` when all checks pass.
    3. Call ``train`` to execute training.
    4. Call ``evaluate`` to run evaluation on a test set.
    """

    __slots__ = ("_config", "_state")

    def __init__(self, config: TrainConfig | None = None) -> None:
        self._config = config or TrainConfig()
        self._state = TrainerState.CREATED

    # --- Properties ---

    @property
    def state(self) -> TrainerState:
        """Current lifecycle state."""
        return self._state

    @property
    def config(self) -> TrainConfig:
        """Training configuration."""
        return self._config

    # --- Validation phase ---

    @abstractmethod
    def check_agent(self, agent: Any) -> None:
        """Validate that *agent* meets training requirements.

        Raises:
            TrainerError: If validation fails.
        """

    @abstractmethod
    def check_dataset(
        self,
        train_data: Any,
        test_data: Any | None = None,
    ) -> None:
        """Validate training (and optional test) data.

        Raises:
            TrainerError: If validation fails.
        """

    @abstractmethod
    def check_reward(self, reward_fn: Any | None = None) -> None:
        """Validate reward function or scoring mechanism.

        Raises:
            TrainerError: If validation fails.
        """

    @abstractmethod
    def check_config(self, config: TrainConfig | dict[str, Any] | None = None) -> None:
        """Validate and optionally update training configuration.

        Implementations may merge *config* into ``self._config``.

        Raises:
            TrainerError: If configuration is invalid.
        """

    def mark_validated(self) -> None:
        """Transition to VALIDATED state after all checks pass."""
        if self._state != TrainerState.CREATED:
            msg = f"Cannot validate from state {self._state!r}"
            raise TrainerError(msg)
        self._state = TrainerState.VALIDATED

    # --- Training phase ---

    @abstractmethod
    async def train(self) -> TrainMetrics:
        """Execute the training loop.

        Must be called after ``mark_validated``.

        Returns:
            Training metrics.

        Raises:
            TrainerError: If training fails or trainer not validated.
        """

    def _require_validated(self) -> None:
        """Guard: raise if not in VALIDATED state."""
        if self._state != TrainerState.VALIDATED:
            msg = f"Trainer must be validated before training (state={self._state!r})"
            raise TrainerError(msg)

    # --- Evaluation phase ---

    @abstractmethod
    async def evaluate(self, test_data: Any | None = None) -> TrainMetrics:
        """Run evaluation on test data.

        Args:
            test_data: Optional test dataset. If *None*, use data from check_dataset.

        Returns:
            Evaluation metrics.

        Raises:
            TrainerError: If evaluation fails.
        """
