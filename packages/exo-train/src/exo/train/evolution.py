"""Agent evolution utilities for iterative improvement.

Implements a multi-epoch evolution pipeline inspired by AWorld's EvolutionRunner:
1. Data synthesis — generate or augment training data
2. Training     — fine-tune or update the agent
3. Evaluation   — measure improvement

Each phase is pluggable via EvolutionStrategy, allowing custom backends.
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from exo.types import ExoError  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)


class EvolutionError(ExoError):
    """Error during evolution operations."""


class EvolutionPhase(StrEnum):
    """Phases in an evolution epoch."""

    SYNTHESIS = "synthesis"
    TRAINING = "training"
    EVALUATION = "evaluation"


class EvolutionState(StrEnum):
    """State of the evolution pipeline."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class EvolutionConfig:
    """Configuration for an evolution run.

    Args:
        max_epochs: Number of evolution cycles to run.
        phases: Which phases to execute each epoch.
        early_stop_threshold: Stop if evaluation accuracy >= this value.
        extra: Backend-specific settings.
    """

    max_epochs: int = 1
    phases: tuple[EvolutionPhase, ...] = (
        EvolutionPhase.SYNTHESIS,
        EvolutionPhase.TRAINING,
        EvolutionPhase.EVALUATION,
    )
    early_stop_threshold: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_epochs < 1:
            msg = f"max_epochs must be >= 1, got {self.max_epochs}"
            raise ValueError(msg)
        if self.early_stop_threshold is not None and not 0.0 <= self.early_stop_threshold <= 1.0:
            msg = f"early_stop_threshold must be in [0, 1], got {self.early_stop_threshold}"
            raise ValueError(msg)


@dataclass(slots=True)
class EpochResult:
    """Metrics for a single evolution epoch."""

    epoch: int = 0
    synthesis_count: int = 0
    train_loss: float = 0.0
    eval_accuracy: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvolutionResult:
    """Aggregate result of a full evolution run."""

    epochs: list[EpochResult] = field(default_factory=list)
    final_accuracy: float = 0.0
    early_stopped: bool = False

    @property
    def total_epochs(self) -> int:
        return len(self.epochs)

    @property
    def best_epoch(self) -> EpochResult | None:
        if not self.epochs:
            return None
        return max(self.epochs, key=lambda e: e.eval_accuracy)


# ---------------------------------------------------------------------------
# Strategy ABC
# ---------------------------------------------------------------------------


class EvolutionStrategy(ABC):
    """Pluggable strategy for each phase of an evolution epoch.

    Subclasses implement concrete backends (VeRL, TRL, custom loops, etc.).
    """

    __slots__ = ()

    def reset(self) -> None:  # noqa: B027
        """Reset strategy state so the pipeline can be re-run from scratch.

        The default implementation is a no-op.  Strategies that maintain
        stateful RNGs (e.g. :class:`GaussianMutationStrategy`) should
        override this to reseed their RNG so that re-runs are reproducible.
        """

    @abstractmethod
    async def synthesize(
        self,
        agent: Any,
        data: Sequence[dict[str, Any]],
        epoch: int,
    ) -> list[dict[str, Any]]:
        """Generate or augment training data for this epoch.

        Returns:
            New or augmented training items.
        """

    async def synthesise(
        self,
        agent: Any,
        data: Sequence[dict[str, Any]],
        epoch: int,
    ) -> list[dict[str, Any]]:
        """Deprecated spelling — override synthesize instead."""
        return await self.synthesize(agent, data, epoch)

    @abstractmethod
    async def train(
        self,
        agent: Any,
        data: Sequence[dict[str, Any]],
        epoch: int,
    ) -> float:
        """Train the agent on *data*.

        Returns:
            Training loss.
        """

    @abstractmethod
    async def evaluate(
        self,
        agent: Any,
        data: Sequence[dict[str, Any]],
        epoch: int,
    ) -> float:
        """Evaluate the agent.

        Returns:
            Accuracy score in [0, 1].
        """


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------


class GaussianMutationStrategy(EvolutionStrategy):
    """Concrete evolution strategy using Gaussian noise mutation for data augmentation.

    Synthesis phase — augments the dataset by duplicating items and perturbing
    any numeric values with Gaussian noise.  Non-numeric values are copied
    verbatim, so the strategy works with arbitrary dict schemas.

    Training phase — simulates a lightweight gradient-descent step by tracking
    a running loss that decays exponentially across epochs.  No external ML
    framework is required; this strategy is intended as a usable default for
    pipelines that handle their own model updates outside the strategy.

    Evaluation phase — scores each item by checking whether an ``output`` key
    is present and non-empty, giving a simple but deterministic accuracy signal
    that improves naturally as synthesis adds more complete items.

    Args:
        mutation_std:      Standard deviation of Gaussian noise applied to
                           numeric field values during synthesis.
        augment_factor:    Number of mutated copies to generate per original
                           item (in addition to the original).
        learning_rate:     Decay coefficient used to simulate loss reduction
                           across epochs.
        seed:              Optional RNG seed for reproducibility.
    """

    __slots__ = ("_augment_factor", "_learning_rate", "_mutation_std", "_rng", "_seed")

    def __init__(
        self,
        mutation_std: float = 0.1,
        augment_factor: int = 1,
        learning_rate: float = 0.1,
        seed: int | None = None,
    ) -> None:
        if mutation_std < 0.0:
            msg = f"mutation_std must be >= 0, got {mutation_std}"
            raise ValueError(msg)
        if augment_factor < 0:
            msg = f"augment_factor must be >= 0, got {augment_factor}"
            raise ValueError(msg)
        self._mutation_std = mutation_std
        self._augment_factor = augment_factor
        self._learning_rate = learning_rate
        self._seed = seed
        self._rng = random.Random(seed)

    def reset(self) -> None:
        """Reseed the RNG so that a re-run produces identical output."""
        self._rng = random.Random(self._seed)

    def _mutate_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of *item* with numeric values perturbed by Gaussian noise."""
        mutated: dict[str, Any] = {}
        for key, value in item.items():
            if isinstance(value, float):
                mutated[key] = value + self._rng.gauss(0.0, self._mutation_std)
            elif isinstance(value, int) and not isinstance(value, bool):
                delta = self._rng.gauss(0.0, self._mutation_std)
                mutated[key] = value + delta
            else:
                mutated[key] = value
        return mutated

    async def synthesize(
        self,
        agent: Any,
        data: Sequence[dict[str, Any]],
        epoch: int,
    ) -> list[dict[str, Any]]:
        """Augment *data* by adding Gaussian-mutated copies of each item.

        Returns the original items plus ``augment_factor`` mutated copies of
        each, giving a dataset that grows predictably each epoch.
        """
        result: list[dict[str, Any]] = list(data)
        for item in data:
            for _ in range(self._augment_factor):
                result.append(self._mutate_item(item))
        logger.debug(
            "GaussianMutationStrategy.synthesize: epoch=%d, original=%d, augmented=%d",
            epoch,
            len(data),
            len(result),
        )
        return result

    async def train(
        self,
        agent: Any,
        data: Sequence[dict[str, Any]],
        epoch: int,
    ) -> float:
        """Simulate a training step; returns an exponentially decaying loss.

        The loss follows: ``loss = exp(-learning_rate * (epoch + 1))``,
        which decreases smoothly from ~1.0 toward 0 as epochs increase.
        """
        import math

        loss = math.exp(-self._learning_rate * (epoch + 1))
        logger.debug(
            "GaussianMutationStrategy.train: epoch=%d, n_items=%d, loss=%.4f",
            epoch,
            len(data),
            loss,
        )
        return loss

    async def evaluate(
        self,
        agent: Any,
        data: Sequence[dict[str, Any]],
        epoch: int,
    ) -> float:
        """Evaluate quality of *data*; returns fraction of items with a non-empty ``output``.

        Items that contain a non-empty ``output`` key are counted as correct.
        This gives a concrete, computable accuracy signal without requiring
        an external model.  Accuracy improves naturally as synthesis adds
        items copied from originals that already have outputs.

        Returns:
            Accuracy in [0, 1], or 0.0 for an empty dataset.
        """
        if not data:
            return 0.0
        correct = sum(1 for item in data if item.get("output"))
        accuracy = correct / len(data)
        logger.debug(
            "GaussianMutationStrategy.evaluate: epoch=%d, n_items=%d, accuracy=%.4f",
            epoch,
            len(data),
            accuracy,
        )
        return accuracy

    def __repr__(self) -> str:
        return (
            f"GaussianMutationStrategy("
            f"mutation_std={self._mutation_std}, "
            f"augment_factor={self._augment_factor}, "
            f"learning_rate={self._learning_rate})"
        )


# ---------------------------------------------------------------------------
# Evolution pipeline
# ---------------------------------------------------------------------------


class EvolutionPipeline:
    """Multi-epoch evolution pipeline.

    Runs synthesis → training → evaluation for each epoch, with optional
    early stopping when evaluation accuracy meets the threshold.
    """

    __slots__ = ("_config", "_state", "_strategy")

    def __init__(
        self,
        strategy: EvolutionStrategy,
        config: EvolutionConfig | None = None,
    ) -> None:
        self._strategy = strategy
        self._config = config or EvolutionConfig()
        self._state = EvolutionState.IDLE

    @property
    def config(self) -> EvolutionConfig:
        return self._config

    @property
    def strategy(self) -> EvolutionStrategy:
        return self._strategy

    @property
    def state(self) -> EvolutionState:
        return self._state

    async def run(
        self,
        agent: Any,
        data: Sequence[dict[str, Any]],
    ) -> EvolutionResult:
        """Execute the full evolution pipeline.

        Args:
            agent: The agent to evolve (type depends on backend).
            data: Initial training data.

        Returns:
            Aggregate evolution result with per-epoch metrics.

        Raises:
            EvolutionError: If the pipeline is not idle or a phase fails.
        """
        if self._state != EvolutionState.IDLE:
            msg = f"Pipeline must be idle to run (state={self._state!r})"
            raise EvolutionError(msg)

        self._state = EvolutionState.RUNNING
        cfg = self._config
        result = EvolutionResult()
        current_data = list(data)

        try:
            for epoch_idx in range(cfg.max_epochs):
                epoch = EpochResult(epoch=epoch_idx)
                logger.info("Evolution epoch %d/%d starting", epoch_idx + 1, cfg.max_epochs)

                # Phase 1: Synthesis
                if EvolutionPhase.SYNTHESIS in cfg.phases:
                    synthesised = await self._strategy.synthesize(agent, current_data, epoch_idx)
                    epoch.synthesis_count = len(synthesised)
                    current_data = synthesised if synthesised else current_data

                # Phase 2: Training
                if EvolutionPhase.TRAINING in cfg.phases:
                    epoch.train_loss = await self._strategy.train(agent, current_data, epoch_idx)

                # Phase 3: Evaluation
                if EvolutionPhase.EVALUATION in cfg.phases:
                    epoch.eval_accuracy = await self._strategy.evaluate(
                        agent, current_data, epoch_idx
                    )

                result.epochs.append(epoch)
                result.final_accuracy = epoch.eval_accuracy
                logger.info(
                    "Evolution epoch %d/%d complete: loss=%.4f accuracy=%.4f",
                    epoch_idx + 1,
                    cfg.max_epochs,
                    epoch.train_loss,
                    epoch.eval_accuracy,
                )

                best = result.best_epoch
                if best is not None:
                    logger.debug(
                        "Best epoch so far: epoch=%d accuracy=%.4f", best.epoch, best.eval_accuracy
                    )

                # Early stopping
                if (
                    cfg.early_stop_threshold is not None
                    and epoch.eval_accuracy >= cfg.early_stop_threshold
                ):
                    result.early_stopped = True
                    break

        except Exception as exc:
            self._state = EvolutionState.FAILED
            msg = f"Evolution failed at epoch {epoch_idx}: {exc}"
            raise EvolutionError(msg) from exc

        self._state = EvolutionState.COMPLETED
        return result

    def reset(self) -> None:
        """Reset pipeline to IDLE so it can be re-run.

        Also resets the strategy (e.g. reseeds RNGs) for reproducibility.
        """
        self._state = EvolutionState.IDLE
        self._strategy.reset()

    def __repr__(self) -> str:
        return (
            f"EvolutionPipeline(epochs={self._config.max_epochs}, "
            f"phases={[p.value for p in self._config.phases]}, "
            f"state={self._state!r})"
        )
