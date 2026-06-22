"""Refinement loop configuration and iteration state tracking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from exo.eval.base import EvalError  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class StopType(StrEnum):
    """Categorised exit reason for loop termination."""

    NONE = "none"
    COMPLETION = "completion"
    MAX_ITERATIONS = "max_iterations"
    TIMEOUT = "timeout"
    MAX_COST = "max_cost"
    MAX_CONSECUTIVE_FAILURES = "max_consecutive_failures"
    SCORE_THRESHOLD = "score_threshold"
    USER_INTERRUPTED = "user_interrupted"
    SYSTEM_ERROR = "system_error"

    def is_success(self) -> bool:
        """Return ``True`` if the stop type indicates a successful outcome."""
        return self in {StopType.COMPLETION, StopType.SCORE_THRESHOLD}

    def is_failure(self) -> bool:
        """Return ``True`` if the stop type indicates a failure."""
        return self in {
            StopType.MAX_CONSECUTIVE_FAILURES,
            StopType.SYSTEM_ERROR,
        }


# ---------------------------------------------------------------------------
# Sub-configurations
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """Configuration for the Analyze (scoring) phase."""

    enabled: bool = True
    scorer_names: tuple[str, ...] = ()
    min_score_threshold: float = 0.5
    parallel: int = 4
    timeout: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_score_threshold <= 1.0:
            raise EvalError(
                f"min_score_threshold must be in [0, 1], got {self.min_score_threshold}",
                context={"min_score_threshold": self.min_score_threshold},
                hint="Set min_score_threshold to a value between 0.0 and 1.0 (e.g. 0.5).",
            )


@dataclass(frozen=True, slots=True)
class ReflectionConfig:
    """Configuration for the Learn (reflection) phase."""

    enabled: bool = True
    level: str = "medium"
    max_history: int = 50


@dataclass(frozen=True, slots=True)
class StopConditionConfig:
    """Configuration for the Halt (stop detection) phase."""

    max_iterations: int = 10
    timeout: float = 0.0
    max_cost: float = 0.0
    max_consecutive_failures: int = 3
    score_threshold: float = 0.0
    # NOTE: enable_user_interrupt is accepted for forward-compatibility but is
    # not yet wired to a USER_INTERRUPTED detector.  Setting it to True has no
    # effect in the current implementation.  A future release will add a
    # signal/event-based detector that raises StopType.USER_INTERRUPTED.
    enable_user_interrupt: bool = False

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise EvalError(
                f"max_iterations must be >= 1, got {self.max_iterations}",
                context={"max_iterations": self.max_iterations},
                hint="Set max_iterations to at least 1 (e.g. max_iterations=10).",
            )


# ---------------------------------------------------------------------------
# RefinementConfig — top-level unified configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RefinementConfig:
    """Unified configuration for the iterative refinement loop.

    Aggregates validation (scoring), reflection, and stop-condition settings.
    """

    validation: ValidationConfig = field(default_factory=ValidationConfig)
    reflection: ReflectionConfig = field(default_factory=ReflectionConfig)
    stop_condition: StopConditionConfig = field(default_factory=StopConditionConfig)
    metadata: dict[str, Any] = field(default_factory=dict)


# Deprecated alias: use RefinementConfig
RalphConfig = RefinementConfig


# ---------------------------------------------------------------------------
# LoopState — mutable iteration tracker
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LoopState:
    """Runtime state for a refinement loop execution.

    Tracks iteration count, timing, cost, and aggregated score/reflection
    history across the iterative refinement lifecycle.
    """

    iteration: int = 0
    start_time: float = field(default_factory=time.monotonic)
    cumulative_cost: float = 0.0
    consecutive_failures: int = 0
    successful_steps: int = 0
    failed_steps: int = 0
    total_tokens: int = 0
    score_history: list[dict[str, float]] = field(default_factory=list)
    reflection_history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---- queries ----------------------------------------------------------

    def elapsed(self) -> float:
        """Seconds since the loop started."""
        return time.monotonic() - self.start_time

    def success_rate(self) -> float:
        """Fraction of successful steps (0.0 when none executed)."""
        total = self.successful_steps + self.failed_steps
        return self.successful_steps / total if total > 0 else 0.0

    def latest_score(self) -> dict[str, float]:
        """Return the most recent score snapshot, or empty dict."""
        return dict(self.score_history[-1]) if self.score_history else {}

    def best_score(self, metric: str) -> float:
        """Return the highest value seen for *metric*, or 0.0."""
        return max(
            (s.get(metric, 0.0) for s in self.score_history),
            default=0.0,
        )

    # ---- mutations --------------------------------------------------------

    def record_score(self, scores: dict[str, float]) -> None:
        """Append a score snapshot for the current iteration."""
        self.score_history.append(dict(scores))

    def record_reflection(self, reflection: dict[str, Any]) -> None:
        """Append a reflection summary for the current iteration."""
        self.reflection_history.append(dict(reflection))

    def record_success(self, *, tokens: int = 0, cost: float = 0.0) -> None:
        """Mark the current step as successful."""
        self.successful_steps += 1
        self.consecutive_failures = 0
        self.total_tokens += tokens
        self.cumulative_cost += cost

    def record_failure(self, *, cost: float = 0.0) -> None:
        """Mark the current step as failed."""
        self.failed_steps += 1
        self.consecutive_failures += 1
        self.cumulative_cost += cost

    # ---- serialisation ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for checkpointing."""
        return {
            "iteration": self.iteration,
            "cumulative_cost": self.cumulative_cost,
            "consecutive_failures": self.consecutive_failures,
            "successful_steps": self.successful_steps,
            "failed_steps": self.failed_steps,
            "total_tokens": self.total_tokens,
            "score_history": [dict(s) for s in self.score_history],
            "reflection_history": [dict(r) for r in self.reflection_history],
            "metadata": dict(self.metadata),
        }

    def __repr__(self) -> str:
        return (
            f"LoopState(iteration={self.iteration}, "
            f"success_rate={self.success_rate():.1%}, "
            f"elapsed={self.elapsed():.1f}s)"
        )
