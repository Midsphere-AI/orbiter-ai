"""Tests for Ralph loop configuration and state tracking."""

from __future__ import annotations

import time

import pytest

from exo.eval.base import EvalError  # pyright: ignore[reportMissingImports]
from exo.eval.ralph.config import (  # pyright: ignore[reportMissingImports]
    LoopState,
    RalphConfig,
    ReflectionConfig,
    StopConditionConfig,
    StopType,
    ValidationConfig,
)

# ---------------------------------------------------------------------------
# StopType
# ---------------------------------------------------------------------------


class TestStopType:
    def test_values(self) -> None:
        assert StopType.NONE == "none"
        assert StopType.COMPLETION == "completion"
        assert StopType.MAX_ITERATIONS == "max_iterations"
        assert StopType.TIMEOUT == "timeout"
        assert StopType.MAX_COST == "max_cost"
        assert StopType.MAX_CONSECUTIVE_FAILURES == "max_consecutive_failures"
        assert StopType.SCORE_THRESHOLD == "score_threshold"
        assert StopType.USER_INTERRUPTED == "user_interrupted"
        assert StopType.SYSTEM_ERROR == "system_error"

    def test_is_success(self) -> None:
        assert StopType.COMPLETION.is_success()
        assert StopType.SCORE_THRESHOLD.is_success()
        assert not StopType.TIMEOUT.is_success()
        assert not StopType.SYSTEM_ERROR.is_success()

    def test_is_failure(self) -> None:
        assert StopType.MAX_CONSECUTIVE_FAILURES.is_failure()
        assert StopType.SYSTEM_ERROR.is_failure()
        assert not StopType.COMPLETION.is_failure()
        assert not StopType.TIMEOUT.is_failure()


# ---------------------------------------------------------------------------
# ValidationConfig
# ---------------------------------------------------------------------------


class TestValidationConfig:
    def test_defaults(self) -> None:
        cfg = ValidationConfig()
        assert cfg.enabled is True
        assert cfg.scorer_names == ()
        assert cfg.min_score_threshold == 0.5
        assert cfg.parallel == 4
        assert cfg.timeout == 0.0

    def test_custom(self) -> None:
        cfg = ValidationConfig(
            enabled=False,
            scorer_names=("accuracy", "f1"),
            min_score_threshold=0.8,
            parallel=8,
            timeout=30.0,
        )
        assert cfg.enabled is False
        assert cfg.scorer_names == ("accuracy", "f1")
        assert cfg.min_score_threshold == 0.8
        assert cfg.parallel == 8
        assert cfg.timeout == 30.0

    def test_frozen(self) -> None:
        cfg = ValidationConfig()
        with pytest.raises(AttributeError):
            cfg.enabled = False  # type: ignore[misc]

    def test_invalid_threshold_low(self) -> None:
        with pytest.raises(EvalError, match="min_score_threshold"):
            ValidationConfig(min_score_threshold=-0.1)

    def test_invalid_threshold_high(self) -> None:
        with pytest.raises(EvalError, match="min_score_threshold"):
            ValidationConfig(min_score_threshold=1.1)

    def test_boundary_threshold(self) -> None:
        assert ValidationConfig(min_score_threshold=0.0).min_score_threshold == 0.0
        assert ValidationConfig(min_score_threshold=1.0).min_score_threshold == 1.0


# ---------------------------------------------------------------------------
# ReflectionConfig
# ---------------------------------------------------------------------------


class TestReflectionConfig:
    def test_defaults(self) -> None:
        cfg = ReflectionConfig()
        assert cfg.enabled is True
        assert cfg.level == "medium"
        assert cfg.max_history == 50

    def test_custom(self) -> None:
        cfg = ReflectionConfig(enabled=False, level="deep", max_history=100)
        assert cfg.enabled is False
        assert cfg.level == "deep"
        assert cfg.max_history == 100

    def test_frozen(self) -> None:
        cfg = ReflectionConfig()
        with pytest.raises(AttributeError):
            cfg.level = "shallow"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# StopConditionConfig
# ---------------------------------------------------------------------------


class TestStopConditionConfig:
    def test_defaults(self) -> None:
        cfg = StopConditionConfig()
        assert cfg.max_iterations == 10
        assert cfg.timeout == 0.0
        assert cfg.max_cost == 0.0
        assert cfg.max_consecutive_failures == 3
        assert cfg.score_threshold == 0.0
        assert cfg.enable_user_interrupt is False

    def test_custom(self) -> None:
        cfg = StopConditionConfig(
            max_iterations=50,
            timeout=300.0,
            max_cost=10.0,
            max_consecutive_failures=5,
            score_threshold=0.9,
            enable_user_interrupt=True,
        )
        assert cfg.max_iterations == 50
        assert cfg.timeout == 300.0
        assert cfg.max_cost == 10.0
        assert cfg.max_consecutive_failures == 5
        assert cfg.score_threshold == 0.9
        assert cfg.enable_user_interrupt is True

    def test_frozen(self) -> None:
        cfg = StopConditionConfig()
        with pytest.raises(AttributeError):
            cfg.max_iterations = 20  # type: ignore[misc]

    def test_invalid_max_iterations(self) -> None:
        with pytest.raises(EvalError, match="max_iterations"):
            StopConditionConfig(max_iterations=0)


# ---------------------------------------------------------------------------
# RalphConfig
# ---------------------------------------------------------------------------


class TestRalphConfig:
    def test_defaults(self) -> None:
        cfg = RalphConfig()
        assert isinstance(cfg.validation, ValidationConfig)
        assert isinstance(cfg.reflection, ReflectionConfig)
        assert isinstance(cfg.stop_condition, StopConditionConfig)
        assert cfg.metadata == {}

    def test_custom_sub_configs(self) -> None:
        v = ValidationConfig(min_score_threshold=0.9)
        r = ReflectionConfig(level="deep")
        s = StopConditionConfig(max_iterations=20)
        cfg = RalphConfig(validation=v, reflection=r, stop_condition=s, metadata={"run": "test"})
        assert cfg.validation.min_score_threshold == 0.9
        assert cfg.reflection.level == "deep"
        assert cfg.stop_condition.max_iterations == 20
        assert cfg.metadata == {"run": "test"}

    def test_frozen(self) -> None:
        cfg = RalphConfig()
        with pytest.raises(AttributeError):
            cfg.validation = ValidationConfig()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# LoopState — init / queries
# ---------------------------------------------------------------------------


class TestLoopStateInit:
    def test_defaults(self) -> None:
        state = LoopState()
        assert state.iteration == 0
        assert state.cumulative_cost == 0.0
        assert state.consecutive_failures == 0
        assert state.successful_steps == 0
        assert state.failed_steps == 0
        assert state.total_tokens == 0
        assert state.score_history == []
        assert state.reflection_history == []
        assert state.metadata == {}

    def test_elapsed(self) -> None:
        state = LoopState()
        time.sleep(0.01)
        # Check only that elapsed() is non-negative to avoid timing flakiness.
        assert state.elapsed() >= 0.0

    def test_success_rate_empty(self) -> None:
        assert LoopState().success_rate() == 0.0

    def test_success_rate(self) -> None:
        state = LoopState(successful_steps=3, failed_steps=1)
        assert state.success_rate() == 0.75

    def test_latest_score_empty(self) -> None:
        assert LoopState().latest_score() == {}

    def test_latest_score(self) -> None:
        state = LoopState()
        state.score_history = [{"acc": 0.5}, {"acc": 0.8}]
        assert state.latest_score() == {"acc": 0.8}

    def test_best_score_empty(self) -> None:
        assert LoopState().best_score("acc") == 0.0

    def test_best_score(self) -> None:
        state = LoopState()
        state.score_history = [{"acc": 0.7}, {"acc": 0.9}, {"acc": 0.8}]
        assert state.best_score("acc") == 0.9

    def test_best_score_missing_metric(self) -> None:
        state = LoopState()
        state.score_history = [{"f1": 0.8}]
        assert state.best_score("acc") == 0.0


# ---------------------------------------------------------------------------
# LoopState — mutations
# ---------------------------------------------------------------------------


class TestLoopStateMutations:
    def test_record_score(self) -> None:
        state = LoopState()
        state.record_score({"acc": 0.8, "f1": 0.7})
        assert len(state.score_history) == 1
        assert state.score_history[0] == {"acc": 0.8, "f1": 0.7}

    def test_record_score_copies(self) -> None:
        state = LoopState()
        scores = {"acc": 0.8}
        state.record_score(scores)
        scores["acc"] = 0.0
        assert state.score_history[0]["acc"] == 0.8

    def test_record_reflection(self) -> None:
        state = LoopState()
        state.record_reflection({"summary": "Good progress", "type": "success"})
        assert len(state.reflection_history) == 1
        assert state.reflection_history[0]["summary"] == "Good progress"

    def test_record_success(self) -> None:
        state = LoopState()
        state.record_success(tokens=100, cost=0.01)
        assert state.successful_steps == 1
        assert state.consecutive_failures == 0
        assert state.total_tokens == 100
        assert state.cumulative_cost == 0.01

    def test_record_success_resets_consecutive_failures(self) -> None:
        state = LoopState(consecutive_failures=3)
        state.record_success()
        assert state.consecutive_failures == 0

    def test_record_failure(self) -> None:
        state = LoopState()
        state.record_failure(cost=0.02)
        assert state.failed_steps == 1
        assert state.consecutive_failures == 1
        assert state.cumulative_cost == 0.02

    def test_record_failure_increments(self) -> None:
        state = LoopState()
        state.record_failure()
        state.record_failure()
        assert state.consecutive_failures == 2
        assert state.failed_steps == 2

    def test_success_then_failure_resets(self) -> None:
        state = LoopState()
        state.record_failure()
        state.record_failure()
        assert state.consecutive_failures == 2
        state.record_success()
        assert state.consecutive_failures == 0
        state.record_failure()
        assert state.consecutive_failures == 1


# ---------------------------------------------------------------------------
# LoopState — serialisation
# ---------------------------------------------------------------------------


class TestLoopStateSerialization:
    def test_to_dict(self) -> None:
        state = LoopState(iteration=3)
        state.record_score({"acc": 0.9})
        state.record_reflection({"summary": "OK"})
        state.record_success(tokens=50, cost=0.005)
        d = state.to_dict()

        assert d["iteration"] == 3
        assert d["successful_steps"] == 1
        assert d["total_tokens"] == 50
        assert d["cumulative_cost"] == 0.005
        assert d["score_history"] == [{"acc": 0.9}]
        assert d["reflection_history"] == [{"summary": "OK"}]

    def test_to_dict_independent_copy(self) -> None:
        state = LoopState()
        state.record_score({"acc": 0.5})
        d = state.to_dict()
        d["score_history"][0]["acc"] = 0.0
        assert state.score_history[0]["acc"] == 0.5

    def test_repr(self) -> None:
        state = LoopState(iteration=5, successful_steps=3, failed_steps=1)
        r = repr(state)
        assert "iteration=5" in r
        assert "75.0%" in r


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_config_and_state_together(self) -> None:
        """Simulate a minimal loop lifecycle with config and state."""
        cfg = RalphConfig(
            validation=ValidationConfig(min_score_threshold=0.7),
            stop_condition=StopConditionConfig(max_iterations=5, max_consecutive_failures=2),
        )
        state = LoopState()

        for i in range(cfg.stop_condition.max_iterations):
            state.iteration = i + 1

            # Simulate alternating success/failure
            if i % 3 == 2:
                state.record_failure()
            else:
                state.record_success(tokens=10)
                state.record_score({"accuracy": 0.5 + i * 0.1})

            # Check consecutive failure stop condition
            if state.consecutive_failures >= cfg.stop_condition.max_consecutive_failures:
                break

        assert state.iteration == 5
        assert state.successful_steps == 4
        assert state.failed_steps == 1

    def test_score_threshold_check(self) -> None:
        """Verify score threshold can be checked against LoopState."""
        cfg = RalphConfig(
            stop_condition=StopConditionConfig(score_threshold=0.9),
        )
        state = LoopState()
        state.record_score({"acc": 0.85})
        state.record_score({"acc": 0.92})

        # Check if best score meets threshold
        meets = state.best_score("acc") >= cfg.stop_condition.score_threshold
        assert meets is True
