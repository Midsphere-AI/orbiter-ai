"""Tests for agent evolution utilities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from exo.train.evolution import (  # pyright: ignore[reportMissingImports]
    EpochResult,
    EvolutionConfig,
    EvolutionError,
    EvolutionPhase,
    EvolutionPipeline,
    EvolutionResult,
    EvolutionState,
    EvolutionStrategy,
)

# ---------------------------------------------------------------------------
# Stub strategy for testing
# ---------------------------------------------------------------------------


class _StubStrategy(EvolutionStrategy):
    """Minimal strategy that tracks calls and returns predictable values."""

    def __init__(self, *, fail_at_epoch: int | None = None) -> None:
        self.synthesise_calls: list[int] = []
        self.train_calls: list[int] = []
        self.evaluate_calls: list[int] = []
        self._fail_at_epoch = fail_at_epoch

    async def synthesize(
        self,
        agent: Any,
        data: Sequence[dict[str, Any]],
        epoch: int,
    ) -> list[dict[str, Any]]:
        self.synthesise_calls.append(epoch)
        if self._fail_at_epoch == epoch:
            msg = f"Synthesis failed at epoch {epoch}"
            raise RuntimeError(msg)
        # Return data with one extra item per epoch
        return [*data, {"input": f"synth-{epoch}", "output": f"out-{epoch}"}]

    async def train(
        self,
        agent: Any,
        data: Sequence[dict[str, Any]],
        epoch: int,
    ) -> float:
        self.train_calls.append(epoch)
        # Decreasing loss over epochs
        return max(0.1, 1.0 - 0.3 * epoch)

    async def evaluate(
        self,
        agent: Any,
        data: Sequence[dict[str, Any]],
        epoch: int,
    ) -> float:
        self.evaluate_calls.append(epoch)
        # Increasing accuracy over epochs
        return min(1.0, 0.5 + 0.15 * epoch)


# ---------------------------------------------------------------------------
# EvolutionError
# ---------------------------------------------------------------------------


class TestEvolutionError:
    def test_is_exception(self) -> None:
        err = EvolutionError("boom")
        assert isinstance(err, Exception)
        assert str(err) == "boom"


# ---------------------------------------------------------------------------
# EvolutionPhase
# ---------------------------------------------------------------------------


class TestEvolutionPhase:
    def test_values(self) -> None:
        assert EvolutionPhase.SYNTHESIS == "synthesis"
        assert EvolutionPhase.TRAINING == "training"
        assert EvolutionPhase.EVALUATION == "evaluation"

    def test_is_str(self) -> None:
        assert isinstance(EvolutionPhase.SYNTHESIS, str)


# ---------------------------------------------------------------------------
# EvolutionState
# ---------------------------------------------------------------------------


class TestEvolutionState:
    def test_values(self) -> None:
        assert EvolutionState.IDLE == "idle"
        assert EvolutionState.RUNNING == "running"
        assert EvolutionState.COMPLETED == "completed"
        assert EvolutionState.FAILED == "failed"


# ---------------------------------------------------------------------------
# EvolutionConfig
# ---------------------------------------------------------------------------


class TestEvolutionConfig:
    def test_defaults(self) -> None:
        cfg = EvolutionConfig()
        assert cfg.max_epochs == 1
        assert len(cfg.phases) == 3
        assert EvolutionPhase.SYNTHESIS in cfg.phases
        assert EvolutionPhase.TRAINING in cfg.phases
        assert EvolutionPhase.EVALUATION in cfg.phases
        assert cfg.early_stop_threshold is None
        assert cfg.extra == {}

    def test_custom(self) -> None:
        cfg = EvolutionConfig(
            max_epochs=5,
            phases=(EvolutionPhase.TRAINING,),
            early_stop_threshold=0.95,
            extra={"lr": 1e-4},
        )
        assert cfg.max_epochs == 5
        assert len(cfg.phases) == 1
        assert cfg.early_stop_threshold == 0.95
        assert cfg.extra["lr"] == 1e-4

    def test_frozen(self) -> None:
        cfg = EvolutionConfig()
        with pytest.raises(AttributeError):
            cfg.max_epochs = 5  # type: ignore[misc]

    def test_invalid_max_epochs(self) -> None:
        with pytest.raises(ValueError, match="max_epochs"):
            EvolutionConfig(max_epochs=0)

    def test_invalid_threshold_low(self) -> None:
        with pytest.raises(ValueError, match="early_stop_threshold"):
            EvolutionConfig(early_stop_threshold=-0.1)

    def test_invalid_threshold_high(self) -> None:
        with pytest.raises(ValueError, match="early_stop_threshold"):
            EvolutionConfig(early_stop_threshold=1.5)

    def test_threshold_boundary(self) -> None:
        cfg_zero = EvolutionConfig(early_stop_threshold=0.0)
        assert cfg_zero.early_stop_threshold == 0.0
        cfg_one = EvolutionConfig(early_stop_threshold=1.0)
        assert cfg_one.early_stop_threshold == 1.0


# ---------------------------------------------------------------------------
# EpochResult
# ---------------------------------------------------------------------------


class TestEpochResult:
    def test_defaults(self) -> None:
        e = EpochResult()
        assert e.epoch == 0
        assert e.synthesis_count == 0
        assert e.train_loss == 0.0
        assert e.eval_accuracy == 0.0
        assert e.extra == {}

    def test_custom(self) -> None:
        e = EpochResult(epoch=2, synthesis_count=10, train_loss=0.3, eval_accuracy=0.85)
        assert e.epoch == 2
        assert e.synthesis_count == 10
        assert e.train_loss == 0.3
        assert e.eval_accuracy == 0.85


# ---------------------------------------------------------------------------
# EvolutionResult
# ---------------------------------------------------------------------------


class TestEvolutionResult:
    def test_defaults(self) -> None:
        r = EvolutionResult()
        assert r.total_epochs == 0
        assert r.final_accuracy == 0.0
        assert r.early_stopped is False
        assert r.best_epoch is None

    def test_with_epochs(self) -> None:
        r = EvolutionResult(
            epochs=[
                EpochResult(epoch=0, eval_accuracy=0.5),
                EpochResult(epoch=1, eval_accuracy=0.8),
                EpochResult(epoch=2, eval_accuracy=0.7),
            ],
            final_accuracy=0.7,
        )
        assert r.total_epochs == 3
        assert r.best_epoch is not None
        assert r.best_epoch.epoch == 1
        assert r.best_epoch.eval_accuracy == 0.8


# ---------------------------------------------------------------------------
# EvolutionStrategy ABC
# ---------------------------------------------------------------------------


class TestEvolutionStrategyABC:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            EvolutionStrategy()  # type: ignore[abstract]

    async def test_concrete_subclass(self) -> None:
        strategy = _StubStrategy()
        items = await strategy.synthesise("agent", [{"x": 1}], 0)
        assert len(items) == 2


# ---------------------------------------------------------------------------
# EvolutionPipeline init
# ---------------------------------------------------------------------------


class TestEvolutionPipelineInit:
    def test_defaults(self) -> None:
        p = EvolutionPipeline(_StubStrategy())
        assert p.config.max_epochs == 1
        assert p.state == EvolutionState.IDLE
        assert isinstance(p.strategy, _StubStrategy)

    def test_custom_config(self) -> None:
        cfg = EvolutionConfig(max_epochs=3)
        p = EvolutionPipeline(_StubStrategy(), cfg)
        assert p.config.max_epochs == 3

    def test_repr(self) -> None:
        p = EvolutionPipeline(_StubStrategy())
        r = repr(p)
        assert "EvolutionPipeline" in r
        assert "idle" in r


# ---------------------------------------------------------------------------
# EvolutionPipeline.run — single epoch
# ---------------------------------------------------------------------------


class TestEvolutionPipelineSingleEpoch:
    async def test_single_epoch(self) -> None:
        strategy = _StubStrategy()
        p = EvolutionPipeline(strategy)
        result = await p.run("agent", [{"input": "q1"}])

        assert result.total_epochs == 1
        assert result.final_accuracy == 0.5
        assert result.early_stopped is False
        assert p.state == EvolutionState.COMPLETED

        assert strategy.synthesise_calls == [0]
        assert strategy.train_calls == [0]
        assert strategy.evaluate_calls == [0]


# ---------------------------------------------------------------------------
# EvolutionPipeline.run — multi-epoch
# ---------------------------------------------------------------------------


class TestEvolutionPipelineMultiEpoch:
    async def test_multi_epoch(self) -> None:
        cfg = EvolutionConfig(max_epochs=3)
        strategy = _StubStrategy()
        p = EvolutionPipeline(strategy, cfg)
        result = await p.run("agent", [{"input": "q1"}])

        assert result.total_epochs == 3
        assert len(strategy.synthesise_calls) == 3
        assert len(strategy.train_calls) == 3
        assert len(strategy.evaluate_calls) == 3

        # Verify epoch metrics
        assert result.epochs[0].epoch == 0
        assert result.epochs[1].epoch == 1
        assert result.epochs[2].epoch == 2

        # Accuracy increases each epoch
        assert result.epochs[0].eval_accuracy < result.epochs[2].eval_accuracy

    async def test_synthesis_grows_data(self) -> None:
        cfg = EvolutionConfig(max_epochs=2)
        strategy = _StubStrategy()
        p = EvolutionPipeline(strategy, cfg)
        result = await p.run("agent", [{"input": "seed"}])

        # Epoch 0: 1 seed + 1 synth = 2 items
        assert result.epochs[0].synthesis_count == 2
        # Epoch 1: 2 items + 1 synth = 3 items
        assert result.epochs[1].synthesis_count == 3


# ---------------------------------------------------------------------------
# EvolutionPipeline.run — early stopping
# ---------------------------------------------------------------------------


class TestEvolutionPipelineEarlyStopping:
    async def test_early_stop(self) -> None:
        # Accuracy at epoch 0=0.5, 1=0.65, 2=0.8 — threshold 0.6
        cfg = EvolutionConfig(max_epochs=5, early_stop_threshold=0.6)
        strategy = _StubStrategy()
        p = EvolutionPipeline(strategy, cfg)
        result = await p.run("agent", [{"input": "q1"}])

        assert result.early_stopped is True
        assert result.total_epochs == 2  # Stopped at epoch 1 (accuracy 0.65 >= 0.6)

    async def test_no_early_stop_without_threshold(self) -> None:
        cfg = EvolutionConfig(max_epochs=3)
        strategy = _StubStrategy()
        p = EvolutionPipeline(strategy, cfg)
        result = await p.run("agent", [{"input": "q1"}])

        assert result.early_stopped is False
        assert result.total_epochs == 3


# ---------------------------------------------------------------------------
# EvolutionPipeline.run — partial phases
# ---------------------------------------------------------------------------


class TestEvolutionPipelinePartialPhases:
    async def test_training_only(self) -> None:
        cfg = EvolutionConfig(phases=(EvolutionPhase.TRAINING,))
        strategy = _StubStrategy()
        p = EvolutionPipeline(strategy, cfg)
        result = await p.run("agent", [{"input": "q1"}])

        assert strategy.synthesise_calls == []
        assert strategy.train_calls == [0]
        assert strategy.evaluate_calls == []
        assert result.epochs[0].synthesis_count == 0
        assert result.epochs[0].eval_accuracy == 0.0

    async def test_evaluation_only(self) -> None:
        cfg = EvolutionConfig(phases=(EvolutionPhase.EVALUATION,))
        strategy = _StubStrategy()
        p = EvolutionPipeline(strategy, cfg)
        result = await p.run("agent", [{"input": "q1"}])

        assert strategy.synthesise_calls == []
        assert strategy.train_calls == []
        assert strategy.evaluate_calls == [0]
        assert result.final_accuracy == 0.5


# ---------------------------------------------------------------------------
# EvolutionPipeline.run — error handling
# ---------------------------------------------------------------------------


class TestEvolutionPipelineErrors:
    async def test_not_idle_raises(self) -> None:
        strategy = _StubStrategy()
        p = EvolutionPipeline(strategy)
        await p.run("agent", [])
        assert p.state == EvolutionState.COMPLETED

        with pytest.raises(EvolutionError, match="must be idle"):
            await p.run("agent", [])

    async def test_reset_allows_rerun(self) -> None:
        strategy = _StubStrategy()
        p = EvolutionPipeline(strategy)
        await p.run("agent", [])
        p.reset()
        assert p.state == EvolutionState.IDLE
        result = await p.run("agent", [])
        assert result.total_epochs == 1

    async def test_strategy_failure(self) -> None:
        strategy = _StubStrategy(fail_at_epoch=0)
        p = EvolutionPipeline(strategy)

        with pytest.raises(EvolutionError, match="Evolution failed"):
            await p.run("agent", [{"input": "q1"}])
        assert p.state == EvolutionState.FAILED

    async def test_strategy_failure_mid_run(self) -> None:
        cfg = EvolutionConfig(max_epochs=3)
        strategy = _StubStrategy(fail_at_epoch=1)
        p = EvolutionPipeline(strategy, cfg)

        with pytest.raises(EvolutionError, match="epoch 1"):
            await p.run("agent", [{"input": "q1"}])
        assert p.state == EvolutionState.FAILED


# ---------------------------------------------------------------------------
# EvolutionPipeline — full lifecycle integration
# ---------------------------------------------------------------------------


class TestEvolutionLifecycle:
    async def test_full_lifecycle(self) -> None:
        """Full evolution: 3 epochs, all phases, verify metrics progression."""
        cfg = EvolutionConfig(max_epochs=3)
        strategy = _StubStrategy()
        p = EvolutionPipeline(strategy, cfg)

        assert p.state == EvolutionState.IDLE
        result = await p.run("my-agent", [{"input": "seed-data"}])
        assert p.state == EvolutionState.COMPLETED

        assert result.total_epochs == 3
        assert not result.early_stopped

        # Verify training loss decreases
        losses = [e.train_loss for e in result.epochs]
        assert losses[0] > losses[-1]

        # Verify accuracy increases
        accuracies = [e.eval_accuracy for e in result.epochs]
        assert accuracies[0] < accuracies[-1]

        # Best epoch should be the last one (highest accuracy)
        assert result.best_epoch is not None
        assert result.best_epoch.epoch == 2

        # Reset and re-run
        p.reset()
        assert p.state == EvolutionState.IDLE
        result2 = await p.run("my-agent", [{"input": "seed-data"}])
        assert result2.total_epochs == 3
