"""Tests for GaussianMutationStrategy concrete EvolutionStrategy implementation."""

from __future__ import annotations

import math

import pytest

from exo.train.evolution import (  # pyright: ignore[reportMissingImports]
    EvolutionConfig,
    EvolutionPhase,
    EvolutionPipeline,
    EvolutionState,
    EvolutionStrategy,
    GaussianMutationStrategy,
)

# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------


class TestGaussianMutationStrategyInit:
    def test_default(self) -> None:
        s = GaussianMutationStrategy()
        assert isinstance(s, EvolutionStrategy)

    def test_custom_params(self) -> None:
        s = GaussianMutationStrategy(
            mutation_std=0.5,
            augment_factor=3,
            learning_rate=0.05,
            seed=42,
        )
        # Just check construction succeeds; internals are private
        assert repr(s).startswith("GaussianMutationStrategy(")

    def test_invalid_mutation_std(self) -> None:
        with pytest.raises(ValueError, match="mutation_std"):
            GaussianMutationStrategy(mutation_std=-0.1)

    def test_invalid_augment_factor(self) -> None:
        with pytest.raises(ValueError, match="augment_factor"):
            GaussianMutationStrategy(augment_factor=-1)

    def test_is_concrete(self) -> None:
        # Should instantiate without error (not abstract)
        s = GaussianMutationStrategy()
        assert s is not None

    def test_repr(self) -> None:
        s = GaussianMutationStrategy(mutation_std=0.2, augment_factor=2, learning_rate=0.1)
        r = repr(s)
        assert "GaussianMutationStrategy" in r
        assert "0.2" in r
        assert "2" in r


# ---------------------------------------------------------------------------
# synthesise
# ---------------------------------------------------------------------------


class TestGaussianMutationStrategySynthesise:
    async def test_empty_data_returns_empty(self) -> None:
        s = GaussianMutationStrategy(augment_factor=2, seed=0)
        result = await s.synthesise("agent", [], epoch=0)
        assert result == []

    async def test_augment_factor_zero(self) -> None:
        data = [{"input": "q1"}, {"input": "q2"}]
        s = GaussianMutationStrategy(augment_factor=0, seed=0)
        result = await s.synthesise("agent", data, epoch=0)
        # No augmentation; returns only originals
        assert len(result) == 2

    async def test_augment_factor_one(self) -> None:
        data = [{"input": "q1", "score": 1.0}]
        s = GaussianMutationStrategy(augment_factor=1, seed=42)
        result = await s.synthesise("agent", data, epoch=0)
        assert len(result) == 2  # 1 original + 1 copy

    async def test_augment_factor_three(self) -> None:
        data = [{"input": "a"}, {"input": "b"}]
        s = GaussianMutationStrategy(augment_factor=3, seed=0)
        result = await s.synthesise("agent", data, epoch=0)
        # 2 originals + 3 copies each = 8
        assert len(result) == 8

    async def test_numeric_values_mutated(self) -> None:
        data = [{"input": "q", "score": 0.5}]
        s = GaussianMutationStrategy(mutation_std=1.0, augment_factor=1, seed=1)
        result = await s.synthesise("agent", data, epoch=0)
        # The copy should have a different score due to Gaussian noise
        # (with std=1.0 and seed=1, the mutation is non-zero)
        original_score = data[0]["score"]
        copy_score = result[1]["score"]  # index 1 is the mutated copy
        assert isinstance(copy_score, float)
        # They differ with high probability (std=1.0)
        assert copy_score != original_score

    async def test_non_numeric_values_unchanged(self) -> None:
        data = [{"input": "hello", "label": "cat", "active": True}]
        s = GaussianMutationStrategy(mutation_std=1.0, augment_factor=1, seed=7)
        result = await s.synthesise("agent", data, epoch=0)
        copy = result[1]
        assert copy["input"] == "hello"
        assert copy["label"] == "cat"
        assert copy["active"] is True

    async def test_reproducible_with_seed(self) -> None:
        data = [{"input": "x", "val": 3.14}]
        s1 = GaussianMutationStrategy(mutation_std=0.5, augment_factor=1, seed=99)
        s2 = GaussianMutationStrategy(mutation_std=0.5, augment_factor=1, seed=99)
        r1 = await s1.synthesise("agent", data, epoch=0)
        r2 = await s2.synthesise("agent", data, epoch=0)
        assert r1[1]["val"] == r2[1]["val"]


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


class TestGaussianMutationStrategyTrain:
    async def test_returns_float(self) -> None:
        s = GaussianMutationStrategy()
        loss = await s.train("agent", [{"input": "q"}], epoch=0)
        assert isinstance(loss, float)

    async def test_loss_decays_over_epochs(self) -> None:
        s = GaussianMutationStrategy(learning_rate=0.5)
        data = [{"input": "q"}]
        losses = [await s.train("agent", data, epoch=e) for e in range(5)]
        # Each loss should be strictly less than the previous
        for i in range(1, len(losses)):
            assert losses[i] < losses[i - 1]

    async def test_loss_approaches_zero(self) -> None:
        s = GaussianMutationStrategy(learning_rate=1.0)
        loss = await s.train("agent", [], epoch=100)
        # exp(-1.0 * 101) ≈ 0
        assert loss < 1e-30

    async def test_loss_formula(self) -> None:
        lr = 0.3
        s = GaussianMutationStrategy(learning_rate=lr)
        for epoch in range(5):
            expected = math.exp(-lr * (epoch + 1))
            actual = await s.train("agent", [{"input": "q"}], epoch=epoch)
            assert abs(actual - expected) < 1e-10


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


class TestGaussianMutationStrategyEvaluate:
    async def test_empty_data_returns_zero(self) -> None:
        s = GaussianMutationStrategy()
        acc = await s.evaluate("agent", [], epoch=0)
        assert acc == 0.0

    async def test_all_with_output(self) -> None:
        data = [{"input": "q", "output": "a"}, {"input": "q2", "output": "b"}]
        s = GaussianMutationStrategy()
        acc = await s.evaluate("agent", data, epoch=0)
        assert acc == 1.0

    async def test_none_with_output(self) -> None:
        data = [{"input": "q"}, {"input": "q2"}]
        s = GaussianMutationStrategy()
        acc = await s.evaluate("agent", data, epoch=0)
        assert acc == 0.0

    async def test_partial_with_output(self) -> None:
        data = [
            {"input": "q1", "output": "a1"},
            {"input": "q2"},
            {"input": "q3", "output": "a3"},
            {"input": "q4"},
        ]
        s = GaussianMutationStrategy()
        acc = await s.evaluate("agent", data, epoch=0)
        assert acc == pytest.approx(0.5)

    async def test_empty_string_output_not_counted(self) -> None:
        data = [{"input": "q", "output": ""}, {"input": "q2", "output": "ans"}]
        s = GaussianMutationStrategy()
        acc = await s.evaluate("agent", data, epoch=0)
        assert acc == pytest.approx(0.5)

    async def test_returns_float_in_range(self) -> None:
        import random

        rng = random.Random(0)
        data = [
            {"input": f"q{i}", **({"output": f"a{i}"} if rng.random() > 0.5 else {})}
            for i in range(20)
        ]
        s = GaussianMutationStrategy()
        acc = await s.evaluate("agent", data, epoch=0)
        assert 0.0 <= acc <= 1.0


# ---------------------------------------------------------------------------
# Integration: GaussianMutationStrategy inside EvolutionPipeline
# ---------------------------------------------------------------------------


class TestGaussianMutationStrategyInPipeline:
    async def test_single_epoch_pipeline(self) -> None:
        data = [{"input": f"q{i}", "output": f"a{i}"} for i in range(5)]
        strategy = GaussianMutationStrategy(augment_factor=1, seed=0)
        pipeline = EvolutionPipeline(strategy)
        result = await pipeline.run("agent", data)

        assert result.total_epochs == 1
        assert pipeline.state == EvolutionState.COMPLETED
        # All original items have output, so accuracy should be 1.0
        assert result.final_accuracy == pytest.approx(1.0)

    async def test_multi_epoch_pipeline(self) -> None:
        data = [{"input": f"q{i}", "output": f"a{i}"} for i in range(3)]
        cfg = EvolutionConfig(max_epochs=3)
        strategy = GaussianMutationStrategy(augment_factor=1, learning_rate=0.5, seed=42)
        pipeline = EvolutionPipeline(strategy, cfg)
        result = await pipeline.run("agent", data)

        assert result.total_epochs == 3
        # Loss should decrease over epochs
        losses = [e.train_loss for e in result.epochs]
        assert losses[0] > losses[-1]

    async def test_training_only_phase(self) -> None:
        cfg = EvolutionConfig(phases=(EvolutionPhase.TRAINING,))
        strategy = GaussianMutationStrategy()
        pipeline = EvolutionPipeline(strategy, cfg)
        result = await pipeline.run("agent", [{"input": "q"}])

        assert result.total_epochs == 1
        assert result.epochs[0].eval_accuracy == 0.0  # Phase skipped

    async def test_pipeline_exported_from_package(self) -> None:
        import exo.train as train_pkg  # pyright: ignore[reportMissingImports]

        assert train_pkg.GaussianMutationStrategy is GaussianMutationStrategy
