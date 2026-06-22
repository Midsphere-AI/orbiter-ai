"""Tests for the ACE (Adaptive Context Engine) memory evolution strategy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from exo.memory.base import ExoMemoryError, MemoryItem  # pyright: ignore[reportMissingImports]
from exo.memory.evolution.ace import (  # pyright: ignore[reportMissingImports]
    ACEStrategy,
    Counters,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockModel:
    """Mock LLM that returns a fixed label or per-content mapping."""

    def __init__(self, label: str = "helpful", mapping: dict[str, str] | None = None) -> None:
        self._label = label
        self._mapping = mapping or {}

    async def __call__(self, prompt: str) -> str:
        for content_fragment, label in self._mapping.items():
            if content_fragment in prompt:
                return label
        return self._label


@pytest.fixture()
def sample_items() -> list[MemoryItem]:
    return [
        MemoryItem(content="always useful fact", memory_type="test", id="m1"),
        MemoryItem(content="sometimes wrong info", memory_type="test", id="m2"),
        MemoryItem(content="neutral observation", memory_type="test", id="m3"),
    ]


# ---------------------------------------------------------------------------
# Counters unit tests
# ---------------------------------------------------------------------------


class TestCounters:
    def test_defaults(self) -> None:
        c = Counters()
        assert c.helpful == 0
        assert c.harmful == 0
        assert c.neutral == 0
        assert c.total == 0

    def test_score_no_feedback(self) -> None:
        c = Counters()
        assert c.score() == 0.5

    def test_score_all_helpful(self) -> None:
        c = Counters(helpful=10, harmful=0, neutral=0)
        assert c.score() == pytest.approx(1.0)

    def test_score_all_harmful(self) -> None:
        c = Counters(helpful=0, harmful=10, neutral=0)
        assert c.score() == pytest.approx(0.0)

    def test_score_balanced(self) -> None:
        c = Counters(helpful=5, harmful=5, neutral=0)
        assert c.score() == pytest.approx(0.5)

    def test_score_mixed(self) -> None:
        c = Counters(helpful=7, harmful=3, neutral=0)
        # raw = (7-3)/10 = 0.4 → mapped = (0.4+1)/2 = 0.7
        assert c.score() == pytest.approx(0.7)

    def test_to_dict(self) -> None:
        c = Counters(helpful=1, harmful=2, neutral=3)
        assert c.to_dict() == {"helpful": 1, "harmful": 2, "neutral": 3}

    def test_from_dict(self) -> None:
        c = Counters.from_dict({"helpful": 5, "harmful": 1, "neutral": 2})
        assert c.helpful == 5
        assert c.harmful == 1
        assert c.neutral == 2

    def test_from_dict_missing_keys(self) -> None:
        c = Counters.from_dict({})
        assert c.helpful == 0
        assert c.harmful == 0
        assert c.neutral == 0


# ---------------------------------------------------------------------------
# ACEStrategy basics
# ---------------------------------------------------------------------------


class TestACEStrategy:
    def test_name(self) -> None:
        strategy = ACEStrategy()
        assert strategy.name == "ace"

    def test_is_evolution_strategy(self) -> None:
        from exo.memory.evolution import (
            MemoryEvolutionStrategy,  # pyright: ignore[reportMissingImports]
        )

        strategy = ACEStrategy()
        assert isinstance(strategy, MemoryEvolutionStrategy)

    def test_record_valid_labels(self) -> None:
        strategy = ACEStrategy()
        strategy.record("x", "helpful")
        strategy.record("x", "harmful")
        strategy.record("x", "neutral")
        c = strategy.get_counters("x")
        assert c.helpful == 1
        assert c.harmful == 1
        assert c.neutral == 1

    def test_record_invalid_label(self) -> None:
        strategy = ACEStrategy()
        with pytest.raises(ExoMemoryError, match="Invalid feedback label"):
            strategy.record("x", "unknown")

    def test_get_counters_creates_default(self) -> None:
        strategy = ACEStrategy()
        c = strategy.get_counters("new_id")
        assert c.total == 0


# ---------------------------------------------------------------------------
# evolve()
# ---------------------------------------------------------------------------


class TestEvolve:
    async def test_keeps_all_without_feedback(self, sample_items: list[MemoryItem]) -> None:
        strategy = ACEStrategy()
        result = await strategy.evolve(sample_items)
        assert len(result) == 3

    async def test_prunes_high_harmful_ratio(self, sample_items: list[MemoryItem]) -> None:
        strategy = ACEStrategy(harmful_threshold=0.5)
        # Make m2 mostly harmful
        for _ in range(8):
            strategy.record("m2", "harmful")
        for _ in range(2):
            strategy.record("m2", "helpful")
        # harmful ratio = 8/10 = 0.8 > 0.5 → pruned
        result = await strategy.evolve(sample_items)
        ids = [item.id for item in result]
        assert "m2" not in ids
        assert "m1" in ids
        assert "m3" in ids

    async def test_keeps_at_threshold(self, sample_items: list[MemoryItem]) -> None:
        strategy = ACEStrategy(harmful_threshold=0.5)
        # Exactly at threshold: harmful ratio = 5/10 = 0.5
        for _ in range(5):
            strategy.record("m1", "harmful")
        for _ in range(5):
            strategy.record("m1", "helpful")
        result = await strategy.evolve(sample_items)
        ids = [item.id for item in result]
        assert "m1" in ids

    async def test_empty_input(self) -> None:
        strategy = ACEStrategy()
        result = await strategy.evolve([])
        assert result == []

    async def test_context_accepted(self, sample_items: list[MemoryItem]) -> None:
        strategy = ACEStrategy()
        result = await strategy.evolve(sample_items, context={"key": "val"})
        assert len(result) == 3


# ---------------------------------------------------------------------------
# reflect()
# ---------------------------------------------------------------------------


class TestReflect:
    async def test_reflect_without_model(self, sample_items: list[MemoryItem]) -> None:
        strategy = ACEStrategy()
        labels = await strategy.reflect(sample_items, "good work")
        # Without model, all should be neutral
        assert all(label == "neutral" for label in labels.values())
        assert len(labels) == 3
        # Counters should be updated
        for item in sample_items:
            c = strategy.get_counters(item.id)
            assert c.neutral == 1

    async def test_reflect_with_model(self, sample_items: list[MemoryItem]) -> None:
        model = MockModel(label="helpful")
        strategy = ACEStrategy()
        labels = await strategy.reflect(sample_items, "great", model=model)
        assert all(label == "helpful" for label in labels.values())
        for item in sample_items:
            c = strategy.get_counters(item.id)
            assert c.helpful == 1

    async def test_reflect_with_mapping(self, sample_items: list[MemoryItem]) -> None:
        model = MockModel(
            label="neutral",
            mapping={
                "always useful": "helpful",
                "sometimes wrong": "harmful",
            },
        )
        strategy = ACEStrategy()
        labels = await strategy.reflect(sample_items, "mixed feedback", model=model)
        assert labels["m1"] == "helpful"
        assert labels["m2"] == "harmful"
        assert labels["m3"] == "neutral"

    async def test_reflect_invalid_model_output(self, sample_items: list[MemoryItem]) -> None:
        model = MockModel(label="INVALID_LABEL")
        strategy = ACEStrategy()
        labels = await strategy.reflect(sample_items, "test", model=model)
        # Invalid labels fall back to neutral
        assert all(label == "neutral" for label in labels.values())


# ---------------------------------------------------------------------------
# curate()
# ---------------------------------------------------------------------------


class TestCurate:
    async def test_curate_keeps_high_score(self, sample_items: list[MemoryItem]) -> None:
        strategy = ACEStrategy()
        for _ in range(10):
            strategy.record("m1", "helpful")
        result = await strategy.curate(sample_items, threshold=0.3)
        ids = [item.id for item in result]
        assert "m1" in ids

    async def test_curate_removes_low_score(self, sample_items: list[MemoryItem]) -> None:
        strategy = ACEStrategy()
        for _ in range(10):
            strategy.record("m2", "harmful")
        result = await strategy.curate(sample_items, threshold=0.3)
        ids = [item.id for item in result]
        assert "m2" not in ids

    async def test_curate_keeps_no_feedback(self, sample_items: list[MemoryItem]) -> None:
        strategy = ACEStrategy()
        # score defaults to 0.5, above 0.3 threshold
        result = await strategy.curate(sample_items, threshold=0.3)
        assert len(result) == 3

    async def test_curate_custom_threshold(self, sample_items: list[MemoryItem]) -> None:
        strategy = ACEStrategy()
        # score 0.5 is below threshold 0.6 for items with no feedback
        result = await strategy.curate(sample_items, threshold=0.6)
        # Default 0.5 < 0.6, so items with no feedback are removed
        assert len(result) == 0

    async def test_curate_empty_input(self) -> None:
        strategy = ACEStrategy()
        result = await strategy.curate([], threshold=0.3)
        assert result == []


# ---------------------------------------------------------------------------
# File persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_and_load(self, tmp_path: Path) -> None:
        path = tmp_path / "counters.json"
        strategy = ACEStrategy(counter_path=path)
        strategy.record("m1", "helpful")
        strategy.record("m1", "helpful")
        strategy.record("m2", "harmful")

        # Load into a fresh instance
        strategy2 = ACEStrategy(counter_path=path)
        c1 = strategy2.get_counters("m1")
        c2 = strategy2.get_counters("m2")
        assert c1.helpful == 2
        assert c2.harmful == 1

    def test_file_format(self, tmp_path: Path) -> None:
        path = tmp_path / "counters.json"
        strategy = ACEStrategy(counter_path=path)
        strategy.record("abc", "helpful")
        data = json.loads(path.read_text())
        assert data == {"abc": {"helpful": 1, "harmful": 0, "neutral": 0}}

    def test_load_missing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "missing.json"
        strategy = ACEStrategy(counter_path=path)
        assert strategy._counters == {}

    def test_load_corrupt_file(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("not json!!!")
        # Should not raise, just warn
        strategy = ACEStrategy(counter_path=path)
        assert strategy._counters == {}

    def test_no_persistence(self) -> None:
        strategy = ACEStrategy(counter_path=None)
        strategy.record("x", "helpful")
        # No file written, counters only in memory
        assert strategy.get_counters("x").helpful == 1

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "sub" / "dir" / "counters.json"
        strategy = ACEStrategy(counter_path=path)
        strategy.record("x", "helpful")
        assert path.exists()


# ---------------------------------------------------------------------------
# Composition with pipeline
# ---------------------------------------------------------------------------


class TestComposition:
    async def test_compose_sequential(self, sample_items: list[MemoryItem]) -> None:
        from exo.memory.evolution import (
            MemoryEvolutionPipeline,  # pyright: ignore[reportMissingImports]
        )

        ace = ACEStrategy()
        for _ in range(10):
            ace.record("m2", "harmful")

        # Use ACE as part of a pipeline
        pipeline = ace >> ace  # double-evolve (idempotent pruning)
        assert isinstance(pipeline, MemoryEvolutionPipeline)
        result = await pipeline.evolve(sample_items)
        ids = [item.id for item in result]
        assert "m2" not in ids
