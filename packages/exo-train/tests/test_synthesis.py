"""Tests for data synthesis utilities."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest

from exo.train.synthesis import (  # pyright: ignore[reportMissingImports]
    DataSynthesiser,
    SynthesisConfig,
    SynthesisError,
    SynthesisPipeline,
    SynthesisResult,
    SynthesisStrategy,
    TemplateSynthesiser,
    augment_add_noise,
    augment_swap_io,
    deduplicate,
    filter_by_score,
    split_dataset,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_items(n: int = 5) -> list[dict[str, Any]]:
    """Create sample trajectory dicts for testing."""
    return [
        {
            "id": f"item-{i}",
            "input": f"question {i}",
            "output": f"answer {i}",
            "score": 0.2 * i,
            "metadata": {},
        }
        for i in range(n)
    ]


class _StubSynthesiser(DataSynthesiser):
    """Returns source items unchanged, capped to num_samples."""

    async def synthesize(
        self,
        source: Sequence[dict[str, Any]],
        config: SynthesisConfig,
    ) -> list[dict[str, Any]]:
        return list(source)[: config.num_samples]


# ---------------------------------------------------------------------------
# SynthesisError
# ---------------------------------------------------------------------------


class TestSynthesisError:
    def test_is_exception(self) -> None:
        err = SynthesisError("boom")
        assert isinstance(err, Exception)
        assert str(err) == "boom"


# ---------------------------------------------------------------------------
# SynthesisStrategy
# ---------------------------------------------------------------------------


class TestSynthesisStrategy:
    def test_values(self) -> None:
        assert SynthesisStrategy.LLM == "llm"
        assert SynthesisStrategy.TEMPLATE == "template"
        assert SynthesisStrategy.AUGMENT == "augment"

    def test_is_str(self) -> None:
        assert isinstance(SynthesisStrategy.LLM, str)


# ---------------------------------------------------------------------------
# SynthesisConfig
# ---------------------------------------------------------------------------


class TestSynthesisConfig:
    def test_defaults(self) -> None:
        cfg = SynthesisConfig()
        assert cfg.strategy == SynthesisStrategy.TEMPLATE
        assert cfg.num_samples == 10
        assert cfg.train_ratio == 0.9
        assert cfg.min_score is None
        assert cfg.seed is None
        assert cfg.extra == {}

    def test_custom(self) -> None:
        cfg = SynthesisConfig(
            strategy=SynthesisStrategy.LLM,
            num_samples=50,
            train_ratio=0.8,
            min_score=0.5,
            seed=42,
            extra={"model": "gpt-4"},
        )
        assert cfg.strategy == SynthesisStrategy.LLM
        assert cfg.num_samples == 50
        assert cfg.train_ratio == 0.8
        assert cfg.min_score == 0.5
        assert cfg.seed == 42
        assert cfg.extra["model"] == "gpt-4"

    def test_frozen(self) -> None:
        cfg = SynthesisConfig()
        with pytest.raises(AttributeError):
            cfg.num_samples = 5  # type: ignore[misc]

    def test_invalid_num_samples(self) -> None:
        with pytest.raises(ValueError, match="num_samples"):
            SynthesisConfig(num_samples=0)

    def test_invalid_train_ratio_low(self) -> None:
        with pytest.raises(ValueError, match="train_ratio"):
            SynthesisConfig(train_ratio=0.0)

    def test_invalid_train_ratio_high(self) -> None:
        with pytest.raises(ValueError, match="train_ratio"):
            SynthesisConfig(train_ratio=1.5)

    def test_train_ratio_one(self) -> None:
        cfg = SynthesisConfig(train_ratio=1.0)
        assert cfg.train_ratio == 1.0


# ---------------------------------------------------------------------------
# SynthesisResult
# ---------------------------------------------------------------------------


class TestSynthesisResult:
    def test_defaults(self) -> None:
        r = SynthesisResult()
        assert r.total == 0
        assert r.train_count == 0
        assert r.test_count == 0
        assert r.metadata == {}

    def test_with_items(self) -> None:
        items = ({"a": 1}, {"b": 2})
        r = SynthesisResult(items=items, train_items=(items[0],), test_items=(items[1],))
        assert r.total == 2
        assert r.train_count == 1
        assert r.test_count == 1

    def test_frozen(self) -> None:
        r = SynthesisResult()
        with pytest.raises(AttributeError):
            r.items = ()  # type: ignore[misc]

    def test_to_json(self) -> None:
        r = SynthesisResult(
            items=({"x": 1},),
            train_items=({"x": 1},),
            test_items=(),
            metadata={"count": 1},
        )
        parsed = json.loads(r.to_json())
        assert parsed["items"] == [{"x": 1}]
        assert parsed["metadata"]["count"] == 1


# ---------------------------------------------------------------------------
# filter_by_score
# ---------------------------------------------------------------------------


class TestFilterByScore:
    def test_filters_below_threshold(self) -> None:
        items = _sample_items(5)
        result = filter_by_score(items, 0.5)
        assert all(it["score"] >= 0.5 for it in result)

    def test_keeps_at_threshold(self) -> None:
        items = [{"score": 0.5}, {"score": 0.4}]
        result = filter_by_score(items, 0.5)
        assert len(result) == 1

    def test_none_score_excluded(self) -> None:
        items = [{"score": None}, {"score": 0.8}]
        result = filter_by_score(items, 0.5)
        assert len(result) == 1

    def test_missing_score_excluded(self) -> None:
        items = [{"input": "no score"}, {"score": 1.0}]
        result = filter_by_score(items, 0.5)
        assert len(result) == 1

    def test_custom_score_key(self) -> None:
        items = [{"quality": 0.9}, {"quality": 0.1}]
        result = filter_by_score(items, 0.5, score_key="quality")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# split_dataset
# ---------------------------------------------------------------------------


class TestSplitDataset:
    def test_basic_split(self) -> None:
        items = _sample_items(10)
        train, test = split_dataset(items, 0.8, seed=42)
        assert len(train) + len(test) == 10
        assert len(train) == 8

    def test_deterministic_with_seed(self) -> None:
        items = _sample_items(10)
        t1, _ = split_dataset(items, 0.8, seed=42)
        t2, _ = split_dataset(items, 0.8, seed=42)
        assert t1 == t2

    def test_ratio_one(self) -> None:
        items = _sample_items(5)
        train, test = split_dataset(items, 1.0, seed=0)
        assert len(train) == 5
        assert len(test) == 0

    def test_invalid_ratio(self) -> None:
        with pytest.raises(ValueError, match="train_ratio"):
            split_dataset([], 0.0)

    def test_single_item(self) -> None:
        items = [{"x": 1}]
        train, test = split_dataset(items, 0.5, seed=0)
        assert len(train) == 1
        assert len(test) == 0


# ---------------------------------------------------------------------------
# deduplicate
# ---------------------------------------------------------------------------


class TestDeduplicate:
    def test_removes_duplicates(self) -> None:
        items = [{"input": "a"}, {"input": "b"}, {"input": "a"}]
        result = deduplicate(items)
        assert len(result) == 2

    def test_preserves_order(self) -> None:
        items = [{"input": "b"}, {"input": "a"}, {"input": "b"}]
        result = deduplicate(items)
        assert result[0]["input"] == "b"
        assert result[1]["input"] == "a"

    def test_custom_key(self) -> None:
        items = [{"id": "1", "x": "a"}, {"id": "2", "x": "a"}]
        result = deduplicate(items, key="x")
        assert len(result) == 1

    def test_missing_key(self) -> None:
        items = [{"a": 1}, {"a": 2}]
        result = deduplicate(items, key="missing")
        # Both stringify to "" so first is kept
        assert len(result) == 1

    def test_empty(self) -> None:
        assert deduplicate([]) == []


# ---------------------------------------------------------------------------
# Augmentation transforms
# ---------------------------------------------------------------------------


class TestAugmentSwapIO:
    def test_swaps(self) -> None:
        item = {"id": "orig", "input": "q", "output": "a", "metadata": {}}
        aug = augment_swap_io(item)
        assert "a" in aug["input"]
        assert aug["output"] == "q"
        assert aug["metadata"]["augmented"] == "swap_io"
        assert aug["id"] != "orig"

    def test_missing_fields(self) -> None:
        aug = augment_swap_io({})
        assert aug["output"] == ""


class TestAugmentAddNoise:
    def test_default_noise(self) -> None:
        item = {"id": "orig", "input": "hello", "metadata": {}}
        aug = augment_add_noise(item)
        assert "hello" in aug["input"]
        assert aug["metadata"]["augmented"] == "noise"
        assert aug["id"] != "orig"

    def test_custom_noise_fn(self) -> None:
        item = {"input": "hello", "metadata": {}}
        aug = augment_add_noise(item, noise_fn=lambda s: s.upper())
        assert aug["input"] == "HELLO"

    def test_missing_input(self) -> None:
        aug = augment_add_noise({})
        assert aug["input"] == " (rephrased)"


# ---------------------------------------------------------------------------
# TemplateSynthesiser
# ---------------------------------------------------------------------------


class TestTemplateSynthesiser:
    async def test_generates_items(self) -> None:
        source = _sample_items(3)
        cfg = SynthesisConfig(num_samples=5, seed=42)
        synth = TemplateSynthesiser()
        result = await synth.synthesise(source, cfg)
        assert len(result) == 5

    async def test_empty_source(self) -> None:
        cfg = SynthesisConfig(num_samples=5)
        synth = TemplateSynthesiser()
        result = await synth.synthesise([], cfg)
        assert result == []

    async def test_custom_transforms(self) -> None:
        source = [{"input": "x", "output": "y", "metadata": {}}]
        cfg = SynthesisConfig(num_samples=2, seed=0)
        synth = TemplateSynthesiser(transforms=[augment_add_noise])
        result = await synth.synthesise(source, cfg)
        assert len(result) == 2
        assert all(it["metadata"].get("augmented") == "noise" for it in result)

    async def test_deterministic_with_seed(self) -> None:
        source = _sample_items(5)
        cfg = SynthesisConfig(num_samples=3, seed=42)
        synth = TemplateSynthesiser()
        r1 = await synth.synthesise(source, cfg)
        r2 = await synth.synthesise(source, cfg)
        # IDs are random UUIDs, so compare everything except id
        for a, b in zip(r1, r2, strict=True):
            assert a["input"] == b["input"]
            assert a["output"] == b["output"]


# ---------------------------------------------------------------------------
# DataSynthesiser ABC
# ---------------------------------------------------------------------------


class TestDataSynthesiserABC:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            DataSynthesiser()  # type: ignore[abstract]

    async def test_concrete_subclass(self) -> None:
        synth = _StubSynthesiser()
        result = await synth.synthesise([{"x": 1}], SynthesisConfig(num_samples=1))
        assert len(result) == 1


# ---------------------------------------------------------------------------
# SynthesisPipeline
# ---------------------------------------------------------------------------


class TestSynthesisPipelineInit:
    def test_defaults(self) -> None:
        p = SynthesisPipeline()
        assert p.config.strategy == SynthesisStrategy.TEMPLATE
        assert isinstance(p.synthesiser, TemplateSynthesiser)

    def test_custom_config(self) -> None:
        cfg = SynthesisConfig(num_samples=20, seed=1)
        p = SynthesisPipeline(cfg)
        assert p.config.num_samples == 20

    def test_custom_synthesiser(self) -> None:
        synth = _StubSynthesiser()
        p = SynthesisPipeline(synthesiser=synth)
        assert p.synthesiser is synth

    def test_repr(self) -> None:
        p = SynthesisPipeline()
        assert "SynthesisPipeline" in repr(p)
        assert "template" in repr(p)


class TestSynthesisPipelineRun:
    async def test_basic_run(self) -> None:
        source = _sample_items(5)
        cfg = SynthesisConfig(num_samples=3, seed=42)
        p = SynthesisPipeline(cfg, synthesiser=_StubSynthesiser())
        result = await p.run(source)
        assert result.total == 3
        assert result.train_count + result.test_count == 3
        assert result.metadata["source_count"] == 5

    async def test_empty_source(self) -> None:
        cfg = SynthesisConfig(num_samples=3)
        p = SynthesisPipeline(cfg, synthesiser=_StubSynthesiser())
        result = await p.run([])
        assert result.total == 0
        assert result.metadata.get("filtered") is True

    async def test_with_score_filter(self) -> None:
        source = _sample_items(5)  # scores: 0.0, 0.2, 0.4, 0.6, 0.8
        cfg = SynthesisConfig(num_samples=2, min_score=0.5, seed=42)
        p = SynthesisPipeline(cfg, synthesiser=_StubSynthesiser())
        result = await p.run(source)
        assert result.total == 2
        assert result.metadata["after_filter"] == 2  # items with score >= 0.5

    async def test_deduplication(self) -> None:
        source = [
            {"id": "1", "input": "same", "output": "a", "score": 0.9},
            {"id": "2", "input": "same", "output": "b", "score": 0.8},
            {"id": "3", "input": "diff", "output": "c", "score": 0.7},
        ]
        cfg = SynthesisConfig(num_samples=2, seed=42)
        p = SynthesisPipeline(cfg, synthesiser=_StubSynthesiser())
        result = await p.run(source)
        assert result.metadata["after_dedup"] == 2

    async def test_train_test_split(self) -> None:
        source = _sample_items(10)
        cfg = SynthesisConfig(num_samples=10, train_ratio=0.8, seed=42)
        p = SynthesisPipeline(cfg, synthesiser=_StubSynthesiser())
        result = await p.run(source)
        assert result.train_count == 8
        assert result.test_count == 2

    async def test_all_filtered_out(self) -> None:
        source = [{"input": "a", "score": 0.1}]
        cfg = SynthesisConfig(num_samples=5, min_score=0.9)
        p = SynthesisPipeline(cfg, synthesiser=_StubSynthesiser())
        result = await p.run(source)
        assert result.total == 0

    async def test_metadata_strategy(self) -> None:
        source = _sample_items(3)
        cfg = SynthesisConfig(num_samples=2, seed=0)
        p = SynthesisPipeline(cfg, synthesiser=_StubSynthesiser())
        result = await p.run(source)
        assert result.metadata["strategy"] == "template"

    async def test_with_template_synthesiser(self) -> None:
        source = _sample_items(3)
        cfg = SynthesisConfig(num_samples=4, seed=42)
        p = SynthesisPipeline(cfg)
        result = await p.run(source)
        assert result.total == 4
        # Template synthesiser generates augmented items
        assert all(isinstance(it, dict) for it in result.items)


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    async def test_full_pipeline(self) -> None:
        """Full flow: filter → dedup → synthesise → split."""
        source = [
            {"id": f"i{i}", "input": f"q{i}", "output": f"a{i}", "score": 0.1 * i, "metadata": {}}
            for i in range(10)
        ]
        cfg = SynthesisConfig(
            num_samples=5,
            min_score=0.3,
            train_ratio=0.8,
            seed=42,
        )
        p = SynthesisPipeline(cfg)
        result = await p.run(source)
        assert result.total == 5
        assert result.train_count + result.test_count == 5
        # Verify JSON serialisation roundtrip
        parsed = json.loads(result.to_json())
        assert len(parsed["items"]) == 5

    async def test_custom_synthesiser_integration(self) -> None:
        source = _sample_items(3)
        cfg = SynthesisConfig(num_samples=2, seed=0)
        p = SynthesisPipeline(cfg, synthesiser=_StubSynthesiser())
        result = await p.run(source)
        assert result.total == 2
        assert result.metadata["synthesised"] == 2
