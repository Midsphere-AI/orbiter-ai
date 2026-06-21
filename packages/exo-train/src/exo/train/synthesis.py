"""Data synthesis utilities for generating training data from trajectories."""

from __future__ import annotations

import json
import random
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from exo.types import ExoError  # pyright: ignore[reportMissingImports]


class SynthesisError(ExoError):
    """Error during data synthesis."""


class SynthesisStrategy(StrEnum):
    """Strategy for generating synthetic data."""

    LLM = "llm"
    TEMPLATE = "template"
    AUGMENT = "augment"


@dataclass(frozen=True, slots=True)
class SynthesisConfig:
    """Configuration for a synthesis pipeline run."""

    strategy: SynthesisStrategy = SynthesisStrategy.TEMPLATE
    num_samples: int = 10
    train_ratio: float = 0.9
    min_score: float | None = None
    seed: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_samples < 1:
            msg = f"num_samples must be >= 1, got {self.num_samples}"
            raise ValueError(msg)
        if not 0.0 < self.train_ratio <= 1.0:
            msg = f"train_ratio must be in (0, 1], got {self.train_ratio}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    """Output of a synthesis pipeline run."""

    items: tuple[dict[str, Any], ...] = ()
    train_items: tuple[dict[str, Any], ...] = ()
    test_items: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.items)

    @property
    def train_count(self) -> int:
        return len(self.train_items)

    @property
    def test_count(self) -> int:
        return len(self.test_items)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(
            {
                "items": list(self.items),
                "train_items": list(self.train_items),
                "test_items": list(self.test_items),
                "metadata": self.metadata,
            },
            indent=2,
        )


# ---------------------------------------------------------------------------
# Trajectory filtering and splitting
# ---------------------------------------------------------------------------


def filter_by_score(
    items: Sequence[dict[str, Any]],
    min_score: float,
    *,
    score_key: str = "score",
) -> list[dict[str, Any]]:
    """Keep only items where *score_key* >= *min_score*."""
    result: list[dict[str, Any]] = []
    for item in items:
        score = item.get(score_key)
        if score is not None and score >= min_score:
            result.append(item)
    return result


def split_dataset(
    items: Sequence[dict[str, Any]],
    train_ratio: float = 0.9,
    *,
    seed: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split *items* into train/test lists by ratio."""
    if not 0.0 < train_ratio <= 1.0:
        msg = f"train_ratio must be in (0, 1], got {train_ratio}"
        raise ValueError(msg)
    ordered = list(items)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    split_idx = max(1, int(len(ordered) * train_ratio))
    return ordered[:split_idx], ordered[split_idx:]


def deduplicate(
    items: Sequence[dict[str, Any]],
    *,
    key: str = "input",
) -> list[dict[str, Any]]:
    """Remove duplicate items based on a key field."""
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        val = str(item.get(key, ""))
        if val not in seen:
            seen.add(val)
            result.append(item)
    return result


# ---------------------------------------------------------------------------
# Augmentation transforms
# ---------------------------------------------------------------------------


def augment_swap_io(item: dict[str, Any]) -> dict[str, Any]:
    """Create an augmented item by making the output a new input prompt."""
    return {
        **item,
        "id": uuid.uuid4().hex[:12],
        "input": f"Given this answer: {item.get('output', '')}\nWhat was the original question?",
        "output": item.get("input", ""),
        "metadata": {**item.get("metadata", {}), "augmented": "swap_io"},
    }


def augment_add_noise(
    item: dict[str, Any],
    *,
    noise_fn: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Create an augmented item with noise applied to the input."""
    original = item.get("input", "")
    noisy = noise_fn(original) if noise_fn is not None else original + " (rephrased)"
    return {
        **item,
        "id": uuid.uuid4().hex[:12],
        "input": noisy,
        "metadata": {**item.get("metadata", {}), "augmented": "noise"},
    }


# ---------------------------------------------------------------------------
# Synthesis pipeline
# ---------------------------------------------------------------------------


class DataSynthesizer(ABC):
    """Abstract base for custom data synthesizers."""

    __slots__ = ()

    @abstractmethod
    async def synthesise(
        self,
        source: Sequence[dict[str, Any]],
        config: SynthesisConfig,
    ) -> list[dict[str, Any]]:
        """Generate synthetic items from *source* data. (Deprecated spelling — override synthesize instead.)"""

    async def synthesize(
        self,
        source: Sequence[dict[str, Any]],
        config: SynthesisConfig,
    ) -> list[dict[str, Any]]:
        """Generate synthetic items from *source* data."""
        return await self.synthesise(source, config)


DataSynthesiser = DataSynthesizer  # Deprecated alias — use DataSynthesizer


class TemplateSynthesizer(DataSynthesizer):
    """Generate items from trajectory items via template transforms."""

    __slots__ = ("_transforms",)

    def __init__(
        self,
        transforms: Sequence[Callable[[dict[str, Any]], dict[str, Any]]] | None = None,
    ) -> None:
        self._transforms: tuple[Callable[[dict[str, Any]], dict[str, Any]], ...] = (
            tuple(transforms) if transforms else (augment_swap_io,)
        )

    async def synthesise(
        self,
        source: Sequence[dict[str, Any]],
        config: SynthesisConfig,
    ) -> list[dict[str, Any]]:
        if not source:
            return []
        rng = random.Random(config.seed)
        result: list[dict[str, Any]] = []
        while len(result) < config.num_samples:
            item = rng.choice(source)
            transform = rng.choice(self._transforms)
            result.append(transform(item))
        return result[: config.num_samples]


TemplateSynthesiser = TemplateSynthesizer  # Deprecated alias — use TemplateSynthesizer


class SynthesisPipeline:
    """Orchestrates data synthesis from trajectory items.

    Pipeline phases:
    1. Filter source items (optional score threshold)
    2. Deduplicate
    3. Synthesize via a DataSynthesizer
    4. Split into train/test sets
    """

    __slots__ = ("_config", "_synthesizer")

    def __init__(
        self,
        config: SynthesisConfig | None = None,
        *,
        synthesizer: DataSynthesizer | None = None,
        synthesiser: DataSynthesizer | None = None,
    ) -> None:
        self._config = config or SynthesisConfig()
        self._synthesizer = synthesizer or synthesiser or TemplateSynthesizer()

    @property
    def config(self) -> SynthesisConfig:
        return self._config

    @property
    def synthesizer(self) -> DataSynthesizer:
        return self._synthesizer

    @property
    def synthesiser(self) -> DataSynthesizer:
        """Deprecated alias — use synthesizer."""
        return self._synthesizer

    async def run(
        self,
        source: Sequence[dict[str, Any]],
    ) -> SynthesisResult:
        """Execute the full synthesis pipeline."""
        cfg = self._config

        # Phase 1: Filter
        filtered: Sequence[dict[str, Any]] = source
        if cfg.min_score is not None:
            filtered = filter_by_score(filtered, cfg.min_score)

        # Phase 2: Deduplicate
        deduped = deduplicate(filtered)

        if not deduped:
            return SynthesisResult(metadata={"source_count": len(source), "filtered": True})

        # Phase 3: Synthesize
        items = await self._synthesizer.synthesize(deduped, cfg)

        # Phase 4: Split
        train, test = split_dataset(items, cfg.train_ratio, seed=cfg.seed)

        return SynthesisResult(
            items=tuple(items),
            train_items=tuple(train),
            test_items=tuple(test),
            metadata={
                "source_count": len(source),
                "after_filter": len(filtered),
                "after_dedup": len(deduped),
                "synthesised": len(items),
                "strategy": str(cfg.strategy),
            },
        )

    def __repr__(self) -> str:
        return (
            f"SynthesisPipeline(strategy={self._config.strategy!r}, "
            f"num_samples={self._config.num_samples})"
        )
