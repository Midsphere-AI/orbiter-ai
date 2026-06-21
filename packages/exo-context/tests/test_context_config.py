"""Tests for ContextConfig — simplified public API and derived internal thresholds."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from exo.context.config import (  # pyright: ignore[reportMissingImports]
    ContextConfig,
    OverflowStrategy,
)

# ── Defaults ──────────────────────────────────────────────────────────


class TestContextConfigDefaults:
    def test_default_values(self) -> None:
        cfg = ContextConfig()
        assert cfg.limit == 20
        assert cfg.overflow == OverflowStrategy.SUMMARIZE
        assert cfg.keep_recent == 5
        assert cfg.token_pressure == 0.8
        assert cfg.cache is False
        assert cfg.neuron_names == ()
        assert cfg.extra == {}

    def test_frozen(self) -> None:
        cfg = ContextConfig()
        with pytest.raises(ValidationError):
            cfg.limit = 99  # type: ignore[misc]

    def test_overflow_string_coercion(self) -> None:
        cfg = ContextConfig(overflow="truncate")  # type: ignore[arg-type]
        assert cfg.overflow == OverflowStrategy.TRUNCATE

    def test_token_pressure_default(self) -> None:
        cfg = ContextConfig()
        assert cfg.token_pressure == 0.8

    def test_token_pressure_custom(self) -> None:
        cfg = ContextConfig(token_pressure=0.5)
        assert cfg.token_pressure == 0.5

    def test_token_pressure_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            ContextConfig(token_pressure=1.5)

    def test_cache_true(self) -> None:
        cfg = ContextConfig(cache=True)
        assert cfg.cache is True


# ── Derived internal thresholds ───────────────────────────────────────


class TestDerivedThresholds:
    """Assert that internal windowing thresholds are correctly derived from public fields.

    These are the thresholds actually used by the context windowing engine.
    """

    def test_default_config_internal_thresholds(self) -> None:
        """ContextConfig() → same windowing values as the old default ContextConfig()."""
        cfg = ContextConfig()
        # history_rounds == limit == 20
        assert cfg._history_rounds == 20
        # summary_threshold == limit == 20 (old new-API path also set this to limit)
        assert cfg._summary_threshold == 20
        # offload_threshold == max(limit, limit*2.5) == max(20,50) == 50
        assert cfg._offload_threshold == 50
        # token_budget_trigger == token_pressure == 0.8
        assert cfg._token_budget_trigger == 0.8
        # enable_snapshots == cache == False
        assert cfg._enable_snapshots is False

    def test_custom_limit_30_truncate(self) -> None:
        """ContextConfig(limit=30, overflow='truncate') — truncate thresholds are high."""
        cfg = ContextConfig(limit=30, overflow="truncate")
        assert cfg._history_rounds == 30
        assert cfg._summary_threshold == 10_000
        assert cfg._offload_threshold == 10_000

    def test_overflow_none_thresholds(self) -> None:
        """overflow='none' sets all windowing thresholds to 10_000 (effectively disabled)."""
        cfg = ContextConfig(limit=20, overflow="none")
        assert cfg._history_rounds == 10_000
        assert cfg._summary_threshold == 10_000
        assert cfg._offload_threshold == 10_000

    def test_overflow_hook_thresholds(self) -> None:
        """overflow='hook' disables built-in cascade — all thresholds are 10_000."""
        cfg = ContextConfig(overflow="hook")
        assert cfg._history_rounds == 10_000
        assert cfg._summary_threshold == 10_000
        assert cfg._offload_threshold == 10_000

    def test_summarize_offload_scales_with_limit(self) -> None:
        """Offload threshold = max(limit, int(limit * 2.5)) for summarize strategy."""
        cfg = ContextConfig(limit=40, overflow="summarize")
        assert cfg._offload_threshold == max(40, int(40 * 2.5))  # == 100

    def test_cache_maps_to_enable_snapshots(self) -> None:
        cfg = ContextConfig(cache=True)
        assert cfg._enable_snapshots is True

    def test_token_pressure_maps_to_budget_trigger(self) -> None:
        cfg = ContextConfig(token_pressure=0.6)
        assert cfg._token_budget_trigger == 0.6


# ── Validation ────────────────────────────────────────────────────────


class TestContextConfigValidation:
    def test_limit_positive(self) -> None:
        with pytest.raises(ValidationError, match="limit"):
            ContextConfig(limit=0)

    def test_keep_recent_positive(self) -> None:
        with pytest.raises(ValidationError, match="keep_recent"):
            ContextConfig(keep_recent=0)

    def test_invalid_overflow(self) -> None:
        with pytest.raises(ValidationError):
            ContextConfig(overflow="invalid")  # type: ignore[arg-type]


# ── Neuron names ──────────────────────────────────────────────────────


class TestNeuronNames:
    def test_list_coerced_to_tuple(self) -> None:
        cfg = ContextConfig(neuron_names=["system", "task"])  # type: ignore[arg-type]
        assert cfg.neuron_names == ("system", "task")
        assert isinstance(cfg.neuron_names, tuple)

    def test_tuple_accepted(self) -> None:
        cfg = ContextConfig(neuron_names=("history",))
        assert cfg.neuron_names == ("history",)

    def test_empty_list(self) -> None:
        cfg = ContextConfig(neuron_names=[])  # type: ignore[arg-type]
        assert cfg.neuron_names == ()


# ── Extra metadata ───────────────────────────────────────────────────


class TestExtra:
    def test_extra_dict(self) -> None:
        cfg = ContextConfig(extra={"custom_key": 42})
        assert cfg.extra["custom_key"] == 42

    def test_extra_default_empty(self) -> None:
        cfg = ContextConfig()
        assert cfg.extra == {}


# ── Serialization ────────────────────────────────────────────────────


class TestSerialization:
    def test_model_dump(self) -> None:
        cfg = ContextConfig(limit=30, neuron_names=["a", "b"])  # type: ignore[arg-type]
        d = cfg.model_dump()
        assert d["limit"] == 30
        assert d["neuron_names"] == ("a", "b")

    def test_model_dump_json(self) -> None:
        cfg = ContextConfig()
        json_str = cfg.model_dump_json()
        assert '"summarize"' in json_str

    def test_roundtrip(self) -> None:
        original = ContextConfig(
            limit=15,
            overflow="truncate",
            cache=True,
            neuron_names=["system", "task"],  # type: ignore[arg-type]
        )
        d = original.model_dump()
        restored = ContextConfig(**d)
        assert restored == original
