"""Tests for exo.observability.cost — CostResult, estimate_cost, CostTracker, etc.

asyncio_mode=auto is set globally; sync tests are fine here.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from unittest.mock import MagicMock, patch

import pytest

from exo.observability.cost import (  # pyright: ignore[reportMissingImports]
    PRICE_TABLE,
    CostResult,
    CostTracker,
    estimate_cost,
    estimate_cost_from_usage,
    register_price,
    stamp_cost_on_current_span,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@dataclass
class _FakeUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int = 0


# ---------------------------------------------------------------------------
# CostResult
# ---------------------------------------------------------------------------


class TestCostResult:
    def test_is_frozen(self) -> None:
        result = CostResult(
            input_cost_usd=1.0,
            output_cost_usd=2.0,
            total_usd=3.0,
            model="openai:gpt-4o",
            priced=True,
            input_tokens=100,
            output_tokens=200,
        )
        with pytest.raises(FrozenInstanceError):
            result.total_usd = 99.0  # type: ignore[misc]

    def test_has_expected_fields(self) -> None:
        result = CostResult(
            input_cost_usd=0.0,
            output_cost_usd=0.0,
            total_usd=0.0,
            model="x",
            priced=False,
        )
        assert result.priced is False
        assert result.input_tokens == 0
        assert result.output_tokens == 0


# ---------------------------------------------------------------------------
# estimate_cost — known models
# ---------------------------------------------------------------------------


class TestEstimateCostKnownModel:
    def test_gpt4o_input_1m_tokens(self) -> None:
        """1 million input tokens at gpt-4o list price ($2.50/1M) → $2.50."""
        result = estimate_cost("openai:gpt-4o", input_tokens=1_000_000, output_tokens=0)
        assert result.priced is True
        assert result.input_cost_usd == pytest.approx(2.50, rel=1e-6)
        assert result.output_cost_usd == pytest.approx(0.0, abs=1e-9)
        assert result.total_usd == pytest.approx(2.50, rel=1e-6)

    def test_gpt4o_output_1m_tokens(self) -> None:
        """1 million output tokens at gpt-4o list price ($10.00/1M) → $10.00."""
        result = estimate_cost("openai:gpt-4o", input_tokens=0, output_tokens=1_000_000)
        assert result.priced is True
        assert result.output_cost_usd == pytest.approx(10.00, rel=1e-6)
        assert result.total_usd == pytest.approx(10.00, rel=1e-6)

    def test_gpt4o_mini_combined(self) -> None:
        """500k in + 500k out at gpt-4o-mini ($0.15/$0.60 per 1M)."""
        expected_input = 0.5 * 0.15
        expected_output = 0.5 * 0.60
        result = estimate_cost("openai:gpt-4o-mini", input_tokens=500_000, output_tokens=500_000)
        assert result.priced is True
        assert result.input_cost_usd == pytest.approx(expected_input, rel=1e-6)
        assert result.output_cost_usd == pytest.approx(expected_output, rel=1e-6)
        assert result.total_usd == pytest.approx(expected_input + expected_output, rel=1e-6)

    def test_claude_sonnet_known_price(self) -> None:
        """Anthropic claude-sonnet-4-6: $3.00/$15.00 per 1M."""
        result = estimate_cost(
            "anthropic:claude-sonnet-4-6",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )
        assert result.priced is True
        assert result.input_cost_usd == pytest.approx(3.00, rel=1e-6)
        assert result.output_cost_usd == pytest.approx(15.00, rel=1e-6)
        assert result.total_usd == pytest.approx(18.00, rel=1e-6)

    def test_gemini_flash_known_price(self) -> None:
        """Google gemini-2.0-flash: $0.10/$0.40 per 1M."""
        result = estimate_cost("google:gemini-2.0-flash", input_tokens=1_000_000, output_tokens=0)
        assert result.priced is True
        assert result.input_cost_usd == pytest.approx(0.10, rel=1e-6)

    def test_result_stores_token_counts(self) -> None:
        result = estimate_cost("openai:gpt-4o", input_tokens=123, output_tokens=456)
        assert result.input_tokens == 123
        assert result.output_tokens == 456

    def test_model_stored_as_given(self) -> None:
        result = estimate_cost("openai:gpt-4o", input_tokens=0, output_tokens=0)
        assert result.model == "openai:gpt-4o"

    def test_case_insensitive_lookup(self) -> None:
        result_lower = estimate_cost("openai:gpt-4o", input_tokens=1_000_000, output_tokens=0)
        result_upper = estimate_cost("OPENAI:GPT-4O", input_tokens=1_000_000, output_tokens=0)
        assert result_lower.priced is True
        assert result_upper.priced is True
        assert result_lower.input_cost_usd == pytest.approx(result_upper.input_cost_usd)

    def test_bare_model_name_lookup(self) -> None:
        """A model string without a provider prefix should still match."""
        result = estimate_cost("gpt-4o", input_tokens=1_000_000, output_tokens=0)
        assert result.priced is True
        assert result.input_cost_usd == pytest.approx(2.50, rel=1e-6)


# ---------------------------------------------------------------------------
# estimate_cost — unknown models
# ---------------------------------------------------------------------------


class TestEstimateCostUnknownModel:
    def test_unknown_model_priced_false(self) -> None:
        result = estimate_cost(
            "provider:totally-unknown-model-xyz", input_tokens=100, output_tokens=50
        )
        assert result.priced is False

    def test_unknown_model_zero_costs(self) -> None:
        result = estimate_cost("unknown:model", input_tokens=1_000_000, output_tokens=1_000_000)
        assert result.total_usd == 0.0
        assert result.input_cost_usd == 0.0
        assert result.output_cost_usd == 0.0

    def test_unknown_model_no_exception(self) -> None:
        # Must not raise
        estimate_cost("completely:bogus-model", input_tokens=0, output_tokens=0)

    def test_empty_model_string(self) -> None:
        result = estimate_cost("", input_tokens=100, output_tokens=100)
        assert result.priced is False
        assert result.total_usd == 0.0


# ---------------------------------------------------------------------------
# estimate_cost_from_usage
# ---------------------------------------------------------------------------


class TestEstimateCostFromUsage:
    def test_reads_usage_fields(self) -> None:
        usage = _FakeUsage(input_tokens=1_000_000, output_tokens=0)
        result = estimate_cost_from_usage("openai:gpt-4o", usage)
        assert result.input_cost_usd == pytest.approx(2.50, rel=1e-6)

    def test_unknown_model_from_usage(self) -> None:
        usage = _FakeUsage(input_tokens=500, output_tokens=250)
        result = estimate_cost_from_usage("openai:nonexistent-model-abc", usage)
        assert result.priced is False
        assert result.total_usd == 0.0


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------


class TestCostTracker:
    def test_initial_state(self) -> None:
        tracker = CostTracker()
        assert tracker.total_usd == 0.0
        assert tracker.by_model == {}
        assert tracker.input_tokens == 0
        assert tracker.output_tokens == 0

    def test_add_single_call(self) -> None:
        tracker = CostTracker()
        usage = _FakeUsage(input_tokens=1_000_000, output_tokens=0)
        result = tracker.add("openai:gpt-4o", usage)
        assert result.priced is True
        assert tracker.total_usd == pytest.approx(2.50, rel=1e-6)
        assert tracker.by_model == {"openai:gpt-4o": pytest.approx(2.50, rel=1e-6)}
        assert tracker.input_tokens == 1_000_000
        assert tracker.output_tokens == 0

    def test_add_multiple_calls_accumulates(self) -> None:
        tracker = CostTracker()
        usage1 = _FakeUsage(input_tokens=1_000_000, output_tokens=0)
        usage2 = _FakeUsage(input_tokens=0, output_tokens=1_000_000)
        tracker.add("openai:gpt-4o", usage1)
        tracker.add("openai:gpt-4o", usage2)
        # 1M input = $2.50, 1M output = $10.00 → total $12.50
        assert tracker.total_usd == pytest.approx(12.50, rel=1e-6)
        assert tracker.input_tokens == 1_000_000
        assert tracker.output_tokens == 1_000_000

    def test_by_model_tracks_multiple_models(self) -> None:
        tracker = CostTracker()
        usage_gpt = _FakeUsage(input_tokens=1_000_000, output_tokens=0)
        usage_claude = _FakeUsage(input_tokens=1_000_000, output_tokens=0)
        tracker.add("openai:gpt-4o", usage_gpt)  # $2.50
        tracker.add("anthropic:claude-sonnet-4-6", usage_claude)  # $3.00
        by_model = tracker.by_model
        assert "openai:gpt-4o" in by_model
        assert "anthropic:claude-sonnet-4-6" in by_model
        assert by_model["openai:gpt-4o"] == pytest.approx(2.50, rel=1e-6)
        assert by_model["anthropic:claude-sonnet-4-6"] == pytest.approx(3.00, rel=1e-6)
        assert tracker.total_usd == pytest.approx(5.50, rel=1e-6)

    def test_by_model_same_model_accumulates(self) -> None:
        tracker = CostTracker()
        usage = _FakeUsage(input_tokens=500_000, output_tokens=0)
        tracker.add("openai:gpt-4o-mini", usage)
        tracker.add("openai:gpt-4o-mini", usage)
        # 500k * 2 = 1M input at $0.15/1M → $0.15
        assert tracker.by_model["openai:gpt-4o-mini"] == pytest.approx(0.15, rel=1e-6)

    def test_by_model_returns_copy(self) -> None:
        """Mutating the returned dict should not affect the tracker."""
        tracker = CostTracker()
        tracker.add("openai:gpt-4o", _FakeUsage(input_tokens=1000, output_tokens=0))
        snapshot = tracker.by_model
        snapshot["evil"] = 999.0
        assert "evil" not in tracker.by_model

    def test_reset_clears_state(self) -> None:
        tracker = CostTracker()
        tracker.add("openai:gpt-4o", _FakeUsage(input_tokens=1_000_000, output_tokens=0))
        tracker.reset()
        assert tracker.total_usd == 0.0
        assert tracker.by_model == {}
        assert tracker.input_tokens == 0
        assert tracker.output_tokens == 0

    def test_unknown_model_still_accumulates_tokens(self) -> None:
        tracker = CostTracker()
        tracker.add("unknown:model", _FakeUsage(input_tokens=300, output_tokens=150))
        assert tracker.total_usd == 0.0
        assert tracker.input_tokens == 300
        assert tracker.output_tokens == 150

    def test_repr_does_not_raise(self) -> None:
        tracker = CostTracker()
        tracker.add("openai:gpt-4o", _FakeUsage(input_tokens=100, output_tokens=50))
        repr_str = repr(tracker)
        assert "CostTracker" in repr_str


# ---------------------------------------------------------------------------
# register_price
# ---------------------------------------------------------------------------


class TestRegisterPrice:
    def test_register_and_estimate(self) -> None:
        """A newly registered model should be priceable immediately."""
        register_price("test:new-model-v9999", input_per_1m=5.0, output_per_1m=20.0)
        result = estimate_cost("test:new-model-v9999", input_tokens=1_000_000, output_tokens=0)
        assert result.priced is True
        assert result.input_cost_usd == pytest.approx(5.0, rel=1e-6)

    def test_override_existing_model(self) -> None:
        """Overriding an existing entry should take effect immediately."""
        original = PRICE_TABLE.get("openai:gpt-4o")
        try:
            register_price("openai:gpt-4o", input_per_1m=99.0, output_per_1m=99.0)
            result = estimate_cost("openai:gpt-4o", input_tokens=1_000_000, output_tokens=0)
            assert result.input_cost_usd == pytest.approx(99.0, rel=1e-6)
        finally:
            # Restore original price so other tests are not affected
            if original is not None:
                PRICE_TABLE["openai:gpt-4o"] = original

    def test_register_without_provider_prefix(self) -> None:
        """A bare model name works as a key."""
        register_price("bare-model-xyz", input_per_1m=1.0, output_per_1m=2.0)
        result = estimate_cost("bare-model-xyz", input_tokens=1_000_000, output_tokens=0)
        assert result.priced is True
        assert result.input_cost_usd == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# stamp_cost_on_current_span
# ---------------------------------------------------------------------------


class TestStampCostOnCurrentSpan:
    def test_does_not_raise_when_otel_absent(self) -> None:
        """Should silently no-op if opentelemetry is not importable."""
        result = CostResult(
            input_cost_usd=1.0,
            output_cost_usd=2.0,
            total_usd=3.0,
            model="openai:gpt-4o",
            priced=True,
            input_tokens=100,
            output_tokens=200,
        )
        import sys

        otel_mod = sys.modules.pop("opentelemetry.trace", None)
        try:
            stamp_cost_on_current_span(result)  # must not raise
        finally:
            if otel_mod is not None:
                sys.modules["opentelemetry.trace"] = otel_mod

    def test_no_op_when_priced_false(self) -> None:
        """When priced=False, the function returns early — no span interactions."""
        result = CostResult(
            input_cost_usd=0.0,
            output_cost_usd=0.0,
            total_usd=0.0,
            model="unknown:model",
            priced=False,
        )
        # If this somehow tried to touch a span it would blow up on a mock;
        # passing without mock means it returned early.
        stamp_cost_on_current_span(result)

    def test_sets_attributes_on_recording_span(self) -> None:
        """When OTel is present and a recording span is active, attributes are set."""
        import importlib

        try:
            importlib.import_module("opentelemetry.trace")
        except ImportError:
            pytest.skip("opentelemetry not installed")

        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        result = CostResult(
            input_cost_usd=2.50,
            output_cost_usd=10.00,
            total_usd=12.50,
            model="openai:gpt-4o",
            priced=True,
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )

        # Patch at the opentelemetry.trace module level (where cost.py imports it).
        import opentelemetry.trace as _otel_trace  # type: ignore[import-untyped]

        original_fn = _otel_trace.get_current_span
        try:
            _otel_trace.get_current_span = lambda: mock_span  # type: ignore[method-assign]
            stamp_cost_on_current_span(result)
        finally:
            _otel_trace.get_current_span = original_fn  # type: ignore[method-assign]

        mock_span.set_attribute.assert_any_call("exo.cost.input_tokens", 1_000_000)
        mock_span.set_attribute.assert_any_call("exo.cost.output_tokens", 1_000_000)
        mock_span.set_attribute.assert_any_call("exo.cost.total_usd", 12.50)

    def test_no_op_on_non_recording_span(self) -> None:
        """When span.is_recording() is False, no set_attribute calls."""
        import importlib

        try:
            importlib.import_module("opentelemetry.trace")
        except ImportError:
            pytest.skip("opentelemetry not installed")

        mock_span = MagicMock()
        mock_span.is_recording.return_value = False

        result = CostResult(
            input_cost_usd=1.0,
            output_cost_usd=2.0,
            total_usd=3.0,
            model="openai:gpt-4o",
            priced=True,
        )

        with patch("opentelemetry.trace.get_current_span", return_value=mock_span):
            stamp_cost_on_current_span(result)

        mock_span.set_attribute.assert_not_called()


# ---------------------------------------------------------------------------
# PRICE_TABLE completeness smoke test
# ---------------------------------------------------------------------------


class TestPriceTable:
    def test_has_at_least_15_entries(self) -> None:
        assert len(PRICE_TABLE) >= 15, f"Expected ≥15 entries, got {len(PRICE_TABLE)}"

    def test_all_tuples_are_positive_floats(self) -> None:
        for model, (inp, out) in PRICE_TABLE.items():
            assert isinstance(inp, float), f"{model}: input price must be float"
            assert isinstance(out, float), f"{model}: output price must be float"
            assert inp >= 0, f"{model}: input price must be >= 0"
            assert out >= 0, f"{model}: output price must be >= 0"

    def test_known_providers_present(self) -> None:
        providers = {k.split(":")[0] for k in PRICE_TABLE}
        assert "openai" in providers
        assert "anthropic" in providers
        assert "google" in providers

    @pytest.mark.parametrize(
        "model",
        [
            "openai:gpt-4o",
            "openai:gpt-4o-mini",
            "anthropic:claude-sonnet-4-6",
            "anthropic:claude-haiku-3",
            "google:gemini-2.0-flash",
        ],
    )
    def test_specific_models_present(self, model: str) -> None:
        assert model in PRICE_TABLE, f"{model!r} not found in PRICE_TABLE"
