"""Tests for guardrail core types: RiskLevel, RiskAssessment, GuardrailError,
GuardrailBackend, GuardrailResult."""

from __future__ import annotations

import pytest

from exo.guardrail.types import (
    GuardrailBackend,
    GuardrailConfigError,
    GuardrailError,
    GuardrailResult,
    RiskAssessment,
    RiskLevel,
)
from exo.types import ExoError


class TestRiskLevel:
    def test_values(self) -> None:
        assert RiskLevel.SAFE == "safe"
        assert RiskLevel.LOW == "low"
        assert RiskLevel.MEDIUM == "medium"
        assert RiskLevel.HIGH == "high"
        assert RiskLevel.CRITICAL == "critical"

    def test_all_members(self) -> None:
        assert set(RiskLevel) == {
            RiskLevel.SAFE,
            RiskLevel.LOW,
            RiskLevel.MEDIUM,
            RiskLevel.HIGH,
            RiskLevel.CRITICAL,
        }

    def test_is_str(self) -> None:
        assert isinstance(RiskLevel.SAFE, str)


class TestRiskAssessment:
    def test_create_safe(self) -> None:
        assessment = RiskAssessment(has_risk=False, risk_level=RiskLevel.SAFE)
        assert assessment.has_risk is False
        assert assessment.risk_level == RiskLevel.SAFE
        assert assessment.risk_type is None
        assert assessment.confidence == 1.0
        assert assessment.details == {}

    def test_create_risky(self) -> None:
        assessment = RiskAssessment(
            has_risk=True,
            risk_level=RiskLevel.HIGH,
            risk_type="prompt_injection",
            confidence=0.95,
            details={"pattern": "ignore previous"},
        )
        assert assessment.has_risk is True
        assert assessment.risk_level == RiskLevel.HIGH
        assert assessment.risk_type == "prompt_injection"
        assert assessment.confidence == 0.95
        assert assessment.details == {"pattern": "ignore previous"}

    def test_frozen(self) -> None:
        assessment = RiskAssessment(has_risk=False, risk_level=RiskLevel.SAFE)
        with pytest.raises(Exception):  # noqa: B017
            assessment.has_risk = True  # type: ignore[misc]

    def test_defaults(self) -> None:
        assessment = RiskAssessment(has_risk=False, risk_level=RiskLevel.SAFE)
        assert assessment.risk_type is None
        assert assessment.confidence == 1.0
        assert assessment.details == {}

    def test_validation_risk_level(self) -> None:
        """RiskLevel field validates against the enum."""
        assessment = RiskAssessment(has_risk=True, risk_level=RiskLevel.CRITICAL)
        assert assessment.risk_level == RiskLevel.CRITICAL

    def test_details_default_factory(self) -> None:
        """Each instance gets its own details dict."""
        a = RiskAssessment(has_risk=False, risk_level=RiskLevel.SAFE)
        b = RiskAssessment(has_risk=False, risk_level=RiskLevel.SAFE)
        assert a.details is not b.details


class TestGuardrailError:
    def test_inherits_exo_error(self) -> None:
        err = GuardrailError("blocked", risk_level=RiskLevel.HIGH)
        assert isinstance(err, ExoError)
        assert isinstance(err, Exception)

    def test_message(self) -> None:
        err = GuardrailError("injection detected", risk_level=RiskLevel.CRITICAL)
        assert str(err) == "injection detected"

    def test_attributes(self) -> None:
        err = GuardrailError(
            "blocked",
            risk_level=RiskLevel.HIGH,
            risk_type="prompt_injection",
            details={"pattern": "ignore"},
        )
        assert err.risk_level == RiskLevel.HIGH
        assert err.risk_type == "prompt_injection"
        assert err.details == {"pattern": "ignore"}

    def test_defaults(self) -> None:
        err = GuardrailError("blocked", risk_level=RiskLevel.HIGH)
        assert err.risk_type is None
        assert err.details == {}

    def test_raises(self) -> None:
        with pytest.raises(GuardrailError, match="blocked"):
            raise GuardrailError("blocked", risk_level=RiskLevel.HIGH)

    def test_context_and_hint_forwarded_to_exo_error(self) -> None:
        """GuardrailError must forward context/hint to ExoError.__init__."""
        err = GuardrailError(
            "blocked",
            risk_level=RiskLevel.HIGH,
            context={"hook": "pre_llm_call"},
            hint="Check the guardrail thresholds.",
        )
        assert err.context == {"hook": "pre_llm_call"}
        assert err.hint == "Check the guardrail thresholds."
        assert "where:" in str(err)
        assert "→" in str(err)


class TestGuardrailConfigError:
    def test_inherits_exo_error(self) -> None:
        err = GuardrailConfigError("bad config")
        assert isinstance(err, ExoError)

    def test_context_and_hint(self) -> None:
        err = GuardrailConfigError(
            "Unknown hook point: 'bad'",
            context={"hook_point": "bad"},
            hint="Use HookPoint.<NAME>.",
        )
        assert err.context == {"hook_point": "bad"}
        assert err.hint == "Use HookPoint.<NAME>."
        rendered = str(err)
        assert "where:" in rendered
        assert "→" in rendered


class TestGuardrailBackend:
    def test_is_abstract(self) -> None:
        """GuardrailBackend cannot be instantiated directly."""
        with pytest.raises(TypeError):
            GuardrailBackend()  # type: ignore[abstract]

    def test_subclass_must_implement_analyze(self) -> None:
        """A subclass without analyze() cannot be instantiated."""

        class Incomplete(GuardrailBackend):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    async def test_subclass_analyze(self) -> None:
        """A concrete subclass can be instantiated and called."""

        class SafeBackend(GuardrailBackend):
            async def analyze(self, data: dict) -> RiskAssessment:
                return RiskAssessment(has_risk=False, risk_level=RiskLevel.SAFE)

        backend = SafeBackend()
        result = await backend.analyze({"messages": []})
        assert result.has_risk is False
        assert result.risk_level == RiskLevel.SAFE

    async def test_subclass_risky(self) -> None:
        """A backend can return a risky assessment."""

        class RiskyBackend(GuardrailBackend):
            async def analyze(self, data: dict) -> RiskAssessment:
                return RiskAssessment(
                    has_risk=True,
                    risk_level=RiskLevel.HIGH,
                    risk_type="test_risk",
                    confidence=0.9,
                    details={"reason": "test"},
                )

        backend = RiskyBackend()
        result = await backend.analyze({"input": "bad stuff"})
        assert result.has_risk is True
        assert result.risk_level == RiskLevel.HIGH
        assert result.risk_type == "test_risk"


class TestGuardrailResult:
    def test_safe_factory(self) -> None:
        result = GuardrailResult.safe()
        assert result.is_safe is True
        assert result.risk_level == RiskLevel.SAFE
        assert result.risk_type is None
        assert result.details == {}

    def test_block_factory(self) -> None:
        result = GuardrailResult.block(
            risk_level=RiskLevel.CRITICAL,
            risk_type="prompt_injection",
            details={"pattern": "ignore previous"},
        )
        assert result.is_safe is False
        assert result.risk_level == RiskLevel.CRITICAL
        assert result.risk_type == "prompt_injection"
        assert result.details == {"pattern": "ignore previous"}

    def test_block_default_details(self) -> None:
        result = GuardrailResult.block(
            risk_level=RiskLevel.HIGH,
            risk_type="pii_leak",
        )
        assert result.details == {}

    def test_frozen(self) -> None:
        result = GuardrailResult.safe()
        with pytest.raises(Exception):  # noqa: B017
            result.is_safe = False  # type: ignore[misc]

    def test_details_default_factory(self) -> None:
        """Each instance gets its own details dict."""
        a = GuardrailResult.safe()
        b = GuardrailResult.safe()
        assert a.details is not b.details

    def test_block_returns_guardrail_result(self) -> None:
        result = GuardrailResult.block(
            risk_level=RiskLevel.MEDIUM,
            risk_type="content_filter",
        )
        assert isinstance(result, GuardrailResult)
