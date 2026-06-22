"""Core guardrail types: risk levels, assessments, backends, and results."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from exo.types import ExoError


class RiskLevel(StrEnum):
    """Severity level of a detected risk."""

    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskAssessment(BaseModel):
    """Result of a backend's risk analysis.

    Attributes:
        has_risk: Whether any risk was detected.
        risk_level: Severity of the detected risk.
        risk_type: Category of risk (e.g., "prompt_injection", "pii_leak").
        confidence: Backend's confidence in the assessment (0.0-1.0).
        details: Additional metadata for logging and auditing.
    """

    model_config = {"frozen": True}

    has_risk: bool
    risk_level: RiskLevel
    risk_type: str | None = None
    # NOTE: confidence is populated by backends (e.g. PatternBackend uses match
    # count; LLMGuardrailBackend forwards the model's self-reported confidence)
    # but is not currently used in blocking decisions inside BaseGuardrail.
    # It is exposed for callers that want to implement confidence-threshold
    # policies at the application layer.
    confidence: float = 1.0
    details: dict[str, Any] = Field(default_factory=dict)


class GuardrailConfigError(ExoError):
    """Raised when a guardrail is misconfigured at construction time.

    Examples: passing ``backend=None``, an unrecognised ``HookPoint`` name.

    Attributes:
        context: Dict of involved identifiers (field name, supplied value, …).
        hint: Actionable fix — e.g. which enum values are valid.
    """


class GuardrailError(ExoError):
    """Raised when a guardrail blocks an operation.

    Attributes:
        risk_level: The risk level that triggered the block.
        risk_type: Category of the detected risk.
        details: Additional context from the risk assessment.
    """

    def __init__(
        self,
        message: str,
        *,
        risk_level: RiskLevel,
        risk_type: str | None = None,
        details: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        hint: str | None = None,
        doc: str | None = None,
    ) -> None:
        super().__init__(message, context=context, hint=hint, doc=doc)
        self.risk_level = risk_level
        self.risk_type = risk_type
        self.details = details or {}


class GuardrailBackend(ABC):
    """Abstract base class for pluggable guardrail detection backends.

    Subclasses implement ``analyze`` to inspect data and return a
    ``RiskAssessment`` indicating whether the data poses a risk.

    Example:
        >>> class MyBackend(GuardrailBackend):
        ...     async def analyze(self, data: dict[str, Any]) -> RiskAssessment:
        ...         return RiskAssessment(has_risk=False, risk_level=RiskLevel.SAFE)
    """

    @abstractmethod
    async def analyze(self, data: dict[str, Any]) -> RiskAssessment:
        """Analyze data for potential risks.

        Args:
            data: Arbitrary data to inspect (e.g., messages, tool arguments).

        Returns:
            A ``RiskAssessment`` describing the detected risk level.
        """


class GuardrailResult(BaseModel):
    """Outcome of a guardrail check.

    Attributes:
        is_safe: Whether the data passed the guardrail check.
        risk_level: Severity of the detected risk.
        risk_type: Category of risk (e.g., "prompt_injection", "pii_leak").
        details: Additional metadata for logging and auditing.
    """

    model_config = {"frozen": True}

    is_safe: bool
    risk_level: RiskLevel
    risk_type: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def safe(cls) -> GuardrailResult:
        """Create a result indicating the data is safe.

        Returns:
            A ``GuardrailResult`` with ``is_safe=True`` and ``risk_level=SAFE``.
        """
        return cls(is_safe=True, risk_level=RiskLevel.SAFE)

    @classmethod
    def block(
        cls,
        risk_level: RiskLevel,
        risk_type: str,
        details: dict[str, Any] | None = None,
    ) -> GuardrailResult:
        """Create a result indicating the data should be blocked.

        Args:
            risk_level: Severity of the detected risk.
            risk_type: Category of the detected risk.
            details: Additional context for logging and auditing.

        Returns:
            A ``GuardrailResult`` with ``is_safe=False`` and the given risk info.
        """
        return cls(
            is_safe=False,
            risk_level=risk_level,
            risk_type=risk_type,
            details=details or {},
        )
