"""Exo Guardrail: Pluggable security detection for agents."""

from importlib.metadata import PackageNotFoundError, version
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from exo.guardrail.base import BaseGuardrail  # pyright: ignore[reportMissingImports]
from exo.guardrail.llm_backend import (  # pyright: ignore[reportMissingImports]
    LLMGuardrailBackend,
)
from exo.guardrail.types import (  # pyright: ignore[reportMissingImports]
    GuardrailBackend,
    GuardrailError,
    GuardrailResult,
    RiskAssessment,
    RiskLevel,
)
from exo.guardrail.user_input import (  # pyright: ignore[reportMissingImports]
    PatternBackend,
    UserInputGuardrail,
)

try:
    __version__: str = version("exo-guardrail")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__: list[str] = [
    "BaseGuardrail",
    "GuardrailBackend",
    "GuardrailError",
    "GuardrailResult",
    "LLMGuardrailBackend",
    "PatternBackend",
    "RiskAssessment",
    "RiskLevel",
    "UserInputGuardrail",
]
