"""Refinement Loop: iterative refinement via Run -> Analyze -> Learn -> Plan -> Halt."""

from exo.eval.ralph.config import (  # pyright: ignore[reportMissingImports]
    LoopState,
    RalphConfig,
    RefinementConfig,
    ReflectionConfig,
    StopConditionConfig,
    StopType,
    ValidationConfig,
)
from exo.eval.ralph.detectors import (  # pyright: ignore[reportMissingImports]
    CompositeDetector,
    ConsecutiveFailureDetector,
    CostLimitDetector,
    MaxIterationDetector,
    ScoreThresholdDetector,
    StopDecision,
    StopDetector,
    TimeoutDetector,
)
from exo.eval.ralph.runner import (  # pyright: ignore[reportMissingImports]
    RalphResult,
    RalphRunner,
    RefinementLoop,
    RefinementResult,
)
from exo.types import (  # pyright: ignore[reportMissingImports]
    RalphIterationEvent,
    RalphStopEvent,
    RefinementIterationEvent,
    RefinementStopEvent,
)

__all__ = [
    "CompositeDetector",
    "ConsecutiveFailureDetector",
    "CostLimitDetector",
    "LoopState",
    "MaxIterationDetector",
    # Deprecated aliases (kept for backward compatibility)
    "RalphConfig",
    "RalphIterationEvent",
    "RalphResult",
    "RalphRunner",
    "RalphStopEvent",
    # New canonical names
    "RefinementConfig",
    "RefinementIterationEvent",
    "RefinementLoop",
    "RefinementResult",
    "RefinementStopEvent",
    "ReflectionConfig",
    "ScoreThresholdDetector",
    "StopConditionConfig",
    "StopDecision",
    "StopDetector",
    "StopType",
    "TimeoutDetector",
    "ValidationConfig",
]
