"""Exo Eval: Evaluation and scoring framework."""

__version__: str = "0.1.0"

from exo.eval.base import (  # pyright: ignore[reportMissingImports]
    EvalCaseResult,
    EvalCriteria,
    EvalError,
    EvalResult,
    EvalStatus,
    EvalTarget,
    Evaluator,
    Scorer,
    ScorerResult,
)
from exo.eval.llm_scorer import (  # pyright: ignore[reportMissingImports]
    ConstraintSatisfactionScorer,
    LLMAsJudgeScorer,
    LogicConsistencyScorer,
    OutputQualityScorer,
    ReasoningValidityScorer,
)
from exo.eval.reflection import (  # pyright: ignore[reportMissingImports]
    GeneralReflector,
    ReflectionHistory,
    ReflectionLevel,
    ReflectionResult,
    ReflectionType,
    Reflector,
)
from exo.eval.scorers import (  # pyright: ignore[reportMissingImports]
    FormatValidationScorer,
    OutputCompletenessScorer,
    OutputCorrectnessScorer,
    OutputLengthScorer,
    OutputRelevanceScorer,
    SchemaValidationScorer,
)
from exo.eval.trajectory_scorers import (  # pyright: ignore[reportMissingImports]
    AnswerAccuracyLLMScorer,
    LabelDistributionScorer,
    TimeCostScorer,
    TrajectoryValidator,
    get_scorer,
    list_scorers,
    scorer_register,
)

__all__: list[str] = [
    "AnswerAccuracyLLMScorer",
    "ConstraintSatisfactionScorer",
    "EvalCaseResult",
    "EvalCriteria",
    "EvalError",
    "EvalResult",
    "EvalStatus",
    "EvalTarget",
    "Evaluator",
    "FormatValidationScorer",
    "GeneralReflector",
    "LLMAsJudgeScorer",
    "LabelDistributionScorer",
    "LogicConsistencyScorer",
    "OutputCompletenessScorer",
    "OutputCorrectnessScorer",
    "OutputLengthScorer",
    "OutputQualityScorer",
    "OutputRelevanceScorer",
    "ReasoningValidityScorer",
    "ReflectionHistory",
    "ReflectionLevel",
    "ReflectionResult",
    "ReflectionType",
    "Reflector",
    "SchemaValidationScorer",
    "Scorer",
    "ScorerResult",
    "TimeCostScorer",
    "TrajectoryValidator",
    "get_scorer",
    "list_scorers",
    "scorer_register",
]
