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
    SchemaValidationScorer,
)
from exo.eval.trajectory import (  # pyright: ignore[reportMissingImports]
    DefaultStrategy,
    TrajectoryAction,
    TrajectoryDataset,
    TrajectoryError,
    TrajectoryExtractor,
    TrajectoryItem,
    TrajectoryReward,
    TrajectoryState,
    TrajectoryStrategy,
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

# ---------------------------------------------------------------------------
# Register all rule-based scorers in the scorer registry.
# trajectory_scorers.py registers its own four classes via @scorer_register.
# The rule-based scorers from scorers.py are registered here so the registry
# is fully populated when the package is imported.
# ---------------------------------------------------------------------------
scorer_register("format")(FormatValidationScorer)
scorer_register("schema")(SchemaValidationScorer)
scorer_register("correctness")(OutputCorrectnessScorer)
scorer_register("length")(OutputLengthScorer)
scorer_register("completeness")(OutputCompletenessScorer)
# LLM-based scorers (require a judge callable at construction time)
scorer_register("llm_judge")(LLMAsJudgeScorer)
scorer_register("output_quality")(OutputQualityScorer)
scorer_register("logic_consistency")(LogicConsistencyScorer)
scorer_register("reasoning_validity")(ReasoningValidityScorer)
scorer_register("constraint_satisfaction")(ConstraintSatisfactionScorer)

__all__: list[str] = [
    "AnswerAccuracyLLMScorer",
    "ConstraintSatisfactionScorer",
    "DefaultStrategy",
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
    "TrajectoryAction",
    "TrajectoryDataset",
    "TrajectoryError",
    "TrajectoryExtractor",
    "TrajectoryItem",
    "TrajectoryReward",
    "TrajectoryState",
    "TrajectoryStrategy",
    "TrajectoryValidator",
    "get_scorer",
    "list_scorers",
    "scorer_register",
]
