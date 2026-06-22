"""Trajectory validation, time cost, accuracy, and label distribution scorers.

Includes a scorer registry with ``@scorer_register()`` decorator for
automatic scorer discovery and factory-based creation.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from typing import Any

from exo.eval.base import EvalError, Scorer, ScorerResult  # pyright: ignore[reportMissingImports]
from exo.eval.llm_scorer import LLMAsJudgeScorer  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scorer registry
# ---------------------------------------------------------------------------

_SCORER_REGISTRY: dict[str, type[Scorer]] = {}


def scorer_register(
    name: str,
) -> Any:
    """Decorator that registers a ``Scorer`` subclass under *name*.

    Usage::

        @scorer_register("my_metric")
        class MyScorer(Scorer): ...
    """

    def decorator(cls: type[Scorer]) -> type[Scorer]:
        _SCORER_REGISTRY[name] = cls
        return cls

    return decorator


def get_scorer(name: str) -> type[Scorer]:
    """Lookup a registered scorer class by *name*.

    Raises:
        EvalError: If *name* is not registered, with a list of available scorers.
    """
    if name not in _SCORER_REGISTRY:
        raise EvalError(
            f"Scorer {name!r} is not registered.",
            context={"name": name},
            hint=(
                f"Available scorers: {list_scorers()}. "
                "Register custom scorers with @scorer_register(name)."
            ),
        )
    return _SCORER_REGISTRY[name]


def list_scorers() -> list[str]:
    """Return all registered scorer names."""
    return sorted(_SCORER_REGISTRY)


# ---------------------------------------------------------------------------
# TrajectoryValidator — trajectory step validation
# ---------------------------------------------------------------------------


@scorer_register("trajectory")
class TrajectoryValidator(Scorer):
    """Validates a trajectory (list of step dicts) for structural integrity.

    Checks each step for required keys and returns the fraction of valid steps.
    Required per-step keys: ``"step"`` (or ``"id"``), ``"action"``.
    """

    __slots__ = ("_name", "_required_keys")

    def __init__(
        self,
        *,
        required_keys: Sequence[str] = ("action",),
        name: str = "trajectory",
    ) -> None:
        self._required_keys = list(required_keys)
        self._name = name

    async def score(self, case_id: str, input: Any, output: Any) -> ScorerResult:
        steps = _parse_trajectory(output)
        if not steps:
            return ScorerResult(
                scorer_name=self._name,
                score=0.0,
                details={"error": "Empty or invalid trajectory"},
            )
        valid = 0
        errors: list[str] = []
        for i, step in enumerate(steps):
            missing = [k for k in self._required_keys if k not in step]
            has_id = "step" in step or "id" in step
            if not missing and has_id:
                valid += 1
            else:
                reasons = []
                if not has_id:
                    reasons.append("missing step/id")
                if missing:
                    reasons.append(f"missing {missing}")
                errors.append(f"step {i}: {', '.join(reasons)}")
        ratio = valid / len(steps)
        logger.debug("TrajectoryValidator case=%s score=%.3f", case_id, ratio)
        return ScorerResult(
            scorer_name=self._name,
            score=ratio,
            details={"valid": valid, "total": len(steps), "errors": errors},
        )


# ---------------------------------------------------------------------------
# TimeCostScorer
# ---------------------------------------------------------------------------


@scorer_register("time_cost")
class TimeCostScorer(Scorer):
    """Scores based on execution time relative to a maximum budget.

    Reads ``_time_cost_ms`` from the output dict.  If the key is **absent**,
    the scorer raises :exc:`KeyError` so callers (e.g. ``RalphRunner._analyze``
    or the ``Evaluator``) can distinguish "timing not available" from a real
    zero-time result.  Previously the scorer silently fell back to ``0 ms``,
    which produced a spurious perfect score of 1.0 for any output that did not
    carry timing information.

    Score formula: ``clamp(1.0 - elapsed / max_ms, 0.0, 1.0)``.
    """

    __slots__ = ("_max_ms", "_name")

    def __init__(self, *, max_ms: float = 30_000.0, name: str = "time_cost") -> None:
        self._max_ms = max_ms
        self._name = name

    async def score(self, case_id: str, input: Any, output: Any) -> ScorerResult:
        if not isinstance(output, dict) or "_time_cost_ms" not in output:
            msg = (
                f"TimeCostScorer (case={case_id!r}): '_time_cost_ms' key missing from output. "
                "Include timing information in the output dict or skip this scorer."
            )
            raise KeyError(msg)
        elapsed = float(output["_time_cost_ms"])
        score = max(0.0, min(1.0, 1.0 - elapsed / self._max_ms)) if self._max_ms > 0 else 0.0
        logger.debug("TimeCostScorer case=%s score=%.3f elapsed_ms=%.1f", case_id, score, elapsed)
        return ScorerResult(
            scorer_name=self._name,
            score=score,
            details={"elapsed_ms": elapsed, "max_ms": self._max_ms},
        )


# ---------------------------------------------------------------------------
# AnswerAccuracyLLMScorer
# ---------------------------------------------------------------------------


@scorer_register("answer_accuracy")
class AnswerAccuracyLLMScorer(LLMAsJudgeScorer):
    """LLM-as-Judge scorer comparing agent output to a reference answer.

    Expects *input* to be a dict with ``question`` and ``answer`` keys
    (configurable via constructor).
    """

    __slots__ = ("_answer_key", "_question_key")

    def __init__(
        self,
        judge: Any = None,
        *,
        question_key: str = "question",
        answer_key: str = "answer",
        name: str = "answer_accuracy",
    ) -> None:
        super().__init__(judge, name=name)
        self._question_key = question_key
        self._answer_key = answer_key

    @staticmethod
    def _default_system_prompt() -> str:
        return (
            "You are an answer accuracy evaluator. Compare the agent's response "
            "to the correct answer. Score 1.0 for a fully correct answer, 0.0 for "
            "completely wrong. Partial credit is allowed. "
            'Return JSON: {"score": <float 0.0-1.0>, "explanation": "<reasoning>"}.'
        )

    def build_prompt(self, case_id: str, input: Any, output: Any) -> str:
        question = input.get(self._question_key, "") if isinstance(input, dict) else str(input)
        answer = input.get(self._answer_key, "") if isinstance(input, dict) else ""
        parts = [
            self._system_prompt,
            "",
            f"[Question]\n{question}",
            f"[Correct Answer]\n{answer}",
            f"[Agent Response]\n{output}",
            '\nReturn JSON: {"score": <float 0.0-1.0>, "explanation": "<reasoning>"}.',
        ]
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# LabelDistributionScorer
# ---------------------------------------------------------------------------


@scorer_register("label_distribution")
class LabelDistributionScorer(Scorer):
    """Evaluates label balance / distribution skew across a dataset.

    Per-case score definition: ``1.0`` if the case carries a non-None label
    (i.e. the case is labelled and contributes to the distribution), ``0.0``
    if the label is absent or the input does not expose one.  This gives a
    meaningful signal at the case level — cases that lack labels are scored as
    failures so callers can detect unlabelled data.

    The real aggregate value is in :meth:`summarize`, which computes label
    counts, per-label fractions, and distribution skew across all scored cases.
    """

    __slots__ = ("_label_key", "_name")

    def __init__(self, *, label_key: str = "label", name: str = "label_distribution") -> None:
        self._label_key = label_key
        self._name = name

    async def score(self, case_id: str, input: Any, output: Any) -> ScorerResult:
        label = input.get(self._label_key, None) if isinstance(input, dict) else None
        # Per-case score: 1.0 when a label is present, 0.0 when absent.
        # This allows downstream criteria to flag unlabelled cases as failures.
        per_case_score = 1.0 if label is not None else 0.0
        logger.debug(
            "LabelDistributionScorer case=%s label=%r score=%.1f", case_id, label, per_case_score
        )
        return ScorerResult(
            scorer_name=self._name,
            score=per_case_score,
            details={"label": label},
        )

    def summarize(self, results: list[ScorerResult]) -> dict[str, Any]:
        """Compute label distribution across all scored cases."""
        labels = [r.details.get("label") for r in results if r.details.get("label") is not None]
        if not labels:
            return {"labels": [], "fractions": [], "skew": 0.0}
        counts = Counter(labels)
        total = len(labels)
        sorted_labels = sorted(counts)
        fractions = [counts[lb] / total for lb in sorted_labels]
        # Simple skew: max_fraction - min_fraction (0 = perfectly balanced)
        skew = max(fractions) - min(fractions) if fractions else 0.0
        return {
            "labels": sorted_labels,
            "fractions": fractions,
            "counts": dict(counts),
            "skew": skew,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_trajectory(output: Any) -> list[dict[str, Any]]:
    """Normalise *output* into a list of step dicts."""
    if isinstance(output, list):
        return [s for s in output if isinstance(s, dict)]
    if isinstance(output, dict) and "trajectory" in output:
        return _parse_trajectory(output["trajectory"])
    return []
