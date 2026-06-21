"""LLM-as-Judge scorers and multi-dimensional quality assessment."""

from __future__ import annotations

import json
from typing import Any

from exo.eval.base import Scorer, ScorerResult  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Judge protocol
# ---------------------------------------------------------------------------


def extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from *text* (supports nested braces).

    Falls back to an empty dict if no valid JSON is found.

    This is the canonical implementation shared by :mod:`exo.eval.llm_scorer`
    and :mod:`exo.eval.reflection`.  Both modules import from here rather than
    duplicating the brace-depth logic.
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])  # type: ignore[no-any-return]
                    except (json.JSONDecodeError, ValueError):
                        break
        start = text.find("{", start + 1)
    return {}


# ---------------------------------------------------------------------------
# LLMAsJudgeScorer
# ---------------------------------------------------------------------------


class LLMAsJudgeScorer(Scorer):
    """Scorer that delegates evaluation to an LLM judge.

    Subclass and override :meth:`build_prompt` and :meth:`parse_response`
    for domain-specific judges.  Or use directly with a custom *system_prompt*
    and a *judge* callable.

    *judge* is an async callable ``(prompt: str) -> str`` — any function that
    takes a prompt and returns the LLM response text.  This keeps the scorer
    decoupled from a specific model provider.
    """

    __slots__ = ("_judge", "_name", "_system_prompt")

    def __init__(
        self,
        judge: Any = None,
        *,
        system_prompt: str | None = None,
        name: str = "llm_judge",
    ) -> None:
        self._judge = judge
        self._system_prompt = system_prompt or self._default_system_prompt()
        self._name = name

    # -- overridable hooks ---------------------------------------------------

    def build_prompt(self, case_id: str, input: Any, output: Any) -> str:
        """Build the user-facing prompt sent to the judge LLM."""
        parts = [self._system_prompt, ""]
        if input is not None:
            parts.append(f"[Input]\n{input}")
        parts.append(f"[Output]\n{output}")
        parts.append('\nReturn a JSON object with at minimum {"score": <float 0.0-1.0>}.')
        return "\n".join(parts)

    def parse_response(self, response: str) -> tuple[float, dict[str, Any]]:
        """Extract score and details from the judge LLM response."""
        data = extract_json(response)
        score = float(data.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        return score, data

    # -- Scorer interface ----------------------------------------------------

    async def score(self, case_id: str, input: Any, output: Any) -> ScorerResult:
        if self._judge is None:
            msg = (
                f"{type(self).__name__}(name={self._name!r}) cannot score without a judge callable. "
                "Pass a judge=(async callable) at construction time."
            )
            raise ValueError(msg)
        prompt = self.build_prompt(case_id, input, output)
        response = await self._judge(prompt)
        score, details = self.parse_response(str(response))
        return ScorerResult(scorer_name=self._name, score=score, details=details)

    @staticmethod
    def _default_system_prompt() -> str:
        return (
            "You are an expert evaluator. Score the output on a scale of 0.0 to 1.0. "
            'Respond with a JSON object: {"score": <float>, "explanation": "<reasoning>"}.'
        )


# ---------------------------------------------------------------------------
# OutputQualityScorer — weighted 5-dimensional quality assessment
# ---------------------------------------------------------------------------

_QUALITY_DIMENSIONS: dict[str, float] = {
    "correctness": 0.40,
    "relevance": 0.20,
    "completeness": 0.20,
    "clarity": 0.10,
    "professionalism": 0.10,
}

_QUALITY_LABELS: list[tuple[float, str]] = [
    (0.90, "Excellent"),
    (0.80, "Good"),
    (0.60, "Medium"),
    (0.40, "Pass"),
    (0.00, "Fail"),
]


def _quality_label(score: float) -> str:
    for threshold, label in _QUALITY_LABELS:
        if score >= threshold:
            return label
    return "Fail"


class OutputQualityScorer(LLMAsJudgeScorer):
    """Weighted 5-dimensional quality scorer.

    Dimensions (default weights):
      correctness (40%), relevance (20%), completeness (20%),
      clarity (10%), professionalism (10%).
    """

    __slots__ = ("_dimensions",)

    def __init__(
        self,
        judge: Any = None,
        *,
        dimensions: dict[str, float] | None = None,
        name: str = "output_quality",
    ) -> None:
        super().__init__(judge, name=name)
        self._dimensions = dimensions or dict(_QUALITY_DIMENSIONS)

    def build_prompt(self, case_id: str, input: Any, output: Any) -> str:
        dim_list = ", ".join(self._dimensions)
        parts = [
            "You are an expert evaluator. Score the output on each dimension "
            f"from 0.0 to 1.0: {dim_list}.",
            "",
        ]
        if input is not None:
            parts.append(f"[Input]\n{input}")
        parts.append(f"[Output]\n{output}")
        dim_schema = ", ".join(f'"{d}": <float>' for d in self._dimensions)
        parts.append(
            f'\nReturn a JSON object: {{"dimension_scores": {{{dim_schema}}}, '
            '"score": <weighted_total>, "quality_label": "<label>", "reason": "<reasoning>"}.'
        )
        return "\n".join(parts)

    def parse_response(self, response: str) -> tuple[float, dict[str, Any]]:
        data = extract_json(response)
        dim_scores = data.get("dimension_scores", {})

        # Compute weighted score from whatever dimensions the LLM returned
        total = 0.0
        for dim, weight in self._dimensions.items():
            total += float(dim_scores.get(dim, 0.0)) * weight
        total = max(0.0, min(1.0, total))

        data["score"] = total
        data["quality_label"] = _quality_label(total)
        return total, data


# ---------------------------------------------------------------------------
# LogicConsistencyScorer
# ---------------------------------------------------------------------------


class LogicConsistencyScorer(LLMAsJudgeScorer):
    """Detects internal contradictions, causal fallacies, data inconsistencies."""

    __slots__ = ()

    def __init__(self, judge: Any = None, *, name: str = "logic_consistency") -> None:
        super().__init__(judge, name=name)

    @staticmethod
    def _default_system_prompt() -> str:
        return (
            "You are a logic evaluator. Analyse the output for internal "
            "contradictions (weight 0.5), causal/temporal errors (weight 0.3), "
            "and numerical/data inconsistencies (weight 0.2). "
            'Return JSON: {"contradiction_score": <float>, "causal_score": <float>, '
            '"data_score": <float>, "score": <weighted_total>, "issues": [<str>]}.'
        )

    def parse_response(self, response: str) -> tuple[float, dict[str, Any]]:
        data = extract_json(response)
        c = float(data.get("contradiction_score", 0.0))
        ca = float(data.get("causal_score", 0.0))
        d = float(data.get("data_score", 0.0))
        total = c * 0.5 + ca * 0.3 + d * 0.2
        total = max(0.0, min(1.0, total))
        data["score"] = total
        return total, data


# ---------------------------------------------------------------------------
# ReasoningValidityScorer
# ---------------------------------------------------------------------------


class ReasoningValidityScorer(LLMAsJudgeScorer):
    """Validates argumentation logic and detects formal/informal fallacies."""

    __slots__ = ()

    def __init__(self, judge: Any = None, *, name: str = "reasoning_validity") -> None:
        super().__init__(judge, name=name)

    @staticmethod
    def _default_system_prompt() -> str:
        return (
            "You are a reasoning evaluator. Assess whether the argument is "
            "logically valid. Classify reasoning type (deductive, inductive, "
            "abductive) and list any fallacies. "
            'Return JSON: {"score": <float 0.0-1.0>, "is_valid": <bool>, '
            '"fallacies": [<str>], "reasoning_type": "<type>", "explanation": "<text>"}.'
        )


# ---------------------------------------------------------------------------
# ConstraintSatisfactionScorer
# ---------------------------------------------------------------------------


class ConstraintSatisfactionScorer(LLMAsJudgeScorer):
    """Binary constraint checking — PASS/FAIL per constraint, no partial credit."""

    __slots__ = ("_constraints",)

    def __init__(
        self,
        constraints: list[str],
        judge: Any = None,
        *,
        name: str = "constraint_satisfaction",
    ) -> None:
        super().__init__(judge, name=name)
        self._constraints = constraints

    def build_prompt(self, case_id: str, input: Any, output: Any) -> str:
        numbered = "\n".join(f"  {i + 1}. {c}" for i, c in enumerate(self._constraints))
        parts = [
            "You are a constraint evaluator. Check whether the output satisfies "
            "each constraint (PASS or FAIL, no partial credit).",
            f"\nConstraints:\n{numbered}",
            "",
        ]
        if input is not None:
            parts.append(f"[Input]\n{input}")
        parts.append(f"[Output]\n{output}")
        parts.append(
            '\nReturn JSON: {"constraint_results": [{"id": <int>, "status": "PASS"|"FAIL"}], '
            '"score": <float 0.0-1.0>}.'
        )
        return "\n".join(parts)

    def parse_response(self, response: str) -> tuple[float, dict[str, Any]]:
        data = extract_json(response)
        # Try to compute score from individual constraints if available
        results = data.get("constraint_results", [])
        if results and isinstance(results, list):
            passed = sum(
                1
                for r in results
                if isinstance(r, dict) and str(r.get("status", "")).upper() == "PASS"
            )
            total = len(self._constraints) or 1
            score = passed / total
        else:
            score = float(data.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        data["score"] = score
        return score, data
