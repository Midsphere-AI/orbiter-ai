"""LLM-as-Judge scorers and multi-dimensional quality assessment."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from exo.eval.base import EvalError, Scorer, ScorerResult  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Judge protocol
# ---------------------------------------------------------------------------


def extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from *text* (supports nested braces).

    Falls back to an empty dict if no valid JSON is found.

    This is the canonical implementation shared by :mod:`exo.eval.llm_scorer`
    and :mod:`exo.eval.reflection`.  Both modules import from here rather than
    duplicating the brace-depth logic.

    String literals are skipped so that ``}`` or ``{`` inside a JSON string
    value does not confuse the brace-depth counter (e.g.
    ``{"key": "a}b"}`` was previously broken by the unescaped ``}``).
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        i = start
        in_string = False
        escape_next = False
        while i < len(text):
            ch = text[i]
            if escape_next:
                escape_next = False
            elif ch == "\\" and in_string:
                escape_next = True
            elif ch == '"':
                in_string = not in_string
            elif not in_string:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])  # type: ignore[no-any-return]
                        except (json.JSONDecodeError, ValueError):
                            break
            i += 1
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

    __slots__ = ("_judge", "_name", "_system_prompt", "_timeout")

    def __init__(
        self,
        judge: Any = None,
        *,
        system_prompt: str | None = None,
        name: str = "llm_judge",
        timeout: float = 0.0,
    ) -> None:
        self._judge = judge
        self._system_prompt = system_prompt or self._default_system_prompt()
        self._name = name
        self._timeout = timeout

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
            raise EvalError(
                f"{type(self).__name__}(name={self._name!r}) cannot score without a judge callable.",
                context={"scorer": self._name, "case_id": case_id},
                hint="Pass judge=(async callable) at construction time.",
            )
        prompt = self.build_prompt(case_id, input, output)
        if self._timeout > 0:
            response = await asyncio.wait_for(self._judge(prompt), timeout=self._timeout)
        else:
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
            n_configured = len(self._constraints)
            n_returned = len(results)
            if n_returned != n_configured:
                # Log a note when the LLM returned a different number of results
                # than the number of configured constraints.  We clamp to the
                # configured count so extras are ignored and the denominator is
                # stable; missing results count as FAILs.
                logger.warning(
                    "ConstraintSatisfactionScorer: LLM returned %d constraint_results "
                    "but %d constraints were configured; clamping to %d.",
                    n_returned,
                    n_configured,
                    n_configured,
                )
                data["constraint_count_mismatch"] = {
                    "configured": n_configured,
                    "returned": n_returned,
                }
            passed = sum(
                1
                for r in results[:n_configured]
                if isinstance(r, dict) and str(r.get("status", "")).upper() == "PASS"
            )
            total = n_configured or 1
            score = passed / total
        else:
            score = float(data.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        data["score"] = score
        return score, data
