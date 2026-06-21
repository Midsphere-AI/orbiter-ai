"""Tests for exo.eval.llm_scorer -- LLM-as-judge and quality scorers."""

from __future__ import annotations

import json
from typing import Any

import pytest

from exo.eval.base import Scorer  # pyright: ignore[reportMissingImports]
from exo.eval.llm_scorer import (  # pyright: ignore[reportMissingImports]
    ConstraintSatisfactionScorer,
    LLMAsJudgeScorer,
    LogicConsistencyScorer,
    OutputQualityScorer,
    ReasoningValidityScorer,
    extract_json,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_judge(response: str | dict[str, Any]):
    """Return an async callable that always returns *response*."""
    text = json.dumps(response) if isinstance(response, dict) else response

    async def _judge(prompt: str) -> str:
        return text

    return _judge


# ---------------------------------------------------------------------------
# extract_json
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_simple_json(self) -> None:
        assert extract_json('here is {"score": 0.5} done') == {"score": 0.5}

    def test_no_json(self) -> None:
        assert extract_json("no json here") == {}

    def test_invalid_json(self) -> None:
        assert extract_json("here is {bad json} done") == {}

    def test_empty_string(self) -> None:
        assert extract_json("") == {}


# ---------------------------------------------------------------------------
# LLMAsJudgeScorer -- base
# ---------------------------------------------------------------------------


class TestLLMAsJudgeScorerInit:
    def test_defaults(self) -> None:
        s = LLMAsJudgeScorer()
        assert s._name == "llm_judge"
        assert s._judge is None
        assert "evaluator" in s._system_prompt.lower()

    def test_custom_name(self) -> None:
        s = LLMAsJudgeScorer(name="my_judge")
        assert s._name == "my_judge"

    def test_custom_system_prompt(self) -> None:
        s = LLMAsJudgeScorer(system_prompt="Custom instructions")
        assert s._system_prompt == "Custom instructions"


class TestLLMAsJudgeScorerScore:
    async def test_no_judge_raises(self) -> None:
        """judge=None must raise ValueError — silent 0.0 was a footgun."""
        s = LLMAsJudgeScorer()
        with pytest.raises(ValueError, match="judge callable"):
            await s.score("c1", "input", "output")

    async def test_with_judge(self) -> None:
        judge = _mock_judge({"score": 0.8, "explanation": "Good"})
        s = LLMAsJudgeScorer(judge, name="test")
        sr = await s.score("c1", "input", "output")
        assert sr.score == 0.8
        assert sr.scorer_name == "test"
        assert sr.details.get("explanation") == "Good"

    async def test_clamps_score(self) -> None:
        judge = _mock_judge({"score": 1.5})
        s = LLMAsJudgeScorer(judge)
        sr = await s.score("c1", None, "output")
        assert sr.score == 1.0

    async def test_clamps_negative(self) -> None:
        judge = _mock_judge({"score": -0.5})
        s = LLMAsJudgeScorer(judge)
        sr = await s.score("c1", None, "output")
        assert sr.score == 0.0

    async def test_missing_score_defaults_zero(self) -> None:
        judge = _mock_judge({"explanation": "No score field"})
        s = LLMAsJudgeScorer(judge)
        sr = await s.score("c1", None, "output")
        assert sr.score == 0.0

    async def test_build_prompt_includes_input_and_output(self) -> None:
        captured: list[str] = []

        async def _judge(prompt: str) -> str:
            captured.append(prompt)
            return '{"score": 0.5}'

        s = LLMAsJudgeScorer(_judge)
        await s.score("c1", "my input", "my output")
        assert "[Input]\nmy input" in captured[0]
        assert "[Output]\nmy output" in captured[0]

    async def test_build_prompt_no_input(self) -> None:
        captured: list[str] = []

        async def _judge(prompt: str) -> str:
            captured.append(prompt)
            return '{"score": 0.5}'

        s = LLMAsJudgeScorer(_judge)
        await s.score("c1", None, "output")
        assert "[Input]" not in captured[0]
        assert "[Output]\noutput" in captured[0]


# ---------------------------------------------------------------------------
# OutputQualityScorer
# ---------------------------------------------------------------------------


class TestOutputQualityScorerInit:
    def test_defaults(self) -> None:
        s = OutputQualityScorer()
        assert s._name == "output_quality"
        assert "correctness" in s._dimensions
        assert len(s._dimensions) == 5

    def test_custom_dimensions(self) -> None:
        dims = {"a": 0.5, "b": 0.5}
        s = OutputQualityScorer(dimensions=dims)
        assert s._dimensions == dims

    def test_custom_name(self) -> None:
        s = OutputQualityScorer(name="quality_v2")
        assert s._name == "quality_v2"


class TestOutputQualityScorerScore:
    async def test_full_score(self) -> None:
        judge = _mock_judge(
            {
                "dimension_scores": {
                    "correctness": 1.0,
                    "relevance": 1.0,
                    "completeness": 1.0,
                    "clarity": 1.0,
                    "professionalism": 1.0,
                },
                "score": 1.0,
            }
        )
        s = OutputQualityScorer(judge)
        sr = await s.score("c1", "q", "a")
        assert sr.score == pytest.approx(1.0)
        assert sr.details.get("quality_label") == "Excellent"

    async def test_weighted_calculation(self) -> None:
        judge = _mock_judge(
            {
                "dimension_scores": {
                    "correctness": 0.5,
                    "relevance": 0.5,
                    "completeness": 0.5,
                    "clarity": 0.5,
                    "professionalism": 0.5,
                },
            }
        )
        s = OutputQualityScorer(judge)
        sr = await s.score("c1", None, "output")
        # All 0.5 x respective weights => 0.5 total
        assert sr.score == pytest.approx(0.5)
        assert sr.details.get("quality_label") == "Pass"

    async def test_mixed_scores(self) -> None:
        judge = _mock_judge(
            {
                "dimension_scores": {
                    "correctness": 1.0,  # x 0.4 = 0.4
                    "relevance": 0.5,  # x 0.2 = 0.1
                    "completeness": 0.5,  # x 0.2 = 0.1
                    "clarity": 0.0,  # x 0.1 = 0.0
                    "professionalism": 0.0,  # x 0.1 = 0.0
                },
            }
        )
        s = OutputQualityScorer(judge)
        sr = await s.score("c1", None, "output")
        assert sr.score == pytest.approx(0.6)
        assert sr.details.get("quality_label") == "Medium"

    async def test_missing_dimensions_default_zero(self) -> None:
        judge = _mock_judge({"dimension_scores": {"correctness": 1.0}})
        s = OutputQualityScorer(judge)
        sr = await s.score("c1", None, "output")
        # Only correctness contributes: 1.0 x 0.4 = 0.4
        assert sr.score == pytest.approx(0.4)

    async def test_quality_labels(self) -> None:
        for val, expected in [
            (0.95, "Excellent"),
            (0.85, "Good"),
            (0.70, "Medium"),
            (0.45, "Pass"),
            (0.10, "Fail"),
        ]:
            judge = _mock_judge(
                {
                    "dimension_scores": {
                        "correctness": val,
                        "relevance": val,
                        "completeness": val,
                        "clarity": val,
                        "professionalism": val,
                    }
                }
            )
            s = OutputQualityScorer(judge)
            sr = await s.score("c1", None, "x")
            assert sr.details.get("quality_label") == expected, (
                f"val={val} expected={expected} got={sr.details.get('quality_label')}"
            )

    async def test_prompt_includes_dimensions(self) -> None:
        captured: list[str] = []

        async def _judge(prompt: str) -> str:
            captured.append(prompt)
            return '{"dimension_scores": {}, "score": 0.0}'

        s = OutputQualityScorer(_judge)
        await s.score("c1", None, "output")
        assert "correctness" in captured[0]
        assert "relevance" in captured[0]


# ---------------------------------------------------------------------------
# LogicConsistencyScorer
# ---------------------------------------------------------------------------


class TestLogicConsistencyScorerInit:
    def test_defaults(self) -> None:
        s = LogicConsistencyScorer()
        assert s._name == "logic_consistency"

    def test_custom_name(self) -> None:
        s = LogicConsistencyScorer(name="logic_v2")
        assert s._name == "logic_v2"


class TestLogicConsistencyScorerScore:
    async def test_weighted_score(self) -> None:
        judge = _mock_judge(
            {
                "contradiction_score": 1.0,  # x 0.5
                "causal_score": 1.0,  # x 0.3
                "data_score": 1.0,  # x 0.2
            }
        )
        s = LogicConsistencyScorer(judge)
        sr = await s.score("c1", None, "The sky is blue.")
        assert sr.score == pytest.approx(1.0)

    async def test_partial_scores(self) -> None:
        judge = _mock_judge(
            {
                "contradiction_score": 0.8,
                "causal_score": 0.6,
                "data_score": 0.4,
            }
        )
        s = LogicConsistencyScorer(judge)
        sr = await s.score("c1", None, "text")
        expected = 0.8 * 0.5 + 0.6 * 0.3 + 0.4 * 0.2
        assert sr.score == pytest.approx(expected)

    async def test_missing_fields_default_zero(self) -> None:
        judge = _mock_judge({})
        s = LogicConsistencyScorer(judge)
        sr = await s.score("c1", None, "text")
        assert sr.score == 0.0

    async def test_no_judge(self) -> None:
        s = LogicConsistencyScorer()
        with pytest.raises(ValueError, match="judge callable"):
            await s.score("c1", None, "text")


# ---------------------------------------------------------------------------
# ReasoningValidityScorer
# ---------------------------------------------------------------------------


class TestReasoningValidityScorerInit:
    def test_defaults(self) -> None:
        s = ReasoningValidityScorer()
        assert s._name == "reasoning_validity"


class TestReasoningValidityScorerScore:
    async def test_valid_reasoning(self) -> None:
        judge = _mock_judge(
            {
                "score": 0.9,
                "is_valid": True,
                "fallacies": [],
                "reasoning_type": "deductive",
                "explanation": "Sound logic",
            }
        )
        s = ReasoningValidityScorer(judge)
        sr = await s.score("c1", None, "All A are B. X is A. Therefore X is B.")
        assert sr.score == 0.9
        assert sr.details.get("is_valid") is True

    async def test_with_fallacies(self) -> None:
        judge = _mock_judge(
            {
                "score": 0.2,
                "is_valid": False,
                "fallacies": ["Ad Hominem"],
                "reasoning_type": "inductive",
            }
        )
        s = ReasoningValidityScorer(judge)
        sr = await s.score("c1", None, "He's wrong because he's dumb.")
        assert sr.score == 0.2
        assert "Ad Hominem" in sr.details.get("fallacies", [])

    async def test_no_judge(self) -> None:
        s = ReasoningValidityScorer()
        with pytest.raises(ValueError, match="judge callable"):
            await s.score("c1", None, "text")


# ---------------------------------------------------------------------------
# ConstraintSatisfactionScorer
# ---------------------------------------------------------------------------


class TestConstraintSatisfactionScorerInit:
    def test_defaults(self) -> None:
        s = ConstraintSatisfactionScorer(["c1", "c2"])
        assert s._name == "constraint_satisfaction"
        assert s._constraints == ["c1", "c2"]

    def test_custom_name(self) -> None:
        s = ConstraintSatisfactionScorer(["c1"], name="cs")
        assert s._name == "cs"


class TestConstraintSatisfactionScorerScore:
    async def test_all_pass(self) -> None:
        judge = _mock_judge(
            {
                "constraint_results": [
                    {"id": 1, "status": "PASS"},
                    {"id": 2, "status": "PASS"},
                ],
                "score": 1.0,
            }
        )
        s = ConstraintSatisfactionScorer(["Must be polite", "Must be concise"], judge)
        sr = await s.score("c1", None, "Thank you. Here it is.")
        assert sr.score == 1.0

    async def test_partial_pass(self) -> None:
        judge = _mock_judge(
            {
                "constraint_results": [
                    {"id": 1, "status": "PASS"},
                    {"id": 2, "status": "FAIL"},
                    {"id": 3, "status": "PASS"},
                ],
                "score": 0.67,
            }
        )
        s = ConstraintSatisfactionScorer(["a", "b", "c"], judge)
        sr = await s.score("c1", None, "output")
        assert sr.score == pytest.approx(2 / 3)

    async def test_all_fail(self) -> None:
        judge = _mock_judge(
            {
                "constraint_results": [
                    {"id": 1, "status": "FAIL"},
                    {"id": 2, "status": "FAIL"},
                ],
            }
        )
        s = ConstraintSatisfactionScorer(["x", "y"], judge)
        sr = await s.score("c1", None, "output")
        assert sr.score == 0.0

    async def test_fallback_to_score_field(self) -> None:
        judge = _mock_judge({"score": 0.75})
        s = ConstraintSatisfactionScorer(["a"], judge)
        sr = await s.score("c1", None, "output")
        assert sr.score == 0.75

    async def test_prompt_includes_constraints(self) -> None:
        captured: list[str] = []

        async def _judge(prompt: str) -> str:
            captured.append(prompt)
            return '{"score": 0.5}'

        s = ConstraintSatisfactionScorer(["Be polite", "Be brief"], _judge)
        await s.score("c1", "q", "a")
        assert "Be polite" in captured[0]
        assert "Be brief" in captured[0]
        assert "[Input]\nq" in captured[0]

    async def test_no_judge(self) -> None:
        s = ConstraintSatisfactionScorer(["a"])
        with pytest.raises(ValueError, match="judge callable"):
            await s.score("c1", None, "text")


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    async def test_all_scorers_follow_protocol(self) -> None:
        """All scorers return ScorerResult with score in [0.0, 1.0]."""
        judge = _mock_judge({"score": 0.5, "dimension_scores": {}})
        scorers: list[Scorer] = [
            LLMAsJudgeScorer(judge),
            OutputQualityScorer(judge),
            LogicConsistencyScorer(judge),
            ReasoningValidityScorer(judge),
            ConstraintSatisfactionScorer(["x"], judge),
        ]
        for s in scorers:
            sr = await s.score("c1", "input", "output")
            assert 0.0 <= sr.score <= 1.0, f"{type(s).__name__}: {sr.score}"
            assert isinstance(sr.scorer_name, str)

    async def test_subclass_with_custom_hooks(self) -> None:
        """Verify subclassing works for custom domain-specific judges."""

        class CustomJudge(LLMAsJudgeScorer):
            def build_prompt(self, case_id: str, input: Any, output: Any) -> str:
                return f"Custom: {output}"

            def parse_response(self, response: str) -> tuple[float, dict[str, Any]]:
                return 0.42, {"custom": True}

        judge = _mock_judge({"score": 0.0})
        s = CustomJudge(judge, name="custom")
        sr = await s.score("c1", None, "output")
        assert sr.score == 0.42
        assert sr.details == {"custom": True}
