"""Tests for trajectory_scorers: trajectory validation, time cost, accuracy, label distribution, registry."""

from __future__ import annotations

import pytest

from exo.eval.base import EvalError, Scorer, ScorerResult  # pyright: ignore[reportMissingImports]
from exo.eval.trajectory_scorers import (  # pyright: ignore[reportMissingImports]
    AnswerAccuracyLLMScorer,
    LabelDistributionScorer,
    TimeCostScorer,
    TrajectoryValidator,
    _parse_trajectory,
    get_scorer,
    list_scorers,
    scorer_register,
)

# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestScorerRegistry:
    def test_register_decorator(self) -> None:
        @scorer_register("test_metric")
        class _TestScorer(Scorer):
            async def score(self, case_id: str, input: object, output: object) -> ScorerResult:
                return ScorerResult(scorer_name="test", score=1.0)

        assert get_scorer("test_metric") is _TestScorer

    def test_builtin_scorers_registered(self) -> None:
        names = list_scorers()
        assert "trajectory" in names
        assert "time_cost" in names
        assert "answer_accuracy" in names
        assert "label_distribution" in names

    def test_get_scorer_missing(self) -> None:
        with pytest.raises(EvalError, match="not registered"):
            get_scorer("nonexistent_scorer")

    def test_list_scorers_sorted(self) -> None:
        names = list_scorers()
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# _parse_trajectory helper
# ---------------------------------------------------------------------------


class TestParseTrajectory:
    def test_list_of_dicts(self) -> None:
        result = _parse_trajectory([{"step": 1}, {"step": 2}])
        assert len(result) == 2

    def test_wrapped_in_dict(self) -> None:
        result = _parse_trajectory({"trajectory": [{"step": 1}]})
        assert len(result) == 1

    def test_empty_list(self) -> None:
        assert _parse_trajectory([]) == []

    def test_non_list(self) -> None:
        assert _parse_trajectory("not a list") == []

    def test_filters_non_dicts(self) -> None:
        result = _parse_trajectory([{"step": 1}, "bad", 42, {"step": 2}])
        assert len(result) == 2


# ---------------------------------------------------------------------------
# TrajectoryValidator tests
# ---------------------------------------------------------------------------


class TestTrajectoryValidator:
    async def test_valid_trajectory(self) -> None:
        scorer = TrajectoryValidator()
        trajectory = [
            {"step": 0, "action": {"content": "hello"}},
            {"step": 1, "action": {"content": "world"}},
        ]
        result = await scorer.score("c1", {}, trajectory)
        assert result.score == 1.0
        assert result.details["valid"] == 2
        assert result.details["total"] == 2

    async def test_empty_trajectory(self) -> None:
        scorer = TrajectoryValidator()
        result = await scorer.score("c1", {}, [])
        assert result.score == 0.0
        assert "error" in result.details

    async def test_missing_action(self) -> None:
        scorer = TrajectoryValidator()
        trajectory = [{"step": 0}]  # no "action"
        result = await scorer.score("c1", {}, trajectory)
        assert result.score == 0.0
        assert len(result.details["errors"]) == 1

    async def test_missing_step_id(self) -> None:
        scorer = TrajectoryValidator()
        trajectory = [{"action": {"content": "x"}}]  # no step/id
        result = await scorer.score("c1", {}, trajectory)
        assert result.score == 0.0

    async def test_id_key_accepted(self) -> None:
        scorer = TrajectoryValidator()
        trajectory = [{"id": "abc", "action": {}}]
        result = await scorer.score("c1", {}, trajectory)
        assert result.score == 1.0

    async def test_partial_valid(self) -> None:
        scorer = TrajectoryValidator()
        trajectory = [
            {"step": 0, "action": {}},
            {"action": {}},  # missing step/id
        ]
        result = await scorer.score("c1", {}, trajectory)
        assert result.score == 0.5

    async def test_custom_required_keys(self) -> None:
        scorer = TrajectoryValidator(required_keys=("action", "state"))
        trajectory = [{"step": 0, "action": {}, "state": {}}]
        result = await scorer.score("c1", {}, trajectory)
        assert result.score == 1.0

    async def test_custom_name(self) -> None:
        scorer = TrajectoryValidator(name="traj_check")
        result = await scorer.score("c1", {}, [{"step": 0, "action": {}}])
        assert result.scorer_name == "traj_check"


# ---------------------------------------------------------------------------
# TimeCostScorer tests
# ---------------------------------------------------------------------------


class TestTimeCostScorer:
    async def test_zero_time(self) -> None:
        scorer = TimeCostScorer()
        result = await scorer.score("c1", {}, {"_time_cost_ms": 0})
        assert result.score == 1.0

    async def test_half_budget(self) -> None:
        scorer = TimeCostScorer(max_ms=10_000)
        result = await scorer.score("c1", {}, {"_time_cost_ms": 5_000})
        assert result.score == 0.5

    async def test_over_budget(self) -> None:
        scorer = TimeCostScorer(max_ms=1_000)
        result = await scorer.score("c1", {}, {"_time_cost_ms": 2_000})
        assert result.score == 0.0

    async def test_no_time_key(self) -> None:
        """Missing _time_cost_ms raises KeyError — no silent perfect score."""
        scorer = TimeCostScorer()
        with pytest.raises(KeyError, match="_time_cost_ms"):
            await scorer.score("c1", {}, {"answer": "hello"})

    async def test_non_dict_output(self) -> None:
        """Non-dict output raises KeyError — no silent perfect score."""
        scorer = TimeCostScorer()
        with pytest.raises(KeyError, match="_time_cost_ms"):
            await scorer.score("c1", {}, "string output")

    async def test_details(self) -> None:
        scorer = TimeCostScorer(max_ms=5_000)
        result = await scorer.score("c1", {}, {"_time_cost_ms": 1_000})
        assert result.details["elapsed_ms"] == 1_000
        assert result.details["max_ms"] == 5_000

    async def test_custom_name(self) -> None:
        scorer = TimeCostScorer(name="latency")
        result = await scorer.score("c1", {}, {"_time_cost_ms": 0})
        assert result.scorer_name == "latency"


# ---------------------------------------------------------------------------
# AnswerAccuracyLLMScorer tests
# ---------------------------------------------------------------------------


class TestAnswerAccuracyLLMScorer:
    async def test_no_judge(self) -> None:
        scorer = AnswerAccuracyLLMScorer()
        with pytest.raises(EvalError, match="judge callable"):
            await scorer.score("c1", {"question": "Q", "answer": "A"}, "response")

    async def test_with_judge(self) -> None:
        async def mock_judge(prompt: str) -> str:
            return '{"score": 0.85, "explanation": "close match"}'

        scorer = AnswerAccuracyLLMScorer(judge=mock_judge)
        result = await scorer.score("c1", {"question": "Q", "answer": "A"}, "response")
        assert result.score == 0.85

    async def test_prompt_includes_question_and_answer(self) -> None:
        scorer = AnswerAccuracyLLMScorer()
        prompt = scorer.build_prompt("c1", {"question": "What is 2+2?", "answer": "4"}, "four")
        assert "What is 2+2?" in prompt
        assert "4" in prompt
        assert "four" in prompt

    async def test_custom_keys(self) -> None:
        scorer = AnswerAccuracyLLMScorer(question_key="q", answer_key="a")
        prompt = scorer.build_prompt("c1", {"q": "Q1", "a": "A1"}, "out")
        assert "Q1" in prompt
        assert "A1" in prompt

    async def test_non_dict_input(self) -> None:
        scorer = AnswerAccuracyLLMScorer()
        prompt = scorer.build_prompt("c1", "raw question", "output")
        assert "raw question" in prompt

    def test_custom_name(self) -> None:
        scorer = AnswerAccuracyLLMScorer(name="accuracy_v2")
        assert scorer._name == "accuracy_v2"


# ---------------------------------------------------------------------------
# LabelDistributionScorer tests
# ---------------------------------------------------------------------------


class TestLabelDistributionScorer:
    async def test_per_case_labelled(self) -> None:
        # Per-case score is 1.0 when a label is present
        scorer = LabelDistributionScorer()
        result = await scorer.score("c1", {"label": "positive"}, "out")
        assert result.score == 1.0
        assert result.details["label"] == "positive"

    async def test_per_case_unlabelled(self) -> None:
        # Per-case score is 0.0 when label is absent
        scorer = LabelDistributionScorer()
        result = await scorer.score("c1", {"text": "hello"}, "out")
        assert result.score == 0.0
        assert result.details["label"] is None

    async def test_no_label(self) -> None:
        scorer = LabelDistributionScorer()
        result = await scorer.score("c1", {"text": "hello"}, "out")
        assert result.details["label"] is None

    async def test_non_dict_input(self) -> None:
        scorer = LabelDistributionScorer()
        result = await scorer.score("c1", "raw", "out")
        assert result.score == 0.0
        assert result.details["label"] is None

    async def test_custom_label_key(self) -> None:
        scorer = LabelDistributionScorer(label_key="category")
        result = await scorer.score("c1", {"category": "A"}, "out")
        assert result.score == 1.0
        assert result.details["label"] == "A"

    async def test_custom_name(self) -> None:
        scorer = LabelDistributionScorer(name="dist_check")
        result = await scorer.score("c1", {}, "out")
        assert result.scorer_name == "dist_check"


class TestLabelDistributionSummarize:
    def test_balanced(self) -> None:
        scorer = LabelDistributionScorer()
        results = [
            ScorerResult(scorer_name="label_distribution", score=0.0, details={"label": "A"}),
            ScorerResult(scorer_name="label_distribution", score=0.0, details={"label": "B"}),
            ScorerResult(scorer_name="label_distribution", score=0.0, details={"label": "A"}),
            ScorerResult(scorer_name="label_distribution", score=0.0, details={"label": "B"}),
        ]
        summary = scorer.summarize(results)
        assert summary["skew"] == 0.0
        assert set(summary["labels"]) == {"A", "B"}

    def test_skewed(self) -> None:
        scorer = LabelDistributionScorer()
        results = [
            ScorerResult(scorer_name="ld", score=0.0, details={"label": "A"}),
            ScorerResult(scorer_name="ld", score=0.0, details={"label": "A"}),
            ScorerResult(scorer_name="ld", score=0.0, details={"label": "A"}),
            ScorerResult(scorer_name="ld", score=0.0, details={"label": "B"}),
        ]
        summary = scorer.summarize(results)
        assert summary["skew"] == 0.5  # 0.75 - 0.25
        assert summary["counts"]["A"] == 3

    def test_empty_results(self) -> None:
        scorer = LabelDistributionScorer()
        summary = scorer.summarize([])
        assert summary["labels"] == []
        assert summary["skew"] == 0.0

    def test_no_labels(self) -> None:
        scorer = LabelDistributionScorer()
        results = [
            ScorerResult(scorer_name="ld", score=0.0, details={"label": None}),
        ]
        summary = scorer.summarize(results)
        assert summary["labels"] == []

    def test_single_label(self) -> None:
        scorer = LabelDistributionScorer()
        results = [
            ScorerResult(scorer_name="ld", score=0.0, details={"label": "X"}),
            ScorerResult(scorer_name="ld", score=0.0, details={"label": "X"}),
        ]
        summary = scorer.summarize(results)
        assert summary["skew"] == 0.0
        assert summary["fractions"] == [1.0]
