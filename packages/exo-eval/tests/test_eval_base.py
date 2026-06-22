"""Tests for exo.eval.base — eval types + base evaluator."""

from __future__ import annotations

from typing import Any

import pytest

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

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _EchoTarget(EvalTarget):
    """Returns the input as-is."""

    async def predict(self, case_id: str, input: Any) -> Any:
        return input


class _ConstantScorer(Scorer):
    """Returns a fixed score for every case."""

    def __init__(self, name: str = "accuracy", score: float = 0.8) -> None:
        self._name = name
        self._score = score

    async def score(self, case_id: str, input: Any, output: Any) -> ScorerResult:
        return ScorerResult(scorer_name=self._name, score=self._score)


class _ToggleScorer(Scorer):
    """Alternates between 1.0 and 0.0 based on the repeat number embedded in case_id.

    The repeat suffix ``-r<n>`` appended by the Evaluator is parsed from
    ``case_id`` so the toggle is stable regardless of asyncio.gather order.
    If no suffix is found, repeat index 0 is assumed.
    """

    def __init__(self, name: str = "toggle") -> None:
        self._name = name

    async def score(self, case_id: str, input: Any, output: Any) -> ScorerResult:
        # Extract repeat index from case_id (e.g. "c1-r0", "c1-r1", "c1-r2").
        repeat_idx = 0
        if "-r" in case_id:
            try:
                repeat_idx = int(case_id.rsplit("-r", 1)[1])
            except ValueError:
                repeat_idx = 0
        val = 1.0 if repeat_idx % 2 == 0 else 0.0
        return ScorerResult(scorer_name=self._name, score=val)


# ---------------------------------------------------------------------------
# EvalStatus
# ---------------------------------------------------------------------------


class TestEvalStatus:
    def test_values(self) -> None:
        assert set(EvalStatus) == {
            EvalStatus.PASSED,
            EvalStatus.FAILED,
            EvalStatus.NOT_EVALUATED,
        }

    def test_str_enum(self) -> None:
        assert str(EvalStatus.PASSED) == "passed"


# ---------------------------------------------------------------------------
# ScorerResult
# ---------------------------------------------------------------------------


class TestScorerResult:
    def test_creation(self) -> None:
        sr = ScorerResult(scorer_name="acc", score=0.9)
        assert sr.scorer_name == "acc"
        assert sr.score == 0.9
        assert sr.status == EvalStatus.NOT_EVALUATED
        assert sr.details == {}

    def test_frozen(self) -> None:
        sr = ScorerResult(scorer_name="acc", score=0.9)
        with pytest.raises(AttributeError):
            sr.score = 0.5  # type: ignore[misc]

    def test_with_details(self) -> None:
        sr = ScorerResult(
            scorer_name="acc",
            score=0.7,
            status=EvalStatus.PASSED,
            details={"reason": "close enough"},
        )
        assert sr.details["reason"] == "close enough"


# ---------------------------------------------------------------------------
# EvalCaseResult
# ---------------------------------------------------------------------------


class TestEvalCaseResult:
    def test_creation(self) -> None:
        cr = EvalCaseResult(case_id="c1", input="hello", output="world")
        assert cr.case_id == "c1"
        assert cr.scores == {}

    def test_with_scores(self) -> None:
        sr = ScorerResult(scorer_name="acc", score=0.8)
        cr = EvalCaseResult(case_id="c1", input="x", output="y", scores={"acc": sr})
        assert cr.scores["acc"].score == 0.8


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------


class TestEvalResult:
    def test_empty(self) -> None:
        r = EvalResult()
        assert r.case_results == []
        assert r.summary == {}
        assert r.pass_at_k == {}

    def test_with_data(self) -> None:
        cr = EvalCaseResult(case_id="c1", input="x", output="y")
        r = EvalResult(
            case_results=[cr],
            summary={"acc": 0.9},
            pass_at_k={1: 1.0},
        )
        assert len(r.case_results) == 1
        assert r.pass_at_k[1] == 1.0


# ---------------------------------------------------------------------------
# EvalCriteria
# ---------------------------------------------------------------------------


class TestEvalCriteria:
    def test_defaults(self) -> None:
        c = EvalCriteria(metric_name="acc")
        assert c.threshold == 0.5

    def test_judge_pass(self) -> None:
        c = EvalCriteria(metric_name="acc", threshold=0.7)
        assert c.judge(0.8) == EvalStatus.PASSED
        assert c.judge(0.7) == EvalStatus.PASSED

    def test_judge_fail(self) -> None:
        c = EvalCriteria(metric_name="acc", threshold=0.7)
        assert c.judge(0.69) == EvalStatus.FAILED

    def test_frozen(self) -> None:
        c = EvalCriteria(metric_name="acc")
        with pytest.raises(AttributeError):
            c.threshold = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# EvalTarget ABC
# ---------------------------------------------------------------------------


class TestEvalTarget:
    async def test_concrete_target(self) -> None:
        t = _EchoTarget()
        result = await t.predict("c1", "hello")
        assert result == "hello"

    def test_abstract(self) -> None:
        with pytest.raises(TypeError):
            EvalTarget()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Scorer ABC
# ---------------------------------------------------------------------------


class TestScorer:
    async def test_concrete_scorer(self) -> None:
        s = _ConstantScorer(score=0.9)
        sr = await s.score("c1", "x", "y")
        assert sr.score == 0.9

    def test_abstract(self) -> None:
        with pytest.raises(TypeError):
            Scorer()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Evaluator init
# ---------------------------------------------------------------------------


class TestEvaluatorInit:
    def test_defaults(self) -> None:
        ev = Evaluator(scorers=[_ConstantScorer()])
        assert repr(ev) == "Evaluator(scorers=1, parallel=4, repeat_times=1)"

    def test_invalid_parallel(self) -> None:
        with pytest.raises(EvalError, match="parallel"):
            Evaluator(scorers=[], parallel=0)

    def test_invalid_repeat_times(self) -> None:
        with pytest.raises(EvalError, match="repeat_times"):
            Evaluator(scorers=[], repeat_times=0)

    def test_with_criteria(self) -> None:
        c = EvalCriteria(metric_name="accuracy", threshold=0.7)
        ev = Evaluator(scorers=[_ConstantScorer()], criteria=[c])
        assert "accuracy" in ev._criteria


# ---------------------------------------------------------------------------
# Evaluator.evaluate
# ---------------------------------------------------------------------------


class TestEvaluatorEvaluate:
    async def test_single_case(self) -> None:
        ev = Evaluator(scorers=[_ConstantScorer(score=0.9)])
        dataset = [{"id": "c1", "input": "hello"}]
        result = await ev.evaluate(_EchoTarget(), dataset)
        assert len(result.case_results) == 1
        assert result.case_results[0].output == "hello"
        assert result.summary["accuracy"] == pytest.approx(0.9)

    async def test_multiple_cases(self) -> None:
        ev = Evaluator(scorers=[_ConstantScorer(score=0.5)])
        dataset = [{"id": f"c{i}", "input": i} for i in range(5)]
        result = await ev.evaluate(_EchoTarget(), dataset)
        assert len(result.case_results) == 5
        assert result.summary["accuracy"] == pytest.approx(0.5)

    async def test_empty_dataset(self) -> None:
        ev = Evaluator(scorers=[_ConstantScorer()])
        result = await ev.evaluate(_EchoTarget(), [])
        assert result.case_results == []
        assert result.summary == {}

    async def test_multiple_scorers(self) -> None:
        ev = Evaluator(
            scorers=[
                _ConstantScorer(name="acc", score=0.8),
                _ConstantScorer(name="rel", score=0.6),
            ]
        )
        dataset = [{"id": "c1", "input": "x"}]
        result = await ev.evaluate(_EchoTarget(), dataset)
        assert result.summary["acc"] == pytest.approx(0.8)
        assert result.summary["rel"] == pytest.approx(0.6)

    async def test_criteria_applied(self) -> None:
        criteria = [EvalCriteria(metric_name="accuracy", threshold=0.7)]
        ev = Evaluator(scorers=[_ConstantScorer(score=0.8)], criteria=criteria)
        dataset = [{"id": "c1", "input": "x"}]
        result = await ev.evaluate(_EchoTarget(), dataset)
        sr = result.case_results[0].scores["accuracy"]
        assert sr.status == EvalStatus.PASSED

    async def test_criteria_fail(self) -> None:
        criteria = [EvalCriteria(metric_name="accuracy", threshold=0.9)]
        ev = Evaluator(scorers=[_ConstantScorer(score=0.5)], criteria=criteria)
        dataset = [{"id": "c1", "input": "x"}]
        result = await ev.evaluate(_EchoTarget(), dataset)
        sr = result.case_results[0].scores["accuracy"]
        assert sr.status == EvalStatus.FAILED

    async def test_parallel_execution(self) -> None:
        ev = Evaluator(scorers=[_ConstantScorer()], parallel=2)
        dataset = [{"id": f"c{i}", "input": i} for i in range(10)]
        result = await ev.evaluate(_EchoTarget(), dataset)
        assert len(result.case_results) == 10

    async def test_case_without_id(self) -> None:
        ev = Evaluator(scorers=[_ConstantScorer()])
        dataset = [{"input": "hello"}]
        result = await ev.evaluate(_EchoTarget(), dataset)
        assert len(result.case_results) == 1
        # Case ID falls back to a generated string
        assert result.case_results[0].case_id.startswith("case-")

    async def test_summarize_skips_sentinel_keys(self) -> None:
        """__error__ sentinel inserted by _run_safe must not appear in summary."""

        class _RaisingTarget(EvalTarget):
            async def predict(self, case_id: str, input: Any) -> Any:
                raise RuntimeError("boom")

        ev = Evaluator(scorers=[_ConstantScorer(score=0.9)])
        dataset = [{"id": "c1", "input": "x"}, {"id": "c2", "input": "y"}]
        result = await ev.evaluate(_RaisingTarget(), dataset)
        # Both cases fail → only __error__ sentinel in scores; summary must be empty.
        assert "__error__" not in result.summary
        assert result.summary == {}

    async def test_repeat_case_ids_include_suffix(self) -> None:
        """With repeat_times > 1, case_ids embed the repeat index for stable scorer dispatch."""
        ev = Evaluator(scorers=[_ConstantScorer()], repeat_times=3)
        dataset = [{"id": "c1", "input": "x"}]
        result = await ev.evaluate(_EchoTarget(), dataset)
        assert len(result.case_results) == 3
        case_ids = {cr.case_id for cr in result.case_results}
        assert case_ids == {"c1-r0", "c1-r1", "c1-r2"}


# ---------------------------------------------------------------------------
# Pass@k
# ---------------------------------------------------------------------------


class TestPassAtK:
    async def test_no_repeats(self) -> None:
        ev = Evaluator(
            scorers=[_ConstantScorer(score=0.8)],
            criteria=[EvalCriteria(metric_name="accuracy", threshold=0.7)],
            repeat_times=1,
        )
        dataset = [{"id": "c1", "input": "x"}]
        result = await ev.evaluate(_EchoTarget(), dataset)
        # repeat_times=1 → no pass@k
        assert result.pass_at_k == {}

    async def test_all_pass(self) -> None:
        ev = Evaluator(
            scorers=[_ConstantScorer(score=0.9)],
            criteria=[EvalCriteria(metric_name="accuracy", threshold=0.5)],
            repeat_times=3,
        )
        dataset = [{"id": "c1", "input": "x"}]
        result = await ev.evaluate(_EchoTarget(), dataset)
        assert len(result.case_results) == 3  # 1 case x 3 repeats
        assert result.pass_at_k[1] == pytest.approx(1.0)
        assert result.pass_at_k[3] == pytest.approx(1.0)

    async def test_none_pass(self) -> None:
        ev = Evaluator(
            scorers=[_ConstantScorer(score=0.1)],
            criteria=[EvalCriteria(metric_name="accuracy", threshold=0.5)],
            repeat_times=3,
        )
        dataset = [{"id": "c1", "input": "x"}]
        result = await ev.evaluate(_EchoTarget(), dataset)
        assert result.pass_at_k[1] == pytest.approx(0.0)
        assert result.pass_at_k[3] == pytest.approx(0.0)

    async def test_partial_pass(self) -> None:
        # Toggle scorer: 1.0, 0.0, 1.0 → with threshold 0.5:
        # pass@1 = 1.0 (first attempt passes)
        # pass@2 = 1.0 (at least one of first 2 passes)
        ev = Evaluator(
            scorers=[_ToggleScorer(name="accuracy")],
            criteria=[EvalCriteria(metric_name="accuracy", threshold=0.5)],
            repeat_times=3,
        )
        dataset = [{"id": "c1", "input": "x"}]
        result = await ev.evaluate(_EchoTarget(), dataset)
        assert result.pass_at_k[1] == pytest.approx(1.0)  # first attempt = 1.0 → pass
        assert result.pass_at_k[2] == pytest.approx(1.0)  # any of first 2 → pass

    async def test_multiple_cases(self) -> None:
        # Scorer: always 0.8 → threshold 0.7 → all pass
        ev = Evaluator(
            scorers=[_ConstantScorer(score=0.8)],
            criteria=[EvalCriteria(metric_name="accuracy", threshold=0.7)],
            repeat_times=2,
        )
        dataset = [
            {"id": "c1", "input": "a"},
            {"id": "c2", "input": "b"},
        ]
        result = await ev.evaluate(_EchoTarget(), dataset)
        assert len(result.case_results) == 4  # 2 cases x 2 repeats
        assert result.pass_at_k[1] == pytest.approx(1.0)
        assert result.pass_at_k[2] == pytest.approx(1.0)

    async def test_no_criteria_no_pass_at_k(self) -> None:
        ev = Evaluator(
            scorers=[_ConstantScorer(score=0.9)],
            repeat_times=3,
        )
        dataset = [{"id": "c1", "input": "x"}]
        result = await ev.evaluate(_EchoTarget(), dataset)
        assert result.pass_at_k == {}


# ---------------------------------------------------------------------------
# Evaluator repr
# ---------------------------------------------------------------------------


class TestEvaluatorRepr:
    def test_repr(self) -> None:
        ev = Evaluator(
            scorers=[_ConstantScorer(), _ConstantScorer()],
            parallel=8,
            repeat_times=3,
        )
        assert "scorers=2" in repr(ev)
        assert "parallel=8" in repr(ev)
        assert "repeat_times=3" in repr(ev)
