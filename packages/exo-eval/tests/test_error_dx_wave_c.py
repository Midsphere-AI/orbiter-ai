"""Wave-C error-DX tests for exo-eval.

Covers the concrete fixes introduced by the Wave-C hardening pass:
- Failed cases in Evaluator.evaluate are recorded, not silently dropped (N-in → N-out)
- EvalSyncError inherits ExoError (carries hint/context fields)
- Scorer failures in RefinementLoop._analyze are recorded as 0.0 with metadata
- get_scorer raises actionable EvalError instead of bare KeyError
- LLMAsJudgeScorer.score and GeneralReflector.analyze respect an optional timeout=
- Config validation raises EvalError (not ValueError) with context and hint
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from exo.eval.base import (  # pyright: ignore[reportMissingImports]
    _CASE_FAILED_SENTINEL,
    EvalError,
    EvalStatus,
    EvalTarget,
    Evaluator,
    Scorer,
    ScorerResult,
)
from exo.eval.llm_scorer import LLMAsJudgeScorer  # pyright: ignore[reportMissingImports]
from exo.eval.reflection import GeneralReflector  # pyright: ignore[reportMissingImports]
from exo.eval.sync import EvalSyncError  # pyright: ignore[reportMissingImports]
from exo.eval.trajectory_scorers import (  # pyright: ignore[reportMissingImports]
    get_scorer,
    list_scorers,
)
from exo.types import ExoError  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _EchoTarget(EvalTarget):
    async def predict(self, case_id: str, input: Any) -> Any:
        return input


class _BombTarget(EvalTarget):
    """Raises on every case."""

    async def predict(self, case_id: str, input: Any) -> Any:
        raise RuntimeError(f"boom for {case_id}")


class _SelectiveBombTarget(EvalTarget):
    """Raises on cases whose id is in *fail_ids*."""

    def __init__(self, fail_ids: set[str]) -> None:
        self._fail_ids = fail_ids

    async def predict(self, case_id: str, input: Any) -> Any:
        if case_id in self._fail_ids:
            raise RuntimeError(f"intentional failure for {case_id}")
        return input


class _ConstantScorer(Scorer):
    def __init__(self, score: float = 1.0) -> None:
        self._score = score

    async def score(self, case_id: str, input: Any, output: Any) -> ScorerResult:
        return ScorerResult(scorer_name="const", score=self._score)


# ---------------------------------------------------------------------------
# P0: Evaluator.evaluate — N-in, N-out with failed-case sentinel
# ---------------------------------------------------------------------------


class TestEvaluatorFailedCaseRecording:
    async def test_all_fail_still_produces_n_results(self) -> None:
        """All cases fail → all still appear as failed EvalCaseResult entries."""
        ev = Evaluator(scorers=[_ConstantScorer()])
        dataset = [{"id": f"c{i}", "input": i} for i in range(5)]
        result = await ev.evaluate(_BombTarget(), dataset)

        # N-in → N-out
        assert len(result.case_results) == 5

        for cr in result.case_results:
            # Sentinel output flags the failure
            assert cr.output == _CASE_FAILED_SENTINEL
            # A special __error__ scorer entry is present
            assert "__error__" in cr.scores
            sr = cr.scores["__error__"]
            assert sr.score == 0.0
            assert sr.status == EvalStatus.FAILED
            assert "error" in sr.details
            assert sr.details["exc_type"] == "RuntimeError"

    async def test_partial_failure_preserves_successes(self) -> None:
        """Some cases fail, some succeed — both groups appear in results."""
        fail_ids = {"c1", "c3"}
        ev = Evaluator(scorers=[_ConstantScorer(score=0.9)])
        dataset = [{"id": f"c{i}", "input": i} for i in range(5)]
        result = await ev.evaluate(_SelectiveBombTarget(fail_ids), dataset)

        assert len(result.case_results) == 5

        by_id = {cr.case_id: cr for cr in result.case_results}
        # Successful cases carry real scores
        for good_id in {"c0", "c2", "c4"}:
            cr = by_id[good_id]
            assert cr.output != _CASE_FAILED_SENTINEL
            assert "const" in cr.scores

        # Failed cases carry the sentinel
        for bad_id in fail_ids:
            cr = by_id[bad_id]
            assert cr.output == _CASE_FAILED_SENTINEL
            assert "__error__" in cr.scores

    async def test_case_id_is_captured_for_failed_case(self) -> None:
        """The failed EvalCaseResult carries the correct case_id."""
        ev = Evaluator(scorers=[_ConstantScorer()])
        dataset = [{"id": "special-case", "input": "x"}]
        result = await ev.evaluate(_BombTarget(), dataset)

        assert result.case_results[0].case_id == "special-case"

    async def test_zero_failures_unchanged(self) -> None:
        """Normal runs produce zero failed sentinels."""
        ev = Evaluator(scorers=[_ConstantScorer()])
        dataset = [{"id": "c1", "input": "hello"}]
        result = await ev.evaluate(_EchoTarget(), dataset)

        assert len(result.case_results) == 1
        assert result.case_results[0].output == "hello"
        assert "__error__" not in result.case_results[0].scores


# ---------------------------------------------------------------------------
# P1: EvalSyncError taxonomy — must be an ExoError subclass
# ---------------------------------------------------------------------------


class TestEvalSyncErrorTaxonomy:
    def test_inherits_exo_error(self) -> None:
        err = EvalSyncError("something broke")
        assert isinstance(err, ExoError)
        assert isinstance(err, Exception)

    def test_carries_hint(self) -> None:
        err = EvalSyncError("no client", hint="Pass score_client= to PlatformSync.")
        assert err.hint == "Pass score_client= to PlatformSync."

    def test_carries_context(self) -> None:
        err = EvalSyncError("no client", context={"dataset": "my-ds"})
        assert err.context["dataset"] == "my-ds"

    def test_str_includes_message(self) -> None:
        err = EvalSyncError("broken link")
        assert "broken link" in str(err)

    def test_pull_dataset_raises_with_hint(self) -> None:
        """pull_dataset raises EvalSyncError with a helpful hint."""
        from exo.eval.sync import PlatformSync  # pyright: ignore[reportMissingImports]

        ps = PlatformSync()
        with pytest.raises(EvalSyncError) as exc_info:
            ps.pull_dataset("my-dataset")

        err = exc_info.value
        assert err.hint is not None
        assert "dataset_client" in err.hint

    async def test_push_scores_raises_with_hint(self) -> None:
        """push_scores raises EvalSyncError with a helpful hint."""
        from exo.eval.base import EvalResult  # pyright: ignore[reportMissingImports]
        from exo.eval.sync import PlatformSync  # pyright: ignore[reportMissingImports]

        ps = PlatformSync()
        with pytest.raises(EvalSyncError) as exc_info:
            await ps.push_scores(EvalResult())

        err = exc_info.value
        assert err.hint is not None
        assert "score_client" in err.hint


# ---------------------------------------------------------------------------
# P0: scorer failures in _analyze are recorded (not silently dropped)
# ---------------------------------------------------------------------------


class TestRefinementLoopScorerFailureRecorded:
    async def test_failing_scorer_recorded_in_state(self) -> None:
        """A scorer that raises must appear in state.metadata['scorer_failures']."""
        from exo.eval.ralph.config import (  # pyright: ignore[reportMissingImports]
            RalphConfig,
            StopConditionConfig,
        )
        from exo.eval.ralph.runner import RefinementLoop  # pyright: ignore[reportMissingImports]

        class _BoomScorer(Scorer):
            async def score(self, case_id: str, input: Any, output: Any) -> ScorerResult:
                raise ValueError("scorer exploded")

        async def execute(inp: str) -> str:
            return "result"

        cfg = RalphConfig(stop_condition=StopConditionConfig(max_iterations=1))
        loop = RefinementLoop(execute, [_BoomScorer()], config=cfg)
        result = await loop.run("input")

        failures = result.state.get("metadata", {}).get("scorer_failures", [])
        assert len(failures) == 1
        assert failures[0]["exc_type"] == "ValueError"
        assert "scorer exploded" in failures[0]["error"]

    async def test_failing_scorer_zero_in_scores_dict(self) -> None:
        """Failing scorer is recorded as 0.0 score so threshold checks see it."""
        from exo.eval.ralph.config import (  # pyright: ignore[reportMissingImports]
            RalphConfig,
            StopConditionConfig,
        )
        from exo.eval.ralph.runner import RefinementLoop  # pyright: ignore[reportMissingImports]

        class _BoomScorer(Scorer):
            async def score(self, case_id: str, input: Any, output: Any) -> ScorerResult:
                raise RuntimeError("explode")

        async def execute(inp: str) -> str:
            return "ok"

        cfg = RalphConfig(stop_condition=StopConditionConfig(max_iterations=1))
        loop = RefinementLoop(execute, [_BoomScorer()], config=cfg)
        result = await loop.run("input")

        # The scorer's name appears in scores with 0.0 — NOT missing entirely.
        assert "_BoomScorer" in result.scores
        assert result.scores["_BoomScorer"] == 0.0


# ---------------------------------------------------------------------------
# P2: get_scorer — actionable EvalError instead of KeyError
# ---------------------------------------------------------------------------


class TestGetScorerActionableError:
    def test_missing_scorer_raises_eval_error(self) -> None:
        with pytest.raises(EvalError, match="not registered"):
            get_scorer("completely_unknown_scorer_xyz")

    def test_error_lists_available_scorers(self) -> None:
        try:
            get_scorer("completely_unknown_scorer_xyz")
        except EvalError as err:
            # hint mentions available scorers
            assert err.hint is not None
            for name in list_scorers():
                assert name in err.hint
            assert "scorer_register" in err.hint
        else:
            pytest.fail("Expected EvalError was not raised")

    def test_error_carries_context(self) -> None:
        try:
            get_scorer("missing_scorer_name")
        except EvalError as err:
            assert err.context.get("name") == "missing_scorer_name"
        else:
            pytest.fail("Expected EvalError was not raised")


# ---------------------------------------------------------------------------
# P1 (resilience): LLMAsJudgeScorer — optional timeout
# ---------------------------------------------------------------------------


class TestLLMAsJudgeScorerTimeout:
    async def test_timeout_triggers_on_slow_judge(self) -> None:
        """A judge that hangs longer than timeout= raises TimeoutError."""

        async def slow_judge(prompt: str) -> str:
            await asyncio.sleep(10)
            return '{"score": 1.0}'

        scorer = LLMAsJudgeScorer(judge=slow_judge, timeout=0.05)
        with pytest.raises(asyncio.TimeoutError):
            await scorer.score("c1", "input", "output")

    async def test_no_timeout_is_default(self) -> None:
        """With timeout=0.0 (default), a fast judge completes normally."""

        async def fast_judge(prompt: str) -> str:
            return '{"score": 0.75}'

        scorer = LLMAsJudgeScorer(judge=fast_judge, timeout=0.0)
        result = await scorer.score("c1", "input", "output")
        assert result.score == 0.75

    async def test_fast_judge_within_timeout(self) -> None:
        """A judge that responds before the deadline completes normally."""

        async def fast_judge(prompt: str) -> str:
            return '{"score": 0.9}'

        scorer = LLMAsJudgeScorer(judge=fast_judge, timeout=5.0)
        result = await scorer.score("c1", "input", "output")
        assert result.score == 0.9


# ---------------------------------------------------------------------------
# P1 (resilience): GeneralReflector — optional timeout
# ---------------------------------------------------------------------------


class TestGeneralReflectorTimeout:
    async def test_timeout_triggers_on_slow_judge(self) -> None:
        """A judge that hangs longer than timeout= raises TimeoutError."""

        async def slow_judge(prompt: str) -> str:
            await asyncio.sleep(10)
            return "{}"

        reflector = GeneralReflector(judge=slow_judge, timeout=0.05)
        with pytest.raises(asyncio.TimeoutError):
            await reflector.analyze({"input": "hi", "output": "there"})

    async def test_fast_judge_within_timeout(self) -> None:
        """A judge that responds before the deadline completes normally."""

        async def fast_judge(prompt: str) -> str:
            return '{"summary": "ok", "key_findings": [], "root_causes": [], "insights": [], "suggestions": []}'

        reflector = GeneralReflector(judge=fast_judge, timeout=5.0)
        result = await reflector.analyze({"input": "hi", "output": "there"})
        assert result["summary"] == "ok"


# ---------------------------------------------------------------------------
# P1: Config validation raises EvalError (not ValueError) with context/hint
# ---------------------------------------------------------------------------


class TestConfigValidationEvalError:
    def test_evaluator_parallel_zero_carries_context(self) -> None:
        with pytest.raises(EvalError) as exc_info:
            Evaluator(scorers=[], parallel=0)
        err = exc_info.value
        assert err.context.get("parallel") == 0
        assert err.hint is not None

    def test_evaluator_repeat_times_zero_carries_context(self) -> None:
        with pytest.raises(EvalError) as exc_info:
            Evaluator(scorers=[], repeat_times=0)
        err = exc_info.value
        assert err.context.get("repeat_times") == 0
        assert err.hint is not None

    def test_validation_config_bad_threshold_carries_context(self) -> None:
        from exo.eval.ralph.config import ValidationConfig  # pyright: ignore[reportMissingImports]

        with pytest.raises(EvalError) as exc_info:
            ValidationConfig(min_score_threshold=-0.5)
        err = exc_info.value
        assert err.context.get("min_score_threshold") == -0.5
        assert err.hint is not None

    def test_stop_condition_config_bad_iterations_carries_context(self) -> None:
        from exo.eval.ralph.config import (
            StopConditionConfig,  # pyright: ignore[reportMissingImports]
        )

        with pytest.raises(EvalError) as exc_info:
            StopConditionConfig(max_iterations=0)
        err = exc_info.value
        assert err.context.get("max_iterations") == 0
        assert err.hint is not None
