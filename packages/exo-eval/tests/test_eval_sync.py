"""Tests for the eval platform sync bridge (sync.py).

All tests are unit-level: no real network calls, no real LLM calls.
asyncio_mode=auto is project-wide so @pytest.mark.asyncio is NOT needed.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

import pytest

from exo.eval.base import (
    EvalCaseResult,
    EvalResult,
    EvalStatus,
    EvalTarget,
    Scorer,
    ScorerResult,
)
from exo.eval.sync import EvalSyncError, PlatformSync
from exo.eval.trajectory import TrajectoryDataset, TrajectoryItem

# ---------------------------------------------------------------------------
# Fake platform clients
# ---------------------------------------------------------------------------


class FakeScoreClient:
    """Records .score(trace_id, name, value) calls (Langfuse-style)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float]] = []

    def score(self, trace_id: str, name: str, value: float) -> None:
        self.calls.append((trace_id, name, value))


class FakeBraintrustScoreClient:
    """Records .log_score(trace_id=..., name=..., score=...) calls (Braintrust-style)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def log_score(self, *, trace_id: str, name: str, score: float) -> None:
        self.calls.append({"trace_id": trace_id, "name": name, "score": score})


class FakeDatasetClient:
    """Returns canned examples for .list_examples(dataset_name)."""

    CANNED: ClassVar[list[dict[str, Any]]] = [
        {"id": "ex-1", "input": "hello", "expected": "world"},
        {"id": "ex-2", "input": "foo", "expected": "bar"},
        {"inputs": "baz", "outputs": "qux", "example_id": "ex-3"},
    ]

    def list_examples(self, dataset_name: str) -> list[dict[str, Any]]:
        return list(self.CANNED)


class FakeBackend:
    """A backend that exposes both client factories."""

    def __init__(self, score_client: Any, dataset_client: Any) -> None:
        self._sc = score_client
        self._dc = dataset_client

    def score_client(self) -> Any:
        return self._sc

    def dataset_client(self) -> Any:
        return self._dc


class ScoreOnlyBackend:
    """Backend that only exposes score_client()."""

    def __init__(self, score_client: Any) -> None:
        self._sc = score_client

    def score_client(self) -> Any:
        return self._sc


class DatasetOnlyBackend:
    """Backend that only exposes dataset_client()."""

    def __init__(self, dataset_client: Any) -> None:
        self._dc = dataset_client

    def dataset_client(self) -> Any:
        return self._dc


# ---------------------------------------------------------------------------
# Trivial EvalTarget and Scorer (no LLM calls)
# ---------------------------------------------------------------------------


class EchoTarget(EvalTarget):
    """Returns the input unchanged."""

    async def predict(self, case_id: str, input: Any) -> Any:
        return input


class ConstantScorer(Scorer):
    """Always returns a fixed score regardless of input/output."""

    def __init__(self, name: str = "constant", value: float = 1.0) -> None:
        self._name = name
        self._value = value

    async def score(self, case_id: str, input: Any, output: Any) -> ScorerResult:
        return ScorerResult(scorer_name=self._name, score=self._value)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_eval_result(
    n_cases: int = 2, scorer_name: str = "accuracy", score: float = 1.0
) -> EvalResult:
    """Build a synthetic EvalResult with *n_cases* case results."""
    case_results = [
        EvalCaseResult(
            case_id=f"case-{i}",
            input=f"in-{i}",
            output=f"out-{i}",
            scores={
                scorer_name: ScorerResult(
                    scorer_name=scorer_name, score=score, status=EvalStatus.PASSED
                )
            },
        )
        for i in range(n_cases)
    ]
    return EvalResult(case_results=case_results, summary={scorer_name: score}, pass_at_k={})


# ---------------------------------------------------------------------------
# pull_dataset
# ---------------------------------------------------------------------------


class TestPullDataset:
    def test_normalises_standard_shape(self) -> None:
        dc = FakeDatasetClient()
        sync = PlatformSync(dataset_client=dc)
        examples = sync.pull_dataset("my-dataset")
        assert len(examples) == 3

    def test_normalise_id_field(self) -> None:
        dc = FakeDatasetClient()
        sync = PlatformSync(dataset_client=dc)
        examples = sync.pull_dataset("any")
        assert examples[0]["case_id"] == "ex-1"
        assert examples[1]["case_id"] == "ex-2"

    def test_normalise_inputs_outputs_aliases(self) -> None:
        dc = FakeDatasetClient()
        sync = PlatformSync(dataset_client=dc)
        examples = sync.pull_dataset("any")
        # Third example uses "inputs"/"outputs" keys
        third = examples[2]
        assert third["input"] == "baz"
        assert third["expected"] == "qux"
        assert third["case_id"] == "ex-3"

    def test_raises_without_dataset_client(self) -> None:
        sync = PlatformSync()
        with pytest.raises(EvalSyncError, match="dataset client"):
            sync.pull_dataset("any")

    def test_backend_factory_used(self) -> None:
        dc = FakeDatasetClient()
        backend = DatasetOnlyBackend(dc)
        sync = PlatformSync(backend)
        examples = sync.pull_dataset("any")
        assert len(examples) == 3

    def test_explicit_client_overrides_backend(self) -> None:
        dc_backend = FakeDatasetClient()
        dc_explicit = FakeDatasetClient()
        dc_explicit.CANNED = [{"id": "override-1", "input": "x", "expected": "y"}]  # type: ignore[attr-defined]
        backend = DatasetOnlyBackend(dc_backend)
        sync = PlatformSync(backend, dataset_client=dc_explicit)
        examples = sync.pull_dataset("any")
        # Should use the explicit client's data (only 1 row)
        assert len(examples) == 1
        assert examples[0]["case_id"] == "override-1"


# ---------------------------------------------------------------------------
# push_scores — Langfuse-style
# ---------------------------------------------------------------------------


class TestPushScoresLangfuse:
    async def test_pushes_correct_count(self) -> None:
        sc = FakeScoreClient()
        sync = PlatformSync(score_client=sc)
        result = _make_eval_result(n_cases=3, scorer_name="exact_match")
        count = await sync.push_scores(result)
        assert count == 3

    async def test_records_calls(self) -> None:
        sc = FakeScoreClient()
        sync = PlatformSync(score_client=sc)
        result = _make_eval_result(n_cases=2, scorer_name="precision", score=0.9)
        await sync.push_scores(result)
        assert len(sc.calls) == 2
        # Each call: (trace_id, name, value)
        names = {c[1] for c in sc.calls}
        assert names == {"precision"}
        values = {c[2] for c in sc.calls}
        assert values == {0.9}

    async def test_default_trace_id_is_case_id(self) -> None:
        sc = FakeScoreClient()
        sync = PlatformSync(score_client=sc)
        result = _make_eval_result(n_cases=2)
        await sync.push_scores(result)
        trace_ids = {c[0] for c in sc.calls}
        assert trace_ids == {"case-0", "case-1"}

    async def test_custom_trace_id_of(self) -> None:
        sc = FakeScoreClient()
        sync = PlatformSync(score_client=sc)
        result = _make_eval_result(n_cases=2)
        await sync.push_scores(result, trace_id_of=lambda cr: f"trace-{cr.case_id}")
        trace_ids = {c[0] for c in sc.calls}
        assert trace_ids == {"trace-case-0", "trace-case-1"}

    async def test_raises_without_score_client(self) -> None:
        sync = PlatformSync()
        result = _make_eval_result()
        with pytest.raises(EvalSyncError, match="score client"):
            await sync.push_scores(result)


# ---------------------------------------------------------------------------
# push_scores — Braintrust-style
# ---------------------------------------------------------------------------


class TestPushScoresBraintrust:
    async def test_pushes_via_log_score(self) -> None:
        sc = FakeBraintrustScoreClient()
        sync = PlatformSync(score_client=sc)
        result = _make_eval_result(n_cases=2, scorer_name="f1", score=0.75)
        count = await sync.push_scores(result)
        assert count == 2
        assert len(sc.calls) == 2
        assert all(c["name"] == "f1" for c in sc.calls)
        assert all(c["score"] == 0.75 for c in sc.calls)
        assert {c["trace_id"] for c in sc.calls} == {"case-0", "case-1"}

    async def test_braintrust_preferred_over_score_if_only_log_score(self) -> None:
        """A client with only log_score must not fail."""
        sc = FakeBraintrustScoreClient()
        sync = PlatformSync(score_client=sc)
        result = _make_eval_result()
        await sync.push_scores(result)  # should not raise

    async def test_unknown_client_raises(self) -> None:
        class BrokenClient:
            pass

        sync = PlatformSync(score_client=BrokenClient())
        result = _make_eval_result()
        with pytest.raises(EvalSyncError, match=r"score.*nor"):
            await sync.push_scores(result)


# ---------------------------------------------------------------------------
# push_scores — multiple scorers per case
# ---------------------------------------------------------------------------


class TestPushScoresMultiScorer:
    async def test_pushes_one_per_scorer_result(self) -> None:
        sc = FakeScoreClient()
        sync = PlatformSync(score_client=sc)
        # Build a result with 2 scorers per case, 2 cases = 4 total
        case_results = [
            EvalCaseResult(
                case_id=f"c{i}",
                input="x",
                output="y",
                scores={
                    "scorer_a": ScorerResult(scorer_name="scorer_a", score=0.8),
                    "scorer_b": ScorerResult(scorer_name="scorer_b", score=0.6),
                },
            )
            for i in range(2)
        ]
        result = EvalResult(case_results=case_results)
        count = await sync.push_scores(result)
        assert count == 4
        assert len(sc.calls) == 4


# ---------------------------------------------------------------------------
# evaluate_and_sync (end-to-end)
# ---------------------------------------------------------------------------


class TestEvaluateAndSync:
    async def test_returns_eval_result(self) -> None:
        sc = FakeScoreClient()
        dc = FakeDatasetClient()
        sync = PlatformSync(score_client=sc, dataset_client=dc)
        result = await sync.evaluate_and_sync(
            EchoTarget(),
            "my-dataset",
            [ConstantScorer("accuracy", 1.0)],
        )
        assert isinstance(result, EvalResult)
        assert len(result.case_results) == len(FakeDatasetClient.CANNED)

    async def test_scores_pushed_after_evaluate(self) -> None:
        sc = FakeScoreClient()
        dc = FakeDatasetClient()
        sync = PlatformSync(score_client=sc, dataset_client=dc)
        await sync.evaluate_and_sync(
            EchoTarget(),
            "any",
            [ConstantScorer("accuracy", 1.0)],
        )
        # 3 cases * 1 scorer = 3 pushes
        assert len(sc.calls) == 3

    async def test_custom_evaluator_respected(self) -> None:
        from exo.eval.base import Evaluator

        sc = FakeScoreClient()
        dc = FakeDatasetClient()
        sync = PlatformSync(score_client=sc, dataset_client=dc)
        ev = Evaluator(scorers=[ConstantScorer("precision", 0.5)], parallel=2)
        result = await sync.evaluate_and_sync(EchoTarget(), "any", scorers=[], evaluator=ev)
        assert all("precision" in cr.scores for cr in result.case_results)

    async def test_raises_without_dataset_client(self) -> None:
        sc = FakeScoreClient()
        sync = PlatformSync(score_client=sc)
        with pytest.raises(EvalSyncError, match="dataset client"):
            await sync.evaluate_and_sync(EchoTarget(), "any", [ConstantScorer()])

    async def test_raises_without_score_client(self) -> None:
        dc = FakeDatasetClient()
        sync = PlatformSync(dataset_client=dc)
        with pytest.raises(EvalSyncError, match="score client"):
            await sync.evaluate_and_sync(EchoTarget(), "any", [ConstantScorer()])

    async def test_full_backend_factory(self) -> None:
        sc = FakeScoreClient()
        dc = FakeDatasetClient()
        backend = FakeBackend(sc, dc)
        sync = PlatformSync(backend)
        result = await sync.evaluate_and_sync(EchoTarget(), "any", [ConstantScorer("recall", 0.9)])
        assert len(result.case_results) > 0
        assert len(sc.calls) > 0


# ---------------------------------------------------------------------------
# export_trajectory
# ---------------------------------------------------------------------------


class TestExportTrajectory:
    def _make_dataset(self) -> TrajectoryDataset:
        ds = TrajectoryDataset()
        ds.append_trajectory(TrajectoryItem(input="hello", output="world"))
        ds.append_trajectory(TrajectoryItem(input="foo", output="bar"))
        return ds

    def test_returns_valid_json(self) -> None:
        ds = self._make_dataset()
        out = PlatformSync.export_trajectory(ds)
        parsed = json.loads(out)
        assert "format" in parsed
        assert "traces" in parsed

    def test_langfuse_format_tag(self) -> None:
        ds = self._make_dataset()
        parsed = json.loads(PlatformSync.export_trajectory(ds, fmt="langfuse"))
        assert parsed["format"] == "langfuse"

    def test_langsmith_format_tag(self) -> None:
        ds = self._make_dataset()
        parsed = json.loads(PlatformSync.export_trajectory(ds, fmt="langsmith"))
        assert parsed["format"] == "langsmith"

    def test_traces_contain_items(self) -> None:
        ds = self._make_dataset()
        parsed = json.loads(PlatformSync.export_trajectory(ds))
        assert len(parsed["traces"]) == 2
        assert parsed["traces"][0]["input"] == "hello"

    def test_empty_dataset_produces_empty_traces(self) -> None:
        ds = TrajectoryDataset()
        parsed = json.loads(PlatformSync.export_trajectory(ds))
        assert parsed["traces"] == []

    def test_unsupported_fmt_raises(self) -> None:
        ds = self._make_dataset()
        with pytest.raises(EvalSyncError, match="Unsupported trajectory export format"):
            PlatformSync.export_trajectory(ds, fmt="datadog")


# ---------------------------------------------------------------------------
# EvalSyncError
# ---------------------------------------------------------------------------


class TestEvalSyncError:
    def test_is_exception(self) -> None:
        err = EvalSyncError("boom")
        assert isinstance(err, Exception)
        assert str(err) == "boom"

    def test_can_raise_and_catch(self) -> None:
        with pytest.raises(EvalSyncError, match="boom"):
            raise EvalSyncError("boom")
