"""Evaluation framework: targets, scorers, criteria, and evaluator."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from exo.types import ExoError

# Sentinel value stored in EvalCaseResult.output when a case failed entirely.
_CASE_FAILED_SENTINEL = "<case-failed>"

logger = logging.getLogger(__name__)


class EvalError(ExoError):
    """Raised when an evaluation fails."""


class EvalStatus(StrEnum):
    """Outcome status for a single metric evaluation."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_EVALUATED = "not_evaluated"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScorerResult:
    """Output from a single scorer applied to one case."""

    scorer_name: str
    score: float
    status: EvalStatus = EvalStatus.NOT_EVALUATED
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvalCaseResult:
    """Result for one evaluation case (one input/output pair)."""

    case_id: str
    input: Any
    output: Any
    scores: dict[str, ScorerResult] = field(default_factory=dict)


@dataclass(slots=True)
class EvalResult:
    """Aggregated result across all cases."""

    case_results: list[EvalCaseResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    pass_at_k: dict[int, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# EvalCriteria
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvalCriteria:
    """Threshold-based pass/fail criteria for a metric."""

    metric_name: str
    threshold: float = 0.5

    def judge(self, value: float) -> EvalStatus:
        """Return PASSED if *value* meets or exceeds *threshold*."""
        return EvalStatus.PASSED if value >= self.threshold else EvalStatus.FAILED


# ---------------------------------------------------------------------------
# ABCs
# ---------------------------------------------------------------------------


class EvalTarget(ABC):
    """Callable evaluation subject — wraps the system under test."""

    @abstractmethod
    async def predict(self, case_id: str, input: Any) -> Any:
        """Run the system under test and return its output."""


class Scorer(ABC):
    """Abstract scorer that evaluates one (input, output) pair."""

    @abstractmethod
    async def score(self, case_id: str, input: Any, output: Any) -> ScorerResult:
        """Score a single case and return a ScorerResult."""


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class Evaluator:
    """Runs an EvalTarget over a dataset and scores results.

    Supports parallel execution via semaphore and *repeat_times* for
    pass@k metric computation.
    """

    __slots__ = ("_criteria", "_parallel", "_repeat_times", "_scorers")

    def __init__(
        self,
        scorers: list[Scorer],
        *,
        criteria: list[EvalCriteria] | None = None,
        parallel: int = 4,
        repeat_times: int = 1,
    ) -> None:
        if parallel < 1:
            raise EvalError(
                f"parallel must be >= 1, got {parallel}",
                context={"parallel": parallel},
                hint="Set parallel to at least 1 (e.g. parallel=4).",
            )
        if repeat_times < 1:
            raise EvalError(
                f"repeat_times must be >= 1, got {repeat_times}",
                context={"repeat_times": repeat_times},
                hint="Set repeat_times to at least 1 (e.g. repeat_times=3).",
            )
        self._scorers = list(scorers)
        self._criteria = {c.metric_name: c for c in (criteria or [])}
        self._parallel = parallel
        self._repeat_times = repeat_times

    # ---- public API -------------------------------------------------------

    async def evaluate(
        self,
        target: EvalTarget,
        dataset: list[dict[str, Any]],
    ) -> EvalResult:
        """Run *target* over *dataset*, score, and return aggregated result."""
        sem = asyncio.Semaphore(self._parallel)
        case_results: list[EvalCaseResult] = []

        async def _run(case: dict[str, Any], repeat: int) -> EvalCaseResult:
            # Build a stable, unique run-level ID that always embeds the repeat
            # index.  This lets scorers distinguish repeat runs (e.g. for
            # deterministic toggle behaviour in tests) and decouples pass@k
            # grouping from asyncio.gather call order.
            base_id = str(case.get("id", f"case-{id(case)}"))
            case_id = f"{base_id}-r{repeat}" if self._repeat_times > 1 else base_id
            async with sem:
                inp = case.get("input")
                output = await target.predict(case_id, inp)
                scores = {}
                for scorer in self._scorers:
                    sr = await scorer.score(case_id, inp, output)
                    criterion = self._criteria.get(sr.scorer_name)
                    if criterion is not None:
                        status = criterion.judge(sr.score)
                        sr = ScorerResult(
                            scorer_name=sr.scorer_name,
                            score=sr.score,
                            status=status,
                            details=sr.details,
                        )
                    scores[sr.scorer_name] = sr
                return EvalCaseResult(case_id=case_id, input=inp, output=output, scores=scores)

        # We need the case_id even when _run raises, so we wrap failures here.
        async def _run_safe(case: dict[str, Any], repeat: int) -> EvalCaseResult:
            base_id = str(case.get("id", f"case-{id(case)}"))
            case_id = f"{base_id}-r{repeat}" if self._repeat_times > 1 else base_id
            try:
                return await _run(case, repeat)
            except Exception as exc:
                logger.warning(
                    "Eval case failed: case_id=%r exc_type=%s: %s",
                    case_id,
                    type(exc).__name__,
                    exc,
                    exc_info=True,
                )
                # Record a failed sentinel so the caller sees N-in / N-out.
                return EvalCaseResult(
                    case_id=case_id,
                    input=case.get("input"),
                    output=_CASE_FAILED_SENTINEL,
                    scores={
                        "__error__": ScorerResult(
                            scorer_name="__error__",
                            score=0.0,
                            status=EvalStatus.FAILED,
                            details={
                                "error": str(exc),
                                "exc_type": type(exc).__name__,
                            },
                        )
                    },
                )

        tasks = [_run_safe(case, r) for case in dataset for r in range(self._repeat_times)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, BaseException):
                # Should not happen — _run_safe catches all Exception; only
                # BaseException subclasses that bypass except Exception land here
                # (e.g. KeyboardInterrupt, SystemExit — never swallow those).
                logger.error("Eval task raised a non-Exception: %s", result, exc_info=result)
                raise result
            case_results.append(result)

        summary = self._summarize(case_results)
        pass_at_k = self._compute_pass_at_k(case_results, dataset)
        return EvalResult(case_results=case_results, summary=summary, pass_at_k=pass_at_k)

    # ---- internal ---------------------------------------------------------

    def _summarize(self, results: list[EvalCaseResult]) -> dict[str, Any]:
        """Compute mean score per scorer across all cases.

        Sentinel keys prefixed with ``"__"`` (e.g. ``"__error__"`` injected
        when a case fails entirely) are excluded so they do not pollute the
        mean-score summary.
        """
        totals: dict[str, list[float]] = defaultdict(list)
        for cr in results:
            for name, sr in cr.scores.items():
                if name.startswith("__"):
                    continue
                totals[name].append(sr.score)
        return {name: sum(vals) / len(vals) for name, vals in totals.items()}

    def _compute_pass_at_k(
        self,
        results: list[EvalCaseResult],
        dataset: list[dict[str, Any]],
    ) -> dict[int, float]:
        """Compute pass@k for k=1..repeat_times."""
        if self._repeat_times <= 1 or not self._criteria:
            return {}

        # Group results by stable base case_id.  Since _run always embeds the
        # repeat index as ``{base_id}-r{n}`` when repeat_times > 1, we strip the
        # suffix to recover the base id for grouping.
        case_ids = [str(c.get("id", f"case-{id(c)}")) for c in dataset]
        groups: dict[str, list[EvalCaseResult]] = defaultdict(list)
        for cr in results:
            base_id = cr.case_id
            if self._repeat_times > 1 and "-r" in base_id:
                base_id = base_id.rsplit("-r", 1)[0]
            groups[base_id].append(cr)

        n_cases = len(case_ids)
        if n_cases == 0:
            return {}

        pass_at: dict[int, float] = {}
        for k in range(1, self._repeat_times + 1):
            passed = 0
            for cid in case_ids:
                group = groups.get(cid, [])
                first_k = group[:k]
                if any(
                    sr.status == EvalStatus.PASSED for cr in first_k for sr in cr.scores.values()
                ):
                    passed += 1
            pass_at[k] = passed / n_cases
        return pass_at

    def __repr__(self) -> str:
        s = len(self._scorers)
        return (
            f"Evaluator(scorers={s}, parallel={self._parallel}, repeat_times={self._repeat_times})"
        )
