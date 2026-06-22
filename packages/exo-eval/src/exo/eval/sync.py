"""Vendor-neutral eval platform sync bridge.

Pushes evaluation scores to external tracing/eval platforms (Langfuse, Braintrust,
LangSmith) and pulls datasets from them, without hard-importing any vendor SDK.

All platform clients are duck-typed:

- *Score client* (Langfuse-style): exposes ``.score(trace_id, name, value, **kw)``
- *Score client* (Braintrust-style): exposes ``.log_score(trace_id=..., name=..., score=...)``
- *Dataset client* (LangSmith-style): exposes ``.list_examples(dataset_name) -> list[dict]``

A *backend* object may optionally expose ``score_client()`` and/or
``dataset_client()`` factory methods; PlatformSync calls them in ``__init__``
and caches the results.  Callers may also pass the clients directly via keyword
arguments, bypassing the backend entirely.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from exo.eval.base import (  # pyright: ignore[reportMissingImports]
    EvalCaseResult,
    EvalResult,
    Evaluator,
    Scorer,
)
from exo.types import ExoError  # pyright: ignore[reportMissingImports]

if TYPE_CHECKING:
    from exo.eval.trajectory import TrajectoryDataset  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)

__all__ = ["EvalSyncError", "PlatformSync"]


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class EvalSyncError(ExoError):
    """Raised when a required platform client is unavailable or a sync fails."""


# ---------------------------------------------------------------------------
# PlatformSync
# ---------------------------------------------------------------------------


class PlatformSync:
    """Vendor-neutral bridge between Exo evaluations and external eval platforms.

    Args:
        backend: An object that may expose ``score_client()`` and/or
            ``dataset_client()`` factory methods.  Pass ``None`` to rely
            entirely on the explicit ``score_client`` / ``dataset_client``
            keyword arguments.
        score_client: Override the score client directly (takes precedence
            over ``backend.score_client()``).
        dataset_client: Override the dataset client directly (takes precedence
            over ``backend.dataset_client()``).
    """

    def __init__(
        self,
        backend: Any = None,
        *,
        score_client: Any = None,
        dataset_client: Any = None,
    ) -> None:
        # Resolve score client: explicit kwarg > backend factory > None
        if score_client is not None:
            self._score_client: Any = score_client
        elif (
            backend is not None
            and hasattr(backend, "score_client")
            and callable(backend.score_client)
        ):
            self._score_client = backend.score_client()
        else:
            self._score_client = None

        # Resolve dataset client: explicit kwarg > backend factory > None
        if dataset_client is not None:
            self._dataset_client: Any = dataset_client
        elif (
            backend is not None
            and hasattr(backend, "dataset_client")
            and callable(backend.dataset_client)
        ):
            self._dataset_client = backend.dataset_client()
        else:
            self._dataset_client = None

    # ------------------------------------------------------------------
    # Dataset pull
    # ------------------------------------------------------------------

    def pull_dataset(self, name: str) -> list[dict[str, Any]]:
        """Pull a dataset from the platform and normalise it for :class:`Evaluator`.

        Each returned dict has the keys ``input``, ``expected``, and ``case_id``
        (the shape consumed by :meth:`Evaluator.evaluate`).

        Args:
            name: The dataset name as registered on the platform.

        Returns:
            List of normalised example dicts.

        Raises:
            EvalSyncError: If no dataset client is available.
        """
        if self._dataset_client is None:
            raise EvalSyncError(
                "No dataset client is configured.",
                context={"dataset": name},
                hint=(
                    "Pass a backend that exposes dataset_client() or supply "
                    "dataset_client= directly to PlatformSync."
                ),
            )

        raw_examples: list[dict[str, Any]] = list(self._dataset_client.list_examples(name))
        return [self._normalise_example(ex) for ex in raw_examples]

    @staticmethod
    def _normalise_example(ex: dict[str, Any]) -> dict[str, Any]:
        """Coerce a raw platform example into the Evaluator-consumable shape."""
        # LangSmith uses "inputs"/"outputs"; Langfuse/Braintrust commonly use
        # "input"/"output" or "expected_output".  We try all known aliases.
        input_val = ex.get("input") or ex.get("inputs") or ex.get("user_input") or ""
        expected_val = (
            ex.get("expected") or ex.get("output") or ex.get("outputs") or ex.get("expected_output")
        )
        case_id = str(
            ex.get("case_id") or ex.get("id") or ex.get("example_id") or ex.get("run_id") or ""
        )
        return {"input": input_val, "expected": expected_val, "case_id": case_id, **ex}

    # ------------------------------------------------------------------
    # Score push
    # ------------------------------------------------------------------

    async def push_scores(
        self,
        result: EvalResult,
        *,
        trace_id_of: Callable[[EvalCaseResult], str] | None = None,
    ) -> int:
        """Push all :class:`ScorerResult` scores from *result* to the platform.

        Supports both Langfuse-style (``.score(trace_id, name, value)``) and
        Braintrust-style (``.log_score(trace_id=..., name=..., score=...)``)
        clients — detected via :func:`hasattr`.

        Args:
            result: The :class:`EvalResult` whose scores to push.
            trace_id_of: Optional callable mapping an :class:`EvalCaseResult` to
                the external trace ID.  Defaults to ``case_result.case_id``.

        Returns:
            The total number of scores pushed.

        Raises:
            EvalSyncError: If no score client is configured.
        """
        if self._score_client is None:
            raise EvalSyncError(
                "No score client is configured.",
                hint=(
                    "Pass a backend that exposes score_client() or supply "
                    "score_client= directly to PlatformSync."
                ),
            )

        _trace_id = trace_id_of if trace_id_of is not None else lambda cr: cr.case_id

        # Collect all (trace_id, name, value) triples to push.
        pushes: list[tuple[str, str, float]] = []
        for case_result in result.case_results:
            trace_id = _trace_id(case_result)
            for scorer_name, sr in case_result.scores.items():
                pushes.append((trace_id, scorer_name, sr.score))

        # _push_one_score calls a synchronous vendor client; offload each call to
        # the default thread-pool executor so the event loop stays unblocked and
        # all scores are pushed concurrently rather than serially.
        loop = asyncio.get_event_loop()
        await asyncio.gather(
            *(
                loop.run_in_executor(None, self._push_one_score, tid, name, val)
                for tid, name, val in pushes
            )
        )

        pushed = len(pushes)
        logger.debug("Pushed %d scores to platform.", pushed)
        return pushed

    def _push_one_score(self, trace_id: str, name: str, value: float) -> None:
        """Dispatch a single score to whichever client style is present."""
        client = self._score_client
        if hasattr(client, "score") and callable(client.score):
            # Langfuse-style: score(trace_id, name, value, comment=None)
            client.score(trace_id, name, value)
        elif hasattr(client, "log_score") and callable(client.log_score):
            # Braintrust-style: log_score(trace_id=..., name=..., score=...)
            client.log_score(trace_id=trace_id, name=name, score=value)
        else:
            raise EvalSyncError(
                f"Score client {type(client)!r} exposes neither .score() nor .log_score().",
                context={"client_type": type(client).__name__},
                hint=(
                    "Wrap your client in a thin adapter that implements either "
                    ".score(trace_id, name, value) (Langfuse-style) or "
                    ".log_score(trace_id=..., name=..., score=...) (Braintrust-style)."
                ),
            )

    # ------------------------------------------------------------------
    # Evaluate + sync in one shot
    # ------------------------------------------------------------------

    async def evaluate_and_sync(
        self,
        target: Any,
        dataset_name: str,
        scorers: list[Scorer],
        *,
        evaluator: Evaluator | None = None,
        trace_id_of: Callable[[EvalCaseResult], str] | None = None,
    ) -> EvalResult:
        """Pull a dataset, run the evaluator, push scores, and return the result.

        Args:
            target: An :class:`~exo.eval.base.EvalTarget` subclass instance.
            dataset_name: Name of the dataset to pull from the platform.
            scorers: List of :class:`~exo.eval.base.Scorer` instances to run.
            evaluator: Optional pre-built :class:`Evaluator`.  When omitted a
                default one is constructed from *scorers*.
            trace_id_of: Forwarded to :meth:`push_scores`.

        Returns:
            The :class:`EvalResult` produced by the evaluator.
        """
        dataset = self.pull_dataset(dataset_name)
        ev = evaluator if evaluator is not None else Evaluator(scorers=scorers)
        eval_result = await ev.evaluate(target, dataset)
        await self.push_scores(eval_result, trace_id_of=trace_id_of)
        return eval_result

    # ------------------------------------------------------------------
    # Trajectory export
    # ------------------------------------------------------------------

    @staticmethod
    def export_trajectory(
        dataset: TrajectoryDataset,
        fmt: str = "langfuse",
    ) -> str:
        """Serialise a :class:`~exo.eval.trajectory.TrajectoryDataset` to JSON.

        The output is a trace-friendly envelope understood by Langfuse and
        LangSmith ingestion endpoints::

            {
                "format": "<fmt>",
                "traces": [<trajectory item dicts …>]
            }

        Args:
            dataset: The trajectory dataset to export.
            fmt: Target platform format tag.  Supported values:
                ``"langfuse"``, ``"langsmith"``.

        Returns:
            A JSON string.

        Raises:
            EvalSyncError: If *fmt* is not a recognised value.
        """
        supported = {"langfuse", "langsmith"}
        if fmt not in supported:
            raise EvalSyncError(
                f"Unsupported trajectory export format {fmt!r}.",
                context={"fmt": fmt},
                hint=f"Use one of the supported formats: {sorted(supported)}.",
            )

        raw = json.loads(dataset.to_json())
        payload: dict[str, Any] = {"format": fmt, "traces": raw}
        return json.dumps(payload, indent=2)
