"""RefinementLoop — iterative refinement loop: Run → Analyze → Learn → Plan → Halt."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from exo.eval.base import (  # pyright: ignore[reportMissingImports]
    EvalError,
    Scorer,
    ScorerResult,
)
from exo.eval.ralph.config import (  # pyright: ignore[reportMissingImports]
    LoopState,
    RefinementConfig,
    StopType,
)
from exo.eval.ralph.detectors import (  # pyright: ignore[reportMissingImports]
    CompositeDetector,
    ConsecutiveFailureDetector,
    CostLimitDetector,
    MaxIterationDetector,
    ScoreThresholdDetector,
    StopDecision,
    TimeoutDetector,
)
from exo.eval.reflection import (  # pyright: ignore[reportMissingImports]
    ReflectionResult,
    Reflector,
)
from exo.types import (  # pyright: ignore[reportMissingImports]
    RefinementIterationEvent,
    RefinementStopEvent,
    StreamEvent,
    TextEvent,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# Async callable: (input: str) -> str
ExecuteFn = Callable[..., Any]
# Async callable: (input: str) -> AsyncIterator[StreamEvent]
StreamExecuteFn = Callable[..., Any]
# Async callable: (prompt: str) -> str  (for re-prompting with reflection)
RePlanFn = Callable[..., Any]

# ---------------------------------------------------------------------------
# Loop result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RefinementResult:
    """Final outcome of a refinement loop execution."""

    output: str
    stop_type: StopType
    reason: str
    iterations: int
    scores: dict[str, float]
    state: dict[str, Any]
    reflections: list[dict[str, Any]] = field(default_factory=list)


# Deprecated alias: use RefinementResult
RalphResult = RefinementResult


# ---------------------------------------------------------------------------
# RefinementLoop
# ---------------------------------------------------------------------------


class RefinementLoop:
    """Implements the 5-phase iterative refinement loop.

    Phases per iteration:
        1. **Run** — execute the agent/task via *execute_fn*
        2. **Analyze** — score the output using configured scorers
        3. **Learn** — reflect on failures to extract actionable insights
        4. **Plan** — re-prompt by appending reflection suggestions to input
        5. **Halt** — check stop conditions; break or continue
    """

    __slots__ = (
        "_config",
        "_detector",
        "_execute_fn",
        "_reflector",
        "_replan_fn",
        "_scorers",
        "_stream_execute_fn",
    )

    def __init__(
        self,
        execute_fn: ExecuteFn,
        scorers: list[Scorer],
        *,
        stream_execute_fn: StreamExecuteFn | None = None,
        config: RefinementConfig | None = None,
        reflector: Reflector | None = None,
        replan_fn: RePlanFn | None = None,
    ) -> None:
        self._execute_fn = execute_fn
        self._stream_execute_fn = stream_execute_fn
        self._scorers = list(scorers)
        self._config = config or RefinementConfig()
        self._reflector = reflector
        self._replan_fn = replan_fn
        self._detector = self._build_detector()

    @classmethod
    def from_agent(cls, agent: Any, scorers: list[Scorer], **kwargs: Any) -> RefinementLoop:
        """Create a RefinementLoop wired to an Agent's run() and stream().

        Convenience factory that creates both ``execute_fn`` (for ``.run()``)
        and ``stream_execute_fn`` (for ``.stream()``) from the same agent.

        Args:
            agent: An ``Agent`` instance.
            scorers: List of scorers for the analyze phase.
            **kwargs: Additional arguments forwarded to ``RefinementLoop.__init__``.

        Returns:
            A configured ``RefinementLoop``.
        """
        from exo.runner import run  # pyright: ignore[reportMissingImports]

        async def _execute(input: str) -> str:
            result = await run(agent, input)
            return result.output

        async def _stream(input: str) -> AsyncIterator[StreamEvent]:
            async for event in run.stream(agent, input):
                yield event

        return cls(
            execute_fn=_execute,
            stream_execute_fn=_stream,
            scorers=scorers,
            **kwargs,
        )

    # ---- public API -------------------------------------------------------

    async def run(self, input: str) -> RefinementResult:
        """Execute the full refinement loop on *input* and return the result."""
        logger.info(
            "RefinementLoop starting: input_len=%d scorers=%d", len(input), len(self._scorers)
        )
        state = LoopState()
        original_input = input
        last_output = ""
        last_scores: dict[str, float] = {}
        reflections: list[dict[str, Any]] = []

        while True:
            state.iteration += 1
            # Rebuild current_input from the original prompt + accumulated feedback
            # blocks (O(k) per iteration vs the previous O(k²) string-growth).
            current_input = self._plan(original_input, state, None)

            # --- Phase 1: Run ---
            output, success = await self._execute(current_input, state)
            last_output = output

            # --- Phase 2: Analyze ---
            if success:
                last_scores = await self._analyze(output, current_input, state)

            # --- Phase 3: Learn ---
            reflection = await self._learn(
                current_input,
                output,
                last_scores,
                success,
                state,
            )
            if reflection is not None:
                reflections.append({"iteration": state.iteration, "summary": reflection.summary})

            # --- Phase 4: Plan (re-prompt) ---
            # Append new feedback block into state; next iteration rebuilds the
            # full prompt from original_input + all accumulated blocks.
            self._plan(original_input, state, reflection)

            # --- Phase 5: Halt ---
            decision = await self._halt(state)
            if decision.should_stop:
                return RefinementResult(
                    output=last_output,
                    stop_type=decision.stop_type,
                    reason=decision.reason,
                    iterations=state.iteration,
                    scores=last_scores,
                    state=state.to_dict(),
                    reflections=reflections,
                )

    async def stream(self, input: str, *, name: str = "ralph") -> AsyncIterator[StreamEvent]:
        """Stream the refinement loop, yielding inner events and lifecycle events.

        Requires ``stream_execute_fn`` to have been provided at construction
        (or use ``from_agent()`` which sets it automatically).

        Args:
            input: The initial input to the loop.
            name: Agent name to stamp on refinement lifecycle events.

        Yields:
            Interleaved inner agent events, ``RefinementIterationEvent``,
            and ``RefinementStopEvent``.

        Raises:
            EvalError: If ``stream_execute_fn`` was not provided.
        """
        if self._stream_execute_fn is None:
            raise EvalError(
                "stream_execute_fn is required for streaming but was not provided.",
                hint=(
                    "Use RefinementLoop.from_agent(agent, scorers) to wire both "
                    "execute_fn and stream_execute_fn automatically, or pass "
                    "stream_execute_fn= to the constructor."
                ),
            )

        logger.info(
            "RefinementLoop.stream starting: input_len=%d scorers=%d",
            len(input),
            len(self._scorers),
        )
        state = LoopState()
        original_input = input

        while True:
            state.iteration += 1
            # Rebuild current_input from the original prompt + accumulated feedback
            # blocks (O(k) per iteration vs the previous O(k²) string-growth).
            current_input = self._plan(original_input, state, None)
            yield RefinementIterationEvent(
                iteration=state.iteration,
                status="started",
                agent_name=name,
            )

            # --- Phase 1: Run (streaming) ---
            output_parts: list[str] = []
            success = True
            try:
                async for event in self._stream_execute_fn(current_input):
                    yield event
                    if isinstance(event, TextEvent):
                        output_parts.append(event.text)
                output = "".join(output_parts)
                state.record_success()
            except Exception as exc:
                logger.warning(
                    "RefinementLoop.stream execute_fn failed at iteration=%d: %s",
                    state.iteration,
                    exc,
                    exc_info=True,
                )
                output = f"[execute error: {type(exc).__name__}: {exc}]"
                state.record_failure()
                success = False

            # --- Phase 2: Analyze ---
            scores = await self._analyze(output, current_input, state) if success else {}

            yield RefinementIterationEvent(
                iteration=state.iteration,
                status="completed" if success else "failed",
                scores=scores,
                agent_name=name,
            )

            # --- Phase 3: Learn ---
            reflection = await self._learn(current_input, output, scores, success, state)

            # --- Phase 4: Plan ---
            # Append new feedback block into state; next iteration rebuilds the
            # full prompt from original_input + all accumulated blocks.
            self._plan(original_input, state, reflection)

            # --- Phase 5: Halt ---
            decision = await self._halt(state)
            if decision.should_stop:
                yield RefinementStopEvent(
                    stop_type=decision.stop_type.value,
                    reason=decision.reason,
                    iterations=state.iteration,
                    final_scores=scores,
                    agent_name=name,
                )
                return

    # ---- phase implementations --------------------------------------------

    async def _execute(self, input: str, state: LoopState) -> tuple[str, bool]:
        """Phase 1: Run the task and track success/failure."""
        try:
            output = await self._execute_fn(input)
            state.record_success()
            return (str(output), True)
        except Exception as exc:
            logger.warning(
                "RefinementLoop._execute failed at iteration=%d: %s",
                state.iteration,
                exc,
                exc_info=True,
            )
            state.record_failure()
            # Preserve exc_type in the output string so reflectors can see it.
            return (f"[execute error: {type(exc).__name__}: {exc}]", False)

    async def _analyze(
        self,
        output: str,
        input: str,
        state: LoopState,
    ) -> dict[str, float]:
        """Phase 2: Score the output with all configured scorers.

        Scorers are fanned out concurrently, bounded by
        ``ValidationConfig.parallel`` (default 4).
        """
        if not self._config.validation.enabled or not self._scorers:
            return {}
        case_id = f"ralph-iter-{state.iteration}"
        sem = asyncio.Semaphore(self._config.validation.parallel)

        async def _run_scorer(scorer: Scorer) -> ScorerResult | tuple[str, Exception]:
            scorer_name = getattr(scorer, "_name", type(scorer).__name__)
            async with sem:
                try:
                    return await scorer.score(case_id, input, output)
                except Exception as exc:
                    return (scorer_name, exc)

        raw_results = await asyncio.gather(*(_run_scorer(s) for s in self._scorers))

        scores: dict[str, float] = {}
        for item in raw_results:
            if isinstance(item, ScorerResult):
                scores[item.scorer_name] = item.score
            else:
                scorer_name, exc = item
                logger.warning(
                    "Scorer %r failed case=%s: %s",
                    scorer_name,
                    case_id,
                    exc,
                    exc_info=True,
                )
                # Record a zero-score sentinel so a scorer failure is
                # distinguishable from a genuine 0.0 score in pass@k / threshold
                # logic — callers can detect "error" in the details dict.
                scores[scorer_name] = 0.0
                failures: list[dict[str, Any]] = state.metadata.setdefault(
                    "scorer_failures", []
                )
                failures.append(
                    {
                        "scorer": scorer_name,
                        "case_id": case_id,
                        "error": str(exc),
                        "exc_type": type(exc).__name__,
                    }
                )
                # Cap to the 100 most recent entries so metadata stays bounded
                # in long-running loops with repeated scorer errors.
                if len(failures) > 100:
                    state.metadata["scorer_failures"] = failures[-100:]
        state.record_score(scores, max_history=self._config.reflection.max_history)
        return scores

    async def _learn(
        self,
        input: str,
        output: str,
        scores: dict[str, float],
        success: bool,
        state: LoopState,
    ) -> ReflectionResult | None:
        """Phase 3: Reflect when the iteration failed or scored poorly."""
        if not self._config.reflection.enabled or self._reflector is None:
            return None

        # Reflect if execution failed or mean score below threshold
        needs_reflection = not success
        if success and scores:
            mean_score = sum(scores.values()) / len(scores)
            needs_reflection = mean_score < self._config.validation.min_score_threshold

        if not needs_reflection:
            return None

        context: dict[str, Any] = {
            "input": input,
            "output": output,
            "scores": scores,
            "success": success,
            "iteration": state.iteration,
        }
        result = await self._reflector.reflect(context)
        state.record_reflection(
            {"summary": result.summary, "suggestions": result.suggestions},
            max_history=self._config.reflection.max_history,
        )
        return result

    def _plan(
        self,
        original_input: str,
        state: LoopState,
        reflection: ReflectionResult | None,
    ) -> str:
        """Phase 4: Re-prompt by appending reflection suggestions.

        Feedback blocks are accumulated on *state.feedback_blocks* (one entry per
        iteration) so each block is written exactly once — O(k) total rather than
        the O(k²) growth caused by re-appending the entire growing *current_input*
        on each round.  The full prompt is reconstructed by joining at call time.
        """
        if reflection is not None and reflection.suggestions:
            suggestions = "\n".join(f"- {s}" for s in reflection.suggestions)
            state.feedback_blocks.append(f"[Previous feedback]\n{suggestions}")
        if not state.feedback_blocks:
            return original_input
        return original_input + "\n\n" + "\n\n".join(state.feedback_blocks)

    async def _halt(self, state: LoopState) -> StopDecision:
        """Phase 5: Check all stop conditions."""
        return await self._detector.check(state, self._config.stop_condition)

    # ---- internal ---------------------------------------------------------

    def _build_detector(self) -> CompositeDetector:
        """Create the composite stop detector from config."""
        return CompositeDetector(
            [
                MaxIterationDetector(),
                TimeoutDetector(),
                CostLimitDetector(),
                ConsecutiveFailureDetector(),
                ScoreThresholdDetector(),
            ]
        )

    def __repr__(self) -> str:
        s = len(self._scorers)
        # repr keeps the deprecated alias name for backward compatibility
        return f"RalphRunner(scorers={s}, config={self._config!r})"


# Deprecated alias: use RefinementLoop
RalphRunner = RefinementLoop
