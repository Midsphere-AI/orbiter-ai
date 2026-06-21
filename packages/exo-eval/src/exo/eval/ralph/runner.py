"""RalphRunner — iterative refinement loop: Run → Analyze → Learn → Plan → Halt."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from exo.eval.base import (  # pyright: ignore[reportMissingImports]
    Scorer,
    ScorerResult,
)
from exo.eval.ralph.config import (  # pyright: ignore[reportMissingImports]
    LoopState,
    RalphConfig,
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
    RalphIterationEvent,
    RalphStopEvent,
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
class RalphResult:
    """Final outcome of a Ralph loop execution."""

    output: str
    stop_type: StopType
    reason: str
    iterations: int
    scores: dict[str, float]
    state: dict[str, Any]
    reflections: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# RalphRunner
# ---------------------------------------------------------------------------


class RalphRunner:
    """Implements the 5-phase Ralph iterative refinement loop.

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
        config: RalphConfig | None = None,
        reflector: Reflector | None = None,
        replan_fn: RePlanFn | None = None,
    ) -> None:
        self._execute_fn = execute_fn
        self._stream_execute_fn = stream_execute_fn
        self._scorers = list(scorers)
        self._config = config or RalphConfig()
        self._reflector = reflector
        self._replan_fn = replan_fn
        self._detector = self._build_detector()

    @classmethod
    def from_agent(cls, agent: Any, scorers: list[Scorer], **kwargs: Any) -> RalphRunner:
        """Create a RalphRunner wired to an Agent's run() and stream().

        Convenience factory that creates both ``execute_fn`` (for ``.run()``)
        and ``stream_execute_fn`` (for ``.stream()``) from the same agent.

        Args:
            agent: An ``Agent`` instance.
            scorers: List of scorers for the analyze phase.
            **kwargs: Additional arguments forwarded to ``RalphRunner.__init__``.

        Returns:
            A configured ``RalphRunner``.
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

    async def run(self, input: str) -> RalphResult:
        """Execute the full Ralph loop on *input* and return the result."""
        logger.info("RalphRunner starting: input_len=%d scorers=%d", len(input), len(self._scorers))
        state = LoopState()
        current_input = input
        last_output = ""
        last_scores: dict[str, float] = {}
        reflections: list[dict[str, Any]] = []

        while True:
            state.iteration += 1

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
            # Pass current_input (not the original) so chained feedback accumulates.
            current_input = self._plan(current_input, reflection)

            # --- Phase 5: Halt ---
            decision = await self._halt(state)
            if decision.should_stop:
                return RalphResult(
                    output=last_output,
                    stop_type=decision.stop_type,
                    reason=decision.reason,
                    iterations=state.iteration,
                    scores=last_scores,
                    state=state.to_dict(),
                    reflections=reflections,
                )

    async def stream(self, input: str, *, name: str = "ralph") -> AsyncIterator[StreamEvent]:
        """Stream the Ralph loop, yielding inner events and Ralph lifecycle events.

        Requires ``stream_execute_fn`` to have been provided at construction
        (or use ``from_agent()`` which sets it automatically).

        Args:
            input: The initial input to the loop.
            name: Agent name to stamp on Ralph lifecycle events.

        Yields:
            Interleaved inner agent events, ``RalphIterationEvent``,
            and ``RalphStopEvent``.

        Raises:
            ValueError: If ``stream_execute_fn`` was not provided.
        """
        if self._stream_execute_fn is None:
            raise ValueError("stream_execute_fn required for streaming")

        logger.info(
            "RalphRunner.stream starting: input_len=%d scorers=%d",
            len(input),
            len(self._scorers),
        )
        state = LoopState()
        current_input = input

        while True:
            state.iteration += 1
            yield RalphIterationEvent(
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
                output = str(exc)
                state.record_failure()
                success = False

            # --- Phase 2: Analyze ---
            scores = await self._analyze(output, current_input, state) if success else {}

            yield RalphIterationEvent(
                iteration=state.iteration,
                status="completed" if success else "failed",
                scores=scores,
                agent_name=name,
            )

            # --- Phase 3: Learn ---
            reflection = await self._learn(current_input, output, scores, success, state)

            # --- Phase 4: Plan ---
            # Pass current_input (not the original) so chained feedback accumulates.
            current_input = self._plan(current_input, reflection)

            # --- Phase 5: Halt ---
            decision = await self._halt(state)
            if decision.should_stop:
                yield RalphStopEvent(
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
            state.record_failure()
            return (str(exc), False)

    async def _analyze(
        self,
        output: str,
        input: str,
        state: LoopState,
    ) -> dict[str, float]:
        """Phase 2: Score the output with all configured scorers."""
        if not self._config.validation.enabled or not self._scorers:
            return {}
        scores: dict[str, float] = {}
        case_id = f"ralph-iter-{state.iteration}"
        for scorer in self._scorers:
            try:
                result: ScorerResult = await scorer.score(case_id, input, output)
                scores[result.scorer_name] = result.score
            except Exception as exc:
                logger.warning("Scorer failed case=%s: %s", case_id, exc, exc_info=True)
        state.record_score(scores)
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
        state.record_reflection({"summary": result.summary, "suggestions": result.suggestions})
        return result

    def _plan(self, current_input: str, reflection: ReflectionResult | None) -> str:
        """Phase 4: Re-prompt by appending reflection suggestions.

        *current_input* is the input from the **previous** iteration (which may
        already contain chained feedback from earlier iterations).  New
        suggestions are appended to it so that feedback accumulates across
        iterations rather than resetting to the original prompt each time.
        """
        if reflection is None or not reflection.suggestions:
            return current_input
        suggestions = "\n".join(f"- {s}" for s in reflection.suggestions)
        return f"{current_input}\n\n[Previous feedback]\n{suggestions}"

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
        return f"RalphRunner(scorers={s}, config={self._config!r})"
