"""Temporal workflows for durable agent execution.

Imported only when ``temporalio`` is installed (guarded by the package
``__init__`` behind :data:`~exo.distributed.temporal._compat.HAS_TEMPORAL`).

Two workflows are provided:

:class:`AgentExecutionWorkflow`
    The original *single-activity* design.  The whole agent (or Swarm) run
    happens inside one activity; the workflow stays thin and just dispatches it
    with the configured timeouts and retry policy.  Crash-resumable at the
    activity level (heartbeat checkpointing) but a retry re-runs the agent from
    the start.  This remains the default and is the only path that supports
    ``Swarm`` tasks.

:class:`AgentLoopWorkflow`
    The *durable step-as-activity* design (Phase 2).  The agent's step loop is
    lifted into the workflow: each LLM turn and each tool call is its own
    activity, so every step is recorded in Temporal history.  A worker crash
    resumes from the **last completed step**, not from step 0 — the real
    checkpoint-restore Temporal is known for.  Adds signals (mid-run message
    injection), queries (live progress), updates (steering), and
    continue-as-new for long runs.  Single-agent only.

Both workflows keep their orchestration code deterministic: all non-determinism
(LLM calls, tool I/O) lives in activities, and :class:`AgentLoopWorkflow`
manipulates only plain JSON dicts + counters so nothing in ``exo._internal`` is
imported into the replay sandbox.
"""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from typing import Any

from exo.distributed.temporal._compat import workflow  # pyright: ignore[reportMissingImports]

# Temporal runs workflow code inside a reimport sandbox that bans restricted
# (non-deterministic) stdlib calls made at *import* time.  The activities module
# pulls in exo internals (``failure`` → ``exo.agent`` → ``exo.skills`` →
# ``pathlib.Path.home``) at import time, which trips the sandbox, so we never
# import it here — activities are dispatched **by name** (the worker registers
# them under these names).  The config module is pure (pydantic + temporalio
# only) and provides the payload→Temporal conversions used in deterministic
# workflow code; it is loaded as a passthrough module to be safe.
with workflow.unsafe.imports_passed_through():
    from exo.distributed.temporal.config import (  # pyright: ignore[reportMissingImports]
        RetryConfig,
        TimeoutConfig,
    )

__all__ = ["AgentExecutionWorkflow", "AgentLoopWorkflow"]

# Activity names — must match the ``@activity.defn`` function names registered
# on the worker (see ``executor.start_temporal_worker``).
_EXECUTE_AGENT_ACTIVITY = "execute_agent_activity"
_LLM_TURN_ACTIVITY = "agent_llm_turn"
_TOOL_CALL_ACTIVITY = "agent_tool_call"

# Fallback heartbeat timeout when the payload carries no explicit timeouts.
_DEFAULT_HEARTBEAT_TIMEOUT = timedelta(seconds=30)

# Defaults for the durable loop's per-activity bounds and continue-as-new.
_DEFAULT_TURN_TIMEOUT = timedelta(seconds=300)
_DEFAULT_MAX_STEPS = 50
#: Roll over to a fresh workflow run once this many history events accumulate,
#: keeping each run well under Temporal's ~50K-event / ~50MB history cap.
_DEFAULT_CONTINUE_AS_NEW_THRESHOLD = 5_000


@workflow.defn
class AgentExecutionWorkflow:
    """Durable workflow that wraps agent execution in a single activity.

    Receives a :class:`~exo.distributed.models.TaskPayload` as JSON and
    delegates to :func:`execute_agent_activity`.  Activity timeouts and the
    retry policy are taken from the payload's ``timeouts`` / ``retry_policy``
    fields when present, falling back to ``timeout_seconds`` otherwise.
    """

    @workflow.run
    async def run(self, payload_json: str) -> str:
        """Execute the agent activity with the given task payload."""
        data: dict[str, Any] = json.loads(payload_json)

        activity_kwargs: dict[str, Any] = {}

        timeouts_data = data.get("timeouts")
        if timeouts_data:
            activity_kwargs.update(TimeoutConfig.from_dict(timeouts_data).to_activity_timeouts())

        # Guarantee a bounding timeout and liveness check even with no config.
        if "start_to_close_timeout" not in activity_kwargs and (
            "schedule_to_close_timeout" not in activity_kwargs
        ):
            activity_kwargs["start_to_close_timeout"] = timedelta(
                seconds=data.get("timeout_seconds", 300.0)
            )
        activity_kwargs.setdefault("heartbeat_timeout", _DEFAULT_HEARTBEAT_TIMEOUT)

        retry_data = data.get("retry_policy")
        if retry_data:
            activity_kwargs["retry_policy"] = RetryConfig.from_dict(retry_data).to_retry_policy()

        return await workflow.execute_activity(
            _EXECUTE_AGENT_ACTIVITY,
            payload_json,
            result_type=str,
            **activity_kwargs,
        )


@workflow.defn
class AgentLoopWorkflow:
    """Durable agent loop — one LLM turn / one tool call per activity.

    The workflow owns the conversation: it holds the running message history
    (as plain JSON dicts), drives the step loop, and records every LLM turn and
    tool call as a Temporal activity.  Because the loop lives in the workflow,
    Temporal replays it from history on recovery, so a crashed worker resumes
    from the last completed step.

    **Input** — a single dict (``LoopInput``) with:

    - ``agent_config`` (dict, required): a single ``Agent.to_dict()`` snapshot.
    - ``input`` (str): the initial user message (used only on the first run;
      ignored when ``messages`` is supplied, e.g. across continue-as-new).
    - ``messages`` (list[dict], optional): a prior message history to resume
      from.
    - ``max_steps`` (int): hard cap on LLM turns for this run (default 50).
    - ``max_retries`` (int): per-LLM-call retry budget passed to the activity.
    - ``timeouts`` / ``retry_policy`` (dict, optional): per-activity bounds,
      same shape as :class:`AgentExecutionWorkflow`.
    - ``total_steps`` (int): steps already taken in prior continue-as-new
      segments (internal bookkeeping; callers leave it at 0).

    **Control plane**

    - ``@workflow.signal inject_message`` — append a user message mid-run;
      it is folded into the history before the next LLM turn.
    - ``@workflow.signal cancel`` — request graceful stop after the current
      step (in addition to Temporal's native workflow cancellation).
    - ``@workflow.query get_progress`` — live ``{step, message_count,
      last_text, done}`` without touching the run.
    - ``@workflow.update steer`` — inject a steering instruction with an ack.
    - ``@workflow.update set_model`` — swap the model for subsequent turns.

    Returns the agent's final text output.
    """

    def __init__(self) -> None:
        self._agent_config: dict[str, Any] = {}
        self._messages: list[dict[str, Any]] = []
        self._injected: list[dict[str, Any]] = []
        self._step: int = 0
        self._total_steps: int = 0
        self._last_text: str = ""
        self._done: bool = False
        self._cancel_requested: bool = False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    @workflow.run
    async def run(self, loop_input: dict[str, Any]) -> str:
        self._agent_config = loop_input["agent_config"]
        max_steps = int(loop_input.get("max_steps", _DEFAULT_MAX_STEPS))
        max_retries = int(loop_input.get("max_retries", 3))
        self._total_steps = int(loop_input.get("total_steps", 0))

        # Resume from a prior history (continue-as-new / explicit resume), or
        # seed a fresh conversation from the initial user input.
        prior = loop_input.get("messages")
        if prior:
            self._messages = list(prior)
        else:
            self._messages = [{"role": "user", "content": loop_input.get("input", "")}]

        turn_kwargs = self._turn_activity_kwargs(loop_input)
        threshold = int(
            loop_input.get("continue_as_new_threshold", _DEFAULT_CONTINUE_AS_NEW_THRESHOLD)
        )

        for _ in range(max_steps):
            if self._cancel_requested:
                break

            self._drain_injected()

            turn = await workflow.execute_activity(
                _LLM_TURN_ACTIVITY,
                {
                    "agent_config": self._agent_config,
                    "messages": self._messages,
                    "max_retries": max_retries,
                },
                result_type=dict,
                **turn_kwargs,
            )
            assistant = turn["assistant"]
            self._messages.append(assistant)
            self._last_text = assistant.get("content") or ""
            self._step += 1
            self._total_steps += 1

            if turn["finished"]:
                self._done = True
                return self._last_text

            tool_calls = assistant.get("tool_calls") or []
            results = await asyncio.gather(
                *[
                    workflow.execute_activity(
                        _TOOL_CALL_ACTIVITY,
                        {"agent_config": self._agent_config, "tool_call": tc},
                        result_type=dict,
                        **turn_kwargs,
                    )
                    for tc in tool_calls
                ]
            )
            self._messages.extend(r["tool_result"] for r in results)

            # Roll over to a fresh run before history grows unbounded.
            if workflow.info().get_current_history_length() >= threshold:
                self._drain_injected()
                workflow.continue_as_new(
                    args=[self._continue_as_new_input(loop_input, max_steps, max_retries)]
                )

        # max_steps exhausted (or cancelled) — return the last text we have.
        self._done = True
        return self._last_text

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    @workflow.signal
    def inject_message(self, text: str) -> None:
        """Queue a user message to fold into the history before the next turn."""
        self._injected.append({"role": "user", "content": text})

    @workflow.signal
    def cancel(self) -> None:
        """Request a graceful stop after the current step completes."""
        self._cancel_requested = True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @workflow.query
    def get_progress(self) -> dict[str, Any]:
        """Return live progress without mutating or blocking the run."""
        return {
            "step": self._step,
            "total_steps": self._total_steps,
            "message_count": len(self._messages),
            "last_text": self._last_text,
            "done": self._done,
            "cancel_requested": self._cancel_requested,
        }

    # ------------------------------------------------------------------
    # Updates (validated, synchronous steering)
    # ------------------------------------------------------------------

    @workflow.update
    def steer(self, instruction: str) -> dict[str, Any]:
        """Inject a steering instruction mid-run and acknowledge synchronously.

        Returns the step at which the instruction was accepted so the caller
        knows when it will take effect (it is applied before the next turn).
        """
        self._injected.append({"role": "user", "content": instruction})
        return {"accepted_at_step": self._step, "queued": len(self._injected)}

    @steer.validator
    def _validate_steer(self, instruction: str) -> None:
        if not instruction or not instruction.strip():
            raise ValueError("steer instruction must be a non-empty string")
        if self._done:
            raise ValueError("cannot steer a run that has already completed")

    @workflow.update
    def set_model(self, model: str) -> str:
        """Swap the model used for subsequent LLM turns; returns the new model."""
        self._agent_config = {**self._agent_config, "model": model}
        return model

    @set_model.validator
    def _validate_set_model(self, model: str) -> None:
        if not model or ":" not in model:
            raise ValueError("model must be a non-empty 'provider:model' string")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _drain_injected(self) -> None:
        """Move any queued injected messages into the durable history."""
        if self._injected:
            self._messages.extend(self._injected)
            self._injected = []

    def _turn_activity_kwargs(self, loop_input: dict[str, Any]) -> dict[str, Any]:
        """Build per-activity timeout/retry kwargs from the payload."""
        kwargs: dict[str, Any] = {}
        timeouts_data = loop_input.get("timeouts")
        if timeouts_data:
            kwargs.update(TimeoutConfig.from_dict(timeouts_data).to_activity_timeouts())
        if "start_to_close_timeout" not in kwargs and "schedule_to_close_timeout" not in kwargs:
            kwargs["start_to_close_timeout"] = _DEFAULT_TURN_TIMEOUT
        kwargs.setdefault("heartbeat_timeout", _DEFAULT_HEARTBEAT_TIMEOUT)
        retry_data = loop_input.get("retry_policy")
        if retry_data:
            kwargs["retry_policy"] = RetryConfig.from_dict(retry_data).to_retry_policy()
        return kwargs

    def _continue_as_new_input(
        self, loop_input: dict[str, Any], max_steps: int, max_retries: int
    ) -> dict[str, Any]:
        """Capture the resume payload for the next continue-as-new segment."""
        return {
            **loop_input,
            "agent_config": self._agent_config,
            "messages": self._messages,
            "input": "",
            "max_steps": max_steps,
            "max_retries": max_retries,
            "total_steps": self._total_steps,
        }
