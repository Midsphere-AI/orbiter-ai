"""Temporal activities for durable agent execution.

This module is imported only when ``temporalio`` is installed (the package
``__init__`` guards it behind :data:`~exo.distributed.temporal._compat.HAS_TEMPORAL`),
so the ``@activity.defn`` decorators always have a real ``activity`` module.

**Resumption semantics (read this before relying on durability).** The whole
agent run executes inside *one* activity.  Temporal re-runs a crashed activity
**from the start** — it does not replay it step-by-step.  What this module adds
is the *coarse, manual* cousin of checkpoint-restore: each batch of streamed
events updates ``activity.heartbeat(...)`` with the current progress, and a
retry reads ``activity.info().heartbeat_details`` so the new attempt can observe
how far the previous one got.  This makes a long run **crash-resumable at the
activity level** (the worker can die and another picks the task up), but it is
*not* Temporal's automatic per-step checkpoint-restore — that requires lifting
the agent loop into the workflow (a later phase).  Do not describe this as "full
state recovery."
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from exo.distributed.temporal._compat import activity  # pyright: ignore[reportMissingImports]
from exo.distributed.temporal.failure import (  # pyright: ignore[reportMissingImports]
    to_application_error,
)

logger = logging.getLogger(__name__)

__all__ = [
    "agent_llm_turn",
    "agent_tool_call",
    "execute_agent_activity",
]

# Heartbeat every N streamed events so Temporal can detect a dead worker and
# so a retry can observe prior progress via heartbeat details.
_HEARTBEAT_EVERY = 10


def _prior_progress() -> dict[str, Any] | None:
    """Return the last heartbeat payload from a previous attempt, if any.

    Returns ``None`` outside an activity context (e.g. direct unit-test calls)
    or on the first attempt, where no heartbeat details exist yet.
    """
    if not activity.in_activity():
        return None
    try:
        details = activity.info().heartbeat_details
    except Exception:  # pragma: no cover - defensive; info() needs a live context
        return None
    if not details:
        return None
    last = details[-1]
    return last if isinstance(last, dict) else None


@activity.defn
async def execute_agent_activity(payload_json: str) -> str:
    """Execute an agent from a serialized ``TaskPayload`` JSON string.

    Streams the agent, heartbeating progress periodically.  Honors Temporal
    cancellation (the activity is cancelled cooperatively between events) and
    maps any failure onto a Temporal :class:`ApplicationError` whose
    ``non_retryable`` flag respects exo's error taxonomy, so deterministic
    failures are not retried.

    Returns the collected text output as a JSON string.
    """
    from exo.agent import Agent  # pyright: ignore[reportMissingImports]
    from exo.distributed.models import TaskPayload  # pyright: ignore[reportMissingImports]
    from exo.runner import run  # pyright: ignore[reportMissingImports]
    from exo.swarm import Swarm  # pyright: ignore[reportMissingImports]
    from exo.types import TextEvent  # pyright: ignore[reportMissingImports]

    payload_data = json.loads(payload_json)
    task = TaskPayload(**payload_data)

    # A retry after partial progress: surface how far the last attempt got.
    # We cannot resume an LLM stream mid-flight, so this attempt restarts from
    # the beginning — but the prior progress is logged for operability.
    prior = _prior_progress()
    if prior is not None:
        logger.info(
            "Activity for task %s retrying after partial progress (prior step=%s, chars=%s); "
            "restarting the run from the beginning",
            task.task_id,
            prior.get("step"),
            prior.get("chars"),
        )

    # Reconstruct agent or swarm.
    agent_config = task.agent_config
    if "agents" in agent_config:
        agent = Swarm.from_dict(agent_config)
    else:
        agent = Agent.from_dict(agent_config)

    # Deserialize task.messages (same as the local execution path in worker.py).
    from exo.distributed.worker import (
        _deserialize_messages,  # pyright: ignore[reportMissingImports]
    )

    messages = _deserialize_messages(task.messages) if task.messages else None

    # TODO(memory-temporal): The local execution path (Worker._execute_task /
    # Worker._run_agent) hydrates memory from task.metadata["memory"], attaches
    # MemoryPersistence, and loads conversation history before the stream.
    # This Temporal activity path does NOT currently do so — it runs the agent
    # without memory hydration or persistence.  Consequence: tasks routed via
    # Temporal lose conversation history and memory writes.
    #
    # Extracting the memory logic into a shared async helper (e.g.
    # exo.distributed.memory.hydrate_agent_memory) and calling it here is the
    # correct fix, but it is non-trivial to test without a live memory backend
    # and a Temporal cluster.  Tracked as a follow-up; for now the gap is
    # documented here and in exo.distributed.memory so operators are not
    # surprised.  Set task.metadata["memory"] = None/omit it to force the
    # local executor path when memory is required.

    text_parts: list[str] = []
    chars = 0
    step = 0
    try:
        async for event in run.stream(
            agent,
            task.input,
            messages=messages,
            detailed=task.detailed,
        ):
            # Cooperative cancellation: Temporal flips this flag when the
            # workflow/activity is cancelled.  Stop promptly and let Temporal
            # record the cancellation.
            if activity.in_activity() and activity.is_cancelled():
                logger.info("Activity for task %s cancelled at step %d", task.task_id, step)
                raise asyncio.CancelledError

            if isinstance(event, TextEvent):
                text_parts.append(event.text)
                chars += len(event.text)

            step += 1
            if step % _HEARTBEAT_EVERY == 0:
                # Checkpoint progress into the heartbeat so a retry (and ops)
                # can observe it via activity.info().heartbeat_details.
                activity.heartbeat({"step": step, "chars": chars})
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        # Map onto a Temporal ApplicationError so RetryPolicy + the
        # non_retryable taxonomy decide whether this attempt is retried.
        raise to_application_error(exc) from exc

    return json.dumps({"output": "".join(text_parts)})


# ---------------------------------------------------------------------------
# Durable agent loop (step-as-activity)
# ---------------------------------------------------------------------------
#
# The activities below back :class:`~exo.distributed.temporal.workflows.AgentLoopWorkflow`,
# which lifts the agent's *step structure* into the workflow so each LLM turn
# and tool call becomes a recorded Temporal history event.  This delivers the
# real per-step checkpoint-restore (a crashed worker resumes from the last
# completed step, not from step 0) that the single-activity design above cannot.
#
# Each activity is **stateless**: it reconstructs the agent from the frozen
# ``agent_config`` carried on the payload, performs exactly one unit of I/O
# (one LLM call, or one tool execution), and returns plain JSON-serialisable
# dicts.  All orchestration state (the message history, step counter) lives in
# the deterministic workflow, never in the activity.


def _reconstruct_agent(agent_config: dict[str, Any]) -> Any:
    """Rebuild an ``Agent`` from a serialized config dict.

    The durable loop is single-agent only; a Swarm config (one carrying an
    ``"agents"`` key) is rejected with a clear error pointing at the
    single-activity :func:`execute_agent_activity` path, which runs a whole
    ``Swarm`` durably (Swarm-as-child-workflows is a later phase).
    """
    from exo.agent import Agent  # pyright: ignore[reportMissingImports]

    if "agents" in agent_config:
        raise ValueError(
            "AgentLoopWorkflow supports a single Agent, not a Swarm. "
            "Use the single-activity executor path (durable=False) for Swarm tasks."
        )
    return Agent.from_dict(agent_config)


@activity.defn
async def agent_llm_turn(turn: dict[str, Any]) -> dict[str, Any]:
    """Run one LLM turn for the durable agent loop.

    Reconstructs the agent, assembles the prompt from the supplied message
    history (prepending the agent's system instructions), and performs a
    *single* LLM completion via :meth:`Agent._call_llm` — which preserves the
    agent's ``PRE_LLM_CALL`` / ``POST_LLM_CALL`` hooks (guardrails included) and
    per-call retry.  No tools are executed here; tool dispatch is the workflow's
    job (see :func:`agent_tool_call`).

    Args:
        turn: ``{"agent_config": dict, "messages": list[dict], "max_retries": int}``
            where ``messages`` are serialized :class:`~exo.types.Message` dicts
            (``model_dump()`` form).

    Returns:
        ``{"assistant": dict, "usage": dict, "finished": bool}`` — the assistant
        message (an :class:`~exo.types.AssistantMessage` ``model_dump()``), the
        call's token usage, and ``finished`` (``True`` when the model produced
        no tool calls, i.e. the run is complete).
    """
    from exo._internal.message_builder import (  # pyright: ignore[reportMissingImports]
        build_messages,
    )
    from exo._internal.run_helpers import (  # pyright: ignore[reportMissingImports]
        resolve_instructions,
    )
    from exo.models import get_provider  # pyright: ignore[reportMissingImports]
    from exo.types import AssistantMessage  # pyright: ignore[reportMissingImports]

    if activity.in_activity():
        activity.heartbeat({"phase": "llm_turn"})

    agent = _reconstruct_agent(turn["agent_config"])
    history = _deserialize_messages(turn.get("messages", []))
    max_retries = int(turn.get("max_retries", 3))

    try:
        instructions = await resolve_instructions(agent)
        msg_list = build_messages(instructions, history)
        tool_schemas = agent.get_tool_schemas() or None
        provider = get_provider(agent.model)
        output = await agent._call_llm(msg_list, tool_schemas, provider, max_retries)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        raise to_application_error(exc) from exc

    assistant = AssistantMessage(content=output.text, tool_calls=output.tool_calls)
    return {
        "assistant": assistant.model_dump(mode="json"),
        "usage": output.usage.model_dump(mode="json"),
        "finished": not output.tool_calls,
    }


@activity.defn
async def agent_tool_call(req: dict[str, Any]) -> dict[str, Any]:
    """Execute a single tool call for the durable agent loop.

    Reconstructs the agent and runs exactly one tool through
    :meth:`Agent._execute_tools`, which preserves the agent's
    ``PRE_TOOL_CALL`` / ``POST_TOOL_CALL`` hooks, ``injected_tool_args``
    stripping, ``ToolContext`` injection, and ``large_output`` offloading.

    Tool failures are returned as an error :class:`~exo.types.ToolResult`
    (matching exo's in-process semantics) rather than failing the activity, so
    the model can observe and recover from them on the next turn.  Only
    infrastructure faults (a dead worker, cancellation) surface as activity
    failures.

    Args:
        req: ``{"agent_config": dict, "tool_call": dict}`` where ``tool_call``
            is a serialized :class:`~exo.types.ToolCall` (``model_dump()`` form).

    Returns:
        ``{"tool_result": dict}`` — a serialized :class:`~exo.types.ToolResult`.
    """
    from exo._internal.output_parser import (  # pyright: ignore[reportMissingImports]
        parse_tool_arguments,
    )
    from exo.types import ToolCall  # pyright: ignore[reportMissingImports]

    if activity.in_activity():
        activity.heartbeat({"phase": "tool_call", "tool": req.get("tool_call", {}).get("name")})

    agent = _reconstruct_agent(req["agent_config"])
    tool_call = ToolCall.model_validate(req["tool_call"])

    try:
        actions = parse_tool_arguments([tool_call])
        results = await agent._execute_tools(actions)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        raise to_application_error(exc) from exc

    return {"tool_result": results[0].model_dump(mode="json")}


def _deserialize_messages(raw: list[dict[str, Any]]) -> list[Any]:
    """Deserialize the workflow's plain message dicts back to ``Message`` models.

    The workflow holds messages as plain JSON dicts (so it never imports exo
    types into the deterministic sandbox); activities round-trip them through
    the real Pydantic models on the way in.
    """
    from exo.types import (  # pyright: ignore[reportMissingImports]
        AssistantMessage,
        SystemMessage,
        ToolResult,
        UserMessage,
    )

    by_role = {
        "user": UserMessage,
        "assistant": AssistantMessage,
        "system": SystemMessage,
        "tool": ToolResult,
    }
    out: list[Any] = []
    for m in raw:
        cls = by_role.get(m.get("role", "user"), UserMessage)
        out.append(cls.model_validate(m))
    return out
