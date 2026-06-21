"""Public entry point for running agents.

Provides ``run()`` (async), ``run.sync()`` (blocking), and
``run.stream()`` (async generator) as the primary API for executing
an ``Agent``.  Internally delegates to
:func:`exo._internal.call_runner.call_runner` for state tracking
and loop detection.

Usage::

    result = await run(agent, "Hello!")
    result = run.sync(agent, "Hello!")
    async for event in run.stream(agent, "Hello!"):
        print(event)
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from exo.agent import Agent  # pyright: ignore[reportMissingImports]
    from exo.models.provider import ModelProvider  # pyright: ignore[reportMissingImports]
    from exo.swarm import Swarm  # pyright: ignore[reportMissingImports]

from exo._internal.call_runner import call_runner
from exo._internal.message_builder import build_messages
from exo._internal.output_parser import OutputParseError, parse_tool_arguments
from exo._internal.planner import prepare_planned_execution
from exo._internal.run_helpers import (
    check_token_budget_pressure,
    drain_ephemeral_messages,
    resolve_instructions,
)
from exo.agent import TaskLoopAbort, _drain_task_loop_queue  # pyright: ignore[reportMissingImports]
from exo.hooks import HookPoint
from exo.observability.logging import get_logger  # pyright: ignore[reportMissingImports]
from exo.observability.metrics import (  # pyright: ignore[reportMissingImports]
    HAS_OTEL,
    _collector,
    _get_meter,
)
from exo.observability.semconv import (  # pyright: ignore[reportMissingImports]
    METRIC_STREAM_EVENTS_EMITTED,
    STREAM_EVENT_TYPE,
)
from exo.ptc import PTC_TOOL_NAME
from exo.types import (
    AssistantMessage,
    ContextEvent,
    ErrorEvent,
    MCPProgressEvent,
    Message,
    MessageContent,
    MessageInjectedEvent,
    RunResult,
    StatusEvent,
    StepEvent,
    StreamEvent,
    TextEvent,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEvent,
    ToolResult,
    ToolResultEvent,
    Usage,
    UsageEvent,
    UserMessage,
)

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Typed Protocol so IDEs can discover run.sync / run.stream
# ---------------------------------------------------------------------------


class _RunCallable:
    """Public ``run`` entry point: ``run(...)``, ``run.sync(...)``, ``run.stream(...)``.

    Implemented as a callable instance (rather than a function with bolted-on
    attributes) so ``.sync`` and ``.stream`` are real, fully-typed methods —
    IDE auto-complete works and no ``# type: ignore`` is needed.
    """

    async def __call__(
        self,
        agent: Agent | Swarm,
        input: MessageContent,
        *,
        messages: Sequence[Message] | None = None,
        provider: ModelProvider | None = None,
        max_retries: int = 3,
        loop_threshold: int = 3,
    ) -> RunResult:
        return await _run_async(
            agent,
            input,
            messages=messages,
            provider=provider,
            max_retries=max_retries,
            loop_threshold=loop_threshold,
        )

    def sync(
        self,
        agent: Agent | Swarm,
        input: MessageContent,
        *,
        messages: Sequence[Message] | None = None,
        provider: ModelProvider | None = None,
        max_retries: int = 3,
        loop_threshold: int = 3,
    ) -> RunResult:
        return _sync(
            agent,
            input,
            messages=messages,
            provider=provider,
            max_retries=max_retries,
            loop_threshold=loop_threshold,
        )

    def stream(
        self,
        agent: Agent | Swarm,
        input: MessageContent,
        *,
        messages: Sequence[Message] | None = None,
        provider: ModelProvider | None = None,
        max_steps: int | None = None,
        detailed: bool = False,
        event_types: set[str] | None = None,
        conversation_id: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        return _stream(
            agent,
            input,
            messages=messages,
            provider=provider,
            max_steps=max_steps,
            detailed=detailed,
            event_types=event_types,
            conversation_id=conversation_id,
        )


async def _run_async(
    agent: Agent | Swarm,
    input: MessageContent,
    *,
    messages: Sequence[Message] | None = None,
    provider: ModelProvider | None = None,
    max_retries: int = 3,
    loop_threshold: int = 3,
) -> RunResult:
    """Execute an agent (or swarm) and return the result.

    This is the primary async API for running agents.  For a blocking
    variant, use ``run.sync()``.

    If *provider* is ``None``, a default provider is resolved from the
    agent's ``provider_name`` using the model registry (if available).

    Args:
        agent: An ``Agent`` (or ``Swarm``) instance.
        input: User query — a string or list of ContentBlock objects.
        messages: Prior conversation history to continue from.
        provider: LLM provider with ``async complete()`` method.
            When ``None``, auto-resolved from the agent's model string.
        max_retries: Retry attempts for transient LLM errors.
        loop_threshold: Consecutive identical tool-call patterns
            before raising a loop error.

    Returns:
        ``RunResult`` with the agent's output, message history,
        usage stats, and step count.
    """
    resolved_provider = provider or _resolve_provider(agent)
    _log.debug(
        "run() starting agent='%s' provider=%s",
        getattr(agent, "name", "?"),
        type(resolved_provider).__name__ if resolved_provider else None,
    )

    # Detect Swarm: delegate to its own run() method
    if hasattr(agent, "flow_order"):
        return await agent.run(
            input,
            messages=messages,
            provider=resolved_provider,
            max_retries=max_retries,
        )

    return await call_runner(
        agent,
        input,
        messages=messages,
        provider=resolved_provider,
        max_retries=max_retries,
        loop_threshold=loop_threshold,
    )


def _sync(
    agent: Agent | Swarm,
    input: MessageContent,
    *,
    messages: Sequence[Message] | None = None,
    provider: ModelProvider | None = None,
    max_retries: int = 3,
    loop_threshold: int = 3,
) -> RunResult:
    """Execute an agent synchronously (blocking wrapper).

    Calls ``run()`` via ``asyncio.run()``.  This is a convenience for
    scripts and notebooks where an event loop is not already running.

    Args:
        agent: An ``Agent`` (or ``Swarm``) instance.
        input: User query — a string or list of ContentBlock objects.
        messages: Prior conversation history to continue from.
        provider: LLM provider with ``async complete()`` method.
        max_retries: Retry attempts for transient LLM errors.
        loop_threshold: Consecutive identical tool-call patterns
            before raising a loop error.

    Returns:
        ``RunResult`` with the agent's output, message history,
        usage stats, and step count.
    """
    return asyncio.run(
        _run_async(
            agent,
            input,
            messages=messages,
            provider=provider,
            max_retries=max_retries,
            loop_threshold=loop_threshold,
        )
    )


async def _stream(
    agent: Agent | Swarm,
    input: MessageContent,
    *,
    messages: Sequence[Message] | None = None,
    provider: ModelProvider | None = None,
    max_steps: int | None = None,
    detailed: bool = False,
    event_types: set[str] | None = None,
    conversation_id: str | None = None,
) -> AsyncIterator[StreamEvent]:
    """Stream agent execution, yielding events in real-time.

    Uses the provider's ``stream()`` method to deliver text deltas
    as ``TextEvent`` objects and emit ``ToolCallEvent`` for each tool
    invocation. When tool calls are detected, tools are executed and
    the LLM is re-streamed with the results — looping until a
    text-only response or *max_steps* is reached.

    When *detailed* is ``True``, additional event types are emitted:
    ``StatusEvent``, ``StepEvent``, ``UsageEvent``, and
    ``ToolResultEvent``.  ``ErrorEvent`` is emitted on errors
    regardless of the *detailed* flag.

    Args:
        agent: An ``Agent`` instance.
        input: User query string.
        messages: Prior conversation history.
        provider: LLM provider with an ``async stream()`` method.
            When ``None``, auto-resolved from the agent's model string.
        max_steps: Maximum LLM-tool round-trips. Defaults to
            ``agent.max_steps``.
        detailed: When ``True``, emit rich event types (StepEvent,
            UsageEvent, ToolResultEvent, StatusEvent) in addition to
            the default TextEvent and ToolCallEvent.
        event_types: When provided, only events whose ``type`` field
            matches one of the given strings are yielded.  When
            ``None`` (default), all events pass through (respecting
            the *detailed* flag).

    Yields:
        ``TextEvent`` for text chunks and ``ToolCallEvent`` for tool
        invocations.  When *detailed* is ``True``, also yields
        ``StepEvent``, ``UsageEvent``, ``ToolResultEvent``, and
        ``StatusEvent``.  ``ErrorEvent`` is yielded on errors
        regardless of the *detailed* flag.
    """
    resolved = provider or _resolve_provider(agent)

    # Track total events emitted for metrics (only recorded when detailed=True).
    events_emitted: dict[str, int] = {}

    def _passes_filter(event: StreamEvent) -> bool:
        passes = event_types is None or event.type in event_types
        if passes and detailed:
            events_emitted[event.type] = events_emitted.get(event.type, 0) + 1
        return passes

    # Hoist OTel counter creation out of the per-step recording function (L-13)
    _otel_counter = None
    if HAS_OTEL:
        meter = _get_meter()
        _otel_counter = meter.create_counter(
            name=METRIC_STREAM_EVENTS_EMITTED,
            unit="1",
            description="Number of streaming events emitted",
        )

    def _record_stream_metrics() -> None:
        """Record total events emitted during this stream run."""
        if not detailed or not events_emitted:
            return
        for evt_type, count in events_emitted.items():
            attrs: dict[str, str] = {STREAM_EVENT_TYPE: evt_type}
            if _otel_counter is not None:
                _otel_counter.add(count, attrs)
            else:
                _collector.add_counter(METRIC_STREAM_EVENTS_EMITTED, float(count), attrs)

    # Detect Swarm: delegate to its stream() method
    if hasattr(agent, "flow_order"):
        async for event in agent.stream(
            input,
            messages=messages,
            provider=resolved,
            detailed=detailed,
            max_steps=max_steps,
            event_types=event_types,
        ):
            yield event
        return

    if resolved is None:
        from exo.agent import AgentError

        raise AgentError(f"Agent '{agent.name}' requires a provider for stream()")

    input, messages = await prepare_planned_execution(
        agent,
        input,
        messages,
        resolved,
        max_retries=3,
    )

    steps = max_steps if max_steps is not None else agent.max_steps

    # Resolve instructions (may be async callable)
    instr = await resolve_instructions(agent)

    # ---- Memory: load history and persist user input before streaming ----
    history: list[Message] = list(messages) if messages else []
    _persistence = getattr(agent, "_memory_persistence", None)
    _active_conv: str | None = None
    _snapshot_loaded = False
    if _persistence is not None:
        import uuid as _uuid

        _active_conv = conversation_id or getattr(agent, "conversation_id", None)
        if _active_conv is None:
            _active_conv = str(_uuid.uuid4())
            if conversation_id is None:
                agent.conversation_id = _active_conv
        from exo.memory.base import (  # pyright: ignore[reportMissingImports]
            HumanMemory,
            MemoryMetadata,
        )

        _persistence.metadata = MemoryMetadata(
            agent_id=agent.name,
            task_id=_active_conv,
        )

        # ---- Snapshot load ----
        _agent_ctx = getattr(agent, "context", None)
        _snap_cfg = getattr(_agent_ctx, "config", _agent_ctx) if _agent_ctx else None
        if (
            _snap_cfg is not None
            and getattr(
                _snap_cfg, "_enable_snapshots", getattr(_snap_cfg, "enable_snapshots", False)
            )
            and not messages
        ):
            try:
                _snap = await _persistence.load_snapshot(
                    agent_name=agent.name,
                    conversation_id=_active_conv,
                )
                if _snap is not None and await _persistence.is_snapshot_fresh(
                    _snap, agent.name, _active_conv, context_config=_snap_cfg
                ):
                    from exo.memory.snapshot import (  # pyright: ignore[reportMissingImports]
                        deserialize_msg_list,
                    )

                    history = deserialize_msg_list(_snap.content)
                    _snapshot_loaded = True
                    _log.debug(
                        "stream snapshot loaded: agent=%s conversation=%s",
                        agent.name,
                        _active_conv,
                    )
            except Exception:
                _log.warning(
                    "stream snapshot load failed, falling back to raw",
                    exc_info=True,
                )
        # ---- end Snapshot load ----

        if not _snapshot_loaded:
            _db_history = await _persistence.load_history(
                agent_name=agent.name,
                conversation_id=_active_conv,
                rounds=max_steps or agent.max_steps,
            )
            history = list(_db_history) + history

        await _persistence.store.add(
            HumanMemory(
                content=input,
                metadata=_persistence.metadata,
            )
        )
        _log.debug(
            "memory stream pre-run: agent=%s conversation=%s snapshot=%s",
            agent.name,
            _active_conv,
            _snapshot_loaded,
        )
    # ---- end Memory ----

    # Build initial message list
    history.append(UserMessage(content=input))
    msg_list = build_messages(instr, history)

    # ---- Context + Token tracking: resolve model info early (needed by hooks) ----
    _agent_context = getattr(agent, "context", None)
    _agent_name = getattr(agent, "name", "")
    model_name = getattr(agent, "model", "") or ""
    _model_name_only = model_name.partition(":")[2] or model_name
    _stream_context_window: int | None = None
    _stream_token_tracker: Any = None
    _update_system_token_info: Any = None
    if _agent_context is not None:
        try:
            from exo.agent import (  # pyright: ignore[reportMissingImports]
                _get_context_window_tokens,
                _update_system_token_info,
            )
            from exo.context.token_tracker import (
                TokenTracker,  # pyright: ignore[reportMissingImports]
            )

            _stream_context_window = _get_context_window_tokens(_model_name_only)
            _stream_token_tracker = TokenTracker()
        except ImportError:
            pass
    # ---- end Token tracking init ----

    # ---- Context: apply windowing and summarization ----
    # Skip initial windowing when loaded from snapshot.
    if _agent_context is not None and not _snapshot_loaded:
        from exo.agent import _apply_context_windowing  # pyright: ignore[reportMissingImports]

        msg_list, _ctx_actions = await _apply_context_windowing(
            msg_list,
            _agent_context,
            resolved,
            hook_manager=agent.hook_manager,
            agent=agent,
            step=-1,
            max_steps=getattr(agent, "max_steps", 0),
            agent_name=_agent_name,
            model_name=_model_name_only,
            context_window_tokens=_stream_context_window,
        )
        for _ca in _ctx_actions:
            _ev = ContextEvent(
                action=_ca.action,
                agent_name=_agent_name,
                before_count=_ca.before_count,
                after_count=_ca.after_count,
                details=_ca.details,
            )
            if _passes_filter(_ev):
                yield _ev
    # ---- end Context ----

    # ---- Long-term memory: inject relevant knowledge into system message ----
    _agent_memory_lt = getattr(agent, "memory", None)
    if _agent_memory_lt is not None:
        try:
            from exo.agent import (
                _inject_long_term_knowledge,  # pyright: ignore[reportMissingImports]
            )

            msg_list = await _inject_long_term_knowledge(_agent_memory_lt, input, msg_list)
        except ImportError:
            pass
    # ---- end Long-term memory ----

    # Fire START hook (parity with run() path)
    await agent.hook_manager.run(HookPoint.START, agent=agent, input=input)

    if detailed:
        _ev = StatusEvent(
            status="starting",
            agent_name=agent.name,
            message=f"Agent '{agent.name}' starting execution",
        )
        if _passes_filter(_ev):
            yield _ev

    for step_num in range(steps):
        step_started_at = time.time()

        # Re-enumerate tool schemas each step so dynamically added/removed
        # tools (via add_tool/remove_tool) take effect without restarting.
        tool_schemas = agent.get_tool_schemas() or None

        # Augment system message with token context info from previous step
        if _stream_token_tracker is not None and _stream_context_window:
            _traj = _stream_token_tracker.get_trajectory(agent.name)
            if _traj:
                _last_input = _traj[-1].prompt_tokens
                msg_list = _update_system_token_info(msg_list, _last_input, _stream_context_window)  # type: ignore[possibly-undefined]

        # ---- Drain TaskLoopQueue (ABORT / STEER / FOLLOWUP events) ----
        _tlq = getattr(agent, "task_loop_queue", None)
        if _tlq is not None and _tlq:
            try:
                _drain_task_loop_queue(_tlq, msg_list)
            except TaskLoopAbort as _tlq_exc:
                _err_ev = ErrorEvent(
                    error=str(_tlq_exc),
                    error_type="TaskLoopAbort",
                    agent_name=agent.name,
                    step_number=step_num + 1,
                    recoverable=False,
                )
                if _passes_filter(_err_ev):
                    yield _err_ev
                _record_stream_metrics()
                raise

        # ---- Drain injected messages ----
        while not agent._injected_messages.empty():
            try:
                _injected = agent._injected_messages.get_nowait()
                msg_list.append(UserMessage(content=_injected))
                _inj_ev = MessageInjectedEvent(content=_injected, agent_name=agent.name)
                if _passes_filter(_inj_ev):
                    yield _inj_ev
            except asyncio.QueueEmpty:
                break

        # ---- Drain ephemeral messages (visible for this call only) ----
        # Ephemerals are collected into a separate batch and concatenated
        # for the LLM call.  msg_list itself is never mutated, keeping the
        # message history append-only (preserves KV-cache prefix).
        _ephemeral_batch = drain_ephemeral_messages(agent)
        _call_messages = msg_list + _ephemeral_batch if _ephemeral_batch else msg_list

        if detailed:
            _ev = StepEvent(
                step_number=step_num + 1,
                agent_name=agent.name,
                status="started",
                started_at=step_started_at,
            )
            if _passes_filter(_ev):
                yield _ev

        try:
            # Accumulate text and tool call deltas from the stream
            text_parts: list[str] = []
            # dict of index -> accumulated tool call data
            tc_acc: dict[int, dict[str, Any]] = {}
            step_usage = Usage()

            await agent.hook_manager.run(
                HookPoint.PRE_LLM_CALL, agent=agent, messages=_call_messages
            )

            async for chunk in resolved.stream(
                _call_messages,
                tools=tool_schemas,
                temperature=agent.temperature,
                max_tokens=agent.max_tokens,
            ):
                # Yield text deltas
                if chunk.delta:
                    text_parts.append(chunk.delta)
                    _ev = TextEvent(text=chunk.delta, agent_name=agent.name)
                    if _passes_filter(_ev):
                        yield _ev

                # Accumulate tool call deltas
                for tcd in chunk.tool_call_deltas:
                    idx = tcd.index
                    if idx not in tc_acc:
                        tc_acc[idx] = {
                            "id": "",
                            "name": "",
                            "arguments": "",
                            "thought_signature": None,
                        }
                    if tcd.id is not None:
                        tc_acc[idx]["id"] = tcd.id
                    if tcd.name is not None:
                        tc_acc[idx]["name"] = tcd.name
                    tc_acc[idx]["arguments"] += tcd.arguments
                    if getattr(tcd, "thought_signature", None) is not None:
                        tc_acc[idx]["thought_signature"] = tcd.thought_signature

                    # Emit incremental delta event when detailed.
                    # Suppress PTC tool deltas — inner tool events are
                    # emitted transparently by the PTC wrapper instead.
                    _is_ptc = agent.ptc and (
                        tcd.name == PTC_TOOL_NAME
                        or tc_acc.get(idx, {}).get("name") == PTC_TOOL_NAME
                    )
                    if detailed and not _is_ptc:
                        _delta_ev = ToolCallDeltaEvent(
                            index=idx,
                            tool_call_id=tcd.id or "",
                            tool_name=tcd.name or "",
                            arguments_delta=tcd.arguments,
                            agent_name=agent.name,
                        )
                        if _passes_filter(_delta_ev):
                            yield _delta_ev

                # Capture usage from final chunk
                if chunk.usage and chunk.usage.total_tokens > 0:
                    step_usage = Usage(
                        input_tokens=chunk.usage.input_tokens,
                        output_tokens=chunk.usage.output_tokens,
                        total_tokens=chunk.usage.total_tokens,
                    )

            if detailed:
                _ev = UsageEvent(
                    usage=step_usage,
                    agent_name=agent.name,
                    step_number=step_num + 1,
                    model=model_name,
                )
                if _passes_filter(_ev):
                    yield _ev

            # Build completed tool calls
            tool_calls = [
                ToolCall(
                    id=data["id"],
                    name=data["name"],
                    arguments=data["arguments"],
                    thought_signature=data.get("thought_signature"),
                )
                for data in tc_acc.values()
                if data["id"]
            ]

            # Normalize ``default_api.<name>`` tool call names that some
            # models emit directly instead of wrapping them in ``__exo_ptc__``.
            # Without this, the malformed name leaks into the SSE event
            # stream and dispatch fails with "unknown tool".
            if tool_calls:
                from exo.ptc import normalize_default_api_tool_calls

                normalize_default_api_tool_calls(tool_calls, agent)

            full_text = "".join(text_parts)
            from types import SimpleNamespace

            _synth = SimpleNamespace(
                content=full_text,
                tool_calls=tool_calls,
                usage=step_usage,
                finish_reason="tool_calls" if tool_calls else "stop",
            )
            await agent.hook_manager.run(HookPoint.POST_LLM_CALL, agent=agent, response=_synth)

            # Record token usage in tracker
            if _stream_token_tracker is not None and step_usage.total_tokens > 0:
                _stream_token_tracker.add_usage(agent.name, step_usage)

            # No tool calls — done streaming
            if not tool_calls:
                if detailed:
                    _ev = StepEvent(
                        step_number=step_num + 1,
                        agent_name=agent.name,
                        status="completed",
                        started_at=step_started_at,
                        completed_at=time.time(),
                        usage=step_usage,
                    )
                    if _passes_filter(_ev):
                        yield _ev
                    _ev = StatusEvent(
                        status="completed",
                        agent_name=agent.name,
                        message=f"Agent '{agent.name}' completed execution",
                    )
                    if _passes_filter(_ev):
                        yield _ev
                # Fire FINISHED hook (parity with run() path)
                await agent.hook_manager.run(HookPoint.FINISHED, agent=agent, output=full_text)
                # Save context snapshot at end of stream.
                await _save_stream_snapshot(
                    agent,
                    _persistence,
                    _active_conv,
                    msg_list,
                    full_text,
                    tool_calls,
                )
                _record_stream_metrics()
                return

            # Yield ToolCallEvent for each tool call.
            # PTC tool is suppressed — inner tool events emitted via queue.
            for tc in tool_calls:
                if agent.ptc and tc.name == PTC_TOOL_NAME:
                    continue
                _ev = ToolCallEvent(
                    tool_name=tc.name,
                    tool_call_id=tc.id,
                    arguments=tc.arguments,
                    agent_name=agent.name,
                )
                if _passes_filter(_ev):
                    yield _ev

            # Execute tools and feed results back
            try:
                actions = parse_tool_arguments(tool_calls)
            except OutputParseError as exc:
                _log.warning("Failed to parse tool arguments on '%s': %s", agent.name, exc)
                actions = []
                tool_results = [
                    ToolResult(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        error=f"Tool '{tc.name}' error: invalid arguments: {exc}",
                    )
                    for tc in tool_calls
                ]
            else:
                tool_results = []
            tool_exec_start = time.time()
            if actions:
                tool_results = await agent._execute_tools(actions)
            tool_exec_end = time.time()

            # Drain MCP progress queues and yield MCPProgressEvent items.
            # Progress notifications are captured by MCPToolWrapper.execute()
            # into per-wrapper asyncio.Queue instances during _execute_tools().
            # They are yielded here (after all tools complete) so the LLM
            # never sees them — only the caller's async-for loop does.
            for action in actions:
                tool = agent.tools.get(action.tool_name)
                if tool is not None and hasattr(tool, "progress_queue"):
                    q = tool.progress_queue
                    while not q.empty():
                        try:
                            progress_evt: MCPProgressEvent = q.get_nowait()
                            # Stamp agent_name if not already set
                            if not progress_evt.agent_name:
                                progress_evt = MCPProgressEvent(
                                    tool_name=progress_evt.tool_name,
                                    progress=progress_evt.progress,
                                    total=progress_evt.total,
                                    message=progress_evt.message,
                                    agent_name=agent.name,
                                )
                            if _passes_filter(progress_evt):
                                yield progress_evt
                        except Exception:
                            break

            # Drain inner agent events pushed by tools via ToolContext.emit()
            # and PTC inner tool events. ToolResultEvent respects the detailed
            # flag — same contract as the non-PTC ToolResultEvent at line 720.
            _queue_size = agent._event_queue.qsize()
            if _queue_size:
                _log.debug(
                    "draining %d inner events from '%s' event queue",
                    _queue_size,
                    agent.name,
                )
            _drained = 0
            while not agent._event_queue.empty():
                try:
                    inner_event = agent._event_queue.get_nowait()
                    _drained += 1
                    if isinstance(inner_event, ToolResultEvent) and not detailed:
                        continue
                    if _passes_filter(inner_event):
                        yield inner_event
                except Exception:
                    _log.warning(
                        "event queue drain interrupted on '%s' after %d/%d events",
                        agent.name,
                        _drained,
                        _queue_size,
                    )
                    break
            if _queue_size and not _drained:
                _log.warning(
                    "event queue reported %d items but drained 0 on '%s'",
                    _queue_size,
                    agent.name,
                )

            # Emit ToolResultEvent for each tool execution when detailed.
            # PTC tool is suppressed — inner tool events emitted via queue.
            if detailed:
                total_tool_duration_ms = (tool_exec_end - tool_exec_start) * 1000
                per_tool_duration_ms = (
                    total_tool_duration_ms / len(tool_results) if tool_results else 0.0
                )
                for action, tr in zip(actions, tool_results, strict=False):
                    if agent.ptc and tr.tool_name == PTC_TOOL_NAME:
                        continue
                    _ev = ToolResultEvent(
                        tool_name=tr.tool_name,
                        tool_call_id=tr.tool_call_id,
                        arguments=action.arguments,
                        result=tr.content,
                        error=tr.error,
                        success=tr.error is None,
                        duration_ms=per_tool_duration_ms,
                        agent_name=agent.name,
                    )
                    if _passes_filter(_ev):
                        yield _ev

            if detailed:
                _ev = StepEvent(
                    step_number=step_num + 1,
                    agent_name=agent.name,
                    status="completed",
                    started_at=step_started_at,
                    completed_at=time.time(),
                    usage=step_usage,
                )
                if _passes_filter(_ev):
                    yield _ev

            # Append assistant message + tool results to conversation
            msg_list.append(AssistantMessage(content=full_text, tool_calls=tool_calls))
            msg_list.extend(tool_results)

            # Apply context windowing every step (CONTEXT_WINDOW hook fires each turn).
            # Token budget check sets force_summarize for aggressive compression.
            if _agent_context is not None:
                _force_summarize, _fill_ratio, _trigger = check_token_budget_pressure(
                    step_usage.input_tokens if _stream_token_tracker is not None else 0,
                    _stream_context_window,
                    _agent_context,
                )
                if _force_summarize:
                    _log.info(
                        "stream token budget trigger: %.0f%% full (%d/%d tokens) on '%s'",
                        100.0 * _fill_ratio,
                        step_usage.input_tokens,
                        _stream_context_window,
                        agent.name,
                    )
                    _tb_ev = ContextEvent(
                        action="token_budget",
                        agent_name=_agent_name,
                        before_count=len(msg_list),
                        after_count=len(msg_list),
                        details={
                            "fill_ratio": _fill_ratio,
                            "input_tokens": step_usage.input_tokens,
                            "context_window_tokens": _stream_context_window,
                            "trigger": _trigger,
                        },
                    )
                    if _passes_filter(_tb_ev):
                        yield _tb_ev

                from exo.agent import (
                    _apply_context_windowing as _acw,  # pyright: ignore[reportMissingImports]
                )

                msg_list, _step_actions = await _acw(
                    msg_list,
                    _agent_context,
                    resolved,
                    force_summarize=_force_summarize,
                    hook_manager=agent.hook_manager,
                    agent=agent,
                    step=step_num,
                    max_steps=getattr(agent, "max_steps", 0),
                    agent_name=_agent_name,
                    model_name=_model_name_only,
                    context_window_tokens=_stream_context_window,
                    last_usage=step_usage,
                    token_tracker=_stream_token_tracker,
                )
                for _sa in _step_actions:
                    _sa_ev = ContextEvent(
                        action=_sa.action,
                        agent_name=_agent_name,
                        before_count=_sa.before_count,
                        after_count=_sa.after_count,
                        details=_sa.details,
                    )
                    if _passes_filter(_sa_ev):
                        yield _sa_ev

        except Exception as exc:
            _ev = ErrorEvent(
                error=str(exc),
                error_type=type(exc).__name__,
                agent_name=agent.name,
                step_number=step_num + 1,
                recoverable=False,
            )
            if _passes_filter(_ev):
                yield _ev
            if detailed:
                _ev = StatusEvent(
                    status="error",
                    agent_name=agent.name,
                    message=str(exc),
                )
                if _passes_filter(_ev):
                    yield _ev
            # Fire ERROR hook (parity with run() path)
            await agent.hook_manager.run(HookPoint.ERROR, agent=agent, error=exc)
            _record_stream_metrics()
            raise


async def _save_stream_snapshot(
    agent: Any,
    persistence: Any | None,
    conversation_id: str | None,
    msg_list: list[Any],
    final_text: str = "",
    tool_calls: list[Any] | None = None,
) -> None:
    """Save a context snapshot at the end of a stream run.

    Fails silently — snapshot save should never break the stream.
    """
    if persistence is None or conversation_id is None:
        return
    _ctx = getattr(agent, "context", None)
    if _ctx is None:
        return
    _cfg = getattr(_ctx, "config", _ctx)
    if not getattr(_cfg, "_enable_snapshots", getattr(_cfg, "enable_snapshots", False)):
        return
    try:
        from exo.types import AssistantMessage  # pyright: ignore[reportMissingImports]

        snap_list = list(msg_list)
        if final_text:
            snap_list.append(AssistantMessage(content=final_text, tool_calls=tool_calls or []))
        await persistence.save_snapshot(
            agent_name=agent.name,
            conversation_id=conversation_id,
            msg_list=snap_list,
            context_config=_cfg,
        )
    except Exception:
        _log.warning("stream snapshot save failed", exc_info=True)


def _resolve_provider(agent: Agent | Swarm) -> ModelProvider | None:
    """Attempt to auto-resolve a provider from the agent's model config.

    Tries the model registry from ``exo.models`` if available.
    For Swarms (which lack a ``.model`` attribute), resolves from the
    first agent in the flow order.

    Returns ``None`` when the agent has no resolvable model (e.g. a
    bare Swarm without per-agent models).  Raises ``AgentError`` when
    a provider name is present but the provider is not registered.
    """
    try:
        from exo.models.provider import get_provider  # pyright: ignore[reportMissingImports]
    except ImportError:
        # exo-models not installed — cannot auto-resolve
        return None

    model = getattr(agent, "model", None)
    if model is None and hasattr(agent, "agents"):
        # Swarm: resolve from the first agent's model
        first = (
            next(iter(agent.agents.values()), None)
            if isinstance(agent.agents, dict)
            else (agent.agents[0] if agent.agents else None)
        )
        if first is not None:
            model = first.model

    if model is None:
        # No model configured — legitimate "return None" path
        return None

    try:
        return get_provider(model)
    except Exception as exc:
        # Distinguish "provider name unknown" (registry lookup failure, raises ModelError
        # with 'not registered' text) from "provider initialisation failed" (e.g. missing
        # API key — an OpenAIError raised inside cls(config)).  Only the former should be
        # surfaced immediately; the latter is a credential issue that the caller's error
        # path will handle after provider=None is returned.
        exc_type_name = type(exc).__name__
        if exc_type_name == "ModelError" or "not registered" in str(exc):
            from exo.agent import AgentError  # pyright: ignore[reportMissingImports]

            raise AgentError(str(exc)) from exc

        _log.warning(
            "Failed to auto-resolve provider for model '%s': %s",
            getattr(agent, "model", "?"),
            exc,
        )
        return None


# The public entry point: a single callable instance exposing run(...),
# run.sync(...) and run.stream(...) as real, typed methods.
run: _RunCallable = _RunCallable()

__all__ = ["run"]
