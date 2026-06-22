"""Agent-lifecycle tracer: turns exo hooks into a nested OTel span tree.

``install_tracing(agent, spec)`` resolves the requested backends, registers
their span processors on the (global) OpenTelemetry ``TracerProvider`` once, and
attaches a :class:`Tracer` to the agent's hook manager. The Tracer opens/closes
spans on the lifecycle hooks so a run produces::

    agent.run {name}                 (START .. FINISHED/ERROR)
      |- gen_ai.chat                 (PRE_LLM_CALL .. POST_LLM_CALL)
      |- tool.execute {tool}         (PRE_TOOL_CALL .. POST_TOOL_CALL)
      |- context.compress            (CONTEXT_WINDOW, one-shot)

Nesting (sub-agents, parallel tools) works via the OpenTelemetry context: each
span is ``attach``-ed as current when opened, so children — including nested
agent runs and tool calls running in their own asyncio tasks — parent correctly.
A per-context frame stack (an immutable tuple in a ContextVar) tracks which span
to close on the matching close-hook; treating it immutably keeps parallel tasks
(which copy the context at creation) isolated.

When OpenTelemetry is not installed, ``install_tracing`` is a no-op.
"""

from __future__ import annotations

import logging
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from exo.observability import semconv  # pyright: ignore[reportMissingImports]
from exo.observability.backends import (  # pyright: ignore[reportMissingImports]
    resolve_backends,
)
from exo.observability.cost import CostTracker  # pyright: ignore[reportMissingImports]
from exo.observability.tracing import (  # pyright: ignore[reportMissingImports]
    HAS_OTEL,
    _get_tracer,
)

_log = logging.getLogger(__name__)

if HAS_OTEL:
    from opentelemetry import context as _otel_context
    from opentelemetry import trace as _otel_trace
    from opentelemetry.trace import Status, StatusCode


@dataclass
class _Frame:
    """An open span awaiting its matching close-hook."""

    kind: str  # "agent" | "llm" | "tool"
    span: Any
    token: Any  # OTel context-attach token
    start: float


# Immutable frame stack, per OTel/asyncio context. Treated immutably (each
# push/pop replaces the tuple) so child tasks that copy the context stay isolated.
_stack: ContextVar[tuple[_Frame, ...]] = ContextVar("exo_tracer_stack", default=())


def _truncate(value: Any, max_chars: int) -> Any:
    if isinstance(value, str) and len(value) > max_chars:
        return value[:max_chars] + "...[truncated]"
    return value


class Tracer:
    """Registers OTel-span callbacks on an agent's hook manager."""

    def __init__(
        self,
        *,
        attribute_max_chars: int = 4000,
        tracer_name: str = "exo",
        provider: Any = None,
    ) -> None:
        self.attribute_max_chars = attribute_max_chars
        self.tracer_name = tracer_name
        self.cost = CostTracker()  # accumulates run cost; per-call cost rides the span
        # Bind to the concrete provider's tracer so spans are emitted to the
        # configured exporter regardless of global tracer-provider state.
        self._otel_tracer = provider.get_tracer(tracer_name) if provider is not None else None

    # -- public ----------------------------------------------------------------

    def install(self, agent: Any, backends: Any = None) -> None:
        """Attach span callbacks to *agent*'s hook manager (no-op without OTel)."""
        if not HAS_OTEL:
            return
        from exo.hooks import HookPoint  # local: exo-core present at call time

        hm = agent.hook_manager
        hm.add(HookPoint.START, self._on_start)
        hm.add(HookPoint.FINISHED, self._on_finished)
        hm.add(HookPoint.ERROR, self._on_error)
        hm.add(HookPoint.PRE_LLM_CALL, self._on_pre_llm)
        hm.add(HookPoint.POST_LLM_CALL, self._on_post_llm)
        hm.add(HookPoint.PRE_TOOL_CALL, self._on_pre_tool)
        hm.add(HookPoint.POST_TOOL_CALL, self._on_post_tool)
        hm.add(HookPoint.CONTEXT_WINDOW, self._on_context_window)

    # -- span stack helpers ----------------------------------------------------

    def _clean(self, attrs: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in attrs.items():
            if value is None:
                continue
            if isinstance(value, str | bool | int | float):
                out[key] = _truncate(value, self.attribute_max_chars)
            else:
                out[key] = _truncate(str(value), self.attribute_max_chars)
        return out

    def _open(self, kind: str, name: str, attrs: dict[str, Any]) -> None:
        tracer = self._otel_tracer or _get_tracer(self.tracer_name)
        span = tracer.start_span(name, attributes=self._clean(attrs))
        token = _otel_context.attach(_otel_trace.set_span_in_context(span))
        frame = _Frame(kind=kind, span=span, token=token, start=time.monotonic())
        _stack.set((*_stack.get(), frame))

    def _end_frame(
        self, frame: _Frame, *, ok: bool, exception: BaseException | None = None
    ) -> None:
        span = frame.span
        try:
            if exception is not None:
                span.record_exception(exception)
            span.set_status(Status(StatusCode.OK if ok else StatusCode.ERROR))
        finally:
            span.end()
            _otel_context.detach(frame.token)

    def _pop(self, kind: str) -> _Frame | None:
        stack = _stack.get()
        if not stack:
            return None
        top = stack[-1]
        _stack.set(stack[:-1])
        if top.kind != kind:
            _log.debug("tracer stack mismatch: expected %r, got %r", kind, top.kind)
        return top

    def _close_until(self, kind: str, *, exception: BaseException | None = None) -> None:
        """Unwind frames LIFO until (and including) the nearest *kind* frame."""
        while True:
            stack = _stack.get()
            if not stack:
                return
            top = stack[-1]
            _stack.set(stack[:-1])
            is_target = top.kind == kind
            self._end_frame(
                top,
                ok=exception is None,
                exception=exception if is_target else None,
            )
            if is_target:
                return

    # -- hook callbacks --------------------------------------------------------
    # IMPORTANT: every callback is wrapped in ``try/except Exception`` so that
    # an OTel SDK bug or exporter error never aborts the user's agent run.
    # Observability MUST fail-open: a broken tracing backend is a warning, not
    # a crash. CancelledError is not an ``Exception``, so it propagates normally.

    async def _on_start(self, *, agent: Any = None, input: Any = None, **_: Any) -> None:
        try:
            name = getattr(agent, "name", "agent")
            attrs = {
                semconv.AGENT_NAME: name,
                semconv.AGENT_MODEL: getattr(agent, "model", None),
                semconv.GEN_AI_OPERATION_NAME: "agent",
            }
            self._open("agent", f"agent.run {name}", attrs)
        except Exception:
            _log.warning(
                "tracer._on_start failed (tracing degraded); agent run continues",
                exc_info=True,
            )

    async def _on_finished(self, *, agent: Any = None, output: Any = None, **_: Any) -> None:
        try:
            self._close_until("agent")
        except Exception:
            _log.warning(
                "tracer._on_finished failed (tracing degraded); agent run continues",
                exc_info=True,
            )

    async def _on_error(self, *, agent: Any = None, error: Any = None, **_: Any) -> None:
        try:
            exc = error if isinstance(error, BaseException) else None
            self._close_until("agent", exception=exc)
        except Exception:
            _log.warning(
                "tracer._on_error failed (tracing degraded); agent run continues",
                exc_info=True,
            )

    async def _on_pre_llm(self, *, agent: Any = None, messages: Any = None, **_: Any) -> None:
        try:
            attrs = {
                semconv.GEN_AI_OPERATION_NAME: "chat",
                semconv.GEN_AI_REQUEST_MODEL: getattr(agent, "model", None),
            }
            self._open("llm", "gen_ai.chat", attrs)
        except Exception:
            _log.warning(
                "tracer._on_pre_llm failed (tracing degraded); agent run continues",
                exc_info=True,
            )

    async def _on_post_llm(self, *, agent: Any = None, response: Any = None, **_: Any) -> None:
        try:
            # Peek at the top frame to annotate it before closing. We use _pop
            # to get the frame, then _end_frame so the OTel token is always
            # detached from the correct frame — kind mismatches are handled by
            # _close_until which unwinds until the "llm" frame is found.
            stack = _stack.get()
            if not stack:
                return
            # Find the nearest "llm" frame and annotate it before closing.
            llm_frame: _Frame | None = None
            for frame in reversed(stack):
                if frame.kind == "llm":
                    llm_frame = frame
                    break
            if llm_frame is None:
                return
            span = llm_frame.span
            if response is not None:
                usage = getattr(response, "usage", None)
                if usage is not None:
                    input_tokens = getattr(usage, "input_tokens", 0)
                    output_tokens = getattr(usage, "output_tokens", 0)
                    total_tokens = input_tokens + output_tokens
                    span.set_attribute(semconv.GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
                    span.set_attribute(semconv.GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)
                    span.set_attribute(semconv.GEN_AI_USAGE_TOTAL_TOKENS, total_tokens)
                finish_reason = getattr(response, "finish_reason", None)
                if finish_reason:
                    span.set_attribute(semconv.GEN_AI_RESPONSE_FINISH_REASONS, str(finish_reason))
                resp_model = getattr(response, "model", None)
                if resp_model:
                    span.set_attribute(semconv.GEN_AI_RESPONSE_MODEL, resp_model)
                # Estimate + stamp cost (vendor-neutral; rides the span to every backend).
                if usage is not None:
                    model = resp_model or getattr(agent, "model", "")
                    cost = self.cost.add(model, usage)
                    if cost.priced:
                        span.set_attribute(semconv.COST_INPUT_TOKENS, cost.input_tokens)
                        span.set_attribute(semconv.COST_OUTPUT_TOKENS, cost.output_tokens)
                        span.set_attribute(semconv.COST_TOTAL_USD, cost.total_usd)
            # Close (and correctly detach tokens) up to and including the "llm" frame.
            self._close_until("llm")
        except Exception:
            _log.warning(
                "tracer._on_post_llm failed (tracing degraded); agent run continues",
                exc_info=True,
            )

    async def _on_pre_tool(
        self, *, agent: Any = None, tool_name: str = "", arguments: Any = None, **_: Any
    ) -> None:
        try:
            attrs = {
                semconv.TOOL_NAME: tool_name,
                semconv.TOOL_ARGUMENTS: None if arguments is None else str(arguments),
            }
            self._open("tool", f"tool.execute {tool_name}", attrs)
        except Exception:
            _log.warning(
                "tracer._on_pre_tool failed (tracing degraded); agent run continues",
                exc_info=True,
            )

    async def _on_post_tool(
        self, *, agent: Any = None, tool_name: str = "", result: Any = None, **_: Any
    ) -> None:
        try:
            stack = _stack.get()
            if not stack:
                return
            # Find the nearest "tool" frame and annotate it before closing.
            tool_frame: _Frame | None = None
            for frame in reversed(stack):
                if frame.kind == "tool":
                    tool_frame = frame
                    break
            if tool_frame is None:
                return
            span = tool_frame.span
            error = getattr(result, "error", None)
            if result is not None:
                success = error is None
                span.set_attribute(semconv.TOOL_STEP_SUCCESS, success)
                content = getattr(result, "content", None)
                if content is not None:
                    span.set_attribute(semconv.TOOL_RESULT, str(content))
                if error:
                    span.set_attribute(semconv.TOOL_ERROR, str(error))
            span.set_attribute(semconv.TOOL_DURATION, time.monotonic() - tool_frame.start)
            # Close (and correctly detach tokens) up to and including the "tool" frame.
            self._close_until("tool", exception=None if error is None else RuntimeError(str(error)))
        except Exception:
            _log.warning(
                "tracer._on_post_tool failed (tracing degraded); agent run continues",
                exc_info=True,
            )

    async def _on_context_window(
        self,
        *,
        agent: Any = None,
        messages: Any = None,
        budget: Any = None,
        current_fill_ratio: Any = None,
        **_: Any,
    ) -> None:
        try:
            # One-shot span (no matching close hook); parents under the current span.
            tracer = self._otel_tracer or _get_tracer(self.tracer_name)
            attrs = self._clean(
                {
                    "exo.context.budget": budget,
                    "exo.context.fill_ratio": current_fill_ratio,
                    "exo.context.message_count": len(messages) if messages is not None else None,
                }
            )
            span = tracer.start_span("context.compress", attributes=attrs)
            span.end()
        except Exception:
            _log.warning(
                "tracer._on_context_window failed (tracing degraded); agent run continues",
                exc_info=True,
            )


# ---------------------------------------------------------------------------
# Provider wiring + install entry point
# ---------------------------------------------------------------------------

_provider_lock = threading.Lock()
_installed_backends: set[str] = set()


def _ensure_provider(backends: list[Any], *, sample_rate: float = 1.0) -> Any:
    """Idempotently ensure a real TracerProvider exists with *backends* attached.

    Returns the configured ``TracerProvider`` so the Tracer can bind to it
    directly (rather than resolving the global at span time, which is fragile
    when other code has installed a proxy/stale global provider).

    ``sample_rate`` (0.0-1.0) is wired to a ``ParentBased(TraceIdRatioBased(...))``
    sampler when creating a new provider.  When an existing provider is reused,
    the sampler is left as-is (it was already configured on first call).
    """
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.sampling import (  # pyright: ignore[reportMissingImports]
        ALWAYS_ON,
        ParentBased,
        TraceIdRatioBased,
    )

    with _provider_lock:
        current = _otel_trace.get_tracer_provider()
        if isinstance(current, TracerProvider):
            provider = current
        else:
            if sample_rate >= 1.0:
                sampler = ALWAYS_ON
            elif sample_rate <= 0.0:
                from opentelemetry.sdk.trace.sampling import (  # pyright: ignore[reportMissingImports]
                    ALWAYS_OFF,
                )

                sampler = ALWAYS_OFF
            else:
                sampler = ParentBased(TraceIdRatioBased(sample_rate))
            provider = TracerProvider(sampler=sampler)
            _otel_trace.set_tracer_provider(provider)
        for backend in backends:
            if backend.name in _installed_backends:
                continue
            processor = backend.build_span_processor()
            if processor is not None:
                provider.add_span_processor(processor)
                _installed_backends.add(backend.name)
        return provider


def install_tracing(
    agent: Any,
    spec: Any = None,
    *,
    attribute_max_chars: int = 4000,
    sample_rate: float = 1.0,
) -> Tracer | None:
    """Resolve *spec*, wire backends onto the provider, attach a Tracer to *agent*.

    Returns the installed :class:`Tracer`, or None when OTel is absent or no
    backend resolved (e.g. tracing disabled / nothing auto-detected).

    Args:
        agent: The agent to attach tracing hooks to.
        spec: Tracing backend spec (string, list, bool, TracingConfig, or None).
        attribute_max_chars: Truncate string span attributes to this length.
        sample_rate: Sampling probability in [0.0, 1.0].  1.0 = always sample
            (default), 0.0 = never sample.  Wired to a
            ``ParentBased(TraceIdRatioBased(sample_rate))`` sampler on the new
            TracerProvider.  Ignored when an existing provider is already
            installed (the sampler is set at provider construction time).
    """
    if not HAS_OTEL:
        return None
    backends = resolve_backends(spec)
    if not backends:
        return None
    # Extract sample_rate from TracingConfig if spec carries one.
    if hasattr(spec, "sample_rate") and sample_rate == 1.0:
        sample_rate = float(spec.sample_rate)
    provider = _ensure_provider(backends, sample_rate=sample_rate)
    tracer = Tracer(attribute_max_chars=attribute_max_chars, provider=provider)
    tracer.install(agent, backends)
    return tracer


def _reset_for_tests() -> None:
    """Clear the installed-backend de-dupe set and reset the global TracerProvider (tests only)."""
    with _provider_lock:
        _installed_backends.clear()
        # Reset the global OTel TracerProvider so each test gets a fresh one.
        # This prevents processors from accumulating across tests.
        try:
            import opentelemetry.trace as _trace

            _trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
            _trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
        except Exception:
            pass  # Best-effort; OTel internals may vary across versions.


__all__ = ["Tracer", "install_tracing"]
