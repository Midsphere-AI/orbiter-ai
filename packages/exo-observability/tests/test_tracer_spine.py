"""Tests for the OTel tracing spine (Tracer + hook wiring + backend resolution).

Span-tree assertions use a real in-process OTel SDK provider via the built-in
``"memory"`` backend's ``InMemorySpanExporter``; they skip gracefully when
``opentelemetry-sdk`` is not installed. Backend-resolution tests need no OTel.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest

from exo import Agent, run, tool
from exo.types import Usage

# ---------------------------------------------------------------------------
# resolve_backends — no OTel SDK required
# ---------------------------------------------------------------------------


class TestResolveBackends:
    def test_off_disables(self) -> None:
        from exo.observability.backends import resolve_backends

        assert resolve_backends("off") == []
        assert resolve_backends(False) == []

    def test_explicit_otlp_when_exporter_missing(self) -> None:
        """'otlp' resolves only when the OTLP exporter extra is importable."""
        from exo.observability.backends import resolve_backends
        from exo.observability.backends.base import OTLPBackend

        backends = resolve_backends("otlp")
        # Either available (exporter installed) -> one backend, or skipped.
        if backends:
            assert isinstance(backends[0], OTLPBackend)

    def test_env_off_beats_autodetect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from exo.observability.backends import resolve_backends

        monkeypatch.setenv("EXO_TRACING", "off")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        assert resolve_backends(None) == []

    def test_autodetect_no_keys_is_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from exo.observability.backends import resolve_backends

        for key in (
            "EXO_TRACING",
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_SECRET_KEY",
            "LANGSMITH_API_KEY",
            "LANGCHAIN_API_KEY",
            "PHOENIX_API_KEY",
            "PHOENIX_COLLECTOR_ENDPOINT",
            "BRAINTRUST_API_KEY",
            "OTEL_EXPORTER_OTLP_ENDPOINT",
        ):
            monkeypatch.delenv(key, raising=False)
        assert resolve_backends(None) == []

    def test_unknown_backend_skipped(self) -> None:
        from exo.observability.backends import resolve_backends

        # A name with no registered/importable backend resolves to nothing.
        assert resolve_backends("definitely-not-a-backend") == []


# ---------------------------------------------------------------------------
# Span tree — requires opentelemetry-sdk
# ---------------------------------------------------------------------------

pytest.importorskip("opentelemetry.sdk.trace")


class _FakeToolCallDelta:
    def __init__(
        self, index: int = 0, id: str | None = None, name: str | None = None, arguments: str = ""
    ) -> None:
        self.index = index
        self.id = id
        self.name = name
        self.arguments = arguments


class _FakeStreamChunk:
    def __init__(
        self,
        delta: str = "",
        tool_call_deltas: list[Any] | None = None,
        finish_reason: str | None = None,
        usage: Usage | None = None,
    ) -> None:
        self.delta = delta
        self.tool_call_deltas = tool_call_deltas or []
        self.finish_reason = finish_reason
        self.usage = usage or Usage()


def _stream_provider(rounds: list[list[_FakeStreamChunk]]) -> Any:
    call_count = 0

    async def stream(messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
        nonlocal call_count
        chunks = rounds[min(call_count, len(rounds) - 1)]
        call_count += 1
        for c in chunks:
            yield c

    mock = AsyncMock()
    mock.stream = stream
    mock.complete = AsyncMock()
    return mock


@pytest.fixture
def memory_spans() -> Any:
    """Yield a fresh in-memory span backend on an isolated, live OTel provider.

    Other test files reset ``_TRACER_PROVIDER`` and/or shut their provider down,
    which leaves the module-level ``get_tracer`` resolving to a stale no-op
    ProxyTracer. We reset OTel's set-once and explicitly install a fresh real
    ``TracerProvider`` so ``install_tracing`` reuses it and span emission
    actually reaches the in-memory exporter.
    """
    from opentelemetry import trace as _trace

    from exo.observability.backends import get_memory_backend
    from exo.observability.tracer import _reset_for_tests

    # Other test files leave a shut-down or proxy global provider behind. Null it
    # so install_tracing builds a fresh live provider; the Tracer binds directly
    # to that provider (not the global), so spans reach the in-memory exporter
    # regardless of OTel's set-once state.
    _trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
    _trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    _reset_for_tests()
    backend = get_memory_backend()
    backend.clear()
    yield backend
    backend.clear()


def _by_name(spans: Any) -> dict[str, Any]:
    return {s.name: s for s in spans}


class TestSpanTree:
    async def test_full_nested_tree(self, memory_spans: Any) -> None:
        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hi {name}!"

        agent = Agent(name="bot", tools=[greet], tracing="memory")

        usage = Usage(input_tokens=10, output_tokens=5, total_tokens=15)
        round1 = [
            _FakeStreamChunk(
                tool_call_deltas=[_FakeToolCallDelta(index=0, id="tc1", name="greet")],
            ),
            _FakeStreamChunk(
                tool_call_deltas=[_FakeToolCallDelta(index=0, arguments='{"name":"Ann"}')],
                finish_reason="tool_calls",
                usage=usage,
            ),
        ]
        round2 = [_FakeStreamChunk(delta="done", finish_reason="stop", usage=usage)]

        provider = _stream_provider([round1, round2])
        async for _ in run.stream(agent, "hi", provider=provider):
            pass

        spans = memory_spans.spans
        names = [s.name for s in spans]
        assert "agent.run bot" in names
        assert "gen_ai.chat" in names
        assert "tool.execute greet" in names

        by_name = _by_name(spans)
        root = by_name["agent.run bot"]
        assert root.parent is None  # root span

        # children parent to the root
        tool_span = by_name["tool.execute greet"]
        assert tool_span.parent is not None
        assert tool_span.parent.span_id == root.context.span_id
        assert tool_span.attributes.get("exo.tool.name") == "greet"

        # at least one llm span carries token usage
        chat_spans = [s for s in spans if s.name == "gen_ai.chat"]
        assert any(s.attributes.get("gen_ai.usage.input_tokens") == 10 for s in chat_spans)
        for s in chat_spans:
            assert s.parent is not None
            assert s.parent.span_id == root.context.span_id

    async def test_error_sets_error_status(self, memory_spans: Any) -> None:
        from opentelemetry.trace import StatusCode

        agent = Agent(name="boom", tracing="memory")

        async def failing_stream(messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
            raise RuntimeError("provider exploded")
            yield  # pragma: no cover - makes this an async generator

        provider = AsyncMock()
        provider.stream = failing_stream

        with pytest.raises(RuntimeError):
            async for _ in run.stream(agent, "hi", provider=provider):
                pass

        root = _by_name(memory_spans.spans).get("agent.run boom")
        assert root is not None
        assert root.status.status_code == StatusCode.ERROR

    async def test_disabled_emits_no_spans(self, memory_spans: Any, monkeypatch: Any) -> None:
        for key in ("EXO_TRACING", "OTEL_EXPORTER_OTLP_ENDPOINT"):
            monkeypatch.delenv(key, raising=False)
        agent = Agent(name="quiet")  # no tracing= and no env -> off
        assert agent._tracer is None

        provider = _stream_provider([[_FakeStreamChunk(delta="ok", finish_reason="stop")]])
        async for _ in run.stream(agent, "hi", provider=provider):
            pass

        assert memory_spans.spans == ()


def test_no_env_means_no_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare Agent() with no tracing arg and no env stays untraced."""
    for key in (
        "EXO_TRACING",
        "LANGFUSE_PUBLIC_KEY",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "BRAINTRUST_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    assert os.environ.get("EXO_TRACING") is None
    agent = Agent(name="plain")
    assert agent._tracer is None
