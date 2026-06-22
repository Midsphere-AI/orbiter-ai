"""Error-DX / resilience tests for exo-observability (Wave B).

Covers:
- P0 fail-open: a tracer hook callback that raises must NOT abort the agent run.
- P0 CancelledError: asyncio.CancelledError must still propagate through hooks.
- P1 taxonomy: ExoError subclasses at every package-boundary raise site.
- P2 misc: HealthResult message carries exception text; braintrust exc_info passthru.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exo.types import ExoError  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Helpers: minimal fake OTel span that explodes on every call
# ---------------------------------------------------------------------------


class _ExplodingSpan:
    """An OTel span-like object that raises RuntimeError on every method call."""

    def start_span(self, *a: Any, **kw: Any) -> _ExplodingSpan:
        raise RuntimeError("simulated OTel SDK failure")

    def set_attribute(self, *a: Any, **kw: Any) -> None:
        raise RuntimeError("simulated OTel SDK failure")

    def set_status(self, *a: Any, **kw: Any) -> None:
        raise RuntimeError("simulated OTel SDK failure")

    def record_exception(self, *a: Any, **kw: Any) -> None:
        raise RuntimeError("simulated OTel SDK failure")

    def end(self, *a: Any, **kw: Any) -> None:
        raise RuntimeError("simulated OTel SDK failure")


# ---------------------------------------------------------------------------
# P0 — Fail-open: broken OTel SDK does not crash hook callbacks
# ---------------------------------------------------------------------------


class TestTracerFailOpen:
    """A Tracer whose _open/_end_frame helpers raise must log warnings but not propagate."""

    @pytest.fixture
    def tracer(self) -> Any:
        pytest.importorskip("opentelemetry.sdk.trace")
        from exo.observability.tracer import Tracer  # pyright: ignore[reportMissingImports]

        return Tracer(tracer_name="exo-test")

    async def test_on_start_swallows_otel_error(self, tracer: Any) -> None:
        """_on_start: OTel SDK failure is caught; no exception propagates."""
        with patch.object(tracer, "_open", side_effect=RuntimeError("boom")):
            # Must not raise
            await tracer._on_start(agent=MagicMock(name="bot", model="openai:gpt-4o"))

    async def test_on_finished_swallows_otel_error(self, tracer: Any) -> None:
        with patch.object(tracer, "_close_until", side_effect=RuntimeError("boom")):
            await tracer._on_finished(agent=MagicMock())

    async def test_on_error_swallows_otel_error(self, tracer: Any) -> None:
        with patch.object(tracer, "_close_until", side_effect=RuntimeError("boom")):
            await tracer._on_error(agent=MagicMock(), error=ValueError("agent error"))

    async def test_on_pre_llm_swallows_otel_error(self, tracer: Any) -> None:
        with patch.object(tracer, "_open", side_effect=RuntimeError("boom")):
            await tracer._on_pre_llm(agent=MagicMock(model="openai:gpt-4o"))

    async def test_on_post_llm_swallows_otel_error(self, tracer: Any) -> None:
        with patch.object(tracer, "_pop", side_effect=RuntimeError("boom")):
            await tracer._on_post_llm(agent=MagicMock(), response=None)

    async def test_on_pre_tool_swallows_otel_error(self, tracer: Any) -> None:
        with patch.object(tracer, "_open", side_effect=RuntimeError("boom")):
            await tracer._on_pre_tool(tool_name="search", arguments={"q": "x"})

    async def test_on_post_tool_swallows_otel_error(self, tracer: Any) -> None:
        with patch.object(tracer, "_pop", side_effect=RuntimeError("boom")):
            await tracer._on_post_tool(tool_name="search", result=None)

    async def test_on_context_window_swallows_otel_error(self, tracer: Any) -> None:
        with patch.object(tracer, "_clean", side_effect=RuntimeError("boom")):
            await tracer._on_context_window(
                agent=MagicMock(), messages=[], budget=4000, current_fill_ratio=0.5
            )

    async def test_warning_is_logged_on_otel_failure(
        self, tracer: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A suppressed OTel error must produce a WARNING log (not silently vanish)."""
        import logging

        with (
            caplog.at_level(logging.WARNING, logger="exo.observability.tracer"),
            patch.object(tracer, "_open", side_effect=RuntimeError("sdk-boom")),
        ):
            await tracer._on_start(agent=MagicMock(name="bot", model="openai:gpt-4o"))

        assert any("tracing degraded" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# P0 — CancelledError must NOT be swallowed
# ---------------------------------------------------------------------------


class TestTracerCancelledError:
    """asyncio.CancelledError is a BaseException — it must propagate through hook callbacks."""

    @pytest.fixture
    def tracer(self) -> Any:
        pytest.importorskip("opentelemetry.sdk.trace")
        from exo.observability.tracer import Tracer  # pyright: ignore[reportMissingImports]

        return Tracer(tracer_name="exo-test")

    async def test_cancelled_error_propagates_from_on_start(self, tracer: Any) -> None:
        with (
            patch.object(tracer, "_open", side_effect=asyncio.CancelledError()),
            pytest.raises(asyncio.CancelledError),
        ):
            await tracer._on_start(agent=MagicMock(name="bot"))

    async def test_cancelled_error_propagates_from_on_pre_llm(self, tracer: Any) -> None:
        with (
            patch.object(tracer, "_open", side_effect=asyncio.CancelledError()),
            pytest.raises(asyncio.CancelledError),
        ):
            await tracer._on_pre_llm(agent=MagicMock())

    async def test_cancelled_error_propagates_from_on_pre_tool(self, tracer: Any) -> None:
        with (
            patch.object(tracer, "_open", side_effect=asyncio.CancelledError()),
            pytest.raises(asyncio.CancelledError),
        ):
            await tracer._on_pre_tool(tool_name="t")


# ---------------------------------------------------------------------------
# P0 — End-to-end: a live agent run survives a broken tracer backend
# ---------------------------------------------------------------------------


class TestAgentRunSurvivesBrokenTracer:
    """Verifies the highest-level claim: a broken OTel exporter must not abort a run."""

    async def test_agent_run_survives_exploding_otel_tracer(self) -> None:
        """An agent run must complete even when every OTel span method raises.

        Uses the same streaming helper pattern as test_tracer_spine.py so the
        response object satisfies the memory/persistence hook's AIMemory schema.
        """
        pytest.importorskip("opentelemetry.sdk.trace")

        from exo import Agent, run  # pyright: ignore[reportMissingImports]
        from exo.types import Usage  # pyright: ignore[reportMissingImports]

        # Build an agent with tracing wired in (memory backend = in-process, no network).
        agent = Agent(name="survivor", tracing="memory")

        # Replace the tracer's internal OTel tracer object with one that explodes
        # so every span method (start_span, set_attribute, end, …) raises.
        if agent._tracer is not None:
            agent._tracer._otel_tracer = _ExplodingSpan()  # type: ignore[assignment]

        # Use the exact same streaming chunk shape that test_tracer_spine.py uses
        # so the accumulated response object satisfies AIMemory(content=str).
        usage = Usage(input_tokens=1, output_tokens=1, total_tokens=2)

        class _Chunk:
            delta = "hello"
            finish_reason: str | None = "stop"

            def __init__(self) -> None:
                self.usage = usage
                self.tool_call_deltas: list[Any] = []

        async def _stream(messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
            yield _Chunk()

        provider = AsyncMock()
        provider.stream = _stream

        # Consume via run.stream (same as test_tracer_spine.py) — this assembles
        # the response through the framework's streaming path which produces a
        # properly-typed content string for the POST_LLM_CALL hook.
        collected: list[str] = []
        async for event in run.stream(agent, "hi", provider=provider):
            from exo import TextEvent  # pyright: ignore[reportMissingImports]

            if isinstance(event, TextEvent):
                collected.append(event.text)

        # The run must produce output and must not have raised.
        assert "hello" in "".join(collected)


# ---------------------------------------------------------------------------
# P1 — Taxonomy: ExoError at every package-boundary raise site
# ---------------------------------------------------------------------------


class TestExoErrorTaxonomy:
    def test_resolve_backends_unsupported_spec_is_exoerror(self) -> None:
        from exo.observability.backends import (  # pyright: ignore[reportMissingImports]
            resolve_backends,
        )

        with pytest.raises(ExoError) as exc_info:
            resolve_backends(42)  # type: ignore[arg-type]

        err = exc_info.value
        assert err.hint is not None
        assert "TracingConfig" in err.hint or "string" in err.hint
        assert err.context.get("spec_type") == "int"

    def test_health_run_unknown_is_exoerror(self) -> None:
        from exo.observability.health import HealthRegistry  # pyright: ignore[reportMissingImports]

        registry = HealthRegistry()
        with pytest.raises(ExoError) as exc_info:
            registry.run("ghost")

        err = exc_info.value
        assert err.hint is not None
        assert err.context.get("check") == "ghost"

    def test_obs_config_bad_logging_type_is_exoerror(self) -> None:
        from exo.observability.config import (  # pyright: ignore[reportMissingImports]
            ObservabilityConfig,
        )

        with pytest.raises(ExoError):
            ObservabilityConfig(logging="bad")  # type: ignore[call-arg]

    def test_obs_config_logging_conflict_is_exoerror(self) -> None:
        from exo.observability.config import (  # pyright: ignore[reportMissingImports]
            LoggingConfig,
            ObservabilityConfig,
        )

        with pytest.raises(ExoError):
            ObservabilityConfig(logging=LoggingConfig(), log_level="DEBUG")

    def test_obs_config_bad_tracing_type_is_exoerror(self) -> None:
        from exo.observability.config import (  # pyright: ignore[reportMissingImports]
            ObservabilityConfig,
        )

        with pytest.raises(ExoError):
            ObservabilityConfig(tracing="bad")  # type: ignore[call-arg]

    def test_obs_config_tracing_conflict_is_exoerror(self) -> None:
        from exo.observability.config import (  # pyright: ignore[reportMissingImports]
            ObservabilityConfig,
            TracingConfig,
        )

        with pytest.raises(ExoError):
            ObservabilityConfig(tracing=TracingConfig(), trace_enabled=True)

    def test_obs_config_bad_metrics_type_is_exoerror(self) -> None:
        from exo.observability.config import (  # pyright: ignore[reportMissingImports]
            ObservabilityConfig,
        )

        with pytest.raises(ExoError):
            ObservabilityConfig(metrics="bad")  # type: ignore[call-arg]

    def test_obs_config_metrics_conflict_is_exoerror(self) -> None:
        from exo.observability.config import (  # pyright: ignore[reportMissingImports]
            MetricsConfig,
            ObservabilityConfig,
        )

        with pytest.raises(ExoError):
            ObservabilityConfig(metrics=MetricsConfig(), metrics_enabled=True)

    def test_langfuse_score_client_missing_sdk_is_exoerror(self) -> None:
        """_LangfuseScoreClient raises ExoError when the langfuse SDK is absent."""
        from exo.observability.backends.langfuse import (  # pyright: ignore[reportMissingImports]
            _LangfuseScoreClient,
        )

        caught: list[ExoError] = []
        with patch("builtins.__import__", side_effect=ImportError("no langfuse")):
            with pytest.raises(ExoError) as exc_info:
                _LangfuseScoreClient(public_key="pk", secret_key="sk", host="http://x")
            caught.append(exc_info.value)

        err = caught[0]
        assert err.hint is not None
        assert "langfuse" in err.hint.lower()

    def test_langsmith_dataset_client_missing_sdk_is_exoerror(self) -> None:
        """_LangSmithDatasetClient raises ExoError when the langsmith SDK is absent."""
        from exo.observability.backends.langsmith import (  # pyright: ignore[reportMissingImports]
            _LangSmithDatasetClient,
        )

        caught: list[ExoError] = []
        with patch("builtins.__import__", side_effect=ImportError("no langsmith")):
            with pytest.raises(ExoError) as exc_info:
                _LangSmithDatasetClient(api_key="k", endpoint="http://x")
            caught.append(exc_info.value)

        err = caught[0]
        assert err.hint is not None
        assert "langsmith" in err.hint.lower()


# ---------------------------------------------------------------------------
# P2 — HealthResult message includes exception type and message
# ---------------------------------------------------------------------------


class TestHealthResultExceptionText:
    def test_run_all_includes_exc_text_in_message(self) -> None:
        from exo.observability.health import (  # pyright: ignore[reportMissingImports]
            HealthRegistry,
            HealthResult,
            HealthStatus,
        )

        class _ExplodingCheck:
            @property
            def name(self) -> str:
                return "boom"

            def check(self) -> HealthResult:
                raise ValueError("disk is full")

        registry = HealthRegistry()
        registry.register(_ExplodingCheck())
        results = registry.run_all()

        assert results["boom"].status == HealthStatus.UNHEALTHY
        # The message must include the exception type and its text
        assert "ValueError" in results["boom"].message
        assert "disk is full" in results["boom"].message
