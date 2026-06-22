"""Tests for exo.distributed.temporal.interceptors.

Strategy: we do NOT spin up a real Temporal server.  Instead we exercise the
inject/extract logic directly by constructing minimal fake Temporal input
objects and a no-op ``next`` chain via ``unittest.mock``.  This gives full
coverage of the round-trip without any network dependency.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exo.distributed._propagation import clear_baggage, get_baggage, set_baggage
from exo.distributed.temporal.interceptors import (
    EXO_BAGGAGE_HEADER_KEY,
    ExoBaggageInterceptor,
    _baggage_to_payload,
    _inject_baggage_into_headers,
    _payload_to_baggage,
    exo_interceptors,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_baggage():
    """Ensure no baggage leaks between tests."""
    clear_baggage()
    yield
    clear_baggage()


def _make_start_workflow_input(headers: dict | None = None) -> MagicMock:
    """Fake StartWorkflowInput with a mutable ``headers`` attribute."""
    obj = MagicMock()
    obj.headers = headers if headers is not None else {}
    return obj


def _make_execute_activity_input(headers: dict | None = None) -> MagicMock:
    """Fake ExecuteActivityInput with a mutable ``headers`` attribute."""
    obj = MagicMock()
    obj.headers = headers if headers is not None else {}
    return obj


# ---------------------------------------------------------------------------
# exo_interceptors factory
# ---------------------------------------------------------------------------


class TestExoInterceptorsFactory:
    def test_returns_list_with_baggage_interceptor(self):
        chain = exo_interceptors(otel=False)
        assert isinstance(chain, list)
        assert any(isinstance(i, ExoBaggageInterceptor) for i in chain)

    def test_otel_false_no_tracing_interceptor(self):
        chain = exo_interceptors(otel=False)
        type_names = [type(i).__name__ for i in chain]
        assert "TracingInterceptor" not in type_names

    def test_otel_true_with_opentelemetry_installed(self):
        """When opentelemetry IS available, TracingInterceptor should be first."""
        try:
            import opentelemetry  # noqa: F401

            otel_available = True
        except ImportError:
            otel_available = False

        chain = exo_interceptors(otel=True)
        type_names = [type(i).__name__ for i in chain]

        if otel_available:
            assert "TracingInterceptor" in type_names
            # OTel interceptor should come before baggage interceptor
            otel_idx = next(i for i, n in enumerate(type_names) if n == "TracingInterceptor")
            baggage_idx = next(i for i, n in enumerate(type_names) if n == "ExoBaggageInterceptor")
            assert otel_idx < baggage_idx, "OTel should be outermost interceptor"
        else:
            assert "TracingInterceptor" not in type_names

        # Baggage interceptor always present
        assert "ExoBaggageInterceptor" in type_names

    def test_raises_when_temporal_not_installed(self):
        """If HAS_TEMPORAL is False, exo_interceptors must raise RuntimeError."""
        with (
            patch("exo.distributed.temporal.interceptors.HAS_TEMPORAL", False),
            pytest.raises(RuntimeError, match="temporalio"),
        ):
            exo_interceptors()

    def test_chain_always_ends_with_baggage_interceptor(self):
        chain = exo_interceptors(otel=True)
        assert isinstance(chain[-1], ExoBaggageInterceptor)

    def test_chain_otel_false_has_exactly_one_interceptor(self):
        chain = exo_interceptors(otel=False)
        assert len(chain) == 1
        assert isinstance(chain[0], ExoBaggageInterceptor)


# ---------------------------------------------------------------------------
# Payload encode / decode helpers
# ---------------------------------------------------------------------------


class TestPayloadHelpers:
    def test_round_trip_simple(self):
        baggage = {"agent_id": "abc", "run_id": "xyz"}
        payload = _baggage_to_payload(baggage)
        decoded = _payload_to_baggage(payload)
        assert decoded == baggage

    def test_encoding_metadata(self):
        payload = _baggage_to_payload({"k": "v"})
        assert payload.metadata["encoding"] == b"json/plain"

    def test_data_is_valid_json(self):
        payload = _baggage_to_payload({"x": "1"})
        parsed = json.loads(payload.data.decode())
        assert parsed == {"x": "1"}

    def test_empty_baggage_round_trip(self):
        payload = _baggage_to_payload({})
        decoded = _payload_to_baggage(payload)
        assert decoded == {}

    def test_malformed_payload_returns_empty(self):
        from temporalio.api.common.v1 import Payload

        p = Payload()
        p.metadata["encoding"] = b"json/plain"
        p.data = b"not-valid-json{{{"
        result = _payload_to_baggage(p)
        assert result == {}

    def test_inject_baggage_into_headers_no_baggage(self):
        clear_baggage()
        headers: dict = {}
        result = _inject_baggage_into_headers(headers)
        assert EXO_BAGGAGE_HEADER_KEY not in result

    def test_inject_baggage_into_headers_with_baggage(self):
        set_baggage("agent_id", "test-agent")
        headers: dict = {}
        result = _inject_baggage_into_headers(headers)
        assert EXO_BAGGAGE_HEADER_KEY in result
        decoded = _payload_to_baggage(result[EXO_BAGGAGE_HEADER_KEY])
        assert decoded["agent_id"] == "test-agent"

    def test_inject_preserves_existing_headers(self):
        from temporalio.api.common.v1 import Payload

        set_baggage("k", "v")
        existing_payload = Payload()
        existing_payload.data = b"existing"
        headers = {"other-key": existing_payload}
        result = _inject_baggage_into_headers(headers)
        assert "other-key" in result
        assert EXO_BAGGAGE_HEADER_KEY in result


# ---------------------------------------------------------------------------
# Client outbound — baggage injection
# ---------------------------------------------------------------------------


class TestClientOutboundInjection:
    def _make_outbound(self):
        """Return (interceptor, fake_next_outbound)."""
        ic = ExoBaggageInterceptor()
        fake_next = MagicMock()
        fake_next.start_workflow = AsyncMock(return_value=MagicMock())
        fake_next.signal_workflow = AsyncMock(return_value=None)
        outbound = ic.intercept_client(fake_next)
        return outbound, fake_next

    async def test_start_workflow_injects_baggage(self):
        set_baggage("agent_id", "agent-1")
        outbound, fake_next = self._make_outbound()
        input_obj = _make_start_workflow_input()

        await outbound.start_workflow(input_obj)

        fake_next.start_workflow.assert_awaited_once()
        assert EXO_BAGGAGE_HEADER_KEY in input_obj.headers
        decoded = _payload_to_baggage(input_obj.headers[EXO_BAGGAGE_HEADER_KEY])
        assert decoded["agent_id"] == "agent-1"

    async def test_start_workflow_no_baggage_leaves_headers_unchanged(self):
        clear_baggage()
        outbound, _fake_next = self._make_outbound()
        input_obj = _make_start_workflow_input()

        await outbound.start_workflow(input_obj)

        assert EXO_BAGGAGE_HEADER_KEY not in input_obj.headers

    async def test_signal_workflow_injects_baggage(self):
        set_baggage("run_id", "run-42")
        outbound, _fake_next = self._make_outbound()
        input_obj = _make_start_workflow_input()

        await outbound.signal_workflow(input_obj)

        assert EXO_BAGGAGE_HEADER_KEY in input_obj.headers
        decoded = _payload_to_baggage(input_obj.headers[EXO_BAGGAGE_HEADER_KEY])
        assert decoded["run_id"] == "run-42"

    async def test_multiple_baggage_keys_all_injected(self):
        set_baggage("a", "1")
        set_baggage("b", "2")
        set_baggage("c", "3")
        outbound, _fake_next = self._make_outbound()
        input_obj = _make_start_workflow_input()

        await outbound.start_workflow(input_obj)

        decoded = _payload_to_baggage(input_obj.headers[EXO_BAGGAGE_HEADER_KEY])
        assert decoded == {"a": "1", "b": "2", "c": "3"}

    async def test_passthrough_methods_delegate_to_next(self):
        """Non-baggage methods (e.g. cancel_workflow) are forwarded unchanged."""
        outbound, fake_next = self._make_outbound()
        fake_next.cancel_workflow = AsyncMock(return_value=None)
        input_obj = MagicMock()

        await outbound.cancel_workflow(input_obj)

        fake_next.cancel_workflow.assert_awaited_once_with(input_obj)


# ---------------------------------------------------------------------------
# Activity inbound — baggage restoration
# ---------------------------------------------------------------------------


class TestActivityInboundRestoration:
    def _make_activity_interceptor(self):
        """Return (_BaggageActivityInboundInterceptor, fake_next)."""
        ic = ExoBaggageInterceptor()
        fake_next = MagicMock()
        fake_next.execute_activity = AsyncMock(return_value="activity-result")
        activity_ic = ic.intercept_activity(fake_next)
        return activity_ic, fake_next

    async def test_baggage_restored_during_activity(self):
        """Baggage in the header is available via get_baggage() inside the activity."""
        activity_ic, fake_next = self._make_activity_interceptor()

        injected = {"agent_id": "restored-agent", "tenant": "acme"}
        payload = _baggage_to_payload(injected)
        input_obj = _make_execute_activity_input({EXO_BAGGAGE_HEADER_KEY: payload})

        # Capture baggage inside the activity by side-effecting the mock
        captured: dict[str, str] = {}

        async def _capture(*_args, **_kwargs):
            captured.update(get_baggage())
            return "ok"

        fake_next.execute_activity = AsyncMock(side_effect=_capture)

        await activity_ic.execute_activity(input_obj)

        assert captured["agent_id"] == "restored-agent"
        assert captured["tenant"] == "acme"

    async def test_baggage_cleared_after_activity(self):
        """Exo baggage is cleared after the activity completes."""
        activity_ic, _fake_next = self._make_activity_interceptor()

        payload = _baggage_to_payload({"key": "value"})
        input_obj = _make_execute_activity_input({EXO_BAGGAGE_HEADER_KEY: payload})

        await activity_ic.execute_activity(input_obj)

        # Baggage should be gone (no prior baggage existed)
        assert get_baggage() == {}

    async def test_prior_baggage_restored_after_activity(self):
        """Pre-existing baggage is restored after the activity, not clobbered."""
        set_baggage("pre_existing", "yes")

        activity_ic, _fake_next = self._make_activity_interceptor()
        payload = _baggage_to_payload({"from_header": "injected"})
        input_obj = _make_execute_activity_input({EXO_BAGGAGE_HEADER_KEY: payload})

        await activity_ic.execute_activity(input_obj)

        current = get_baggage()
        assert current.get("pre_existing") == "yes"
        # The injected key from the header should not bleed into the restored state
        assert "from_header" not in current

    async def test_missing_header_passes_through(self):
        """No header present — activity runs normally, no baggage set."""
        activity_ic, fake_next = self._make_activity_interceptor()
        input_obj = _make_execute_activity_input({})

        result = await activity_ic.execute_activity(input_obj)

        assert result == "activity-result"
        fake_next.execute_activity.assert_awaited_once()

    async def test_malformed_header_does_not_break_activity(self):
        """Malformed exo-baggage header — activity still runs, no crash."""
        from temporalio.api.common.v1 import Payload

        activity_ic, _fake_next = self._make_activity_interceptor()
        bad_payload = Payload()
        bad_payload.data = b"<<<not json>>>"
        input_obj = _make_execute_activity_input({EXO_BAGGAGE_HEADER_KEY: bad_payload})

        result = await activity_ic.execute_activity(input_obj)

        assert result == "activity-result"

    async def test_baggage_cleared_even_on_activity_exception(self):
        """Baggage is cleaned up even when the activity raises."""
        activity_ic, fake_next = self._make_activity_interceptor()

        payload = _baggage_to_payload({"k": "v"})
        input_obj = _make_execute_activity_input({EXO_BAGGAGE_HEADER_KEY: payload})

        fake_next.execute_activity = AsyncMock(side_effect=RuntimeError("boom"))

        with pytest.raises(RuntimeError, match="boom"):
            await activity_ic.execute_activity(input_obj)

        assert get_baggage() == {}


# ---------------------------------------------------------------------------
# ExoBaggageInterceptor structural tests
# ---------------------------------------------------------------------------


class TestExoBaggageInterceptorStructure:
    def test_intercept_client_returns_outbound_wrapper(self):
        ic = ExoBaggageInterceptor()
        fake_next = MagicMock()
        outbound = ic.intercept_client(fake_next)
        # Should have start_workflow and signal_workflow
        assert hasattr(outbound, "start_workflow")
        assert hasattr(outbound, "signal_workflow")

    def test_intercept_activity_returns_activity_interceptor(self):
        from exo.distributed.temporal.interceptors import (
            _BaggageActivityInboundInterceptor,
        )

        ic = ExoBaggageInterceptor()
        fake_next = MagicMock()
        fake_next.init = MagicMock()
        activity_ic = ic.intercept_activity(fake_next)
        assert isinstance(activity_ic, _BaggageActivityInboundInterceptor)

    def test_header_key_constant(self):
        assert EXO_BAGGAGE_HEADER_KEY == "exo-baggage"
