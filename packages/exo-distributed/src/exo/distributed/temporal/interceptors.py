"""Exo baggage + OTel tracing interceptors for Temporal.

This module provides :class:`ExoBaggageInterceptor` and the
:func:`exo_interceptors` factory, giving you two complementary behaviours in a
single interceptor chain:

1. **OTel span continuity** (via ``temporalio.contrib.opentelemetry.TracingInterceptor``
   when OpenTelemetry is installed and ``otel=True``) — creates linked spans so
   that a Temporal ``start_workflow`` → activity execution shows up as one
   connected trace in your APM tool.

2. **Exo baggage propagation** (via :class:`ExoBaggageInterceptor`) — serialises
   the current :func:`exo.distributed._propagation.get_baggage` dict into a
   Temporal header on every *outbound* client call (``start_workflow``,
   ``signal_workflow``) and deserialises it back into the async context on every
   *inbound* activity execution, making baggage keys like ``agent_id`` and
   ``run_id`` available inside activities without any extra plumbing.

Header key
----------
Exo baggage is stored under the Temporal header key ``"exo-baggage"``.  The
value is a ``temporalio.api.common.v1.Payload`` protobuf message with
``metadata["encoding"] = b"json/plain"`` and ``data`` set to the UTF-8
JSON-encoded ``dict[str, str]``.

Workflow-inbound limitation
---------------------------
The workflow-inbound side (``WorkflowInboundInterceptor.execute_workflow``) is
intentionally *not* implemented for baggage restoration.  Temporal workflows run
inside a deterministic sandbox — calling ``set_baggage`` (which writes to a
``contextvars.ContextVar``) from inside a workflow handler would be meaningless
on replay and would violate determinism.  Baggage is only useful across the
process boundary that the activity hop represents, which is precisely what this
interceptor covers via ``execute_activity``.  If you need workflow-scoped
metadata, store it in the workflow state directly.

Usage
-----
Pass the result of :func:`exo_interceptors` to *both* the Temporal client and
the Temporal worker::

    from temporalio.client import Client
    from temporalio.worker import Worker
    from exo.distributed.temporal.interceptors import exo_interceptors  # pyright: ignore[reportMissingImports]

    interceptors = exo_interceptors(otel=True)

    client = await Client.connect("localhost:7233", interceptors=interceptors)
    worker = Worker(client, task_queue="exo-tasks", interceptors=interceptors, ...)
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from exo.distributed._propagation import (  # pyright: ignore[reportMissingImports]
    clear_baggage,
    get_baggage,
    set_baggage,
)
from exo.distributed.temporal._compat import (  # pyright: ignore[reportMissingImports]
    HAS_TEMPORAL,
    ActivityInboundInterceptor,
    Interceptor,
    Payload,
)

logger = logging.getLogger(__name__)

__all__ = [
    "EXO_BAGGAGE_HEADER_KEY",
    "ExoBaggageInterceptor",
    "exo_interceptors",
]

#: Temporal header key used to carry exo baggage across the client→workflow and
#: workflow→activity process boundaries.
EXO_BAGGAGE_HEADER_KEY = "exo-baggage"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _baggage_to_payload(baggage: dict[str, str]) -> Payload:
    """Encode a baggage dict as a Temporal ``Payload`` (JSON bytes)."""
    p = Payload()
    p.metadata["encoding"] = b"json/plain"
    p.data = json.dumps(baggage).encode()
    return p


def _payload_to_baggage(payload: Payload) -> dict[str, str]:
    """Decode a Temporal ``Payload`` back to a baggage dict.

    Returns an empty dict if the payload cannot be decoded.
    """
    try:
        return json.loads(payload.data.decode())
    except Exception:
        logger.debug("exo-baggage payload could not be decoded; ignoring", exc_info=True)
        return {}


def _inject_baggage_into_headers(
    headers: Mapping[str, Payload],
) -> Mapping[str, Payload]:
    """Return a copy of *headers* with the current exo baggage injected.

    If there is no current baggage the original mapping is returned unchanged.
    """
    baggage = get_baggage()
    if not baggage:
        return headers
    try:
        payload = _baggage_to_payload(baggage)
    except Exception:
        logger.warning("Failed to encode exo baggage; skipping header injection", exc_info=True)
        return headers
    return {**headers, EXO_BAGGAGE_HEADER_KEY: payload}


# ---------------------------------------------------------------------------
# Client outbound interceptor
# ---------------------------------------------------------------------------


class _BaggageClientOutboundInterceptor:
    """Wraps the next client outbound interceptor to inject exo baggage.

    We deliberately avoid importing ``temporalio.client.OutboundInterceptor``
    at module level (it lives outside ``_compat``) so that this file is safe
    to import even when temporalio is not installed — the class is only ever
    instantiated by :meth:`ExoBaggageInterceptor.intercept_client`, which is
    only called when temporalio is present.
    """

    def __init__(self, next: Any) -> None:
        self.next = next

    # ---- inject baggage into workflow/signal headers ----

    async def start_workflow(self, input: Any) -> Any:
        input.headers = _inject_baggage_into_headers(input.headers)
        return await self.next.start_workflow(input)

    async def signal_workflow(self, input: Any) -> Any:
        input.headers = _inject_baggage_into_headers(input.headers)
        return await self.next.signal_workflow(input)

    # ---- pass-through everything else ----

    async def cancel_workflow(self, input: Any) -> None:
        await self.next.cancel_workflow(input)

    async def describe_workflow(self, input: Any) -> Any:
        return await self.next.describe_workflow(input)

    def fetch_workflow_history_events(self, input: Any) -> Any:
        return self.next.fetch_workflow_history_events(input)

    def list_workflows(self, input: Any) -> Any:
        return self.next.list_workflows(input)

    async def count_workflows(self, input: Any) -> Any:
        return await self.next.count_workflows(input)

    async def query_workflow(self, input: Any) -> Any:
        return await self.next.query_workflow(input)

    async def terminate_workflow(self, input: Any) -> None:
        await self.next.terminate_workflow(input)

    async def start_activity(self, input: Any) -> Any:
        return await self.next.start_activity(input)

    async def cancel_activity(self, input: Any) -> None:
        await self.next.cancel_activity(input)

    async def terminate_activity(self, input: Any) -> None:
        await self.next.terminate_activity(input)

    async def describe_activity(self, input: Any) -> Any:
        return await self.next.describe_activity(input)

    def list_activities(self, input: Any) -> Any:
        return self.next.list_activities(input)

    async def count_activities(self, input: Any) -> Any:
        return await self.next.count_activities(input)

    async def start_workflow_update(self, input: Any) -> Any:
        return await self.next.start_workflow_update(input)

    async def start_update_with_start_workflow(self, input: Any) -> Any:
        return await self.next.start_update_with_start_workflow(input)

    async def heartbeat_async_activity(self, input: Any) -> None:
        await self.next.heartbeat_async_activity(input)

    async def complete_async_activity(self, input: Any) -> None:
        await self.next.complete_async_activity(input)

    async def fail_async_activity(self, input: Any) -> None:
        await self.next.fail_async_activity(input)

    async def report_cancellation_async_activity(self, input: Any) -> None:
        await self.next.report_cancellation_async_activity(input)

    async def create_schedule(self, input: Any) -> Any:
        return await self.next.create_schedule(input)

    def list_schedules(self, input: Any) -> Any:
        return self.next.list_schedules(input)

    async def backfill_schedule(self, input: Any) -> None:
        await self.next.backfill_schedule(input)

    async def delete_schedule(self, input: Any) -> None:
        await self.next.delete_schedule(input)

    async def describe_schedule(self, input: Any) -> Any:
        return await self.next.describe_schedule(input)

    async def pause_schedule(self, input: Any) -> None:
        await self.next.pause_schedule(input)

    async def trigger_schedule(self, input: Any) -> None:
        await self.next.trigger_schedule(input)

    async def unpause_schedule(self, input: Any) -> None:
        await self.next.unpause_schedule(input)

    async def update_schedule(self, input: Any) -> None:
        await self.next.update_schedule(input)

    async def update_worker_build_id_compatibility(self, input: Any) -> None:
        await self.next.update_worker_build_id_compatibility(input)

    async def get_worker_build_id_compatibility(self, input: Any) -> Any:
        return await self.next.get_worker_build_id_compatibility(input)

    async def get_worker_task_reachability(self, input: Any) -> Any:
        return await self.next.get_worker_task_reachability(input)

    async def start_nexus_operation(self, input: Any) -> Any:
        return await self.next.start_nexus_operation(input)

    async def describe_nexus_operation(self, input: Any) -> Any:
        return await self.next.describe_nexus_operation(input)

    async def get_nexus_operation_result(self, input: Any) -> Any:
        return await self.next.get_nexus_operation_result(input)

    async def cancel_nexus_operation(self, input: Any) -> None:
        await self.next.cancel_nexus_operation(input)

    async def terminate_nexus_operation(self, input: Any) -> None:
        await self.next.terminate_nexus_operation(input)

    def list_nexus_operations(self, input: Any) -> Any:
        return self.next.list_nexus_operations(input)

    async def count_nexus_operations(self, input: Any) -> Any:
        return await self.next.count_nexus_operations(input)


# ---------------------------------------------------------------------------
# Activity inbound interceptor
# ---------------------------------------------------------------------------


class _BaggageActivityInboundInterceptor(ActivityInboundInterceptor):  # type: ignore[misc]
    """Restores exo baggage from a Temporal header for the duration of an activity."""

    async def execute_activity(self, input: Any) -> Any:
        header_payload: Payload | None = input.headers.get(EXO_BAGGAGE_HEADER_KEY)
        if header_payload is None:
            return await self.next.execute_activity(input)

        baggage = _payload_to_baggage(header_payload)
        if not baggage:
            return await self.next.execute_activity(input)

        # Save the current baggage so we can restore it after the activity.
        # In a normal asyncio task each activity runs in its own context, but
        # being defensive here costs nothing.
        old_baggage = get_baggage()
        try:
            for key, value in baggage.items():
                set_baggage(key, value)
            logger.debug("restored %d exo baggage entries for activity execution", len(baggage))
            return await self.next.execute_activity(input)
        finally:
            clear_baggage()
            for key, value in old_baggage.items():
                set_baggage(key, value)


# ---------------------------------------------------------------------------
# Main interceptor
# ---------------------------------------------------------------------------


class ExoBaggageInterceptor(Interceptor):  # type: ignore[misc]
    """Temporal interceptor that propagates exo baggage across the workflow→activity boundary.

    Implements both ``temporalio.client.Interceptor`` (for outbound client
    calls) and ``temporalio.worker.Interceptor`` (for inbound activity
    execution) — the same dual-inheritance pattern used by
    ``temporalio.contrib.opentelemetry.TracingInterceptor``.

    **What is covered**

    * ``start_workflow`` and ``signal_workflow`` — the current
      :func:`~exo.distributed._propagation.get_baggage` dict is JSON-encoded
      into the ``"exo-baggage"`` Temporal header before the call is forwarded.
    * ``execute_activity`` - the ``"exo-baggage"`` header is decoded and each
      key-value pair is installed via :func:`~exo.distributed._propagation.set_baggage`
      for the duration of the activity, then cleared and restored afterward.

    **What is deferred**

    The workflow-inbound side (``workflow_interceptor_class`` /
    ``WorkflowInboundInterceptor.execute_workflow``) is intentionally not
    implemented.  Temporal workflows run inside a deterministic replay sandbox;
    writing to a ``ContextVar`` inside that sandbox has no effect on replay and
    violates the determinism contract.  Baggage is only meaningful at the
    activity hop (a real process boundary), which is exactly what this
    interceptor handles.
    """

    # -- client side --

    def intercept_client(self, next: Any) -> Any:
        """Wrap the client outbound interceptor to inject exo baggage."""
        return _BaggageClientOutboundInterceptor(next)

    # -- worker side --

    def intercept_activity(self, next: ActivityInboundInterceptor) -> ActivityInboundInterceptor:  # type: ignore[override]
        """Wrap the activity inbound interceptor to restore exo baggage."""
        return _BaggageActivityInboundInterceptor(next)

    # workflow_interceptor_class intentionally not overridden — see class docstring.


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def exo_interceptors(*, otel: bool = True) -> list[Any]:
    """Return the recommended Temporal interceptor chain for exo applications.

    Pass the returned list to **both** ``Client.connect(..., interceptors=...)``
    and ``Worker(..., interceptors=...)``.  The order matters: OTel spans are
    created first (outermost), then exo baggage is injected/extracted.

    Args:
        otel: When ``True`` (the default) and ``opentelemetry`` is importable,
            prepend ``TracingInterceptor()`` so that Temporal operations appear
            as linked spans in your APM tool.  If OpenTelemetry is not installed
            this flag is silently ignored and only the baggage interceptor is
            returned.

    Returns:
        A list of :class:`~temporalio.worker.Interceptor` instances ready to
        pass to ``Client.connect`` or ``Worker``.

    Raises:
        RuntimeError: If ``temporalio`` is not installed
            (``exo.distributed.temporal._compat.HAS_TEMPORAL`` is ``False``).
    """
    if not HAS_TEMPORAL:
        raise RuntimeError(
            "exo_interceptors() requires the 'temporalio' package. "
            "Install it with: pip install 'exo-distributed[temporal]'"
        )

    chain: list[Any] = []

    if otel:
        try:
            from temporalio.contrib.opentelemetry import TracingInterceptor

            chain.append(TracingInterceptor())
            logger.debug("exo_interceptors: OTel TracingInterceptor included")
        except ImportError:
            logger.debug(
                "exo_interceptors: opentelemetry not installed; OTel TracingInterceptor omitted"
            )

    chain.append(ExoBaggageInterceptor())
    return chain
