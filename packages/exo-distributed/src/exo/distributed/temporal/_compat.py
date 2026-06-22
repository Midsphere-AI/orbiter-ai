"""Shared optional-import shim for the Temporal integration.

Every module in :mod:`exo.distributed.temporal` imports the temporalio symbols
it needs from here so the optional dependency is resolved in exactly one place.
When ``temporalio`` is not installed, :data:`HAS_TEMPORAL` is ``False`` and the
re-exported names are ``None`` — the package ``__init__`` guards the import of
any submodule that defines workflow/activity/converter code behind that flag, so
those names are only ever dereferenced when temporalio is present.

Install the optional dependency group::

    pip install exo-distributed[temporal]
"""

from __future__ import annotations

import os

# Default Temporal connection settings (env-overridable).
_DEFAULT_TEMPORAL_HOST = "localhost:7233"
_DEFAULT_TEMPORAL_NAMESPACE = "default"
_DEFAULT_TASK_QUEUE = "exo-tasks"

try:
    from temporalio import activity, workflow
    from temporalio.api.common.v1 import Payload
    from temporalio.client import Client as TemporalClient
    from temporalio.client import TLSConfig
    from temporalio.common import RetryPolicy
    from temporalio.contrib.pydantic import pydantic_data_converter
    from temporalio.converter import (
        DataConverter,
        DefaultFailureConverter,
        FailureConverter,
        PayloadCodec,
    )
    from temporalio.exceptions import ApplicationError, CancelledError, FailureError
    from temporalio.worker import (
        ActivityInboundInterceptor,
        Interceptor,
        WorkflowInboundInterceptor,
    )
    from temporalio.worker import (
        Worker as TemporalWorker,
    )

    HAS_TEMPORAL = True
except ImportError:  # pragma: no cover - exercised only without temporalio
    HAS_TEMPORAL = False
    activity = None  # type: ignore[assignment]
    workflow = None  # type: ignore[assignment]
    TemporalClient = None  # type: ignore[assignment,misc]
    TLSConfig = None  # type: ignore[assignment,misc]
    TemporalWorker = None  # type: ignore[assignment,misc]
    RetryPolicy = None  # type: ignore[assignment,misc]
    pydantic_data_converter = None  # type: ignore[assignment]
    DataConverter = None  # type: ignore[assignment,misc]
    DefaultFailureConverter = None  # type: ignore[assignment,misc]
    FailureConverter = None  # type: ignore[assignment,misc]
    PayloadCodec = None  # type: ignore[assignment,misc]
    ApplicationError = None  # type: ignore[assignment,misc]
    CancelledError = None  # type: ignore[assignment,misc]
    FailureError = None  # type: ignore[assignment,misc]
    Interceptor = None  # type: ignore[assignment,misc]
    ActivityInboundInterceptor = None  # type: ignore[assignment,misc]
    WorkflowInboundInterceptor = None  # type: ignore[assignment,misc]
    Payload = None  # type: ignore[assignment,misc]


def _get_temporal_host() -> str:
    """Resolve Temporal host from environment or default."""
    return os.environ.get("TEMPORAL_HOST", _DEFAULT_TEMPORAL_HOST)


def _get_temporal_namespace() -> str:
    """Resolve Temporal namespace from environment or default."""
    return os.environ.get("TEMPORAL_NAMESPACE", _DEFAULT_TEMPORAL_NAMESPACE)


__all__ = [
    "HAS_TEMPORAL",
    "_DEFAULT_TASK_QUEUE",
    "_DEFAULT_TEMPORAL_HOST",
    "_DEFAULT_TEMPORAL_NAMESPACE",
    "ActivityInboundInterceptor",
    "ApplicationError",
    "CancelledError",
    "DataConverter",
    "DefaultFailureConverter",
    "FailureConverter",
    "FailureError",
    "Interceptor",
    "Payload",
    "PayloadCodec",
    "RetryPolicy",
    "TLSConfig",
    "TemporalClient",
    "TemporalWorker",
    "WorkflowInboundInterceptor",
    "_get_temporal_host",
    "_get_temporal_namespace",
    "activity",
    "pydantic_data_converter",
    "workflow",
]
