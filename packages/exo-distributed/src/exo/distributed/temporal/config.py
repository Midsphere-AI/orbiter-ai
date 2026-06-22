"""Pydantic configuration models for the Temporal integration.

This module provides four frozen Pydantic models that capture every tunable
aspect of a Temporal connection or worker:

- :class:`ConnectionConfig` — server address, namespace, API key, TLS certs.
- :class:`RetryConfig` — per-activity/workflow retry policy.
- :class:`TimeoutConfig` — activity and workflow timeout windows.
- :class:`WorkerTuningConfig` — worker-level concurrency knobs.

All models are **frozen** (immutable after construction) and follow the
``to_*/from_dict`` pattern so callers can round-trip them through JSON
payloads, environment variables, or kwargs dicts without touching temporalio
objects directly.  The temporalio library is accessed only inside methods that
actually need it; every method that does raises a clear :class:`ImportError`
(mentioning ``pip install exo-distributed[temporal]``) when the optional
dependency is absent.
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta
from typing import Any

from pydantic import BaseModel, Field

from exo.distributed.temporal._compat import (  # pyright: ignore[reportMissingImports]
    _DEFAULT_TEMPORAL_HOST,
    _DEFAULT_TEMPORAL_NAMESPACE,
    HAS_TEMPORAL,
    RetryPolicy,
    TLSConfig,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ConnectionConfig",
    "RetryConfig",
    "TimeoutConfig",
    "WorkerTuningConfig",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEMPORAL_INSTALL_HINT = "pip install exo-distributed[temporal]"


def _require_temporal(feature: str) -> None:
    """Raise an :class:`ImportError` when temporalio is not installed."""
    if not HAS_TEMPORAL:
        msg = f"{feature} requires temporalio. Install it with: {_TEMPORAL_INSTALL_HINT}"
        raise ImportError(msg)


# ---------------------------------------------------------------------------
# ConnectionConfig
# ---------------------------------------------------------------------------


class ConnectionConfig(BaseModel):
    """Temporal server connection settings.

    All fields can be supplied directly or loaded from the environment via
    :meth:`from_env`.

    Environment variables recognised by :meth:`from_env`:

    - ``TEMPORAL_HOST`` — server address (default ``localhost:7233``).
    - ``TEMPORAL_NAMESPACE`` — Temporal namespace (default ``default``).
    - ``TEMPORAL_API_KEY`` — Temporal Cloud API key (enables TLS automatically).
    - ``TEMPORAL_TLS`` — any truthy string (``"1"``, ``"true"``, ``"yes"``) enables
      plain TLS without client certs.
    - ``TEMPORAL_TLS_SERVER_CA_CERT_PATH`` — path to PEM CA cert for server
      validation.
    - ``TEMPORAL_TLS_CLIENT_CERT_PATH`` — path to PEM client certificate (mTLS).
    - ``TEMPORAL_TLS_CLIENT_KEY_PATH`` — path to PEM client private key (mTLS).

    Security note: :meth:`__repr__` redacts ``api_key`` and
    ``client_private_key`` so they never appear in logs or tracebacks.
    """

    model_config = {"frozen": True}

    host: str = _DEFAULT_TEMPORAL_HOST
    namespace: str = _DEFAULT_TEMPORAL_NAMESPACE
    api_key: str | None = None
    tls: bool = False
    server_root_ca_cert: bytes | None = None
    client_cert: bytes | None = None
    client_private_key: bytes | None = None
    tls_domain: str | None = None

    # ------------------------------------------------------------------
    # Repr security
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        api_key_display = "***" if self.api_key is not None else None
        key_display = "***" if self.client_private_key is not None else None
        return (
            f"ConnectionConfig("
            f"host={self.host!r}, "
            f"namespace={self.namespace!r}, "
            f"api_key={api_key_display!r}, "
            f"tls={self.tls!r}, "
            f"server_root_ca_cert={'<bytes>' if self.server_root_ca_cert else None}, "
            f"client_cert={'<bytes>' if self.client_cert else None}, "
            f"client_private_key={key_display!r}, "
            f"tls_domain={self.tls_domain!r}"
            f")"
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> ConnectionConfig:
        """Build a :class:`ConnectionConfig` from environment variables.

        Reads ``TEMPORAL_HOST``, ``TEMPORAL_NAMESPACE``, ``TEMPORAL_API_KEY``,
        ``TEMPORAL_TLS``, and cert file paths (see class docstring for details).
        Falls back to the built-in defaults for any variable that is unset.
        """
        host: str = os.environ.get("TEMPORAL_HOST") or _DEFAULT_TEMPORAL_HOST
        namespace: str = os.environ.get("TEMPORAL_NAMESPACE") or _DEFAULT_TEMPORAL_NAMESPACE
        api_key = os.environ.get("TEMPORAL_API_KEY") or None

        tls_raw = os.environ.get("TEMPORAL_TLS", "").strip().lower()
        tls = tls_raw in {"1", "true", "yes", "on"}

        server_root_ca_cert: bytes | None = None
        ca_path = os.environ.get("TEMPORAL_TLS_SERVER_CA_CERT_PATH")
        if ca_path:
            server_root_ca_cert = _read_file_bytes(ca_path, "TEMPORAL_TLS_SERVER_CA_CERT_PATH")

        client_cert: bytes | None = None
        cert_path = os.environ.get("TEMPORAL_TLS_CLIENT_CERT_PATH")
        if cert_path:
            client_cert = _read_file_bytes(cert_path, "TEMPORAL_TLS_CLIENT_CERT_PATH")

        client_private_key: bytes | None = None
        key_path = os.environ.get("TEMPORAL_TLS_CLIENT_KEY_PATH")
        if key_path:
            client_private_key = _read_file_bytes(key_path, "TEMPORAL_TLS_CLIENT_KEY_PATH")

        return cls(
            host=host,
            namespace=namespace,
            api_key=api_key,
            tls=tls,
            server_root_ca_cert=server_root_ca_cert,
            client_cert=client_cert,
            client_private_key=client_private_key,
        )

    # ------------------------------------------------------------------
    # Temporal kwargs
    # ------------------------------------------------------------------

    def to_connect_kwargs(self) -> dict[str, Any]:
        """Produce kwargs ready for ``TemporalClient.connect(**kwargs)``.

        TLS selection rules (applied in order):

        1. Both ``client_cert`` **and** ``client_private_key`` set → mTLS via
           :class:`TLSConfig` (optionally including ``server_root_ca_cert`` and
           ``tls_domain``).
        2. Only ``server_root_ca_cert`` set → server-CA-only TLS via
           :class:`TLSConfig`.
        3. ``tls=True`` **or** ``api_key`` is set (Temporal Cloud requires TLS
           when using an API key) → plain ``tls=True``.
        4. Otherwise → no TLS kwarg emitted (Temporal default: plain text).

        Raises:
            ImportError: If temporalio is not installed.
        """
        _require_temporal("ConnectionConfig.to_connect_kwargs")

        kwargs: dict[str, Any] = {
            "target_host": self.host,
            "namespace": self.namespace,
        }

        if self.api_key is not None:
            kwargs["api_key"] = self.api_key

        # TLS selection
        if self.client_cert is not None and self.client_private_key is not None:
            # Full mTLS — optionally include server CA and domain.
            tls_kwargs: dict[str, Any] = {
                "client_cert": self.client_cert,
                "client_private_key": self.client_private_key,
            }
            if self.server_root_ca_cert is not None:
                tls_kwargs["server_root_ca_cert"] = self.server_root_ca_cert
            if self.tls_domain is not None:
                tls_kwargs["domain"] = self.tls_domain
            kwargs["tls"] = TLSConfig(**tls_kwargs)  # type: ignore[misc]
        elif self.server_root_ca_cert is not None:
            # Server CA validation only.
            kwargs["tls"] = TLSConfig(server_root_ca_cert=self.server_root_ca_cert)  # type: ignore[misc]
        elif self.tls or self.api_key is not None:
            # Plain TLS (or Temporal Cloud API-key mode).
            kwargs["tls"] = True

        return kwargs


# ---------------------------------------------------------------------------
# RetryConfig
# ---------------------------------------------------------------------------


class RetryConfig(BaseModel):
    """Per-task or per-activity retry policy.

    Maps directly onto Temporal's :class:`temporalio.common.RetryPolicy`.
    All interval values are in **seconds** so the model can be stored in JSON
    without importing temporalio.

    Args:
        initial_interval_seconds: Backoff interval for the first retry.
        backoff_coefficient: Multiplier applied to each successive interval.
        maximum_interval_seconds: Cap on the backoff interval.  ``None`` means
            Temporal uses its default (100x ``initial_interval``).
        maximum_attempts: Maximum total attempts (0 = unlimited).
        non_retryable_error_types: Error type names that abort retries
            immediately.
    """

    model_config = {"frozen": True}

    initial_interval_seconds: float = 1.0
    backoff_coefficient: float = 2.0
    maximum_interval_seconds: float | None = None
    maximum_attempts: int = 0
    non_retryable_error_types: list[str] = Field(default_factory=list)

    # ------------------------------------------------------------------
    # Temporal object
    # ------------------------------------------------------------------

    def to_retry_policy(self) -> Any:
        """Convert to a :class:`temporalio.common.RetryPolicy` instance.

        Raises:
            ImportError: If temporalio is not installed.
        """
        _require_temporal("RetryConfig.to_retry_policy")

        max_interval = (
            timedelta(seconds=self.maximum_interval_seconds)
            if self.maximum_interval_seconds is not None
            else None
        )
        non_retryable: list[str] | None = (
            self.non_retryable_error_types if self.non_retryable_error_types else None
        )
        return RetryPolicy(  # type: ignore[misc]
            initial_interval=timedelta(seconds=self.initial_interval_seconds),
            backoff_coefficient=self.backoff_coefficient,
            maximum_interval=max_interval,
            maximum_attempts=self.maximum_attempts,
            non_retryable_error_types=non_retryable,
        )

    # ------------------------------------------------------------------
    # JSON round-trip
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON payloads."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RetryConfig:
        """Deserialise from a plain dict (reverse of :meth:`to_dict`)."""
        return cls(**d)


# ---------------------------------------------------------------------------
# TimeoutConfig
# ---------------------------------------------------------------------------


class TimeoutConfig(BaseModel):
    """Activity and workflow timeout windows.

    All values are in **seconds**; ``None`` means "let Temporal decide".

    Activity timeouts map onto the kwargs accepted by
    ``workflow.execute_activity()``:

    - ``schedule_to_close_seconds`` → ``schedule_to_close_timeout``
    - ``start_to_close_seconds`` → ``start_to_close_timeout``
    - ``schedule_to_start_seconds`` → ``schedule_to_start_timeout``
    - ``heartbeat_seconds`` → ``heartbeat_timeout``

    Workflow timeouts map onto the kwargs accepted by
    ``client.start_workflow()``:

    - ``execution_timeout_seconds`` → ``execution_timeout``
    - ``run_timeout_seconds`` → ``run_timeout``
    - ``task_timeout_seconds`` → ``task_timeout``
    """

    model_config = {"frozen": True}

    # Activity timeouts
    schedule_to_close_seconds: float | None = None
    start_to_close_seconds: float | None = None
    schedule_to_start_seconds: float | None = None
    heartbeat_seconds: float | None = None

    # Workflow timeouts
    execution_timeout_seconds: float | None = None
    run_timeout_seconds: float | None = None
    task_timeout_seconds: float | None = None

    # ------------------------------------------------------------------
    # Temporal kwargs
    # ------------------------------------------------------------------

    def to_activity_timeouts(self) -> dict[str, timedelta]:
        """Produce a dict of non-None activity timeout kwargs.

        Keys match the real ``workflow.execute_activity()`` parameter names.
        Only fields that are not ``None`` are included.
        """
        mapping: list[tuple[float | None, str]] = [
            (self.schedule_to_close_seconds, "schedule_to_close_timeout"),
            (self.start_to_close_seconds, "start_to_close_timeout"),
            (self.schedule_to_start_seconds, "schedule_to_start_timeout"),
            (self.heartbeat_seconds, "heartbeat_timeout"),
        ]
        return {kwarg: timedelta(seconds=value) for value, kwarg in mapping if value is not None}

    def to_workflow_timeouts(self) -> dict[str, timedelta]:
        """Produce a dict of non-None workflow timeout kwargs.

        Keys match the real ``client.start_workflow()`` parameter names.
        Only fields that are not ``None`` are included.
        """
        mapping: list[tuple[float | None, str]] = [
            (self.execution_timeout_seconds, "execution_timeout"),
            (self.run_timeout_seconds, "run_timeout"),
            (self.task_timeout_seconds, "task_timeout"),
        ]
        return {kwarg: timedelta(seconds=value) for value, kwarg in mapping if value is not None}

    # ------------------------------------------------------------------
    # JSON round-trip
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON payloads."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TimeoutConfig:
        """Deserialise from a plain dict (reverse of :meth:`to_dict`)."""
        return cls(**d)


# ---------------------------------------------------------------------------
# WorkerTuningConfig
# ---------------------------------------------------------------------------


class WorkerTuningConfig(BaseModel):
    """Worker-level concurrency knobs.

    Each field corresponds to a ``temporalio.worker.Worker.__init__`` kwarg.
    ``None`` values are omitted from :meth:`to_worker_kwargs`, so Temporal's
    own defaults apply for any unset field.

    Args:
        max_concurrent_activities: Maximum parallel activity executions.
        max_concurrent_workflow_tasks: Maximum parallel workflow task
            executions.
        max_concurrent_local_activities: Maximum parallel local-activity
            executions.
        max_cached_workflows: Maximum number of cached workflow state
            machines (default 1000 in temporalio).
    """

    model_config = {"frozen": True}

    max_concurrent_activities: int | None = None
    max_concurrent_workflow_tasks: int | None = None
    max_concurrent_local_activities: int | None = None
    max_cached_workflows: int | None = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_concurrency(cls, concurrency: int) -> WorkerTuningConfig:
        """Convenience constructor that sets ``max_concurrent_activities``.

        Use this when you only care about activity concurrency and want
        Temporal defaults for everything else.

        Args:
            concurrency: Number of activities that may run in parallel.
        """
        return cls(max_concurrent_activities=concurrency)

    # ------------------------------------------------------------------
    # Temporal kwargs
    # ------------------------------------------------------------------

    def to_worker_kwargs(self) -> dict[str, Any]:
        """Produce a dict of non-None worker kwargs.

        Keys match the real ``temporalio.worker.Worker.__init__`` parameter
        names.  Only fields that are not ``None`` are included so that
        Temporal's built-in defaults apply for anything not explicitly set.
        """
        mapping: list[tuple[int | None, str]] = [
            (self.max_concurrent_activities, "max_concurrent_activities"),
            (self.max_concurrent_workflow_tasks, "max_concurrent_workflow_tasks"),
            (self.max_concurrent_local_activities, "max_concurrent_local_activities"),
            (self.max_cached_workflows, "max_cached_workflows"),
        ]
        return {kwarg: value for value, kwarg in mapping if value is not None}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_file_bytes(path: str, env_var: str) -> bytes:
    """Read raw bytes from *path*, raising a clear error on failure."""
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError as exc:
        msg = f"Cannot read {env_var}={path!r}: {exc}"
        raise ValueError(msg) from exc
