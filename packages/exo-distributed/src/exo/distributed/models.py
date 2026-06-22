"""Data models for distributed task execution."""

from __future__ import annotations

from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class TaskStatus(StrEnum):
    """Status of a distributed task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class TaskPayload(BaseModel):
    """Payload describing a task to be queued for distributed execution."""

    model_config = {"frozen": True}

    task_id: str = Field(default_factory=lambda: uuid4().hex)
    agent_config: dict[str, Any] = Field(default_factory=dict)
    input: str = ""
    messages: list[dict[str, Any]] = Field(default_factory=list)
    model: str | None = None
    detailed: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float = 0.0
    timeout_seconds: float = 300.0
    retry_policy: dict[str, Any] | None = None
    """Serialized retry configuration for durable execution backends.

    When set, durable backends (e.g. the Temporal executor) apply it as the
    per-task retry policy.  Round-trips through
    :meth:`exo.distributed.temporal.RetryConfig.to_dict` /
    :meth:`~exo.distributed.temporal.RetryConfig.from_dict`.  Backend-agnostic:
    the local executor ignores it.
    """
    timeouts: dict[str, Any] | None = None
    """Serialized timeout configuration for durable execution backends.

    Carries the activity- and workflow-level timeouts applied by durable
    backends.  Round-trips through
    :meth:`exo.distributed.temporal.TimeoutConfig.to_dict` /
    :meth:`~exo.distributed.temporal.TimeoutConfig.from_dict`.  The local
    executor uses only :attr:`timeout_seconds`.
    """


class TaskResult(BaseModel):
    """Result and status tracking for a distributed task."""

    model_config = {"frozen": True}

    task_id: str = ""
    status: TaskStatus = TaskStatus.PENDING
    result: dict[str, Any] | None = None
    error: str | None = None
    started_at: float | None = None
    completed_at: float | None = None
    worker_id: str | None = None
    retries: int = 0
