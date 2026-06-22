"""Optional Temporal integration for durable workflow execution.

When ``temporalio`` is installed, provides :class:`TemporalExecutor` as an
alternative execution backend for the distributed worker.  Temporal wraps agent
execution in a durable workflow with heartbeating activities, so tasks survive
worker crashes and are retried per exo's error taxonomy.

This package preserves the public import surface of the former single-module
``exo.distributed.temporal`` (``HAS_TEMPORAL``, ``TemporalExecutor``,
``AgentExecutionWorkflow``, ``execute_agent_activity`` and the env helpers) and
adds the Phase-1 production surface: grouped connection/retry/timeout/tuning
configs, an AES payload codec + claim-check, an ``ExoError``-aware failure
converter, and OTel/baggage interceptors.

**Two execution paths.** The original :class:`AgentExecutionWorkflow` runs a
whole agent (or ``Swarm``) inside a *single activity* — crash-resumable via
heartbeat checkpointing, and the only path that supports ``Swarm``.  The
Phase-2 :class:`AgentLoopWorkflow` (opt in with ``TemporalExecutor(durable=True)``)
lifts the agent's step loop into the workflow so each LLM turn
(:func:`agent_llm_turn`) and tool call (:func:`agent_tool_call`) becomes its own
activity recorded in Temporal history.  A crashed worker then resumes from the
**last completed step**, and the run accepts signals (mid-run message
injection), queries (live progress), and updates (steering / model swap), with
continue-as-new for long runs.  Single-agent only; ``Swarm`` as child workflows
remains a later phase.  Workers serving the durable loop must use
:func:`build_workflow_runner` (the executor wires this automatically) so the
``exo`` package is passed through Temporal's reimport sandbox.

Configuration objects (:class:`ConnectionConfig`, :class:`RetryConfig`,
:class:`TimeoutConfig`, :class:`WorkerTuningConfig`) and the error-classification
helpers import fine without ``temporalio``; the codec, converter, interceptor,
activity and workflow objects require it.

Install the optional dependency group::

    pip install exo-distributed[temporal]
"""

from __future__ import annotations

from exo.distributed.temporal._compat import (  # pyright: ignore[reportMissingImports]
    HAS_TEMPORAL,
    _get_temporal_host,
    _get_temporal_namespace,
)
from exo.distributed.temporal.config import (  # pyright: ignore[reportMissingImports]
    ConnectionConfig,
    RetryConfig,
    TimeoutConfig,
    WorkerTuningConfig,
)
from exo.distributed.temporal.executor import (  # pyright: ignore[reportMissingImports]
    TemporalExecutor,
    build_workflow_runner,
)
from exo.distributed.temporal.failure import (  # pyright: ignore[reportMissingImports]
    NON_RETRYABLE_ERROR_TYPES,
    is_retryable,
    to_application_error,
)

if HAS_TEMPORAL:
    from exo.distributed.temporal.activities import (  # pyright: ignore[reportMissingImports]
        agent_llm_turn,
        agent_tool_call,
        execute_agent_activity,
    )
    from exo.distributed.temporal.codec import (  # pyright: ignore[reportMissingImports]
        AESGCMPayloadCodec,
        BlobStore,
        ChainedPayloadCodec,
        ClaimCheckCodec,
        InMemoryBlobStore,
        RedisBlobStore,
    )
    from exo.distributed.temporal.converter import (  # pyright: ignore[reportMissingImports]
        build_data_converter,
        pydantic_data_converter,
    )
    from exo.distributed.temporal.interceptors import (  # pyright: ignore[reportMissingImports]
        ExoBaggageInterceptor,
        exo_interceptors,
    )
    from exo.distributed.temporal.workflows import (  # pyright: ignore[reportMissingImports]
        AgentExecutionWorkflow,
        AgentLoopWorkflow,
    )
else:  # pragma: no cover - exercised only without temporalio
    execute_agent_activity = None  # type: ignore[assignment]
    agent_llm_turn = None  # type: ignore[assignment]
    agent_tool_call = None  # type: ignore[assignment]
    AgentExecutionWorkflow = None  # type: ignore[assignment]
    AgentLoopWorkflow = None  # type: ignore[assignment]
    AESGCMPayloadCodec = None  # type: ignore[assignment,misc]
    ChainedPayloadCodec = None  # type: ignore[assignment,misc]
    ClaimCheckCodec = None  # type: ignore[assignment,misc]
    BlobStore = None  # type: ignore[assignment,misc]
    InMemoryBlobStore = None  # type: ignore[assignment,misc]
    RedisBlobStore = None  # type: ignore[assignment,misc]
    build_data_converter = None  # type: ignore[assignment]
    pydantic_data_converter = None  # type: ignore[assignment]
    ExoBaggageInterceptor = None  # type: ignore[assignment,misc]
    exo_interceptors = None  # type: ignore[assignment]

__all__ = [
    "HAS_TEMPORAL",
    "NON_RETRYABLE_ERROR_TYPES",
    "AESGCMPayloadCodec",
    "AgentExecutionWorkflow",
    "AgentLoopWorkflow",
    "BlobStore",
    "ChainedPayloadCodec",
    "ClaimCheckCodec",
    "ConnectionConfig",
    "ExoBaggageInterceptor",
    "InMemoryBlobStore",
    "RedisBlobStore",
    "RetryConfig",
    "TemporalExecutor",
    "TimeoutConfig",
    "WorkerTuningConfig",
    "_get_temporal_host",
    "_get_temporal_namespace",
    "agent_llm_turn",
    "agent_tool_call",
    "build_data_converter",
    "build_workflow_runner",
    "execute_agent_activity",
    "exo_interceptors",
    "is_retryable",
    "pydantic_data_converter",
    "to_application_error",
]
