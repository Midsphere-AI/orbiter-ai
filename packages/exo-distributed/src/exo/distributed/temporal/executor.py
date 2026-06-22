"""``TemporalExecutor`` — the Temporal control plane for the distributed worker.

Submits agent tasks as durable Temporal workflows instead of executing them
directly, so tasks survive worker crashes and are retried per their policy.

Construction follows exo's additive namespace pattern: the flat ``host`` /
``namespace`` kwargs still work, or pass a grouped
:class:`~exo.distributed.temporal.config.ConnectionConfig`; supplying both a
config and a conflicting flat kwarg raises :exc:`ValueError`.  Production
concerns — TLS/mTLS/API-key connections, at-rest payload encryption, OTel trace
continuity, and worker tuning — are wired through the grouped configs.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from exo.distributed.cancel import CancellationToken  # pyright: ignore[reportMissingImports]
from exo.distributed.events import EventPublisher  # pyright: ignore[reportMissingImports]
from exo.distributed.models import TaskPayload  # pyright: ignore[reportMissingImports]
from exo.distributed.store import TaskStore  # pyright: ignore[reportMissingImports]
from exo.distributed.temporal._compat import (  # pyright: ignore[reportMissingImports]
    _DEFAULT_TASK_QUEUE,
    HAS_TEMPORAL,
    TemporalClient,
    TemporalWorker,
    _get_temporal_host,
    _get_temporal_namespace,
)
from exo.distributed.temporal.config import (  # pyright: ignore[reportMissingImports]
    ConnectionConfig,
    RetryConfig,
    TimeoutConfig,
    WorkerTuningConfig,
)
from exo.distributed.temporal.failure import (  # pyright: ignore[reportMissingImports]
    NON_RETRYABLE_ERROR_TYPES,
)

if TYPE_CHECKING:
    from exo.distributed.temporal._compat import (  # pyright: ignore[reportMissingImports]
        PayloadCodec,
    )

logger = logging.getLogger(__name__)

__all__ = ["TemporalExecutor", "build_workflow_runner"]


def build_workflow_runner() -> Any:
    """Build a Temporal workflow sandbox runner that passes through ``exo``.

    exo's workflows live under the ``exo.*`` namespace, so the sandbox's
    reimport of a workflow module runs ``exo/__init__.py`` — which eagerly
    imports ``exo.agent``/``exo.skills`` and trips the sandbox on a module-load
    ``pathlib.Path.home()`` call.  Marking ``exo`` as a passthrough module loads
    it once from the host instead of reimporting it in the sandbox.  This is
    safe: the durable workflow only calls pure ``config`` conversions and
    manipulates plain dicts in deterministic code; all exo I/O happens in
    activities, which run outside the sandbox.

    Both the executor's worker and any test/embedding worker should pass this as
    ``Worker(..., workflow_runner=build_workflow_runner())`` so the durable
    workflow validates and replays correctly.
    """
    from temporalio.worker.workflow_sandbox import (  # pyright: ignore[reportMissingImports]
        SandboxedWorkflowRunner,
        SandboxRestrictions,
    )

    return SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules("exo")
    )


class TemporalExecutor:
    """Alternative execution backend using Temporal for durable workflows.

    Args:
        host: Temporal server address.  Mutually exclusive with *connection=*.
            Defaults to ``TEMPORAL_HOST`` env var or ``localhost:7233``.
        namespace: Temporal namespace.  Mutually exclusive with *connection=*.
            Defaults to ``TEMPORAL_NAMESPACE`` env var or ``default``.
        task_queue: Temporal task queue name (default ``exo-tasks``).
        connection: Grouped connection config (:class:`ConnectionConfig`) —
            supplies host/namespace plus TLS/mTLS/API-key settings together.
            Raises :exc:`ValueError` if *host*/*namespace* are also set.
        codec: Optional payload codec (e.g.
            :class:`~exo.distributed.temporal.codec.AESGCMPayloadCodec`) to
            encrypt agent payloads at rest in Temporal's history.
        tuning: Worker concurrency tuning (:class:`WorkerTuningConfig`).
        interceptors: When ``True`` (default), install exo's OTel + baggage
            interceptors so traces span the workflow→activity boundary.
        retry: Default per-task **activity** retry policy.  Defaults to a policy
            that marks exo's deterministic errors non-retryable
            (:data:`NON_RETRYABLE_ERROR_TYPES`).
        timeouts: Default per-task timeouts (:class:`TimeoutConfig`), covering
            both activity- and workflow-level timeouts.
        workflow_retry: Optional retry policy for the **whole workflow run**
            (distinct from the per-activity *retry*).
        durable: When ``True``, tasks run through :class:`AgentLoopWorkflow` —
            the durable *step-as-activity* loop where each LLM turn and tool
            call is its own activity, so a crashed worker resumes from the last
            completed step (and the run accepts signals/queries/updates).  When
            ``False`` (default), tasks run through the single-activity
            :class:`AgentExecutionWorkflow` (the only path that supports Swarm).
        max_steps: Hard cap on LLM turns per durable run (default 50).  Ignored
            unless *durable* is ``True``.
    """

    def __init__(
        self,
        *,
        host: str | None = None,
        namespace: str | None = None,
        task_queue: str = _DEFAULT_TASK_QUEUE,
        connection: ConnectionConfig | None = None,
        codec: PayloadCodec | None = None,
        tuning: WorkerTuningConfig | None = None,
        interceptors: bool = True,
        retry: RetryConfig | None = None,
        timeouts: TimeoutConfig | None = None,
        workflow_retry: RetryConfig | None = None,
        durable: bool = False,
        max_steps: int = 50,
    ) -> None:
        if not HAS_TEMPORAL:
            msg = (
                "temporalio is not installed. "
                "Install it with: pip install exo-distributed[temporal]"
            )
            raise ImportError(msg)

        # --- grouped vs flat connection resolution (additive namespace pattern) ---
        if connection is not None:
            if host is not None:
                raise ValueError(
                    "Cannot specify both 'connection=' and 'host='. "
                    "Set host via ConnectionConfig(host=...) instead."
                )
            if namespace is not None:
                raise ValueError(
                    "Cannot specify both 'connection=' and 'namespace='. "
                    "Set namespace via ConnectionConfig(namespace=...) instead."
                )
            self._connection = connection
        else:
            self._connection = ConnectionConfig(
                host=host or _get_temporal_host(),
                namespace=namespace or _get_temporal_namespace(),
            )

        self._task_queue = task_queue
        self._codec = codec
        self._tuning = tuning
        self._interceptors_enabled = interceptors
        self._default_retry = retry or RetryConfig(
            non_retryable_error_types=list(NON_RETRYABLE_ERROR_TYPES)
        )
        self._default_timeouts = timeouts
        self._workflow_retry = workflow_retry
        self._durable = durable
        self._max_steps = max_steps

        self._client: Any | None = None
        self._temporal_worker: Any | None = None

    @property
    def host(self) -> str:
        return self._connection.host

    @property
    def namespace(self) -> str:
        return self._connection.namespace

    @property
    def task_queue(self) -> str:
        return self._task_queue

    @property
    def connection_config(self) -> ConnectionConfig:
        """Read-only view of the resolved connection settings."""
        return self._connection

    @property
    def durable(self) -> bool:
        """Whether tasks run through the durable step-as-activity loop."""
        return self._durable

    def _build_interceptors(self) -> list[Any]:
        from exo.distributed.temporal.interceptors import (  # pyright: ignore[reportMissingImports]
            exo_interceptors,
        )

        return exo_interceptors() if self._interceptors_enabled else []

    async def connect(self) -> None:
        """Connect to the Temporal server with the configured converter/interceptors."""
        from exo.distributed.temporal.converter import (  # pyright: ignore[reportMissingImports]
            build_data_converter,
        )

        kwargs = self._connection.to_connect_kwargs()
        kwargs["data_converter"] = build_data_converter(self._codec)
        interceptors = self._build_interceptors()
        if interceptors:
            kwargs["interceptors"] = interceptors

        logger.debug(
            "TemporalExecutor connecting to %s (namespace=%s, tls=%s, codec=%s)",
            self._connection.host,
            self._connection.namespace,
            "tls" in kwargs,
            self._codec is not None,
        )
        self._client = await TemporalClient.connect(**kwargs)  # pyright: ignore[reportOptionalMemberAccess]
        logger.info("TemporalExecutor connected to %s", self._connection.host)

    async def disconnect(self) -> None:
        """Clean up Temporal resources.

        Stops the Temporal worker (if running) before releasing the client
        reference so the worker task is not orphaned.
        """
        await self.stop_temporal_worker()
        self._client = None
        self._temporal_worker = None
        logger.debug("TemporalExecutor disconnected")

    def _build_payload_json(self, task: TaskPayload) -> str:
        """Serialize a task, injecting default retry/timeout config when unset."""
        data = task.model_dump()
        if data.get("retry_policy") is None:
            data["retry_policy"] = self._default_retry.to_dict()
        if data.get("timeouts") is None and self._default_timeouts is not None:
            data["timeouts"] = self._default_timeouts.to_dict()
        return json.dumps(data)

    def _start_workflow_kwargs(self) -> dict[str, Any]:
        """Build workflow-level kwargs (timeouts + whole-run retry)."""
        kwargs: dict[str, Any] = {}
        if self._default_timeouts is not None:
            kwargs.update(self._default_timeouts.to_workflow_timeouts())
        if self._workflow_retry is not None:
            kwargs["retry_policy"] = self._workflow_retry.to_retry_policy()
        return kwargs

    def _build_loop_input(self, task: TaskPayload) -> dict[str, Any]:
        """Build the :class:`AgentLoopWorkflow` input dict from a task payload."""
        loop_input: dict[str, Any] = {
            "agent_config": task.agent_config,
            "input": task.input,
            "max_steps": self._max_steps,
        }
        if task.retry_policy is not None:
            loop_input["retry_policy"] = task.retry_policy
        elif self._default_retry is not None:
            loop_input["retry_policy"] = self._default_retry.to_dict()
        if task.timeouts is not None:
            loop_input["timeouts"] = task.timeouts
        elif self._default_timeouts is not None:
            loop_input["timeouts"] = self._default_timeouts.to_dict()
        return loop_input

    def workflow_id_for(self, task_id: str) -> str:
        """Return the deterministic Temporal workflow id for a task id."""
        return f"exo-task-{task_id}"

    async def execute_task(
        self,
        task: TaskPayload,
        store: TaskStore,
        publisher: EventPublisher,
        token: CancellationToken,
        worker_id: str,
    ) -> str:
        """Submit a task as a Temporal workflow and wait for the result.

        Routes to :class:`AgentLoopWorkflow` (durable step-as-activity loop)
        when the executor was constructed with ``durable=True``, otherwise to
        the single-activity :class:`AgentExecutionWorkflow`.  Either way the
        *token* is honoured **bidirectionally**: a cancel flipped on the token
        mid-run propagates to Temporal via ``handle.cancel()`` so the in-flight
        workflow actually stops, rather than running to completion.

        Args:
            task: The task payload to execute.
            store: Task status store for state updates.
            publisher: Event publisher (events published by the activity).
            token: Cancellation token (checked before starting and watched
                concurrently while the workflow runs).
            worker_id: ID of the submitting worker.

        Returns:
            The text output from agent execution.
        """
        from exo.distributed.temporal.workflows import (  # pyright: ignore[reportMissingImports]
            AgentExecutionWorkflow,
            AgentLoopWorkflow,
        )

        if self._client is None:
            msg = "TemporalExecutor is not connected. Call connect() first."
            raise RuntimeError(msg)

        if token.cancelled:
            return ""

        workflow_id = self.workflow_id_for(task.task_id)

        if self._durable:
            logger.info(
                "TemporalExecutor starting durable loop %s (task=%s)", workflow_id, task.task_id
            )
            handle = await self._client.start_workflow(
                AgentLoopWorkflow.run,
                self._build_loop_input(task),
                id=workflow_id,
                task_queue=self._task_queue,
                **self._start_workflow_kwargs(),
            )
            return await self._await_with_cancel(handle, token, workflow_id)

        payload_json = self._build_payload_json(task)
        logger.info("TemporalExecutor starting workflow %s (task=%s)", workflow_id, task.task_id)
        handle = await self._client.start_workflow(
            AgentExecutionWorkflow.run,
            payload_json,
            id=workflow_id,
            task_queue=self._task_queue,
            **self._start_workflow_kwargs(),
        )

        result_json = await self._await_with_cancel(handle, token, workflow_id)
        if not result_json:
            return ""
        result_data = json.loads(result_json)
        logger.debug("TemporalExecutor workflow %s completed", workflow_id)
        return result_data.get("output", "")

    async def _await_with_cancel(
        self, handle: Any, token: CancellationToken, workflow_id: str
    ) -> str:
        """Await a workflow result, cancelling the handle if the token trips.

        Runs the result and a token-watcher concurrently.  If cancellation is
        requested first, ``handle.cancel()`` is sent to Temporal and an empty
        string is returned (the task is reported cancelled upstream).
        """
        result_task = asyncio.ensure_future(handle.result())
        cancel_task = asyncio.ensure_future(token.wait())
        try:
            done, _ = await asyncio.wait(
                {result_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if result_task in done:
                return result_task.result()

            # Cancellation won the race — propagate to Temporal.
            logger.info("TemporalExecutor cancelling workflow %s (token tripped)", workflow_id)
            await handle.cancel()
            try:
                await result_task
            except Exception:
                logger.debug("Workflow %s ended after cancel request", workflow_id)
            return ""
        finally:
            for t in (result_task, cancel_task):
                if not t.done():
                    t.cancel()
            await asyncio.gather(result_task, cancel_task, return_exceptions=True)

    # ------------------------------------------------------------------
    # Durable-loop control plane (signals / queries / updates)
    # ------------------------------------------------------------------

    def _require_handle(self, task_id: str) -> Any:
        """Return a workflow handle for *task_id*, or raise if not connected."""
        if self._client is None:
            msg = "TemporalExecutor is not connected. Call connect() first."
            raise RuntimeError(msg)
        return self._client.get_workflow_handle(self.workflow_id_for(task_id))

    async def inject_message(self, task_id: str, text: str) -> None:
        """Inject a user message into a running durable run (fire-and-forget)."""
        from exo.distributed.temporal.workflows import (  # pyright: ignore[reportMissingImports]
            AgentLoopWorkflow,
        )

        await self._require_handle(task_id).signal(AgentLoopWorkflow.inject_message, text)

    async def cancel_task(self, task_id: str) -> None:
        """Request a graceful stop of a running durable run via signal."""
        from exo.distributed.temporal.workflows import (  # pyright: ignore[reportMissingImports]
            AgentLoopWorkflow,
        )

        await self._require_handle(task_id).signal(AgentLoopWorkflow.cancel)

    async def get_progress(self, task_id: str) -> dict[str, Any]:
        """Query live progress of a running durable run."""
        from exo.distributed.temporal.workflows import (  # pyright: ignore[reportMissingImports]
            AgentLoopWorkflow,
        )

        return await self._require_handle(task_id).query(AgentLoopWorkflow.get_progress)

    async def steer(self, task_id: str, instruction: str) -> dict[str, Any]:
        """Send a validated steering instruction to a running durable run."""
        from exo.distributed.temporal.workflows import (  # pyright: ignore[reportMissingImports]
            AgentLoopWorkflow,
        )

        return await self._require_handle(task_id).execute_update(
            AgentLoopWorkflow.steer, instruction
        )

    async def set_model(self, task_id: str, model: str) -> str:
        """Swap the model of a running durable run for subsequent turns."""
        from exo.distributed.temporal.workflows import (  # pyright: ignore[reportMissingImports]
            AgentLoopWorkflow,
        )

        return await self._require_handle(task_id).execute_update(
            AgentLoopWorkflow.set_model, model
        )

    async def start_temporal_worker(self) -> None:
        """Start a Temporal worker that processes agent execution workflows.

        Registers the workflow and activity with Temporal — applying the
        configured concurrency tuning and interceptors — and runs until stopped.
        """
        from exo.distributed.temporal.activities import (  # pyright: ignore[reportMissingImports]
            agent_llm_turn,
            agent_tool_call,
            execute_agent_activity,
        )
        from exo.distributed.temporal.workflows import (  # pyright: ignore[reportMissingImports]
            AgentExecutionWorkflow,
            AgentLoopWorkflow,
        )

        if self._client is None:
            msg = "TemporalExecutor is not connected. Call connect() first."
            raise RuntimeError(msg)

        worker_kwargs: dict[str, Any] = {}
        if self._tuning is not None:
            worker_kwargs.update(self._tuning.to_worker_kwargs())
        interceptors = self._build_interceptors()
        if interceptors:
            worker_kwargs["interceptors"] = interceptors

        # Register both execution paths + their activities so a single worker
        # serves single-activity and durable-loop tasks on the same queue.
        self._temporal_worker = TemporalWorker(  # pyright: ignore[reportOptionalCall]
            self._client,
            task_queue=self._task_queue,
            workflows=[AgentExecutionWorkflow, AgentLoopWorkflow],
            activities=[execute_agent_activity, agent_llm_turn, agent_tool_call],
            workflow_runner=build_workflow_runner(),
            **worker_kwargs,
        )
        await self._temporal_worker.run()  # pyright: ignore[reportOptionalMemberAccess]

    async def stop_temporal_worker(self) -> None:
        """Request graceful shutdown of the Temporal worker."""
        if self._temporal_worker is not None:
            self._temporal_worker.shutdown()
