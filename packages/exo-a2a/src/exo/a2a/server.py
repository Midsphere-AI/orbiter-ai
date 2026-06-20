"""A2A server — FastAPI-based agent serving with agent card discovery."""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from exo.a2a.types import (  # pyright: ignore[reportMissingImports]
    A2ATaskStatus,
    AgentCapabilities,
    AgentCard,
    ServingConfig,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
)
from exo.types import ExoError  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)


class A2AServerError(ExoError):
    """Raised for A2A server-level errors."""


# ---------------------------------------------------------------------------
# Task store abstraction
# ---------------------------------------------------------------------------


@runtime_checkable
class TaskStore(Protocol):
    """Minimal storage interface for A2A task state."""

    async def get(self, task_id: str) -> dict[str, Any] | None: ...
    async def save(self, task_id: str, data: dict[str, Any]) -> None: ...
    async def delete(self, task_id: str) -> None: ...


class InMemoryTaskStore:
    """Simple in-memory task store for development and testing."""

    __slots__ = ("_tasks",)

    def __init__(self) -> None:
        self._tasks: dict[str, dict[str, Any]] = {}

    async def get(self, task_id: str) -> dict[str, Any] | None:
        return self._tasks.get(task_id)

    async def save(self, task_id: str, data: dict[str, Any]) -> None:
        self._tasks[task_id] = data

    async def delete(self, task_id: str) -> None:
        self._tasks.pop(task_id, None)

    def __repr__(self) -> str:
        return f"InMemoryTaskStore(tasks={len(self._tasks)})"


# ---------------------------------------------------------------------------
# Agent executor — wraps an Exo Agent for A2A task execution
# ---------------------------------------------------------------------------


class AgentExecutor:
    """Wraps an agent for A2A task execution.

    Accepts any object with a ``run(input, ...)`` async method and a
    ``name`` attribute (i.e. an Exo ``Agent``).
    """

    __slots__ = ("_agent", "_streaming")

    def __init__(self, agent: Any, *, streaming: bool = False) -> None:
        self._agent = agent
        self._streaming = streaming

    async def execute(self, text: str, *, provider: Any = None) -> str:
        """Run the wrapped agent and return text output.

        Handles both ``AgentOutput`` (has ``.text``) returned by ``Agent.run()``
        and ``RunResult`` (has ``.output``) returned by the public ``run()``
        function or a ``Swarm``.  Falls back gracefully when neither attribute
        is present.

        Priority: ``.text`` (AgentOutput) → ``.output`` (RunResult/Swarm) → str().
        """
        kwargs: dict[str, Any] = {}
        if provider is not None:
            kwargs["provider"] = provider
        result = await self._agent.run(text, **kwargs)
        # AgentOutput exposes .text; RunResult (and Swarm output) exposes .output.
        # Use _has_attr sentinel to distinguish "attribute absent" from "value is None".
        _sentinel = object()
        text_val = getattr(result, "text", _sentinel)
        if text_val is not _sentinel:
            return text_val or ""
        output_val = getattr(result, "output", _sentinel)
        if output_val is not _sentinel:
            return output_val or ""
        return str(result) if result is not None else ""

    @property
    def agent_name(self) -> str:
        return getattr(self._agent, "name", "unknown")

    def __repr__(self) -> str:
        return f"AgentExecutor(agent={self.agent_name!r}, streaming={self._streaming})"


# ---------------------------------------------------------------------------
# A2A server
# ---------------------------------------------------------------------------


class A2AServer:
    """FastAPI-based A2A server with agent card discovery.

    Exposes:
    - ``GET /.well-known/agent-card`` — agent card JSON
    - ``POST /`` — task execution (send text, get response)

    Args:
        executor: An ``AgentExecutor`` wrapping the agent to serve.
        config: Server configuration (host, port, skills, etc.).
        task_store: Task state storage. Defaults to ``InMemoryTaskStore``.
        provider: Optional LLM provider passed through to the agent.
    """

    __slots__ = ("_agent_card", "_app", "_config", "_executor", "_provider", "_task_store")

    def __init__(
        self,
        executor: AgentExecutor,
        config: ServingConfig | None = None,
        *,
        task_store: TaskStore | None = None,
        provider: Any = None,
    ) -> None:
        self._executor = executor
        self._config = config or ServingConfig()
        self._task_store: TaskStore = task_store or InMemoryTaskStore()
        self._provider = provider
        self._agent_card = self._build_agent_card()
        self._app: Any = None

    def _build_agent_card(self) -> AgentCard:
        """Construct the agent card from executor + config."""
        cfg = self._config
        return AgentCard(
            name=self._executor.agent_name,
            description=f"A2A agent: {self._executor.agent_name}",
            version=cfg.version,
            url=f"http://{cfg.host}:{cfg.port}{cfg.endpoint}",
            capabilities=AgentCapabilities(streaming=cfg.streaming),
            skills=cfg.skills,
            default_input_modes=cfg.input_modes,
            default_output_modes=cfg.output_modes,
            supported_transports=cfg.transports,
        )

    @property
    def agent_card(self) -> AgentCard:
        return self._agent_card

    @property
    def task_store(self) -> TaskStore:
        return self._task_store

    def build_app(self) -> Any:
        """Create and return the FastAPI application."""
        try:
            from fastapi import FastAPI
            from fastapi.responses import JSONResponse
        except ImportError as exc:
            raise A2AServerError("fastapi is required for A2AServer: pip install fastapi") from exc

        app = FastAPI(
            title=f"A2A: {self._executor.agent_name}",
            version=self._config.version,
        )

        server = self

        @app.get("/.well-known/agent-card")
        async def get_agent_card() -> JSONResponse:
            return JSONResponse(server._agent_card.model_dump(mode="json"))

        @app.post("/")
        async def execute_task(payload: dict[str, Any]) -> JSONResponse:
            text = payload.get("text", "")
            task_id = payload.get("task_id") or str(uuid.uuid4())

            logger.info("A2A run start: task_id=%s agent=%s", task_id, server._executor.agent_name)

            # Save initial task state
            status = A2ATaskStatus(state=TaskState.SUBMITTED)
            await server._task_store.save(
                task_id,
                {
                    "task_id": task_id,
                    "status": status.model_dump(),
                    "text": text,
                },
            )

            # Update to working
            working = TaskStatusUpdateEvent(
                task_id=task_id,
                status=A2ATaskStatus(state=TaskState.WORKING),
            )
            await server._task_store.save(
                task_id,
                {
                    "task_id": task_id,
                    "status": working.status.model_dump(),
                    "text": text,
                },
            )

            try:
                result = await server._executor.execute(text, provider=server._provider)

                # Save completed state
                completed_status = A2ATaskStatus(state=TaskState.COMPLETED)
                artifact = TaskArtifactUpdateEvent(task_id=task_id, text=result, last_chunk=True)
                await server._task_store.save(
                    task_id,
                    {
                        "task_id": task_id,
                        "status": completed_status.model_dump(),
                        "result": result,
                    },
                )

                logger.info(
                    "A2A run complete: task_id=%s agent=%s", task_id, server._executor.agent_name
                )
                return JSONResponse(
                    {
                        "task_id": task_id,
                        "status": completed_status.model_dump(),
                        "artifact": artifact.model_dump(),
                    }
                )

            except Exception as exc:
                logger.error(
                    "A2A run failed: task_id=%s agent=%s error=%s",
                    task_id,
                    server._executor.agent_name,
                    exc,
                    exc_info=True,
                )
                failed_status = A2ATaskStatus(state=TaskState.FAILED, reason=str(exc))
                await server._task_store.save(
                    task_id,
                    {
                        "task_id": task_id,
                        "status": failed_status.model_dump(),
                        "error": str(exc),
                    },
                )
                return JSONResponse(
                    {
                        "task_id": task_id,
                        "status": failed_status.model_dump(),
                    },
                    status_code=500,
                )

        @app.get("/tasks/{task_id}")
        async def get_task(task_id: str) -> JSONResponse:
            data = await server._task_store.get(task_id)
            if data is None:
                return JSONResponse({"error": f"Task {task_id!r} not found"}, status_code=404)
            return JSONResponse(data)

        if server._config.streaming:

            @app.post("/stream")
            async def stream_task(payload: dict[str, Any]) -> Any:
                from starlette.responses import StreamingResponse

                text = payload.get("text", "")
                task_id = payload.get("task_id") or str(uuid.uuid4())

                async def _generate() -> AsyncIterator[str]:
                    import json

                    logger.info(
                        "A2A stream start: task_id=%s agent=%s",
                        task_id,
                        server._executor.agent_name,
                    )
                    yield (
                        json.dumps(
                            TaskStatusUpdateEvent(
                                task_id=task_id,
                                status=A2ATaskStatus(state=TaskState.WORKING),
                            ).model_dump()
                        )
                        + "\n"
                    )

                    try:
                        result = await server._executor.execute(text, provider=server._provider)
                        logger.info(
                            "A2A stream complete: task_id=%s agent=%s",
                            task_id,
                            server._executor.agent_name,
                        )
                        yield (
                            json.dumps(
                                TaskArtifactUpdateEvent(
                                    task_id=task_id, text=result, last_chunk=True
                                ).model_dump()
                            )
                            + "\n"
                        )
                        yield (
                            json.dumps(
                                TaskStatusUpdateEvent(
                                    task_id=task_id,
                                    status=A2ATaskStatus(state=TaskState.COMPLETED),
                                ).model_dump()
                            )
                            + "\n"
                        )
                    except Exception as exc:
                        logger.error(
                            "A2A stream failed: task_id=%s agent=%s error=%s",
                            task_id,
                            server._executor.agent_name,
                            exc,
                            exc_info=True,
                        )
                        yield (
                            json.dumps(
                                TaskStatusUpdateEvent(
                                    task_id=task_id,
                                    status=A2ATaskStatus(state=TaskState.FAILED, reason=str(exc)),
                                ).model_dump()
                            )
                            + "\n"
                        )

                return StreamingResponse(_generate(), media_type="application/x-ndjson")

        self._app = app
        return app

    def __repr__(self) -> str:
        return (
            f"A2AServer(agent={self._executor.agent_name!r}, "
            f"host={self._config.host!r}, port={self._config.port})"
        )
