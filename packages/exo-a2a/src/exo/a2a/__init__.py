"""Exo A2A: Agent-to-Agent protocol."""

from exo.a2a.client import (  # pyright: ignore[reportMissingImports]
    A2AClient,
    A2AClientError,
    ClientManager,
    RemoteAgent,
)
from exo.a2a.server import (  # pyright: ignore[reportMissingImports]
    A2AServer,
    A2AServerError,
    AgentExecutor,
    InMemoryTaskStore,
    TaskStore,
)
from exo.a2a.types import (  # pyright: ignore[reportMissingImports]
    A2ATaskStatus,
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    ClientConfig,
    ServingConfig,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TransportMode,
)

__version__: str = "0.1.0"

__all__: list[str] = [
    "A2AClient",
    "A2AClientError",
    "A2AServer",
    "A2AServerError",
    "A2ATaskStatus",
    "AgentCapabilities",
    "AgentCard",
    "AgentExecutor",
    "AgentSkill",
    "ClientConfig",
    "ClientManager",
    "InMemoryTaskStore",
    "RemoteAgent",
    "ServingConfig",
    "TaskArtifactUpdateEvent",
    "TaskState",
    "TaskStatus",
    "TaskStatusUpdateEvent",
    "TaskStore",
    "TransportMode",
]
