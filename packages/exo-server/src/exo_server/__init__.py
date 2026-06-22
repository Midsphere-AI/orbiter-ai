"""Exo Server: Web UI and API server."""

from importlib.metadata import PackageNotFoundError, version

from exo_server.agents import AgentInfo, WorkspaceFile, WorkspaceFileContent
from exo_server.app import (
    ChatRequest,
    ChatResponse,
    InjectRequest,
    create_app,
    register_agent,
    serve,
)
from exo_server.sessions import (
    AppendMessageRequest,
    CreateSessionRequest,
    Session,
    SessionMessage,
    SessionSummary,
    UpdateSessionRequest,
)

try:
    __version__: str = version("exo-server")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__: list[str] = [
    "AgentInfo",
    "AppendMessageRequest",
    "ChatRequest",
    "ChatResponse",
    "CreateSessionRequest",
    "InjectRequest",
    "Session",
    "SessionMessage",
    "SessionSummary",
    "UpdateSessionRequest",
    "WorkspaceFile",
    "WorkspaceFileContent",
    "__version__",
    "create_app",
    "register_agent",
    "serve",
]
