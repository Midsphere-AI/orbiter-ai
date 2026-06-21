"""Sandbox interface for safe agent execution."""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any

from exo.types import ExoError  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)


class SandboxError(ExoError):
    """Raised for sandbox-level errors."""


# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------

_VALID_TRANSITIONS: dict[str, set[str]] = {
    "init": {"running", "closed"},
    "running": {"idle", "error", "closed"},
    "idle": {"running", "error", "closed"},
    "error": {"running", "closed"},
    "closed": set(),
}


class SandboxStatus(StrEnum):
    """Lifecycle states for a sandbox."""

    CLOSED = "closed"
    ERROR = "error"
    IDLE = "idle"
    INIT = "init"
    RUNNING = "running"


# ---------------------------------------------------------------------------
# Sandbox ABC
# ---------------------------------------------------------------------------


class Sandbox(ABC):
    """Abstract sandbox providing isolated execution for agents.

    Subclasses implement ``start``, ``stop``, and ``cleanup`` to manage the
    concrete environment (local process, Kubernetes pod, etc.).
    """

    __slots__ = (
        "_agents",
        "_mcp_config",
        "_sandbox_id",
        "_status",
        "_timeout",
        "_workspace",
    )

    def __init__(
        self,
        *,
        sandbox_id: str | None = None,
        workspace: list[str] | None = None,
        mcp_config: dict[str, Any] | None = None,
        agents: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._sandbox_id = sandbox_id or uuid.uuid4().hex[:12]
        self._workspace = list(workspace) if workspace else []
        self._mcp_config = dict(mcp_config) if mcp_config else {}
        self._agents = dict(agents) if agents else {}
        self._timeout = timeout
        self._status = SandboxStatus.INIT

    # -- properties ---------------------------------------------------------

    @property
    def sandbox_id(self) -> str:
        return self._sandbox_id

    @property
    def status(self) -> SandboxStatus:
        return self._status

    @property
    def workspace(self) -> list[str]:
        return list(self._workspace)

    @property
    def mcp_config(self) -> dict[str, Any]:
        return dict(self._mcp_config)

    @property
    def agents(self) -> dict[str, Any]:
        return dict(self._agents)

    @property
    def timeout(self) -> float:
        return self._timeout

    # -- status transitions -------------------------------------------------

    def _transition(self, target: SandboxStatus) -> None:
        allowed = _VALID_TRANSITIONS.get(self._status.value, set())
        if target.value not in allowed:
            msg = f"Cannot transition from {self._status!r} to {target!r}"
            raise SandboxError(msg)
        logger.debug("Sandbox %s: %s -> %s", self._sandbox_id, self._status.value, target.value)
        self._status = target

    # -- abstract lifecycle -------------------------------------------------

    @abstractmethod
    async def start(self) -> None:
        """Start the sandbox environment."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop the sandbox (may be restarted later)."""

    @abstractmethod
    async def cleanup(self) -> None:
        """Release all resources and close the sandbox permanently."""

    @abstractmethod
    async def run_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a named tool within this sandbox.

        Raises :class:`SandboxError` if the sandbox is not running or the
        tool cannot be found/executed.
        """

    # -- convenience --------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            "sandbox_id": self._sandbox_id,
            "status": self._status.value,
            "workspace": self._workspace,
            "timeout": self._timeout,
        }

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(sandbox_id={self._sandbox_id!r}, status={self._status.value!r})"
        )
