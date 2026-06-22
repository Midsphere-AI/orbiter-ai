"""Sandbox interface for safe agent execution."""

from __future__ import annotations

import inspect
import logging
import uuid
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any

from exo.types import ExoError  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)

#: A registered tool handler: ``async (sandbox, arguments) -> result``.
ToolHandler = Callable[["Sandbox", dict[str, Any]], Awaitable[Any]]


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
        "_registered_tools",
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
        self._registered_tools: dict[str, ToolHandler] = {}

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

    # -- custom tool registration -------------------------------------------

    def register_tool(self, tool_or_name: Any, handler: ToolHandler | None = None) -> None:
        """Register a custom tool for :meth:`run_tool` dispatch.

        Two equivalent forms are accepted, so the same call works on every
        sandbox backend:

        * ``register_tool(tool)`` — register a ``Tool``-like object (anything
          with a non-empty string ``.name`` and a callable
          ``.execute(**arguments)``; ``execute`` may be sync or async).
        * ``register_tool("name", handler)`` — register an async *handler* with
          signature ``async (sandbox, arguments) -> result``.

        Registered tools take priority over a backend's built-in dispatch.

        Raises
        ------
        SandboxError
            If the arguments are malformed or *name* is already registered.
        """
        if handler is None:
            tool = tool_or_name
            name = getattr(tool, "name", None)
            if not isinstance(name, str) or not name:
                msg = (
                    "register_tool(tool) expects a Tool with a non-empty string .name; "
                    "to register a bare callable use register_tool(name, handler)."
                )
                raise SandboxError(msg)
            if not callable(getattr(tool, "execute", None)):
                msg = f"Tool {name!r} has no callable .execute(**arguments) method."
                raise SandboxError(msg)
            key = name
            resolved = self._tool_to_handler(tool)
        else:
            if not isinstance(tool_or_name, str) or not tool_or_name:
                msg = "register_tool(name, handler): name must be a non-empty string."
                raise SandboxError(msg)
            if not callable(handler):
                msg = "register_tool(name, handler): handler must be callable."
                raise SandboxError(msg)
            key = tool_or_name
            resolved = handler

        if key in self._registered_tools:
            msg = f"Tool {key!r} is already registered"
            raise SandboxError(msg)
        self._registered_tools[key] = resolved
        logger.debug("Sandbox %s: registered tool %r", self._sandbox_id, key)

    def unregister_tool(self, name: str) -> None:
        """Remove a previously registered tool.

        Raises :class:`SandboxError` if no tool with *name* is registered.
        """
        if name not in self._registered_tools:
            msg = f"Tool {name!r} is not registered"
            raise SandboxError(msg)
        del self._registered_tools[name]
        logger.debug("Sandbox %s: unregistered tool %r", self._sandbox_id, name)

    @property
    def registered_tools(self) -> list[str]:
        """Names of all registered custom tools."""
        return list(self._registered_tools)

    @staticmethod
    def _tool_to_handler(tool: Any) -> ToolHandler:
        """Adapt a ``Tool``-like object to the ``(sandbox, arguments)`` handler form."""

        async def _handler(_sandbox: Sandbox, arguments: dict[str, Any]) -> Any:
            result = tool.execute(**arguments)
            if inspect.isawaitable(result):
                result = await result
            return result

        return _handler

    async def _run_registered(self, tool_name: str, arguments: dict[str, Any]) -> tuple[bool, Any]:
        """Dispatch to a registered tool if one matches *tool_name*.

        Returns ``(handled, result)``. ``handled`` is ``False`` when nothing is
        registered under *tool_name*, letting the caller fall through to its own
        built-in dispatch. Handler errors are wrapped in :class:`SandboxError`.
        """
        handler = self._registered_tools.get(tool_name)
        if handler is None:
            return False, None
        try:
            return True, await handler(self, arguments)
        except SandboxError:
            raise
        except Exception as exc:
            raise SandboxError(
                f"Registered tool {tool_name!r} failed.",
                context={"sandbox_id": self._sandbox_id, "tool": tool_name},
                hint=f"Check the registered handler for {tool_name!r}.",
            ) from exc

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
            "registered_tools": list(self._registered_tools),
        }

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(sandbox_id={self._sandbox_id!r}, status={self._status.value!r})"
        )
