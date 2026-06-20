"""@mcp_server() class decorator and MCPServerRegistry for exposing tools as MCP servers."""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from typing import Any

from exo.types import ExoError  # pyright: ignore[reportMissingImports]
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


class MCPServerError(ExoError):
    """Error raised by MCP server operations."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class MCPServerRegistry:
    """Singleton registry for @mcp_server-decorated classes.

    Stores class references and lazily-created singleton instances.
    """

    __slots__ = ("_classes", "_instances")

    def __init__(self) -> None:
        self._classes: dict[str, type] = {}
        self._instances: dict[str, Any] = {}

    def register(self, name: str, cls: type) -> None:
        """Register a server class by name."""
        self._classes[name] = cls

    def get_class(self, name: str) -> type:
        """Get a registered server class.

        Raises:
            MCPServerError: If the name is not registered.
        """
        if name not in self._classes:
            raise MCPServerError(f"MCP server '{name}' not registered")
        return self._classes[name]

    def get_instance(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Get or create a singleton instance of a registered server.

        Raises:
            MCPServerError: If the name is not registered.
        """
        if name not in self._instances:
            cls = self.get_class(name)
            self._instances[name] = cls(*args, **kwargs)
        return self._instances[name]

    @property
    def names(self) -> list[str]:
        """All registered server names."""
        return list(self._classes)

    def has(self, name: str) -> bool:
        """Check if a server name is registered."""
        return name in self._classes

    def clear(self) -> None:
        """Remove all registrations and instances."""
        self._classes.clear()
        self._instances.clear()

    def __len__(self) -> int:
        return len(self._classes)

    def __repr__(self) -> str:
        return f"MCPServerRegistry(servers={sorted(self._classes)})"


# Module-level global registry
server_registry = MCPServerRegistry()


# ---------------------------------------------------------------------------
# @mcp_server() decorator
# ---------------------------------------------------------------------------


def _register_methods(instance: Any, mcp: FastMCP) -> list[str]:
    """Discover public methods on *instance* and register them as MCP tools.

    Returns the list of registered tool names.
    """
    tool_names: list[str] = []

    for method_name, method in inspect.getmembers(instance, inspect.ismethod):
        if method_name.startswith("_") or method_name in ("run", "run_async", "stop"):
            continue

        description = (inspect.getdoc(method) or f"{method_name} tool").strip().split("\n")[0]
        is_async = asyncio.iscoroutinefunction(method)

        # Register as MCP tool; description param exists at runtime (mcp>=1.0)
        tool_decorator = mcp.tool(name=method_name, description=description)  # pyright: ignore[reportCallIssue]
        if is_async:

            @tool_decorator
            @functools.wraps(method)
            async def async_wrapper(*args: Any, _m: Any = method, **kwargs: Any) -> Any:
                return await _m(*args, **kwargs)

        else:

            @tool_decorator
            @functools.wraps(method)
            def sync_wrapper(*args: Any, _m: Any = method, **kwargs: Any) -> Any:
                return _m(*args, **kwargs)

        tool_names.append(method_name)

    return tool_names


def mcp_server(
    name: str | None = None,
    *,
    transport: str = "stdio",
) -> Any:
    """Class decorator that converts a Python class into an MCP server.

    Public methods (non-underscored, excluding ``run``/``stop``) are
    registered as MCP tools via FastMCP.

    After decoration the class gains:

    * ``_mcp`` -- the ``FastMCP`` instance
    * ``_tool_names`` -- list of registered tool names
    * ``run(**kwargs)`` -- start the server (``transport`` kwarg overrides default)
    * ``stop()`` -- placeholder for graceful shutdown

    The class is also registered in the module-level ``server_registry``.

    Args:
        name: Server name. Defaults to the class name.
        transport: Default transport mode (``"stdio"`` or ``"sse"``).

    Returns:
        The decorated class.

    Example::

        @mcp_server(name="calculator")
        class Calculator:
            \"\"\"A simple calculator server.\"\"\"

            def add(self, a: int, b: int) -> int:
                \"\"\"Add two numbers.\"\"\"
                return a + b
    """

    def decorator(cls: type) -> type:
        server_name = name or cls.__name__
        server_desc = (cls.__doc__ or f"{server_name} MCP Server").strip()
        default_transport = transport

        original_init = cls.__init__

        @functools.wraps(original_init)
        def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)

            mcp = FastMCP(server_name, instructions=server_desc)
            tool_names = _register_methods(self, mcp)

            self._mcp = mcp
            self._tool_names = tool_names

            logger.info(
                "MCP server %r created with tools: %s",
                server_name,
                ", ".join(tool_names) if tool_names else "(none)",
            )

        cls.__init__ = new_init  # type: ignore[assignment]

        def run(self: Any, *, transport: str = default_transport, **kwargs: Any) -> None:
            """Run the MCP server.

            Args:
                transport: "stdio" or "sse".
                **kwargs: Passed to ``FastMCP.run()``.
            """
            if not hasattr(self, "_mcp") or self._mcp is None:
                raise MCPServerError("MCP server not initialized")
            self._mcp.run(transport=transport, **kwargs)

        async def run_async(
            self: Any, *, transport: str = default_transport, **kwargs: Any
        ) -> None:
            """Run the MCP server asynchronously, tracking the task for clean shutdown.

            The running coroutine is stored in ``_mcp_task`` so that
            :meth:`stop` can cancel it.

            Args:
                transport: "stdio" or "sse".
                **kwargs: Passed to the matching ``FastMCP.run_*_async()`` method.
            """
            if not hasattr(self, "_mcp") or self._mcp is None:
                raise MCPServerError("MCP server not initialized")
            if getattr(self, "_mcp_stopped", False):
                raise MCPServerError(f"MCP server {server_name!r} has already been stopped")

            async def _run() -> None:
                t = transport.replace("-", "_")
                runner = getattr(self._mcp, f"run_{t}_async", None)
                if runner is None:
                    raise MCPServerError(f"No async runner for transport {transport!r}")
                await runner(**kwargs)

            task = asyncio.ensure_future(_run())
            self._mcp_task: asyncio.Task[None] = task  # pyright: ignore[reportAttributeAccessIssue]
            try:
                await task
            except asyncio.CancelledError:
                logger.info("MCP server %r task cancelled", server_name)
                raise
            finally:
                self._mcp_task = None  # pyright: ignore[reportAttributeAccessIssue]

        async def stop(self: Any) -> None:
            """Stop the MCP server and release resources.

            This method is idempotent — calling it multiple times is safe.
            It cancels any background task started by :meth:`run_async`, waits
            for it to finish, and sets a stopped flag so subsequent calls are
            no-ops.
            """
            if getattr(self, "_mcp_stopped", False):
                logger.debug("MCP server %r already stopped", server_name)
                return

            self._mcp_stopped = True  # pyright: ignore[reportAttributeAccessIssue]
            logger.info("Stopping MCP server %r", server_name)

            task: asyncio.Task[None] | None = getattr(self, "_mcp_task", None)
            if task is not None and not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
                except (asyncio.CancelledError, TimeoutError):
                    pass
                except Exception as exc:
                    logger.warning("MCP server %r task raised on cancel: %s", server_name, exc)
                self._mcp_task = None  # pyright: ignore[reportAttributeAccessIssue]

            # Close the underlying FastMCP session manager if available.
            mcp: FastMCP | None = getattr(self, "_mcp", None)
            if mcp is not None:
                session_manager = getattr(mcp, "session_manager", None)
                if session_manager is not None:
                    shutdown = getattr(session_manager, "shutdown", None)
                    if shutdown is not None:
                        try:
                            await shutdown()
                        except Exception as exc:
                            logger.warning(
                                "MCP server %r session_manager.shutdown() raised: %s",
                                server_name,
                                exc,
                            )

        cls.run = run  # type: ignore[attr-defined]
        cls.run_async = run_async  # type: ignore[attr-defined]
        cls.stop = stop  # type: ignore[attr-defined]

        server_registry.register(server_name, cls)
        return cls

    return decorator
