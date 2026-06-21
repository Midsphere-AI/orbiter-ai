"""E2B cloud sandbox for remote agent execution."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from exo.sandbox.tools import (  # pyright: ignore[reportMissingImports]
        CodeTool,
        FilesystemTool,
        ShellTool,
        TerminalTool,
    )

from exo.sandbox.base import (  # pyright: ignore[reportMissingImports]
    Sandbox,
    SandboxError,
    SandboxStatus,
)

logger = logging.getLogger(__name__)

#: Type alias for registered tool handlers.
#: A handler receives the E2B sandbox instance and the tool arguments dict,
#: and returns any result.
ToolHandler = Callable[["E2BSandbox", dict[str, Any]], Awaitable[Any]]


class E2BSandbox(Sandbox):
    """Sandbox backed by `E2B <https://e2b.dev>`_ cloud sandboxes.

    Requires the ``e2b`` extra (``pip install exo-sandbox[e2b]``).
    Lifecycle: ``start`` creates (or connects to) an E2B sandbox,
    ``stop`` and ``cleanup`` kill it and release resources.

    Unlike :class:`KubernetesSandbox`, :meth:`run_tool` dispatches to real
    E2B operations (shell commands, file read/write/list) rather than
    returning stub metadata.
    """

    __slots__ = (
        "_api_key",
        "_e2b_sandbox",
        "_e2b_sandbox_id",
        "_existing_sandbox_id",
        "_metadata",
        "_registered_tools",
        "_template",
    )

    def __init__(
        self,
        *,
        sandbox_id: str | None = None,
        workspace: list[str] | None = None,
        mcp_config: dict[str, Any] | None = None,
        agents: dict[str, Any] | None = None,
        timeout: float = 300.0,
        api_key: str | None = None,
        template: str | None = None,
        existing_sandbox_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            sandbox_id=sandbox_id,
            workspace=workspace,
            mcp_config=mcp_config,
            agents=agents,
            timeout=timeout,
        )
        self._api_key = api_key or os.environ.get("E2B_API_KEY")
        self._template = template or os.environ.get("E2B_TEMPLATE_ID")
        self._existing_sandbox_id = existing_sandbox_id
        self._metadata = dict(metadata) if metadata else {}
        self._e2b_sandbox: Any = None
        self._e2b_sandbox_id: str | None = None
        self._registered_tools: dict[str, ToolHandler] = {}

    # -- properties ---------------------------------------------------------

    @property
    def api_key(self) -> str | None:
        return self._api_key

    @property
    def template(self) -> str | None:
        return self._template

    @property
    def e2b_sandbox_id(self) -> str | None:
        return self._e2b_sandbox_id

    @property
    def existing_sandbox_id(self) -> str | None:
        return self._existing_sandbox_id

    # -- E2B helpers --------------------------------------------------------

    def _load_e2b(self) -> Any:
        """Lazy-load the ``e2b`` package.

        Returns the ``e2b`` module so callers can access ``e2b.Sandbox``.
        Raises :class:`SandboxError` if the package is missing or no API
        key is configured.
        """
        try:
            import e2b  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            msg = "e2b package is required: pip install exo-sandbox[e2b]"
            raise SandboxError(msg) from exc
        if not self._api_key:
            msg = "E2B API key is required (pass api_key= or set E2B_API_KEY)"
            raise SandboxError(msg)
        return e2b

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        """Create or connect to an E2B sandbox."""
        self._transition(SandboxStatus.RUNNING)
        e2b_mod = self._load_e2b()

        try:
            if self._existing_sandbox_id:
                sb = await asyncio.to_thread(
                    e2b_mod.Sandbox.connect,
                    self._existing_sandbox_id,
                    api_key=self._api_key,
                )
                logger.info("Connected to existing E2B sandbox %s", self._existing_sandbox_id)
            else:
                create_kwargs: dict[str, Any] = {
                    "timeout": int(self._timeout),
                    "api_key": self._api_key,
                }
                if self._metadata:
                    create_kwargs["metadata"] = self._metadata
                if self._template:
                    sb = await asyncio.to_thread(
                        e2b_mod.Sandbox.create, self._template, **create_kwargs
                    )
                else:
                    sb = await asyncio.to_thread(e2b_mod.Sandbox.create, **create_kwargs)
                logger.info("Created E2B sandbox %s", sb.sandbox_id)

            self._e2b_sandbox = sb
            self._e2b_sandbox_id = sb.sandbox_id
        except SandboxError:
            raise
        except asyncio.CancelledError:
            logger.warning("Sandbox %s: start cancelled, cleaning up", self._sandbox_id)
            await self._kill_sandbox()
            raise
        except Exception as exc:
            self._status = SandboxStatus.ERROR
            logger.error("Failed to start E2B sandbox %s: %s", self._sandbox_id, exc)
            msg = f"Failed to start E2B sandbox: {exc}"
            raise SandboxError(msg) from exc

    async def stop(self) -> None:
        """Kill the E2B sandbox permanently.

        E2B sandboxes cannot be restarted after being killed — the remote
        process is destroyed and its state is lost.  This method transitions
        the sandbox to ``CLOSED`` to make the terminal semantics explicit.
        Use the async context manager (``async with E2BSandbox(...) as sb``)
        or ``cleanup()`` for the same effect with cleaner intent.
        """
        self._transition(SandboxStatus.CLOSED)
        await self._kill_sandbox()

    async def cleanup(self) -> None:
        """Release all E2B resources permanently."""
        self._transition(SandboxStatus.CLOSED)
        await self._kill_sandbox()

    async def _kill_sandbox(self) -> None:
        """Kill the remote E2B sandbox if it exists."""
        if self._e2b_sandbox is None:
            return
        try:
            await asyncio.to_thread(self._e2b_sandbox.kill)
            logger.info("Killed E2B sandbox %s", self._e2b_sandbox_id)
        except Exception:
            logger.warning("Failed to kill E2B sandbox %s", self._e2b_sandbox_id)
        self._e2b_sandbox = None
        self._e2b_sandbox_id = None

    # -- tool registration --------------------------------------------------

    def register_tool(self, name: str, handler: ToolHandler) -> None:
        """Register a custom tool handler for this sandbox.

        Registered handlers take precedence over built-in dispatch in
        :meth:`run_tool`.  A handler is an async callable with signature::

            async def my_handler(sandbox: E2BSandbox, arguments: dict[str, Any]) -> Any:
                ...

        Parameters
        ----------
        name:
            Tool name used for dispatch in :meth:`run_tool`.
        handler:
            Async callable ``(sandbox, arguments) -> result``.

        Raises
        ------
        SandboxError
            If a tool with the same *name* is already registered.
        """
        if name in self._registered_tools:
            msg = f"Tool {name!r} is already registered"
            raise SandboxError(msg)
        self._registered_tools[name] = handler
        logger.debug("Sandbox %s: registered tool %r", self._sandbox_id, name)

    def unregister_tool(self, name: str) -> None:
        """Remove a previously registered tool handler.

        Raises
        ------
        SandboxError
            If no tool with *name* is registered.
        """
        if name not in self._registered_tools:
            msg = f"Tool {name!r} is not registered"
            raise SandboxError(msg)
        del self._registered_tools[name]
        logger.debug("Sandbox %s: unregistered tool %r", self._sandbox_id, name)

    @property
    def registered_tools(self) -> list[str]:
        """Return the names of all registered tool handlers."""
        return list(self._registered_tools)

    # -- tool execution -----------------------------------------------------

    async def run_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool within the E2B sandbox.

        Dispatch order:

        1. **Registered tools** — added via :meth:`register_tool`.
        2. **Built-in tools** — ``"shell"`` / ``"command"``,
           ``"file_read"``, ``"file_write"``, ``"file_list"``.
        3. **Fallback** — returns a metadata dict.

        Raises :class:`SandboxError` if the sandbox is not running.
        """
        if self._status != SandboxStatus.RUNNING:
            msg = f"Sandbox must be running to call tools (status={self._status!r})"
            raise SandboxError(msg)

        # 1. Registered tool handlers take priority
        handler = self._registered_tools.get(tool_name)
        if handler is not None:
            try:
                return await handler(self, arguments)
            except SandboxError:
                raise
            except Exception as exc:
                msg = f"Registered tool {tool_name!r} failed: {exc}"
                raise SandboxError(msg) from exc

        # 2. Built-in dispatch
        sb = self._e2b_sandbox
        try:
            if tool_name in ("shell", "command"):
                command = arguments.get("command", "")
                result = await asyncio.to_thread(sb.commands.run, command)
                return {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "exit_code": result.exit_code,
                    "sandbox_id": self._sandbox_id,
                    "e2b_sandbox_id": self._e2b_sandbox_id,
                    "status": "ok",
                }
            elif tool_name == "file_read":
                path = arguments.get("path", "")
                content = await asyncio.to_thread(sb.files.read, path)
                return {
                    "content": content,
                    "path": path,
                    "sandbox_id": self._sandbox_id,
                    "status": "ok",
                }
            elif tool_name == "file_write":
                path = arguments.get("path", "")
                content = arguments.get("content", "")
                await asyncio.to_thread(sb.files.write, path, content)
                return {
                    "path": path,
                    "bytes_written": len(content),
                    "sandbox_id": self._sandbox_id,
                    "status": "ok",
                }
            elif tool_name == "file_list":
                path = arguments.get("path", "/")
                entries = await asyncio.to_thread(sb.files.list, path)
                return {
                    "path": path,
                    "entries": entries,
                    "sandbox_id": self._sandbox_id,
                    "status": "ok",
                }
            else:
                return {
                    "tool": tool_name,
                    "arguments": arguments,
                    "sandbox_id": self._sandbox_id,
                    "e2b_sandbox_id": self._e2b_sandbox_id,
                    "status": "ok",
                }
        except Exception as exc:
            msg = f"E2B tool {tool_name!r} failed: {exc}"
            raise SandboxError(msg) from exc

    # -- tool factories -----------------------------------------------------

    def filesystem_tool(self, allowed_directories: list[str] | None = None) -> FilesystemTool:
        """Return a :class:`FilesystemTool` wired to this E2B sandbox."""
        from exo.sandbox.tools import FilesystemTool  # pyright: ignore[reportMissingImports]

        return FilesystemTool(allowed_directories=allowed_directories, sandbox=self)

    def terminal_tool(
        self,
        *,
        blacklist: frozenset[str] | None = None,
        timeout: float = 30.0,
    ) -> TerminalTool:
        """Return a :class:`TerminalTool` wired to this E2B sandbox."""
        from exo.sandbox.tools import TerminalTool  # pyright: ignore[reportMissingImports]

        return TerminalTool(blacklist=blacklist, timeout=timeout, sandbox=self)

    def shell_tool(
        self,
        *,
        allowed_commands: list[str] | None = None,
        timeout: float = 30.0,
    ) -> ShellTool:
        """Return a :class:`ShellTool` wired to this E2B sandbox."""
        from exo.sandbox.tools import ShellTool  # pyright: ignore[reportMissingImports]

        return ShellTool(allowed_commands=allowed_commands, timeout=timeout, sandbox=self)

    def code_tool(
        self,
        *,
        blocked_names: frozenset[str] | None = None,
        timeout: float = 10.0,
    ) -> CodeTool:
        """Return a :class:`CodeTool` wired to this E2B sandbox."""
        from exo.sandbox.tools import CodeTool  # pyright: ignore[reportMissingImports]

        return CodeTool(sandbox=self, blocked_names=blocked_names, timeout=timeout)

    # -- context manager ----------------------------------------------------

    async def __aenter__(self) -> E2BSandbox:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.cleanup()

    # -- introspection ------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update(
            {
                "template": self._template,
                "e2b_sandbox_id": self._e2b_sandbox_id,
                "existing_sandbox_id": self._existing_sandbox_id,
                "api_key": "***" if self._api_key else None,
                "registered_tools": list(self._registered_tools),
            }
        )
        return info
