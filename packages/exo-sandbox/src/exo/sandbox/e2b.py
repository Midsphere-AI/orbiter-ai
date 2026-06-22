"""E2B cloud sandbox for remote agent execution."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from pathlib import Path
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

#: Built-in tool names dispatched by :meth:`E2BSandbox.run_tool`.
_BUILTIN_TOOL_NAMES: frozenset[str] = frozenset(
    {"shell", "command", "code", "file_read", "file_write", "file_list"}
)


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

    @property
    def e2b_sandbox(self) -> Any:
        """Return the underlying ``e2b.Sandbox`` handle for direct SDK access.

        Use this to reach E2B features not wrapped by :meth:`run_tool` — e.g.
        ``sandbox.e2b_sandbox.files.write(...)`` to upload files, long-running
        commands, PTY sessions, or watch handlers. The object is the live
        instance returned by ``e2b.Sandbox.create``/``connect``.

        Raises
        ------
        SandboxError
            If the sandbox has not been started yet (call :meth:`start` or use
            ``async with E2BSandbox(...)`` first), or has already been closed.
        """
        if self._e2b_sandbox is None:
            msg = (
                "E2B sandbox is not running — no underlying handle is available. "
                "Call `await sandbox.start()` (or use `async with E2BSandbox(...)`) "
                "before accessing `.e2b_sandbox`."
            )
            raise SandboxError(msg)
        return self._e2b_sandbox

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
        # Load (and validate) the e2b package and API key BEFORE transitioning
        # to RUNNING.  If _load_e2b() raises SandboxError (missing package or no
        # API key), the object stays in its current state rather than getting
        # stuck in RUNNING.
        e2b_mod = self._load_e2b()
        self._transition(SandboxStatus.RUNNING)

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
            raise SandboxError(
                "Failed to start E2B sandbox.",
                context={
                    "sandbox_id": self._sandbox_id,
                    "template": self._template or "(default)",
                    "existing_sandbox_id": self._existing_sandbox_id,
                },
                hint="Check E2B_API_KEY is valid and the template ID exists in your E2B account.",
                doc="https://e2b.dev/docs",
            ) from exc

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
            await asyncio.wait_for(
                asyncio.to_thread(self._e2b_sandbox.kill),
                timeout=10.0,
            )
            logger.info("Killed E2B sandbox %s", self._e2b_sandbox_id)
        except Exception as exc:
            # Log with the exception object so the diagnostic signal is not lost.
            # Do NOT swallow CancelledError — it will not reach here because
            # CancelledError is not a subclass of Exception.
            logger.warning(
                "Failed to kill E2B sandbox %s: %s",
                self._e2b_sandbox_id,
                exc,
            )
        finally:
            self._e2b_sandbox = None
            self._e2b_sandbox_id = None

    # ``register_tool`` / ``unregister_tool`` / ``registered_tools`` are
    # inherited from :class:`~exo.sandbox.base.Sandbox` and shared across all
    # backends. Both call forms work here:
    #   sandbox.register_tool("name", handler)   # async (sandbox, args) -> result
    #   sandbox.register_tool(my_tool)           # a Tool with .name + .execute

    # -- tool execution -----------------------------------------------------

    async def run_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool within the E2B sandbox.

        Dispatch order:

        1. **Registered tools** — added via :meth:`register_tool`.
        2. **Built-in tools** — ``"shell"`` / ``"command"``, ``"code"``,
           ``"file_read"``, ``"file_write"``, ``"file_list"``.
        3. **Unknown name** — raises :class:`SandboxError` listing valid tools.

        Raises :class:`SandboxError` if the sandbox is not running.
        """
        if self._status != SandboxStatus.RUNNING:
            msg = f"Sandbox must be running to call tools (status={self._status!r})"
            raise SandboxError(msg)

        # 1. Registered tool handlers take priority
        handled, result = await self._run_registered(tool_name, arguments)
        if handled:
            return result

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
                    "bytes_written": len(content.encode("utf-8")),
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
            elif tool_name == "code":
                # Run Python in the remote VM. base64 the source so arbitrary
                # quoting/newlines survive the shell round-trip intact.
                code: str = arguments.get("code", "")
                encoded = base64.b64encode(code.encode()).decode()
                result = await asyncio.to_thread(
                    sb.commands.run, f"echo {encoded} | base64 -d | python3"
                )
                return {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "exit_code": result.exit_code,
                    "sandbox_id": self._sandbox_id,
                    "e2b_sandbox_id": self._e2b_sandbox_id,
                    "status": "ok",
                }
            else:
                known = sorted({*_BUILTIN_TOOL_NAMES, *self._registered_tools})
                msg = (
                    f"Unknown tool {tool_name!r}. "
                    f"Built-in tools are {sorted(_BUILTIN_TOOL_NAMES)}; "
                    f"register custom tools with sandbox.register_tool(name, handler). "
                    f"Available here: {known}."
                )
                raise SandboxError(msg)
        except SandboxError:
            raise
        except Exception as exc:
            raise SandboxError(
                f"E2B tool {tool_name!r} failed.",
                context={"sandbox_id": self._sandbox_id, "e2b_sandbox_id": self._e2b_sandbox_id},
                hint=f"Check the E2B sandbox logs or verify the sandbox is healthy (tool={tool_name!r}).",
            ) from exc

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
        """Return a :class:`TerminalTool` wired to this E2B sandbox.

        .. deprecated::
            Prefer :meth:`shell_tool` (``mode="blacklist"``) directly.
            This method is kept for backwards compatibility.
        """
        import warnings

        from exo.sandbox.tools import TerminalTool  # pyright: ignore[reportMissingImports]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
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

    # -- convenience: files & commands --------------------------------------
    #
    # These cover the everyday operations directly, so you don't have to reach
    # for the raw ``e2b_sandbox`` handle. Each runs the blocking E2B SDK call on
    # a worker thread and raises a teaching :class:`SandboxError` if the sandbox
    # isn't running yet (via the ``e2b_sandbox`` accessor).

    async def upload(self, local_path: str | Path, remote_path: str) -> str:
        """Upload a local file into the sandbox. Returns the *remote_path*.

        Example::

            await sandbox.upload("data.csv", "/home/user/data.csv")
        """
        data = await asyncio.to_thread(Path(local_path).read_bytes)
        await asyncio.to_thread(self.e2b_sandbox.files.write, remote_path, data)
        logger.debug("Uploaded %s -> %s", local_path, remote_path)
        return remote_path

    async def download(self, remote_path: str, local_path: str | Path) -> str:
        """Download a file from the sandbox to local disk. Returns the local path.

        Example::

            await sandbox.download("/home/user/out.txt", "out.txt")
        """
        content = await asyncio.to_thread(self.e2b_sandbox.files.read, remote_path)
        dest = Path(local_path)
        if isinstance(content, bytes):
            await asyncio.to_thread(dest.write_bytes, content)
        else:
            await asyncio.to_thread(dest.write_text, content)
        logger.debug("Downloaded %s -> %s", remote_path, local_path)
        return str(dest)

    async def write_file(self, remote_path: str, content: str | bytes) -> str:
        """Write *content* to a file in the sandbox. Returns the *remote_path*."""
        await asyncio.to_thread(self.e2b_sandbox.files.write, remote_path, content)
        return remote_path

    async def read_file(self, remote_path: str) -> str:
        """Read and return the contents of a file in the sandbox."""
        return await asyncio.to_thread(self.e2b_sandbox.files.read, remote_path)

    async def list_files(self, remote_path: str = "/") -> list[Any]:
        """List directory entries at *remote_path* in the sandbox."""
        return await asyncio.to_thread(self.e2b_sandbox.files.list, remote_path)

    async def run(self, command: str, *, background: bool = False) -> Any:
        """Run a shell command in the sandbox.

        With ``background=False`` (default) this waits for the command and
        returns ``{"stdout": ..., "stderr": ..., "exit_code": ...}``. With
        ``background=True`` it returns the live E2B command handle so you can
        stream or kill it later.

        Example::

            result = await sandbox.run("python script.py")
            print(result["stdout"])
        """
        sb = self.e2b_sandbox
        if background:
            return await asyncio.to_thread(sb.commands.run, command, background=True)
        result = await asyncio.to_thread(sb.commands.run, command)
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
        }

    # -- context manager ----------------------------------------------------

    async def __aenter__(self) -> E2BSandbox:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.cleanup()

    # -- introspection ------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        # ``registered_tools`` comes from the base describe().
        info.update(
            {
                "template": self._template,
                "e2b_sandbox_id": self._e2b_sandbox_id,
                "existing_sandbox_id": self._existing_sandbox_id,
                "api_key": "***" if self._api_key else None,
            }
        )
        return info
