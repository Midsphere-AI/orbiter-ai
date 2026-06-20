"""Built-in sandbox tools: filesystem access and terminal execution."""

from __future__ import annotations

import asyncio
import logging
import platform
import shlex
import sys
from pathlib import Path
from typing import Any

from exo.tool import Tool, ToolError  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FilesystemTool
# ---------------------------------------------------------------------------

_FS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["read", "write", "list"],
            "description": "Filesystem action to perform.",
        },
        "path": {
            "type": "string",
            "description": "Absolute or relative file/directory path.",
        },
        "content": {
            "type": "string",
            "description": "Content to write (required for 'write' action).",
        },
    },
    "required": ["action", "path"],
}


class FilesystemTool(Tool):
    """Sandboxed filesystem tool with allowed-directory restrictions.

    Only paths that resolve within one of ``allowed_directories`` are
    permitted.  This prevents agents from reading or writing files outside
    the designated workspace.

    When a ``sandbox`` is provided, all operations are delegated to the
    sandbox via :meth:`~Sandbox.run_tool` instead of executing locally.
    """

    name = "filesystem"
    description = "Read, write, or list files within the sandbox workspace."
    parameters = _FS_SCHEMA

    def __init__(
        self,
        allowed_directories: list[str] | None = None,
        *,
        sandbox: Any | None = None,
    ) -> None:
        self._allowed: list[Path] = [Path(d).resolve() for d in (allowed_directories or [])]
        self._sandbox = sandbox

    # -- path validation ----------------------------------------------------

    def _validate_path(self, raw: str) -> Path:
        """Resolve *raw* and ensure it falls inside an allowed directory."""
        resolved = Path(raw).resolve()
        if not self._allowed:
            return resolved
        for allowed in self._allowed:
            try:
                resolved.relative_to(allowed)
                return resolved
            except ValueError:
                continue
        msg = f"Path {str(resolved)!r} is outside allowed directories"
        raise ToolError(msg)

    # -- execute ------------------------------------------------------------

    async def execute(self, **kwargs: Any) -> str | dict[str, Any]:
        action: str = kwargs.get("action", "")
        raw_path: str = kwargs.get("path", "")
        content: str | None = kwargs.get("content")

        if action not in {"read", "write", "list"}:
            raise ToolError(f"Unknown filesystem action: {action!r}")

        if self._sandbox is not None:
            return await self._execute_via_sandbox(action, raw_path, content)

        path = self._validate_path(raw_path)
        logger.debug("FilesystemTool: %s %s", action, path)

        if action == "read":
            return await self._read(path)
        if action == "write":
            if content is None:
                raise ToolError("'content' is required for write action")
            return await self._write(path, content)
        # action == "list"
        return await self._list(path)

    async def _execute_via_sandbox(
        self, action: str, path: str, content: str | None
    ) -> str | dict[str, Any]:
        """Delegate filesystem operations to the sandbox."""
        try:
            if action == "read":
                result = await self._sandbox.run_tool("file_read", {"path": path})
                return result.get("content", "") if isinstance(result, dict) else result
            if action == "write":
                if content is None:
                    raise ToolError("'content' is required for write action")
                result = await self._sandbox.run_tool(
                    "file_write", {"path": path, "content": content}
                )
                return result  # type: ignore[no-any-return]
            # action == "list"
            result = await self._sandbox.run_tool("file_list", {"path": path})
            return result  # type: ignore[no-any-return]
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"Sandbox filesystem {action} failed: {exc}") from exc

    async def _read(self, path: Path) -> str:
        try:
            return await asyncio.to_thread(path.read_text, encoding="utf-8")
        except FileNotFoundError as exc:
            raise ToolError(f"File not found: {path}") from exc
        except Exception as exc:
            raise ToolError(f"Read failed: {exc}") from exc

    async def _write(self, path: Path, content: str) -> str:
        try:
            await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
            await asyncio.to_thread(path.write_text, content, encoding="utf-8")
            return f"Wrote {len(content)} chars to {path}"
        except Exception as exc:
            raise ToolError(f"Write failed: {exc}") from exc

    async def _list(self, path: Path) -> dict[str, Any]:
        try:
            entries = await asyncio.to_thread(lambda: list(path.iterdir()))
            items = [
                {"name": e.name, "type": "dir" if e.is_dir() else "file"}
                for e in sorted(entries, key=lambda e: e.name)
            ]
            return {"directory": str(path), "entries": items}
        except FileNotFoundError as exc:
            raise ToolError(f"Directory not found: {path}") from exc
        except NotADirectoryError as exc:
            raise ToolError(f"Not a directory: {path}") from exc
        except Exception as exc:
            raise ToolError(f"List failed: {exc}") from exc


# ---------------------------------------------------------------------------
# TerminalTool
# ---------------------------------------------------------------------------

_DANGEROUS_COMMANDS: frozenset[str] = frozenset(
    {
        "rm",
        "rmdir",
        "mkfs",
        "dd",
        "shutdown",
        "reboot",
        "halt",
        "poweroff",
        "kill",
        "killall",
        "pkill",
        "format",
        # Windows equivalents
        "del",
        "erase",
        "rd",
    }
)

_TERMINAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": "Shell command to execute.",
        },
    },
    "required": ["command"],
}


class TerminalTool(Tool):
    """Sandboxed terminal tool with command filtering and timeout.

    Dangerous commands (``rm``, ``shutdown``, etc.) are blocked by default.
    Custom blacklists can be provided.  All commands run with a configurable
    timeout to prevent runaway processes.

    When a ``sandbox`` is provided, commands are executed remotely via
    :meth:`~Sandbox.run_tool` instead of on the local machine.
    """

    name = "terminal"
    description = "Execute a shell command in the sandbox."
    parameters = _TERMINAL_SCHEMA

    def __init__(
        self,
        *,
        blacklist: frozenset[str] | None = None,
        timeout: float = 30.0,
        sandbox: Any | None = None,
    ) -> None:
        self._blacklist: frozenset[str] = (
            blacklist if blacklist is not None else _DANGEROUS_COMMANDS
        )
        self._timeout = timeout
        self._sandbox = sandbox

    @property
    def platform(self) -> str:
        """Return the current platform identifier."""
        return sys.platform

    @staticmethod
    def _split_command_segments(command: str) -> list[str]:
        """Split *command* on shell control operators to get individual command segments.

        First splits on literal newlines (each newline introduces a new shell statement),
        then tokenizes each resulting line with ``shlex`` so that quoted strings are
        handled correctly, and finally collects the text between shell control tokens
        (``|``, ``||``, ``&&``, ``;``, ``&``).  Returns each non-empty segment as a
        string so the caller can inspect the leading executable of every chained
        sub-command.
        """
        # Shell control operators that introduce a new command context.
        _control = frozenset({"|", "||", "&&", ";", "&"})

        segments: list[str] = []

        # A newline in the shell is a command delimiter equivalent to ';', so
        # pre-split on newlines before tokenising each line individually.
        for line in command.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                lex = shlex.shlex(line, posix=True)
                lex.whitespace_split = True
                tokens = list(lex)
            except ValueError:
                # shlex couldn't parse (e.g. unterminated quote) — treat the
                # whole line as a single opaque segment.
                segments.append(line)
                continue

            current: list[str] = []
            for tok in tokens:
                if tok in _control:
                    if current:
                        segments.append(" ".join(current))
                        current = []
                else:
                    current.append(tok)

            if current:
                segments.append(" ".join(current))

        return [s for s in segments if s.strip()]

    def _check_command(self, command: str) -> None:
        """Raise ``ToolError`` if any sub-command in *command* is blacklisted.

        Security rationale: the command is executed via ``asyncio.create_subprocess_shell``
        which passes the string to ``/bin/sh -c``.  Checking only ``parts[0]`` of the
        whole string is insufficient — an attacker can chain blocked commands using shell
        control operators (``; && || | &``) or embed them in command substitutions
        (``$(…)`` / backticks).

        Defense strategy:
        - Reject immediately if the command contains command-substitution syntax
          (``$(`` or backticks) because the substituted commands cannot be safely
          validated without executing them.
        - Otherwise, split on all shell control operators and check the leading
          executable of **every** resulting segment against the blocklist.
        """
        stripped = command.strip()
        if not stripped:
            raise ToolError("Empty command")

        # Block command substitution — $(...) and backtick forms inject arbitrary
        # commands that we cannot inspect without running the shell.
        if "$(" in stripped or "`" in stripped:
            logger.warning(
                "TerminalTool: blocked command with command substitution (sandbox policy)"
            )
            raise ToolError("Command substitution ('$()' / backticks) is not permitted")

        blacklist_lower = {b.lower() for b in self._blacklist}

        for segment in self._split_command_segments(stripped):
            seg_parts = segment.strip().split()
            if not seg_parts:
                continue
            base = seg_parts[0].rsplit("/", maxsplit=1)[-1]  # strip path prefix
            base = base.rsplit("\\", maxsplit=1)[-1]
            if base.lower() in blacklist_lower:
                logger.warning("TerminalTool: blocked command %r (sandbox policy)", base)
                raise ToolError(f"Command {base!r} is blocked by sandbox policy")

    async def execute(self, **kwargs: Any) -> str | dict[str, Any]:
        command: str = kwargs.get("command", "")
        if not command.strip():
            raise ToolError("Empty command")

        self._check_command(command)

        if self._sandbox is not None:
            try:
                result = await self._sandbox.run_tool("command", {"command": command})
                return result  # type: ignore[no-any-return]
            except ToolError:
                raise
            except Exception as exc:
                raise ToolError(f"Sandbox command execution failed: {exc}") from exc

        logger.debug("TerminalTool: executing %r (timeout=%.1fs)", command, self._timeout)

        if platform.system() == "Windows":
            prog: str | None = None  # use default shell
        else:
            prog = "/bin/sh"

        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                executable=prog,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self._timeout)
        except TimeoutError as exc:
            if proc is not None:
                proc.kill()
                await proc.wait()
            logger.warning(
                "TerminalTool: command timed out after %.1fs: %r", self._timeout, command
            )
            raise ToolError(f"Command timed out after {self._timeout}s") from exc
        except asyncio.CancelledError:
            if proc is not None:
                proc.kill()
                await proc.wait()
            logger.debug("TerminalTool: cancelled, killed subprocess for %r", command)
            raise
        except Exception as exc:
            if proc is not None and proc.returncode is None:
                proc.kill()
                await proc.wait()
            logger.error("TerminalTool: execution failed for %r: %s", command, exc)
            raise ToolError(f"Command execution failed: {exc}") from exc

        logger.debug("TerminalTool: %r exited with code %s", command, proc.returncode)
        return {
            "exit_code": proc.returncode,
            "stdout": stdout.decode(errors="replace") if stdout else "",
            "stderr": stderr.decode(errors="replace") if stderr else "",
            "platform": self.platform,
        }


# ---------------------------------------------------------------------------
# ShellTool (allowlist-based shell command execution)
# ---------------------------------------------------------------------------

_DEFAULT_ALLOWED_COMMANDS: list[str] = [
    "ls",
    "cat",
    "grep",
    "find",
    "echo",
    "wc",
    "sort",
    "head",
    "tail",
    "diff",
]

_MAX_OUTPUT_CHARS = 10_000

_SHELL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": "Shell command to execute (must start with an allowed command).",
        },
    },
    "required": ["command"],
}


class ShellTool(Tool):
    """Allowlist-based shell tool for safe command execution.

    Only commands whose base executable is in the allowlist are permitted.
    Uses ``asyncio.create_subprocess_exec()`` (no shell) for safety.
    Output is truncated to 10,000 characters.

    When a ``sandbox`` is provided, commands are executed remotely via
    :meth:`~Sandbox.run_tool` instead of on the local machine.
    """

    name = "shell"
    description = "Execute a shell command from the configured allowlist."
    parameters = _SHELL_SCHEMA

    def __init__(
        self,
        *,
        allowed_commands: list[str] | None = None,
        timeout: float = 30.0,
        sandbox: Any | None = None,
    ) -> None:
        self._allowed: list[str] = (
            list(allowed_commands)
            if allowed_commands is not None
            else list(_DEFAULT_ALLOWED_COMMANDS)
        )
        self._timeout = timeout
        self._sandbox = sandbox

    def _validate_command(self, command: str) -> list[str]:
        """Parse and validate *command* against the allowlist."""
        stripped = command.strip()
        if not stripped:
            raise ToolError("Empty command")

        try:
            parts = shlex.split(stripped)
        except ValueError as exc:
            raise ToolError(f"Invalid command syntax: {exc}") from exc

        if not parts:
            raise ToolError("Empty command")

        base = parts[0].rsplit("/", maxsplit=1)[-1]
        if base not in self._allowed:
            raise ToolError(f"Command {base!r} is not in the allowed list: {self._allowed}")
        return parts

    @staticmethod
    def _truncate(text: str) -> str:
        """Truncate *text* to ``_MAX_OUTPUT_CHARS``."""
        if len(text) <= _MAX_OUTPUT_CHARS:
            return text
        return text[:_MAX_OUTPUT_CHARS] + f"\n... (truncated at {_MAX_OUTPUT_CHARS} chars)"

    async def execute(self, **kwargs: Any) -> str | dict[str, Any]:
        command: str = kwargs.get("command", "")
        parts = self._validate_command(command)

        if self._sandbox is not None:
            try:
                result = await self._sandbox.run_tool("shell", {"command": command})
                return result  # type: ignore[no-any-return]
            except ToolError:
                raise
            except Exception as exc:
                raise ToolError(f"Sandbox shell execution failed: {exc}") from exc

        try:
            proc = await asyncio.create_subprocess_exec(
                *parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self._timeout)
        except TimeoutError as exc:
            proc.kill()  # type: ignore[union-attr]
            raise ToolError(f"Command timed out after {self._timeout}s") from exc
        except FileNotFoundError as exc:
            raise ToolError(f"Command not found: {parts[0]!r}") from exc
        except Exception as exc:
            raise ToolError(f"Command execution failed: {exc}") from exc

        stdout_str = stdout.decode(errors="replace") if stdout else ""
        stderr_str = stderr.decode(errors="replace") if stderr else ""

        return {
            "exit_code": proc.returncode,
            "stdout": self._truncate(stdout_str),
            "stderr": self._truncate(stderr_str),
        }


def shell_tool(
    allowed_commands: list[str] | None = None,
    *,
    timeout: float = 30.0,
    sandbox: Any | None = None,
) -> ShellTool:
    """Factory that creates an allowlist-based shell tool."""
    return ShellTool(allowed_commands=allowed_commands, timeout=timeout, sandbox=sandbox)


# ---------------------------------------------------------------------------
# CodeTool (sandboxed code execution)
# ---------------------------------------------------------------------------

_DEFAULT_BLOCKED_NAMES: frozenset[str] = frozenset(
    {"__import__", "open", "os", "sys", "subprocess", "shutil"}
)

_CODE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "code": {
            "type": "string",
            "description": "Python code to execute.",
        },
    },
    "required": ["code"],
}


def _build_runner_script(code: str, blocked_names: list[str]) -> str:
    """Build a Python script that executes *code* in a restricted namespace."""
    import json

    escaped_code = json.dumps(code)
    escaped_blocked = json.dumps(blocked_names)
    return (
        "import builtins as _b\n"
        f"_code = {escaped_code}\n"
        f"_blocked = set({escaped_blocked})\n"
        "_ns = {'__builtins__': {k: v for k, v in vars(_b).items() if k not in _blocked}}\n"
        "exec(_code, _ns)\n"
    )


class CodeTool(Tool):
    """Sandboxed Python code execution tool.

    When a ``Sandbox`` is provided, execution is delegated to the sandbox.
    Otherwise, code runs locally in a restricted namespace with stdout
    capture and a configurable timeout.
    """

    name = "code"
    description = "Execute Python code in a sandboxed environment."
    parameters = _CODE_SCHEMA

    def __init__(
        self,
        *,
        sandbox: Any | None = None,
        blocked_names: frozenset[str] | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._sandbox = sandbox
        self._blocked: frozenset[str] = (
            blocked_names if blocked_names is not None else _DEFAULT_BLOCKED_NAMES
        )
        self._timeout = timeout

    async def execute(self, **kwargs: Any) -> str | dict[str, Any]:
        code: str = kwargs.get("code", "")
        if not code.strip():
            raise ToolError("Empty code")

        if self._sandbox is not None:
            try:
                result = await self._sandbox.run_tool("code", {"code": code})
                return result  # type: ignore[no-any-return]
            except Exception as exc:
                raise ToolError(f"Sandbox execution failed: {exc}") from exc

        blocked_list = list(self._blocked)
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                _build_runner_script(code, blocked_list),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self._timeout)
        except TimeoutError as exc:
            proc.kill()  # type: ignore[union-attr]
            raise ToolError(f"Code execution timed out after {self._timeout}s") from exc

        stdout_str = stdout.decode(errors="replace") if stdout else ""
        stderr_str = stderr.decode(errors="replace") if stderr else ""

        if proc.returncode != 0:
            return (
                f"Error: {stderr_str.strip()}"
                if stderr_str.strip()
                else f"Error: exit code {proc.returncode}"
            )

        return stdout_str if stdout_str else "(no output)"


def code_tool(
    sandbox: Any | None = None,
    *,
    blocked_names: frozenset[str] | None = None,
    timeout: float = 10.0,
) -> CodeTool:
    """Factory that creates a sandboxed code execution tool."""
    return CodeTool(sandbox=sandbox, blocked_names=blocked_names, timeout=timeout)
