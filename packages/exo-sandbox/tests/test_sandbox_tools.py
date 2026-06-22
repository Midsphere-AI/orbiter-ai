"""Tests for built-in sandbox tools (FilesystemTool and TerminalTool)."""

from __future__ import annotations

import asyncio
import sys
import warnings
from pathlib import Path

import pytest

from exo.sandbox.tools import (  # pyright: ignore[reportMissingImports]
    _DANGEROUS_COMMANDS,
    CodeTool,
    FilesystemTool,
    ShellTool,
    TerminalTool,
)
from exo.tool import ToolError

# Suppress DeprecationWarning for TerminalTool throughout this module — the
# behavioural tests below test TerminalTool's functionality, not the warning.
# The dedicated TestTerminalToolDeprecation class verifies the warning itself.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="exo.sandbox.tools")

# ---------------------------------------------------------------------------
# FilesystemTool — path validation
# ---------------------------------------------------------------------------


class TestFilesystemPathValidation:
    def test_no_restrictions(self, tmp_path: Path) -> None:
        tool = FilesystemTool()
        result = tool._validate_path(str(tmp_path))
        assert result == tmp_path.resolve()

    def test_allowed_directory_accepted(self, tmp_path: Path) -> None:
        child = tmp_path / "sub" / "file.txt"
        tool = FilesystemTool(allowed_directories=[str(tmp_path)])
        result = tool._validate_path(str(child))
        assert result == child.resolve()

    def test_outside_allowed_rejected(self, tmp_path: Path) -> None:
        tool = FilesystemTool(allowed_directories=[str(tmp_path / "allowed")])
        with pytest.raises(ToolError, match="outside allowed directories"):
            tool._validate_path("/etc/passwd")

    def test_multiple_allowed_directories(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        tool = FilesystemTool(allowed_directories=[str(dir_a), str(dir_b)])
        assert tool._validate_path(str(dir_a / "f.txt")) == (dir_a / "f.txt").resolve()
        assert tool._validate_path(str(dir_b / "g.txt")) == (dir_b / "g.txt").resolve()

    def test_path_traversal_blocked(self, tmp_path: Path) -> None:
        allowed = tmp_path / "workspace"
        allowed.mkdir()
        tool = FilesystemTool(allowed_directories=[str(allowed)])
        with pytest.raises(ToolError, match="outside allowed directories"):
            tool._validate_path(str(allowed / ".." / "secret.txt"))


# ---------------------------------------------------------------------------
# FilesystemTool — read
# ---------------------------------------------------------------------------


class TestFilesystemRead:
    async def test_read_file(self, tmp_path: Path) -> None:
        f = tmp_path / "hello.txt"
        f.write_text("hello world", encoding="utf-8")
        tool = FilesystemTool(allowed_directories=[str(tmp_path)])
        result = await tool.execute(action="read", path=str(f))
        assert result == "hello world"

    async def test_read_missing_file(self, tmp_path: Path) -> None:
        tool = FilesystemTool(allowed_directories=[str(tmp_path)])
        with pytest.raises(ToolError, match="File not found"):
            await tool.execute(action="read", path=str(tmp_path / "nope.txt"))


# ---------------------------------------------------------------------------
# FilesystemTool — write
# ---------------------------------------------------------------------------


class TestFilesystemWrite:
    async def test_write_file(self, tmp_path: Path) -> None:
        f = tmp_path / "out.txt"
        tool = FilesystemTool(allowed_directories=[str(tmp_path)])
        result = await tool.execute(action="write", path=str(f), content="data")
        assert "4 chars" in result
        assert f.read_text(encoding="utf-8") == "data"

    async def test_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        f = tmp_path / "a" / "b" / "c.txt"
        tool = FilesystemTool(allowed_directories=[str(tmp_path)])
        await tool.execute(action="write", path=str(f), content="nested")
        assert f.read_text(encoding="utf-8") == "nested"

    async def test_write_missing_content(self, tmp_path: Path) -> None:
        tool = FilesystemTool(allowed_directories=[str(tmp_path)])
        with pytest.raises(ToolError, match=r"content.*required"):
            await tool.execute(action="write", path=str(tmp_path / "x.txt"))


# ---------------------------------------------------------------------------
# FilesystemTool — list
# ---------------------------------------------------------------------------


class TestFilesystemList:
    async def test_list_directory(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("a", encoding="utf-8")
        (tmp_path / "b_dir").mkdir()
        tool = FilesystemTool(allowed_directories=[str(tmp_path)])
        result = await tool.execute(action="list", path=str(tmp_path))
        assert isinstance(result, dict)
        names = [e["name"] for e in result["entries"]]
        assert "a.txt" in names
        assert "b_dir" in names
        # Check types
        by_name = {e["name"]: e for e in result["entries"]}
        assert by_name["a.txt"]["type"] == "file"
        assert by_name["b_dir"]["type"] == "dir"

    async def test_list_missing_directory(self, tmp_path: Path) -> None:
        tool = FilesystemTool(allowed_directories=[str(tmp_path)])
        with pytest.raises(ToolError, match="Directory not found"):
            await tool.execute(action="list", path=str(tmp_path / "nope"))

    async def test_list_file_not_dir(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("x", encoding="utf-8")
        tool = FilesystemTool(allowed_directories=[str(tmp_path)])
        with pytest.raises(ToolError, match="Not a directory"):
            await tool.execute(action="list", path=str(f))


# ---------------------------------------------------------------------------
# FilesystemTool — unknown action
# ---------------------------------------------------------------------------


class TestFilesystemUnknownAction:
    async def test_unknown_action(self, tmp_path: Path) -> None:
        tool = FilesystemTool()
        with pytest.raises(ToolError, match="Unknown filesystem action"):
            await tool.execute(action="delete", path="/tmp/x")


# ---------------------------------------------------------------------------
# FilesystemTool — schema
# ---------------------------------------------------------------------------


class TestFilesystemSchema:
    def test_schema(self) -> None:
        tool = FilesystemTool()
        schema = tool.to_schema()
        assert schema["function"]["name"] == "filesystem"
        assert "action" in schema["function"]["parameters"]["properties"]
        assert "path" in schema["function"]["parameters"]["properties"]


# ---------------------------------------------------------------------------
# TerminalTool — command filtering
# ---------------------------------------------------------------------------


class TestTerminalCommandFiltering:
    def test_dangerous_command_blocked(self) -> None:
        tool = TerminalTool()
        with pytest.raises(ToolError, match="blocked by sandbox policy"):
            tool._check_command("rm -rf /")

    def test_safe_command_allowed(self) -> None:
        tool = TerminalTool()
        tool._check_command("echo hello")  # should not raise

    def test_full_path_command_stripped(self) -> None:
        tool = TerminalTool()
        with pytest.raises(ToolError, match="blocked by sandbox policy"):
            tool._check_command("/usr/bin/rm -rf /")

    def test_case_insensitive_blocking(self) -> None:
        tool = TerminalTool()
        with pytest.raises(ToolError, match="blocked by sandbox policy"):
            tool._check_command("RM -rf /")

    def test_custom_blacklist(self) -> None:
        tool = TerminalTool(blacklist=frozenset({"curl"}))
        with pytest.raises(ToolError, match="blocked by sandbox policy"):
            tool._check_command("curl http://evil.com")
        # rm should be allowed with custom blacklist
        tool._check_command("rm -rf /")

    def test_empty_command_rejected(self) -> None:
        tool = TerminalTool()
        with pytest.raises(ToolError, match="Empty command"):
            tool._check_command("")

    def test_whitespace_only_command(self) -> None:
        tool = TerminalTool()
        with pytest.raises(ToolError, match="Empty command"):
            tool._check_command("   ")

    def test_default_blacklist_contents(self) -> None:
        assert "rm" in _DANGEROUS_COMMANDS
        assert "shutdown" in _DANGEROUS_COMMANDS
        assert "kill" in _DANGEROUS_COMMANDS

    # ------------------------------------------------------------------
    # B-5: command-injection via shell chaining / substitution
    # ------------------------------------------------------------------

    def test_semicolon_chaining_blocked(self) -> None:
        """echo is safe but rm after ; must still be blocked (B-5)."""
        tool = TerminalTool()
        with pytest.raises(ToolError, match="blocked by sandbox policy"):
            tool._check_command("echo hello ; rm -rf /")

    def test_double_ampersand_chaining_blocked(self) -> None:
        tool = TerminalTool()
        with pytest.raises(ToolError, match="blocked by sandbox policy"):
            tool._check_command("echo ok && rm -rf /tmp")

    def test_double_pipe_or_chaining_blocked(self) -> None:
        tool = TerminalTool()
        with pytest.raises(ToolError, match="blocked by sandbox policy"):
            tool._check_command("false || rm -rf /")

    def test_pipe_blocked_command_in_pipeline(self) -> None:
        tool = TerminalTool()
        with pytest.raises(ToolError, match="blocked by sandbox policy"):
            tool._check_command("cat /etc/passwd | rm -rf /tmp/x")

    def test_background_operator_blocked(self) -> None:
        tool = TerminalTool()
        with pytest.raises(ToolError, match="blocked by sandbox policy"):
            tool._check_command("rm -rf / &")

    def test_newline_chaining_blocked(self) -> None:
        tool = TerminalTool()
        with pytest.raises(ToolError, match="blocked by sandbox policy"):
            tool._check_command("echo hi\nrm -rf /")

    def test_command_substitution_dollar_blocked(self) -> None:
        tool = TerminalTool()
        with pytest.raises(ToolError, match="not permitted"):
            tool._check_command("echo $(rm -rf /)")

    def test_command_substitution_backtick_blocked(self) -> None:
        tool = TerminalTool()
        with pytest.raises(ToolError, match="not permitted"):
            tool._check_command("echo `rm -rf /`")

    def test_safe_pipeline_allowed(self) -> None:
        """A pipeline of safe commands should not raise."""
        tool = TerminalTool()
        tool._check_command("ls | grep foo")  # neither ls nor grep are in the default blocklist

    def test_safe_and_chain_allowed(self) -> None:
        tool = TerminalTool()
        tool._check_command("echo hello && echo world")  # should not raise


# ---------------------------------------------------------------------------
# TerminalTool — execution
# ---------------------------------------------------------------------------


class TestTerminalExecution:
    async def test_echo_command(self) -> None:
        tool = TerminalTool()
        result = await tool.execute(command="echo hello")
        assert isinstance(result, dict)
        assert result["exit_code"] == 0
        assert "hello" in result["stdout"]
        assert "platform" in result

    async def test_command_stderr(self) -> None:
        tool = TerminalTool()
        result = await tool.execute(command="echo error >&2")
        assert "error" in result["stderr"]

    async def test_nonzero_exit(self) -> None:
        tool = TerminalTool()
        result = await tool.execute(command="exit 42")
        assert result["exit_code"] == 42

    async def test_empty_command(self) -> None:
        tool = TerminalTool()
        with pytest.raises(ToolError, match="Empty command"):
            await tool.execute(command="")

    async def test_blocked_command(self) -> None:
        tool = TerminalTool()
        with pytest.raises(ToolError, match="blocked by sandbox policy"):
            await tool.execute(command="rm -rf /tmp/test")

    async def test_timeout(self) -> None:
        # Use a generous timeout (0.5s) so subprocess creation overhead on a
        # loaded CI machine doesn't race with the deadline, while keeping the
        # sleep target long enough (30s) that it never completes before the
        # timeout fires.
        tool = TerminalTool(timeout=0.5)
        with pytest.raises(ToolError, match="timed out"):
            await tool.execute(command="sleep 30")


# ---------------------------------------------------------------------------
# TerminalTool — platform detection
# ---------------------------------------------------------------------------


class TestTerminalPlatform:
    def test_platform_property(self) -> None:

        tool = TerminalTool()
        assert tool.platform == sys.platform


# ---------------------------------------------------------------------------
# TerminalTool — schema
# ---------------------------------------------------------------------------


class TestTerminalSchema:
    def test_schema(self) -> None:
        tool = TerminalTool()
        schema = tool.to_schema()
        assert schema["function"]["name"] == "terminal"
        assert "command" in schema["function"]["parameters"]["properties"]


# ---------------------------------------------------------------------------
# ShellTool — invalid mode
# ---------------------------------------------------------------------------


class TestShellToolInvalidMode:
    def test_invalid_mode_raises_tool_error(self) -> None:
        """ShellTool.__init__ must raise ToolError (not ValueError) for bad mode."""
        with pytest.raises(ToolError, match="mode must be 'allowlist' or 'blacklist'"):
            ShellTool(mode="sudo")  # type: ignore[arg-type]

    def test_invalid_mode_hint_present(self) -> None:
        with pytest.raises(ToolError) as exc_info:
            ShellTool(mode="bad")  # type: ignore[arg-type]
        # ToolError renders hint into str() — verify the hint is accessible
        err = exc_info.value
        assert err.hint is not None
        assert "allowlist" in err.hint


# ---------------------------------------------------------------------------
# ShellTool (allowlist mode) — zombie reap on timeout
# ---------------------------------------------------------------------------


class TestShellToolAllowlistTimeoutReap:
    """Verify that the allowlist path reaps the subprocess on timeout (no zombie)."""

    async def test_subprocess_reaped_after_timeout(self) -> None:
        """The process must be waited on after kill so it doesn't become a zombie.

        Strategy: use a very short timeout against a real ``sleep`` subprocess.
        After the ToolError we verify that the process has been reaped by
        calling wait() a second time — it must return immediately (returncode
        already set) and not block.
        """
        tool = ShellTool(allowed_commands=["sleep"], timeout=0.05)
        proc_ref: list[asyncio.subprocess.Process] = []

        original_exec = asyncio.create_subprocess_exec

        async def capturing_exec(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
            proc = await original_exec(*args, **kwargs)  # type: ignore[arg-type]
            proc_ref.append(proc)
            return proc

        # Patch asyncio.create_subprocess_exec at the point where tools.py calls it.
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(asyncio, "create_subprocess_exec", capturing_exec)
            with pytest.raises(ToolError, match="timed out"):
                await tool.execute(command="sleep 10")

        # After the ToolError the process must have been reaped (returncode set).
        assert proc_ref, "No subprocess was captured — test setup issue"
        proc = proc_ref[0]
        # returncode is None only if the process has not been waited on yet.
        assert proc.returncode is not None, (
            "Process was not reaped after timeout — zombie leak detected"
        )

    async def test_timeout_error_carries_context(self) -> None:
        tool = ShellTool(allowed_commands=["sleep"], timeout=0.05)
        with pytest.raises(ToolError) as exc_info:
            await tool.execute(command="sleep 10")
        err = exc_info.value
        assert err.context is not None
        assert "command" in err.context


# ---------------------------------------------------------------------------
# CodeTool — zombie reap on timeout
# ---------------------------------------------------------------------------


# Child code that is guaranteed to outlive the timeout regardless of system
# load *and* of ``blocked_names``: a bare ``while True: pass`` needs no imports
# or builtins, so it cannot exit early (the old flake, where blocked imports made
# the child crash before the timeout, returning instead of raising). The process
# is SIGKILL'd at the timeout, so the brief CPU spin is bounded by ``timeout``.
_NEVER_RETURNS = "while True:\n    pass"


class TestCodeToolTimeoutReap:
    """Verify that CodeTool reaps the subprocess on timeout (no zombie)."""

    async def test_subprocess_reaped_after_timeout(self) -> None:
        tool = CodeTool(timeout=0.2, allow_unsafe_local=True)
        proc_ref: list[asyncio.subprocess.Process] = []

        original_exec = asyncio.create_subprocess_exec

        async def capturing_exec(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
            proc = await original_exec(*args, **kwargs)  # type: ignore[arg-type]
            proc_ref.append(proc)
            return proc

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(asyncio, "create_subprocess_exec", capturing_exec)
            with pytest.raises(ToolError, match="timed out"):
                await tool.execute(code=_NEVER_RETURNS)

        assert proc_ref, "No subprocess was captured — test setup issue"
        proc = proc_ref[0]
        assert proc.returncode is not None, (
            "Process was not reaped after timeout — zombie leak detected"
        )

    async def test_timeout_error_carries_context(self) -> None:
        tool = CodeTool(timeout=0.2, allow_unsafe_local=True)
        with pytest.raises(ToolError) as exc_info:
            await tool.execute(code=_NEVER_RETURNS)
        err = exc_info.value
        assert err.context is not None
        assert "timeout" in err.context

    async def test_basic_execution(self) -> None:
        tool = CodeTool(allow_unsafe_local=True)
        result = await tool.execute(code="print('hello from code tool')")
        assert "hello from code tool" in result  # type: ignore[operator]

    async def test_blocked_name_prevents_import(self) -> None:
        tool = CodeTool(allow_unsafe_local=True)
        # os is blocked by default — accessing it should cause a NameError
        result = await tool.execute(code="print(os.getcwd())")
        assert "Error" in str(result)

    async def test_no_sandbox_no_unsafe_flag_raises(self) -> None:
        """CodeTool without a sandbox and allow_unsafe_local=False must refuse."""
        tool = CodeTool()
        with pytest.raises(ToolError, match="allow_unsafe_local"):
            await tool.execute(code="print('hello')")

    async def test_allow_unsafe_local_flag_permits_execution(self) -> None:
        """allow_unsafe_local=True must permit local execution."""
        tool = CodeTool(allow_unsafe_local=True)
        result = await tool.execute(code="print('unsafe ok')")
        assert "unsafe ok" in str(result)


# ---------------------------------------------------------------------------
# E2B _kill_sandbox — logs exception object (not silent swallow)
# ---------------------------------------------------------------------------


class TestE2BKillSandboxLogging:
    """Verify _kill_sandbox logs the exception and clears internal state."""

    async def test_kill_failure_logs_exc_object(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging
        from unittest.mock import MagicMock

        from exo.sandbox.e2b import E2BSandbox  # pyright: ignore[reportMissingImports]

        sb = E2BSandbox(sandbox_id="kf1", api_key="k")
        mock_e2b = MagicMock()
        mock_e2b.kill.side_effect = RuntimeError("E2B API gone")
        sb._e2b_sandbox = mock_e2b
        sb._e2b_sandbox_id = "remote-abc"

        with caplog.at_level(logging.WARNING, logger="exo.sandbox.e2b"):
            await sb._kill_sandbox()

        # The exception message must appear in the log (not swallowed silently)
        combined = " ".join(caplog.messages)
        assert "E2B API gone" in combined or any(
            "E2B API gone" in r.message for r in caplog.records
        )

        # Internal state must be cleared regardless of the kill failure
        assert sb._e2b_sandbox is None
        assert sb._e2b_sandbox_id is None


# ---------------------------------------------------------------------------
# ShellTool (allowlist) — path-traversal bypass fix (finding #2)
# ---------------------------------------------------------------------------


class TestShellToolAllowlistPathTraversal:
    """Executables with path separators must be rejected even if basename matches."""

    def test_absolute_path_rejected(self) -> None:
        tool = ShellTool(allowed_commands=["ls"])
        with pytest.raises(ToolError, match="bare binary name"):
            tool._validate_command("/tmp/evil_ls -la")

    def test_relative_path_rejected(self) -> None:
        tool = ShellTool(allowed_commands=["ls"])
        with pytest.raises(ToolError, match="bare binary name"):
            tool._validate_command("../ls -la")

    def test_unix_path_with_allowed_basename_rejected(self) -> None:
        tool = ShellTool(allowed_commands=["ls"])
        with pytest.raises(ToolError, match="bare binary name"):
            tool._validate_command("/bin/ls -la")

    def test_bare_binary_accepted(self) -> None:
        # A bare name with no slashes is fine.
        tool = ShellTool(allowed_commands=["ls"])
        parts = tool._validate_command("ls -la")
        assert parts[0] == "ls"


# ---------------------------------------------------------------------------
# ShellTool — response schema consistency (finding #12)
# ---------------------------------------------------------------------------


class TestShellToolResponseSchema:
    """Allowlist and blacklist modes must both include 'platform' in the response."""

    async def test_allowlist_response_has_platform(self) -> None:
        tool = ShellTool(allowed_commands=["echo"])
        result = await tool.execute(command="echo hi")
        assert isinstance(result, dict)
        assert "platform" in result

    async def test_blacklist_response_has_platform(self) -> None:
        tool = ShellTool(mode="blacklist")
        result = await tool.execute(command="echo hi")
        assert isinstance(result, dict)
        assert "platform" in result


# ---------------------------------------------------------------------------
# TerminalTool — deprecation warning (finding #13)
# ---------------------------------------------------------------------------


class TestTerminalToolDeprecation:
    def test_deprecation_warning_emitted(self) -> None:
        with pytest.warns(DeprecationWarning, match="TerminalTool is deprecated"):
            TerminalTool()
