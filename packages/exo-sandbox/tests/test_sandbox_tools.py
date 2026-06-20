"""Tests for built-in sandbox tools (FilesystemTool and TerminalTool)."""

from __future__ import annotations

from pathlib import Path

import pytest

from exo.sandbox.tools import (  # pyright: ignore[reportMissingImports]
    _DANGEROUS_COMMANDS,
    FilesystemTool,
    TerminalTool,
)
from exo.tool import ToolError

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
        tool = TerminalTool(timeout=0.1)
        with pytest.raises(ToolError, match="timed out"):
            await tool.execute(command="sleep 10")


# ---------------------------------------------------------------------------
# TerminalTool — platform detection
# ---------------------------------------------------------------------------


class TestTerminalPlatform:
    def test_platform_property(self) -> None:
        import sys

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
