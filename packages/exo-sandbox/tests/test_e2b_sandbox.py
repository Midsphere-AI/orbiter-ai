"""Tests for E2BSandbox."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from exo.sandbox.base import (  # pyright: ignore[reportMissingImports]
    SandboxError,
    SandboxStatus,
)
from exo.sandbox.e2b import E2BSandbox  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_e2b_sandbox() -> MagicMock:
    """Create a mock E2B Sandbox instance."""
    sb = MagicMock()
    sb.sandbox_id = "e2b-remote-abc123"

    # commands.run returns a result with stdout/stderr/exit_code
    cmd_result = MagicMock()
    cmd_result.stdout = "hello\n"
    cmd_result.stderr = ""
    cmd_result.exit_code = 0
    sb.commands.run.return_value = cmd_result

    # files.read returns content
    sb.files.read.return_value = "file content"

    # files.write succeeds
    sb.files.write.return_value = None

    # files.list returns entries
    entry_a = MagicMock()
    entry_a.name = "a.txt"
    entry_b = MagicMock()
    entry_b.name = "b.txt"
    sb.files.list.return_value = [entry_a, entry_b]

    # kill succeeds
    sb.kill.return_value = None

    return sb


def _mock_e2b_module(mock_sandbox: MagicMock | None = None) -> MagicMock:
    """Create a mock ``e2b`` module."""
    if mock_sandbox is None:
        mock_sandbox = _mock_e2b_sandbox()
    mod = MagicMock()
    mod.Sandbox.create.return_value = mock_sandbox
    mod.Sandbox.connect.return_value = mock_sandbox
    return mod


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------


class TestInit:
    def test_defaults(self) -> None:
        sb = E2BSandbox(api_key="test-key")
        assert sb.status == SandboxStatus.INIT
        assert sb.api_key == "test-key"
        assert sb.template is None
        assert sb.e2b_sandbox_id is None
        assert sb.existing_sandbox_id is None
        assert sb.timeout == 300.0

    def test_custom_values(self) -> None:
        sb = E2BSandbox(
            sandbox_id="test-123",
            api_key="my-key",
            template="tmpl-abc",
            existing_sandbox_id="existing-xyz",
            timeout=120.0,
            workspace=["ws1"],
            mcp_config={"key": "val"},
            agents={"a": "b"},
            metadata={"user": "test"},
        )
        assert sb.sandbox_id == "test-123"
        assert sb.api_key == "my-key"
        assert sb.template == "tmpl-abc"
        assert sb.existing_sandbox_id == "existing-xyz"
        assert sb.timeout == 120.0
        assert sb.workspace == ["ws1"]
        assert sb.mcp_config == {"key": "val"}
        assert sb.agents == {"a": "b"}

    def test_env_var_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("E2B_API_KEY", "env-key-123")
        sb = E2BSandbox()
        assert sb.api_key == "env-key-123"

    def test_env_var_template(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("E2B_TEMPLATE_ID", "env-tmpl-456")
        sb = E2BSandbox(api_key="k")
        assert sb.template == "env-tmpl-456"

    def test_constructor_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("E2B_API_KEY", "env-key")
        sb = E2BSandbox(api_key="explicit-key")
        assert sb.api_key == "explicit-key"


# ---------------------------------------------------------------------------
# TestLoadE2B
# ---------------------------------------------------------------------------


class TestLoadE2B:
    def test_missing_e2b_package(self) -> None:
        sb = E2BSandbox(api_key="k")
        with (
            patch.dict("sys.modules", {"e2b": None}),
            pytest.raises(SandboxError, match="e2b package is required"),
        ):
            sb._load_e2b()

    def test_missing_api_key(self) -> None:
        sb = E2BSandbox()  # no api_key, no env var
        mock_mod = _mock_e2b_module()
        with (
            patch.dict("sys.modules", {"e2b": mock_mod}),
            pytest.raises(SandboxError, match="E2B API key is required"),
        ):
            sb._load_e2b()


# ---------------------------------------------------------------------------
# TestStart
# ---------------------------------------------------------------------------


class TestStart:
    async def test_start_creates_sandbox(self) -> None:
        mock_sb = _mock_e2b_sandbox()
        mock_mod = _mock_e2b_module(mock_sb)
        sb = E2BSandbox(sandbox_id="s1", api_key="k")

        with patch.object(E2BSandbox, "_load_e2b", return_value=mock_mod):
            await sb.start()

        assert sb.status == SandboxStatus.RUNNING
        assert sb.e2b_sandbox_id == "e2b-remote-abc123"
        mock_mod.Sandbox.create.assert_called_once()

    async def test_start_with_template(self) -> None:
        mock_sb = _mock_e2b_sandbox()
        mock_mod = _mock_e2b_module(mock_sb)
        sb = E2BSandbox(sandbox_id="s2", api_key="k", template="my-tmpl")

        with patch.object(E2BSandbox, "_load_e2b", return_value=mock_mod):
            await sb.start()

        call_args = mock_mod.Sandbox.create.call_args
        assert call_args[0][0] == "my-tmpl"

    async def test_start_connects_to_existing(self) -> None:
        mock_sb = _mock_e2b_sandbox()
        mock_mod = _mock_e2b_module(mock_sb)
        sb = E2BSandbox(sandbox_id="s3", api_key="k", existing_sandbox_id="exist-99")

        with patch.object(E2BSandbox, "_load_e2b", return_value=mock_mod):
            await sb.start()

        assert sb.status == SandboxStatus.RUNNING
        mock_mod.Sandbox.connect.assert_called_once()
        call_args = mock_mod.Sandbox.connect.call_args
        assert call_args[0][0] == "exist-99"
        mock_mod.Sandbox.create.assert_not_called()

    async def test_start_error_sets_error_status(self) -> None:
        mock_mod = _mock_e2b_module()
        mock_mod.Sandbox.create.side_effect = RuntimeError("API error")
        sb = E2BSandbox(sandbox_id="s4", api_key="k")

        with (
            patch.object(E2BSandbox, "_load_e2b", return_value=mock_mod),
            pytest.raises(SandboxError, match="Failed to start"),
        ):
            await sb.start()
        assert sb.status == SandboxStatus.ERROR

    async def test_start_with_metadata(self) -> None:
        mock_sb = _mock_e2b_sandbox()
        mock_mod = _mock_e2b_module(mock_sb)
        sb = E2BSandbox(sandbox_id="s5", api_key="k", metadata={"user": "123"})

        with patch.object(E2BSandbox, "_load_e2b", return_value=mock_mod):
            await sb.start()

        call_kwargs = mock_mod.Sandbox.create.call_args[1]
        assert call_kwargs["metadata"] == {"user": "123"}


# ---------------------------------------------------------------------------
# TestStop
# ---------------------------------------------------------------------------


class TestStop:
    async def test_stop_kills_sandbox(self) -> None:
        mock_sb = _mock_e2b_sandbox()
        mock_mod = _mock_e2b_module(mock_sb)
        sb = E2BSandbox(sandbox_id="s6", api_key="k")

        with patch.object(E2BSandbox, "_load_e2b", return_value=mock_mod):
            await sb.start()

        await sb.stop()

        # stop() is terminal — transitions to CLOSED (E2B sandboxes cannot be restarted)
        assert sb.status == SandboxStatus.CLOSED
        assert sb.e2b_sandbox_id is None
        mock_sb.kill.assert_called_once()

    async def test_stop_no_sandbox(self) -> None:
        mock_sb = _mock_e2b_sandbox()
        mock_mod = _mock_e2b_module(mock_sb)
        sb = E2BSandbox(sandbox_id="s7", api_key="k")

        with patch.object(E2BSandbox, "_load_e2b", return_value=mock_mod):
            await sb.start()

        sb._e2b_sandbox = None
        await sb.stop()
        # stop() is terminal even when no remote sandbox object exists
        assert sb.status == SandboxStatus.CLOSED


# ---------------------------------------------------------------------------
# TestCleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    async def test_cleanup_closes(self) -> None:
        mock_sb = _mock_e2b_sandbox()
        mock_mod = _mock_e2b_module(mock_sb)
        sb = E2BSandbox(sandbox_id="s8", api_key="k")

        with patch.object(E2BSandbox, "_load_e2b", return_value=mock_mod):
            await sb.start()

        await sb.cleanup()

        assert sb.status == SandboxStatus.CLOSED
        mock_sb.kill.assert_called_once()

    async def test_cleanup_tolerates_kill_errors(self) -> None:
        mock_sb = _mock_e2b_sandbox()
        mock_sb.kill.side_effect = RuntimeError("gone")
        mock_mod = _mock_e2b_module(mock_sb)
        sb = E2BSandbox(sandbox_id="s9", api_key="k")

        with patch.object(E2BSandbox, "_load_e2b", return_value=mock_mod):
            await sb.start()

        await sb.cleanup()  # should not raise
        assert sb.status == SandboxStatus.CLOSED


# ---------------------------------------------------------------------------
# TestRunTool
# ---------------------------------------------------------------------------


class TestRunTool:
    async def _make_running(self) -> tuple[E2BSandbox, MagicMock]:
        """Helper: create a running E2BSandbox with a mock backend."""
        mock_sb = _mock_e2b_sandbox()
        sb = E2BSandbox(sandbox_id="rt", api_key="k")
        sb._status = SandboxStatus.RUNNING
        sb._e2b_sandbox = mock_sb
        sb._e2b_sandbox_id = mock_sb.sandbox_id
        return sb, mock_sb

    async def test_run_shell_command(self) -> None:
        sb, mock_sb = await self._make_running()
        result = await sb.run_tool("shell", {"command": "echo hi"})

        assert result["stdout"] == "hello\n"
        assert result["stderr"] == ""
        assert result["exit_code"] == 0
        assert result["status"] == "ok"
        assert result["e2b_sandbox_id"] == "e2b-remote-abc123"
        mock_sb.commands.run.assert_called_once_with("echo hi")

    async def test_run_command_alias(self) -> None:
        sb, mock_sb = await self._make_running()
        result = await sb.run_tool("command", {"command": "ls"})

        assert result["status"] == "ok"
        assert result["exit_code"] == 0
        mock_sb.commands.run.assert_called_once_with("ls")

    async def test_run_file_read(self) -> None:
        sb, mock_sb = await self._make_running()
        result = await sb.run_tool("file_read", {"path": "/tmp/test.txt"})

        assert result["content"] == "file content"
        assert result["path"] == "/tmp/test.txt"
        assert result["status"] == "ok"
        mock_sb.files.read.assert_called_once_with("/tmp/test.txt")

    async def test_run_file_write(self) -> None:
        sb, mock_sb = await self._make_running()
        result = await sb.run_tool("file_write", {"path": "/tmp/out.txt", "content": "data"})

        assert result["path"] == "/tmp/out.txt"
        assert result["bytes_written"] == 4
        assert result["status"] == "ok"
        mock_sb.files.write.assert_called_once_with("/tmp/out.txt", "data")

    async def test_run_file_list(self) -> None:
        sb, mock_sb = await self._make_running()
        result = await sb.run_tool("file_list", {"path": "/home/user"})

        assert result["path"] == "/home/user"
        assert len(result["entries"]) == 2
        assert result["status"] == "ok"
        mock_sb.files.list.assert_called_once_with("/home/user")

    async def test_run_file_list_default_path(self) -> None:
        sb, mock_sb = await self._make_running()
        result = await sb.run_tool("file_list", {})

        assert result["path"] == "/"
        mock_sb.files.list.assert_called_once_with("/")

    async def test_run_unknown_tool(self) -> None:
        sb, _ = await self._make_running()
        result = await sb.run_tool("custom_thing", {"foo": "bar"})

        assert result["tool"] == "custom_thing"
        assert result["arguments"] == {"foo": "bar"}
        assert result["status"] == "ok"

    async def test_run_tool_not_running(self) -> None:
        sb = E2BSandbox(sandbox_id="s10", api_key="k")
        with pytest.raises(SandboxError, match="Sandbox must be running"):
            await sb.run_tool("shell", {"command": "echo"})

    async def test_run_tool_error_converts_to_sandbox_error(self) -> None:
        sb, mock_sb = await self._make_running()
        mock_sb.commands.run.side_effect = RuntimeError("connection lost")

        with pytest.raises(SandboxError, match="E2B tool 'shell' failed"):
            await sb.run_tool("shell", {"command": "echo"})


# ---------------------------------------------------------------------------
# TestContextManager
# ---------------------------------------------------------------------------


class TestContextManager:
    async def test_async_context_manager(self) -> None:
        mock_sb = _mock_e2b_sandbox()
        mock_mod = _mock_e2b_module(mock_sb)
        sb = E2BSandbox(sandbox_id="cm1", api_key="k")

        with patch.object(E2BSandbox, "_load_e2b", return_value=mock_mod):
            async with sb as s:
                assert s.status == SandboxStatus.RUNNING
                assert s is sb

        assert sb.status == SandboxStatus.CLOSED


# ---------------------------------------------------------------------------
# TestDescribe
# ---------------------------------------------------------------------------


class TestDescribe:
    def test_describe_before_start(self) -> None:
        sb = E2BSandbox(
            sandbox_id="d1",
            api_key="secret",
            template="tmpl-1",
        )
        info = sb.describe()
        assert info["sandbox_id"] == "d1"
        assert info["template"] == "tmpl-1"
        assert info["e2b_sandbox_id"] is None
        assert info["existing_sandbox_id"] is None
        assert info["api_key"] == "***"

    async def test_describe_after_start(self) -> None:
        mock_sb = _mock_e2b_sandbox()
        mock_mod = _mock_e2b_module(mock_sb)
        sb = E2BSandbox(sandbox_id="d2", api_key="k")

        with patch.object(E2BSandbox, "_load_e2b", return_value=mock_mod):
            await sb.start()

        info = sb.describe()
        assert info["e2b_sandbox_id"] == "e2b-remote-abc123"

    def test_describe_no_api_key(self) -> None:
        sb = E2BSandbox(sandbox_id="d3")
        info = sb.describe()
        assert info["api_key"] is None

    def test_repr(self) -> None:
        sb = E2BSandbox(sandbox_id="r1", api_key="k")
        r = repr(sb)
        assert "E2BSandbox" in r
        assert "r1" in r


# ---------------------------------------------------------------------------
# TestToolFactories
# ---------------------------------------------------------------------------


class TestToolFactories:
    """Test that E2BSandbox factory methods return tools wired to the sandbox."""

    def _make_sandbox(self) -> E2BSandbox:
        sb = E2BSandbox(sandbox_id="tf", api_key="k")
        sb._status = SandboxStatus.RUNNING
        sb._e2b_sandbox = _mock_e2b_sandbox()
        sb._e2b_sandbox_id = "e2b-remote-abc123"
        return sb

    def test_filesystem_tool_factory(self) -> None:
        sb = self._make_sandbox()
        tool = sb.filesystem_tool()
        assert tool.name == "filesystem"
        assert tool._sandbox is sb

    def test_filesystem_tool_with_dirs(self) -> None:
        sb = self._make_sandbox()
        tool = sb.filesystem_tool(allowed_directories=["/tmp"])
        assert tool._sandbox is sb

    def test_terminal_tool_factory(self) -> None:
        sb = self._make_sandbox()
        tool = sb.terminal_tool()
        assert tool.name == "terminal"
        assert tool._sandbox is sb

    def test_shell_tool_factory(self) -> None:
        sb = self._make_sandbox()
        tool = sb.shell_tool()
        assert tool.name == "shell"
        assert tool._sandbox is sb

    def test_code_tool_factory(self) -> None:
        sb = self._make_sandbox()
        tool = sb.code_tool()
        assert tool.name == "code"
        assert tool._sandbox is sb

    async def test_filesystem_read_delegates(self) -> None:
        sb = self._make_sandbox()
        tool = sb.filesystem_tool()
        result = await tool.execute(action="read", path="/tmp/test.txt")
        assert result == "file content"

    async def test_filesystem_write_delegates(self) -> None:
        sb = self._make_sandbox()
        tool = sb.filesystem_tool()
        result = await tool.execute(action="write", path="/tmp/out.txt", content="data")
        assert isinstance(result, dict)
        assert result["status"] == "ok"

    async def test_filesystem_list_delegates(self) -> None:
        sb = self._make_sandbox()
        tool = sb.filesystem_tool()
        result = await tool.execute(action="list", path="/home/user")
        assert isinstance(result, dict)
        assert result["status"] == "ok"

    async def test_terminal_delegates(self) -> None:
        sb = self._make_sandbox()
        tool = sb.terminal_tool()
        result = await tool.execute(command="echo hi")
        assert isinstance(result, dict)
        assert result["stdout"] == "hello\n"
        assert result["exit_code"] == 0

    async def test_shell_delegates(self) -> None:
        sb = self._make_sandbox()
        tool = sb.shell_tool(allowed_commands=["echo", "ls"])
        result = await tool.execute(command="echo hi")
        assert isinstance(result, dict)
        assert result["stdout"] == "hello\n"

    async def test_code_delegates(self) -> None:
        sb = self._make_sandbox()
        tool = sb.code_tool()
        # CodeTool delegates to sandbox.run_tool("code", ...)
        result = await tool.execute(code="print('hi')")
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# TestSandboxAwareTools
# ---------------------------------------------------------------------------


class TestSandboxAwareTools:
    """Test that tools with sandbox=None still work locally."""

    async def test_filesystem_local_read(self, tmp_path: Path) -> None:
        from exo.sandbox.tools import FilesystemTool  # pyright: ignore[reportMissingImports]

        f = tmp_path / "test.txt"
        f.write_text("local content")
        tool = FilesystemTool()  # no sandbox
        result = await tool.execute(action="read", path=str(f))
        assert result == "local content"

    async def test_terminal_local_echo(self) -> None:
        from exo.sandbox.tools import TerminalTool  # pyright: ignore[reportMissingImports]

        tool = TerminalTool()  # no sandbox
        result = await tool.execute(command="echo local")
        assert isinstance(result, dict)
        assert "local" in result["stdout"]

    async def test_shell_local_echo(self) -> None:
        from exo.sandbox.tools import ShellTool  # pyright: ignore[reportMissingImports]

        tool = ShellTool()  # no sandbox
        result = await tool.execute(command="echo local")
        assert isinstance(result, dict)
        assert "local" in result["stdout"]


# ---------------------------------------------------------------------------
# TestToolRegistration
# ---------------------------------------------------------------------------


class TestToolRegistration:
    """Test register_tool / unregister_tool and run_tool dispatch."""

    def _make_running(self) -> E2BSandbox:
        mock_sb = _mock_e2b_sandbox()
        sb = E2BSandbox(sandbox_id="reg", api_key="k")
        sb._status = SandboxStatus.RUNNING
        sb._e2b_sandbox = mock_sb
        sb._e2b_sandbox_id = mock_sb.sandbox_id
        return sb

    def test_register_tool(self) -> None:
        sb = E2BSandbox(sandbox_id="r1", api_key="k")

        async def handler(sandbox: E2BSandbox, args: dict) -> str:
            return "ok"

        sb.register_tool("my_tool", handler)
        assert "my_tool" in sb.registered_tools

    def test_register_duplicate_raises(self) -> None:
        sb = E2BSandbox(sandbox_id="r2", api_key="k")

        async def handler(sandbox: E2BSandbox, args: dict) -> str:
            return "ok"

        sb.register_tool("dup", handler)
        with pytest.raises(SandboxError, match="already registered"):
            sb.register_tool("dup", handler)

    def test_unregister_tool(self) -> None:
        sb = E2BSandbox(sandbox_id="r3", api_key="k")

        async def handler(sandbox: E2BSandbox, args: dict) -> str:
            return "ok"

        sb.register_tool("temp", handler)
        assert "temp" in sb.registered_tools
        sb.unregister_tool("temp")
        assert "temp" not in sb.registered_tools

    def test_unregister_missing_raises(self) -> None:
        sb = E2BSandbox(sandbox_id="r4", api_key="k")
        with pytest.raises(SandboxError, match="not registered"):
            sb.unregister_tool("nonexistent")

    def test_registered_tools_property(self) -> None:
        sb = E2BSandbox(sandbox_id="r5", api_key="k")

        async def h1(sandbox: E2BSandbox, args: dict) -> str:
            return "a"

        async def h2(sandbox: E2BSandbox, args: dict) -> str:
            return "b"

        sb.register_tool("tool_a", h1)
        sb.register_tool("tool_b", h2)
        assert set(sb.registered_tools) == {"tool_a", "tool_b"}

    async def test_run_tool_dispatches_to_registered(self) -> None:
        sb = self._make_running()

        async def custom_handler(sandbox: E2BSandbox, args: dict) -> dict:
            return {"custom": True, "received": args}

        sb.register_tool("custom_op", custom_handler)
        result = await sb.run_tool("custom_op", {"key": "value"})
        assert result == {"custom": True, "received": {"key": "value"}}

    async def test_registered_takes_priority_over_builtin(self) -> None:
        """A registered 'shell' handler overrides the built-in shell dispatch."""
        sb = self._make_running()

        async def override_shell(sandbox: E2BSandbox, args: dict) -> dict:
            return {"overridden": True}

        sb.register_tool("shell", override_shell)
        result = await sb.run_tool("shell", {"command": "echo hi"})
        assert result == {"overridden": True}
        # Built-in E2B commands.run should NOT have been called
        sb._e2b_sandbox.commands.run.assert_not_called()

    async def test_registered_handler_receives_sandbox(self) -> None:
        sb = self._make_running()
        received_sandbox = None

        async def capture_sandbox(sandbox: E2BSandbox, args: dict) -> str:
            nonlocal received_sandbox
            received_sandbox = sandbox
            return "done"

        sb.register_tool("inspect", capture_sandbox)
        await sb.run_tool("inspect", {})
        assert received_sandbox is sb

    async def test_registered_handler_error_wraps(self) -> None:
        sb = self._make_running()

        async def failing_handler(sandbox: E2BSandbox, args: dict) -> None:
            raise ValueError("boom")

        sb.register_tool("fail", failing_handler)
        with pytest.raises(SandboxError, match="Registered tool 'fail' failed"):
            await sb.run_tool("fail", {})

    async def test_builtin_still_works_without_registration(self) -> None:
        """Built-in tools work when no registered tool shadows them."""
        sb = self._make_running()
        result = await sb.run_tool("shell", {"command": "echo hi"})
        assert result["stdout"] == "hello\n"

    def test_describe_includes_registered_tools(self) -> None:
        sb = E2BSandbox(sandbox_id="r6", api_key="k")

        async def handler(sandbox: E2BSandbox, args: dict) -> str:
            return "ok"

        sb.register_tool("custom_a", handler)
        sb.register_tool("custom_b", handler)
        info = sb.describe()
        assert set(info["registered_tools"]) == {"custom_a", "custom_b"}
