"""Tests for Sandbox ABC, SandboxStatus, and the state machine."""

from __future__ import annotations

from typing import Any

import pytest

from exo.sandbox.base import (  # pyright: ignore[reportMissingImports]
    Sandbox,
    SandboxError,
    SandboxStatus,
)

# ---------------------------------------------------------------------------
# Minimal concrete Sandbox for testing the ABC
# ---------------------------------------------------------------------------


class _StubSandbox(Sandbox):
    """Minimal Sandbox subclass for testing the ABC contract."""

    async def start(self) -> None:
        self._transition(SandboxStatus.RUNNING)

    async def stop(self) -> None:
        self._transition(SandboxStatus.IDLE)

    async def cleanup(self) -> None:
        self._transition(SandboxStatus.CLOSED)

    async def run_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if self._status != SandboxStatus.RUNNING:
            raise SandboxError(f"Sandbox must be running to call tools (status={self._status!r})")
        return {"tool": tool_name, "arguments": arguments, "status": "ok"}


# ---------------------------------------------------------------------------
# SandboxStatus enum
# ---------------------------------------------------------------------------


class TestSandboxStatus:
    def test_values(self) -> None:
        assert SandboxStatus.INIT == "init"
        assert SandboxStatus.RUNNING == "running"
        assert SandboxStatus.IDLE == "idle"
        assert SandboxStatus.ERROR == "error"
        assert SandboxStatus.CLOSED == "closed"

    def test_is_str_enum(self) -> None:
        assert isinstance(SandboxStatus.INIT, str)


# ---------------------------------------------------------------------------
# Sandbox ABC
# ---------------------------------------------------------------------------


class TestSandboxABC:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            Sandbox()  # type: ignore[abstract]

    def test_default_init(self) -> None:
        sb = _StubSandbox()
        assert sb.status == SandboxStatus.INIT
        assert sb.workspace == []
        assert sb.mcp_config == {}
        assert sb.agents == {}
        assert sb.timeout == 30.0
        assert len(sb.sandbox_id) == 12

    def test_custom_init(self) -> None:
        sb = _StubSandbox(
            sandbox_id="test-123",
            workspace=["/tmp/ws"],
            mcp_config={"mcpServers": {}},
            agents={"a1": {"model": "gpt-4"}},
            timeout=60.0,
        )
        assert sb.sandbox_id == "test-123"
        assert sb.workspace == ["/tmp/ws"]
        assert sb.mcp_config == {"mcpServers": {}}
        assert sb.agents == {"a1": {"model": "gpt-4"}}
        assert sb.timeout == 60.0

    def test_properties_return_copies(self) -> None:
        sb = _StubSandbox(workspace=["/a"], mcp_config={"k": "v"}, agents={"a": 1})
        ws = sb.workspace
        ws.append("/b")
        assert sb.workspace == ["/a"]

        mc = sb.mcp_config
        mc["new"] = True
        assert "new" not in sb.mcp_config

        ag = sb.agents
        ag["x"] = 2
        assert "x" not in sb.agents


# ---------------------------------------------------------------------------
# Status transitions
# ---------------------------------------------------------------------------


class TestStatusTransitions:
    def test_init_to_running(self) -> None:
        sb = _StubSandbox()
        sb._transition(SandboxStatus.RUNNING)
        assert sb.status == SandboxStatus.RUNNING

    def test_init_to_closed(self) -> None:
        sb = _StubSandbox()
        sb._transition(SandboxStatus.CLOSED)
        assert sb.status == SandboxStatus.CLOSED

    def test_running_to_idle(self) -> None:
        sb = _StubSandbox()
        sb._transition(SandboxStatus.RUNNING)
        sb._transition(SandboxStatus.IDLE)
        assert sb.status == SandboxStatus.IDLE

    def test_running_to_error(self) -> None:
        sb = _StubSandbox()
        sb._transition(SandboxStatus.RUNNING)
        sb._transition(SandboxStatus.ERROR)
        assert sb.status == SandboxStatus.ERROR

    def test_running_to_closed(self) -> None:
        sb = _StubSandbox()
        sb._transition(SandboxStatus.RUNNING)
        sb._transition(SandboxStatus.CLOSED)
        assert sb.status == SandboxStatus.CLOSED

    def test_idle_to_running(self) -> None:
        sb = _StubSandbox()
        sb._transition(SandboxStatus.RUNNING)
        sb._transition(SandboxStatus.IDLE)
        sb._transition(SandboxStatus.RUNNING)
        assert sb.status == SandboxStatus.RUNNING

    def test_error_to_running(self) -> None:
        sb = _StubSandbox()
        sb._transition(SandboxStatus.RUNNING)
        sb._transition(SandboxStatus.ERROR)
        sb._transition(SandboxStatus.RUNNING)
        assert sb.status == SandboxStatus.RUNNING

    def test_closed_cannot_transition(self) -> None:
        sb = _StubSandbox()
        sb._transition(SandboxStatus.CLOSED)
        with pytest.raises(SandboxError, match="Cannot transition"):
            sb._transition(SandboxStatus.RUNNING)

    def test_init_to_idle_invalid(self) -> None:
        sb = _StubSandbox()
        with pytest.raises(SandboxError, match="Cannot transition"):
            sb._transition(SandboxStatus.IDLE)

    def test_init_to_error_invalid(self) -> None:
        sb = _StubSandbox()
        with pytest.raises(SandboxError, match="Cannot transition"):
            sb._transition(SandboxStatus.ERROR)


# ---------------------------------------------------------------------------
# Stub sandbox lifecycle
# ---------------------------------------------------------------------------


class TestStubSandboxLifecycle:
    async def test_start(self) -> None:
        sb = _StubSandbox()
        await sb.start()
        assert sb.status == SandboxStatus.RUNNING

    async def test_start_stop(self) -> None:
        sb = _StubSandbox()
        await sb.start()
        await sb.stop()
        assert sb.status == SandboxStatus.IDLE

    async def test_start_cleanup(self) -> None:
        sb = _StubSandbox()
        await sb.start()
        await sb.cleanup()
        assert sb.status == SandboxStatus.CLOSED

    async def test_start_stop_start(self) -> None:
        sb = _StubSandbox()
        await sb.start()
        await sb.stop()
        await sb.start()
        assert sb.status == SandboxStatus.RUNNING

    async def test_run_tool_while_running(self) -> None:
        sb = _StubSandbox()
        await sb.start()
        result = await sb.run_tool("echo", {"message": "hello"})
        assert result == {"tool": "echo", "arguments": {"message": "hello"}, "status": "ok"}

    async def test_run_tool_not_running(self) -> None:
        sb = _StubSandbox()
        with pytest.raises(SandboxError, match="must be running"):
            await sb.run_tool("echo", {})


# ---------------------------------------------------------------------------
# describe / repr
# ---------------------------------------------------------------------------


class TestDescribeRepr:
    def test_describe(self) -> None:
        sb = _StubSandbox(sandbox_id="d1", workspace=["/x"], timeout=5.0)
        d = sb.describe()
        assert d["sandbox_id"] == "d1"
        assert d["status"] == "init"
        assert d["workspace"] == ["/x"]
        assert d["timeout"] == 5.0

    def test_repr(self) -> None:
        sb = _StubSandbox(sandbox_id="r1")
        assert "_StubSandbox" in repr(sb)
        assert "r1" in repr(sb)
        assert "init" in repr(sb)
