"""Tests for exo.mcp.server — @mcp_server() decorator and MCPServerRegistry."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from exo.mcp.server import (  # pyright: ignore[reportMissingImports]
    MCPServerError,
    MCPServerRegistry,
    _register_methods,
    mcp_server,
)

# ---------------------------------------------------------------------------
# MCPServerRegistry tests
# ---------------------------------------------------------------------------


class TestMCPServerRegistryInit:
    def test_empty_registry(self) -> None:
        reg = MCPServerRegistry()
        assert len(reg) == 0
        assert reg.names == []

    def test_repr(self) -> None:
        reg = MCPServerRegistry()
        assert "MCPServerRegistry" in repr(reg)


class TestMCPServerRegistryRegister:
    def test_register_class(self) -> None:
        reg = MCPServerRegistry()

        class Foo:
            pass

        reg.register("foo", Foo)
        assert reg.has("foo")
        assert reg.get_class("foo") is Foo

    def test_register_multiple(self) -> None:
        reg = MCPServerRegistry()

        class A:
            pass

        class B:
            pass

        reg.register("a", A)
        reg.register("b", B)
        assert len(reg) == 2
        assert sorted(reg.names) == ["a", "b"]

    def test_register_overwrite(self) -> None:
        reg = MCPServerRegistry()

        class V1:
            pass

        class V2:
            pass

        reg.register("svc", V1)
        reg.register("svc", V2)
        assert reg.get_class("svc") is V2

    def test_get_class_missing(self) -> None:
        reg = MCPServerRegistry()
        with pytest.raises(MCPServerError, match="not registered"):
            reg.get_class("missing")


class TestMCPServerRegistryInstance:
    def test_get_instance(self) -> None:
        reg = MCPServerRegistry()

        class Svc:
            pass

        reg.register("svc", Svc)
        inst = reg.get_instance("svc")
        assert isinstance(inst, Svc)

    def test_get_instance_singleton(self) -> None:
        reg = MCPServerRegistry()

        class Svc:
            pass

        reg.register("svc", Svc)
        inst1 = reg.get_instance("svc")
        inst2 = reg.get_instance("svc")
        assert inst1 is inst2

    def test_get_instance_with_args(self) -> None:
        reg = MCPServerRegistry()

        class Svc:
            def __init__(self, value: int) -> None:
                self.value = value

        reg.register("svc", Svc)
        inst = reg.get_instance("svc", 42)
        assert inst.value == 42

    def test_get_instance_missing(self) -> None:
        reg = MCPServerRegistry()
        with pytest.raises(MCPServerError, match="not registered"):
            reg.get_instance("missing")


class TestMCPServerRegistryClear:
    def test_clear(self) -> None:
        reg = MCPServerRegistry()

        class Svc:
            pass

        reg.register("svc", Svc)
        reg.get_instance("svc")
        reg.clear()
        assert len(reg) == 0
        assert not reg.has("svc")

    def test_has_unregistered(self) -> None:
        reg = MCPServerRegistry()
        assert not reg.has("unknown")


# ---------------------------------------------------------------------------
# @mcp_server() decorator tests
# ---------------------------------------------------------------------------


class TestMCPServerDecoratorInit:
    def test_creates_mcp_instance(self) -> None:
        mock_fast_mcp_cls = MagicMock()
        mock_fast_mcp = MagicMock()
        mock_fast_mcp.tool.return_value = lambda fn: fn
        mock_fast_mcp_cls.return_value = mock_fast_mcp

        with patch("exo.mcp.server.FastMCP", mock_fast_mcp_cls):

            @mcp_server(name="test-srv")
            class TestSrv:
                pass

            instance = TestSrv()

        assert hasattr(instance, "_mcp")
        assert instance._mcp is mock_fast_mcp  # pyright: ignore[reportAttributeAccessIssue]
        mock_fast_mcp_cls.assert_called_once_with("test-srv", instructions="test-srv MCP Server")

    def test_uses_class_docstring(self) -> None:
        mock_fast_mcp_cls = MagicMock()
        mock_fast_mcp = MagicMock()
        mock_fast_mcp.tool.return_value = lambda fn: fn
        mock_fast_mcp_cls.return_value = mock_fast_mcp

        with patch("exo.mcp.server.FastMCP", mock_fast_mcp_cls):

            @mcp_server(name="doc-srv")
            class DocServer:
                """A well-documented server."""

                pass

            DocServer()

        mock_fast_mcp_cls.assert_called_once_with(
            "doc-srv", instructions="A well-documented server."
        )

    def test_preserves_original_init(self) -> None:
        mock_fast_mcp_cls = MagicMock()
        mock_fast_mcp = MagicMock()
        mock_fast_mcp.tool.return_value = lambda fn: fn
        mock_fast_mcp_cls.return_value = mock_fast_mcp

        with patch("exo.mcp.server.FastMCP", mock_fast_mcp_cls):

            @mcp_server(name="init-srv")
            class InitSrv:
                def __init__(self, value: int = 10) -> None:
                    self.value = value

            inst = InitSrv(42)

        assert inst.value == 42
        assert hasattr(inst, "_mcp")


class TestMCPServerDecoratorMethods:
    def test_registers_public_methods_as_tools(self) -> None:
        registered_tools: list[str] = []
        mock_fast_mcp_cls = MagicMock()
        mock_fast_mcp = MagicMock()

        def track_tool(name: str, description: str) -> Any:
            registered_tools.append(name)
            return lambda fn: fn

        mock_fast_mcp.tool.side_effect = track_tool
        mock_fast_mcp_cls.return_value = mock_fast_mcp

        with patch("exo.mcp.server.FastMCP", mock_fast_mcp_cls):

            @mcp_server(name="calc")
            class Calculator:
                def add(self, a: int, b: int) -> int:
                    """Add two numbers."""
                    return a + b

                def subtract(self, a: int, b: int) -> int:
                    """Subtract b from a."""
                    return a - b

            Calculator()

        assert "add" in registered_tools
        assert "subtract" in registered_tools

    def test_skips_private_methods(self) -> None:
        registered_tools: list[str] = []
        mock_fast_mcp_cls = MagicMock()
        mock_fast_mcp = MagicMock()

        def track_tool(name: str, description: str) -> Any:
            registered_tools.append(name)
            return lambda fn: fn

        mock_fast_mcp.tool.side_effect = track_tool
        mock_fast_mcp_cls.return_value = mock_fast_mcp

        with patch("exo.mcp.server.FastMCP", mock_fast_mcp_cls):

            @mcp_server(name="priv")
            class PrivServer:
                def public_method(self) -> str:
                    return "public"

                def _private_method(self) -> str:
                    return "private"

                def __dunder(self) -> str:
                    return "dunder"

            PrivServer()

        assert "public_method" in registered_tools
        assert "_private_method" not in registered_tools
        assert "__dunder" not in registered_tools

    def test_skips_run_and_stop(self) -> None:
        registered_tools: list[str] = []
        mock_fast_mcp_cls = MagicMock()
        mock_fast_mcp = MagicMock()

        def track_tool(name: str, description: str) -> Any:
            registered_tools.append(name)
            return lambda fn: fn

        mock_fast_mcp.tool.side_effect = track_tool
        mock_fast_mcp_cls.return_value = mock_fast_mcp

        with patch("exo.mcp.server.FastMCP", mock_fast_mcp_cls):

            @mcp_server(name="skip")
            class SkipServer:
                def greet(self) -> str:
                    return "hi"

            SkipServer()

        # run and stop should NOT be in registered tools (they're added by decorator)
        assert "run" not in registered_tools
        assert "stop" not in registered_tools
        assert "greet" in registered_tools

    def test_tool_names_attribute(self) -> None:
        mock_fast_mcp_cls = MagicMock()
        mock_fast_mcp = MagicMock()
        mock_fast_mcp.tool.return_value = lambda fn: fn
        mock_fast_mcp_cls.return_value = mock_fast_mcp

        with patch("exo.mcp.server.FastMCP", mock_fast_mcp_cls):

            @mcp_server(name="names")
            class NamesSrv:
                def foo(self) -> str:
                    return "foo"

                def bar(self) -> str:
                    return "bar"

            inst = NamesSrv()

        assert hasattr(inst, "_tool_names")
        assert sorted(inst._tool_names) == ["bar", "foo"]  # pyright: ignore[reportAttributeAccessIssue]


class TestMCPServerDecoratorRun:
    def test_run_method_added(self) -> None:
        mock_fast_mcp_cls = MagicMock()
        mock_fast_mcp = MagicMock()
        mock_fast_mcp.tool.return_value = lambda fn: fn
        mock_fast_mcp_cls.return_value = mock_fast_mcp

        with patch("exo.mcp.server.FastMCP", mock_fast_mcp_cls):

            @mcp_server(name="run-test")
            class RunServer:
                pass

            inst = RunServer()

        assert hasattr(inst, "run")
        inst.run(transport="stdio")  # pyright: ignore[reportAttributeAccessIssue]
        mock_fast_mcp.run.assert_called_once_with(transport="stdio")

    def test_run_default_transport(self) -> None:
        mock_fast_mcp_cls = MagicMock()
        mock_fast_mcp = MagicMock()
        mock_fast_mcp.tool.return_value = lambda fn: fn
        mock_fast_mcp_cls.return_value = mock_fast_mcp

        with patch("exo.mcp.server.FastMCP", mock_fast_mcp_cls):

            @mcp_server(name="transport", transport="sse")
            class SSEServer:
                pass

            inst = SSEServer()

        inst.run()  # pyright: ignore[reportAttributeAccessIssue]
        mock_fast_mcp.run.assert_called_once_with(transport="sse")

    def test_run_not_initialized_raises(self) -> None:
        mock_fast_mcp_cls = MagicMock()
        mock_fast_mcp = MagicMock()
        mock_fast_mcp.tool.return_value = lambda fn: fn
        mock_fast_mcp_cls.return_value = mock_fast_mcp

        with patch("exo.mcp.server.FastMCP", mock_fast_mcp_cls):

            @mcp_server(name="no-init")
            class NoInitSrv:
                pass

        # Create instance via __new__ to skip __init__
        inst = NoInitSrv.__new__(NoInitSrv)
        with pytest.raises(MCPServerError, match="not initialized"):
            inst.run()  # pyright: ignore[reportAttributeAccessIssue]

    async def test_stop_method_added(self) -> None:
        mock_fast_mcp_cls = MagicMock()
        mock_fast_mcp = MagicMock()
        mock_fast_mcp.tool.return_value = lambda fn: fn
        mock_fast_mcp_cls.return_value = mock_fast_mcp

        with patch("exo.mcp.server.FastMCP", mock_fast_mcp_cls):

            @mcp_server(name="stop-test")
            class StopServer:
                pass

            inst = StopServer()

        assert hasattr(inst, "stop")
        await inst.stop()  # pyright: ignore[reportAttributeAccessIssue]


# ---------------------------------------------------------------------------
# _register_methods helper tests
# ---------------------------------------------------------------------------


class TestRegisterMethods:
    def test_registers_sync_methods(self) -> None:
        mock_mcp = MagicMock()
        mock_mcp.tool.return_value = lambda fn: fn

        class SyncSrv:
            def greet(self, name: str) -> str:
                """Say hello."""
                return f"Hi {name}"

        inst = SyncSrv()
        names = _register_methods(inst, mock_mcp)
        assert "greet" in names

    def test_registers_async_methods(self) -> None:
        mock_mcp = MagicMock()
        mock_mcp.tool.return_value = lambda fn: fn

        class AsyncSrv:
            async def fetch(self, url: str) -> str:
                """Fetch URL."""
                return f"Response from {url}"

        inst = AsyncSrv()
        names = _register_methods(inst, mock_mcp)
        assert "fetch" in names

    def test_no_public_methods(self) -> None:
        mock_mcp = MagicMock()
        mock_mcp.tool.return_value = lambda fn: fn

        class EmptySrv:
            def _internal(self) -> None:
                pass

        inst = EmptySrv()
        names = _register_methods(inst, mock_mcp)
        assert names == []
