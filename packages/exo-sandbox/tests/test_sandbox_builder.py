"""Tests for SandboxBuilder fluent API and lazy evaluation."""

from __future__ import annotations

from typing import Any

from exo.sandbox.base import (  # pyright: ignore[reportMissingImports]
    LocalSandbox,
    Sandbox,
    SandboxStatus,
)
from exo.sandbox.builder import SandboxBuilder  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------


class TestBuilderBasic:
    def test_default_builds_local_sandbox(self) -> None:
        sb = SandboxBuilder().build()
        assert isinstance(sb, LocalSandbox)

    def test_build_returns_same_instance(self) -> None:
        builder = SandboxBuilder()
        sb1 = builder.build()
        sb2 = builder.build()
        assert sb1 is sb2

    def test_reset_allows_rebuild(self) -> None:
        builder = SandboxBuilder()
        sb1 = builder.build()
        builder.reset()
        sb2 = builder.build()
        assert sb1 is not sb2

    def test_repr_pending(self) -> None:
        builder = SandboxBuilder()
        assert "pending" in repr(builder)
        assert "LocalSandbox" in repr(builder)

    def test_repr_built(self) -> None:
        builder = SandboxBuilder()
        builder.build()
        assert "built" in repr(builder)


# ---------------------------------------------------------------------------
# Fluent method chaining
# ---------------------------------------------------------------------------


class TestBuilderChaining:
    def test_all_setters_return_self(self) -> None:
        builder = SandboxBuilder()
        result = (
            builder.with_sandbox_id("id-1")
            .with_workspace(["/ws"])
            .with_mcp_config({"k": "v"})
            .with_agents({"a": {"m": "gpt-4"}})
            .with_timeout(99.0)
            .with_extra(custom_key="custom_val")
        )
        assert result is builder

    def test_sandbox_id(self) -> None:
        sb = SandboxBuilder().with_sandbox_id("my-id").build()
        assert sb.sandbox_id == "my-id"

    def test_workspace(self) -> None:
        sb = SandboxBuilder().with_workspace(["/a", "/b"]).build()
        assert sb.workspace == ["/a", "/b"]

    def test_mcp_config(self) -> None:
        sb = SandboxBuilder().with_mcp_config({"s": "local"}).build()
        assert sb.mcp_config == {"s": "local"}

    def test_agents(self) -> None:
        sb = SandboxBuilder().with_agents({"x": {"model": "o1"}}).build()
        assert sb.agents == {"x": {"model": "o1"}}

    def test_timeout(self) -> None:
        sb = SandboxBuilder().with_timeout(120.0).build()
        assert sb.timeout == 120.0

    def test_full_chain(self) -> None:
        sb = (
            SandboxBuilder()
            .with_sandbox_id("full")
            .with_workspace(["/tmp"])
            .with_mcp_config({"cfg": True})
            .with_agents({"ag1": {}})
            .with_timeout(45.0)
            .build()
        )
        assert sb.sandbox_id == "full"
        assert sb.workspace == ["/tmp"]
        assert sb.mcp_config == {"cfg": True}
        assert sb.agents == {"ag1": {}}
        assert sb.timeout == 45.0

    def test_workspace_makes_copy(self) -> None:
        original = ["/a"]
        builder = SandboxBuilder().with_workspace(original)
        original.append("/b")
        sb = builder.build()
        assert sb.workspace == ["/a"]

    def test_mcp_config_makes_copy(self) -> None:
        original: dict[str, Any] = {"k": "v"}
        builder = SandboxBuilder().with_mcp_config(original)
        original["new"] = True
        sb = builder.build()
        assert "new" not in sb.mcp_config

    def test_agents_makes_copy(self) -> None:
        original: dict[str, Any] = {"a": {}}
        builder = SandboxBuilder().with_agents(original)
        original["b"] = {}
        sb = builder.build()
        assert "b" not in sb.agents


# ---------------------------------------------------------------------------
# Custom sandbox class
# ---------------------------------------------------------------------------


class _StubSandbox(Sandbox):
    """Minimal Sandbox subclass for testing."""

    async def start(self) -> None:
        self._transition(SandboxStatus.RUNNING)

    async def stop(self) -> None:
        self._transition(SandboxStatus.IDLE)

    async def cleanup(self) -> None:
        self._transition(SandboxStatus.CLOSED)

    async def run_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        return {"tool": tool_name, "arguments": arguments, "status": "ok"}


class TestBuilderCustomClass:
    def test_with_sandbox_class_at_init(self) -> None:
        sb = SandboxBuilder(sandbox_class=_StubSandbox).build()
        assert isinstance(sb, _StubSandbox)

    def test_with_sandbox_class_setter(self) -> None:
        sb = SandboxBuilder().with_sandbox_class(_StubSandbox).build()
        assert isinstance(sb, _StubSandbox)

    def test_custom_class_receives_params(self) -> None:
        sb = (
            SandboxBuilder(sandbox_class=_StubSandbox)
            .with_sandbox_id("custom")
            .with_timeout(77.0)
            .build()
        )
        assert sb.sandbox_id == "custom"
        assert sb.timeout == 77.0


# ---------------------------------------------------------------------------
# with_extra
# ---------------------------------------------------------------------------


class TestBuilderExtra:
    def test_extra_kwargs_passed(self) -> None:
        sb = SandboxBuilder().with_sandbox_id("e").with_extra(workspace=["/x"]).build()
        assert sb.workspace == ["/x"]

    def test_extra_overrides_nothing_when_explicit(self) -> None:
        sb = SandboxBuilder().with_timeout(10.0).with_extra(timeout=999.0).build()
        # with_extra kwargs are applied after the named kwargs,
        # so extra wins if both set the same key
        assert sb.timeout == 999.0


# ---------------------------------------------------------------------------
# Lazy evaluation via __getattr__
# ---------------------------------------------------------------------------


class TestBuilderLazy:
    def test_lazy_property_access(self) -> None:
        builder = SandboxBuilder().with_sandbox_id("lazy-1")
        # Not built yet
        assert builder._built is None
        # Access a Sandbox property — triggers build
        assert builder.sandbox_id == "lazy-1"
        assert builder._built is not None

    def test_lazy_describe(self) -> None:
        builder = SandboxBuilder().with_sandbox_id("lazy-d")
        d = builder.describe()
        assert d["sandbox_id"] == "lazy-d"
        assert builder._built is not None

    async def test_lazy_start(self) -> None:
        builder = SandboxBuilder().with_sandbox_id("lazy-s")
        await builder.start()
        assert builder._built is not None
        assert builder.status == SandboxStatus.RUNNING

    async def test_lazy_context_manager(self) -> None:
        builder = SandboxBuilder().with_sandbox_id("lazy-ctx")
        # __aenter__/__aexit__ are on LocalSandbox, not the builder
        sandbox = builder.build()
        async with sandbox as sb:
            assert sb.status == SandboxStatus.RUNNING
        assert sb.status == SandboxStatus.CLOSED

    async def test_lazy_run_tool(self) -> None:
        # Use _StubSandbox which has a trivial run_tool implementation
        builder = SandboxBuilder(sandbox_class=_StubSandbox).with_sandbox_id("lazy-t")
        await builder.start()
        result = await builder.run_tool("test", {"a": 1})
        assert result["tool"] == "test"
        assert result["status"] == "ok"

    def test_lazy_repr_still_works(self) -> None:
        builder = SandboxBuilder()
        r = repr(builder)
        assert "SandboxBuilder" in r
        # __repr__ is on builder itself, not forwarded
        assert builder._built is None

    def test_builder_methods_not_forwarded(self) -> None:
        builder = SandboxBuilder()
        # Builder methods should be accessible without triggering build
        assert hasattr(builder, "with_workspace")
        assert hasattr(builder, "build")
        assert hasattr(builder, "reset")
        assert builder._built is None

    async def test_lazy_full_lifecycle(self) -> None:
        builder = (
            SandboxBuilder()
            .with_sandbox_id("lazy-full")
            .with_workspace(["/tmp"])
            .with_timeout(50.0)
        )
        # Nothing built yet
        assert builder._built is None

        # Start triggers lazy build + start
        await builder.start()
        assert builder._built is not None
        assert builder.status == SandboxStatus.RUNNING

        # Stop via lazy delegation
        await builder.stop()
        assert builder.status == SandboxStatus.IDLE

        # Cleanup via lazy delegation
        await builder.cleanup()
        assert builder.status == SandboxStatus.CLOSED
