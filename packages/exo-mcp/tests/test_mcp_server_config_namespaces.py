"""Tests for MCPServerConfig namespace kwargs (StdioConfig / HttpConfig).

Covers:
- StdioConfig and HttpConfig dataclasses
- Grouped kwargs populate flat attrs
- Flat kwargs still work unchanged
- Conflict (grouped + flat) raises MCPClientError
- to_dict / from_dict round-trip is unchanged
- Read accessor properties return correct views
"""

from __future__ import annotations

import pytest

from exo.mcp.client import (  # pyright: ignore[reportMissingImports]
    HttpConfig,
    MCPClientError,
    MCPServerConfig,
    MCPTransport,
    StdioConfig,
)

# ---------------------------------------------------------------------------
# StdioConfig / HttpConfig dataclasses
# ---------------------------------------------------------------------------


def test_stdio_config_basic() -> None:
    cfg = StdioConfig(command="python", args=["-m", "server"], env={"KEY": "val"}, cwd="/tmp")
    assert cfg.command == "python"
    assert cfg.args == ["-m", "server"]
    assert cfg.env == {"KEY": "val"}
    assert cfg.cwd == "/tmp"


def test_stdio_config_defaults() -> None:
    cfg = StdioConfig(command="npx")
    assert cfg.args == []
    assert cfg.env is None
    assert cfg.cwd is None


def test_http_config_basic() -> None:
    cfg = HttpConfig(url="http://localhost:8080", headers={"Authorization": "Bearer tok"})
    assert cfg.url == "http://localhost:8080"
    assert cfg.headers == {"Authorization": "Bearer tok"}


def test_http_config_defaults() -> None:
    cfg = HttpConfig(url="http://example.com")
    assert cfg.headers is None


def test_stdio_config_frozen() -> None:
    cfg = StdioConfig(command="python")
    with pytest.raises((AttributeError, TypeError)):
        cfg.command = "other"  # type: ignore[misc]


def test_http_config_frozen() -> None:
    cfg = HttpConfig(url="http://example.com")
    with pytest.raises((AttributeError, TypeError)):
        cfg.url = "http://other.com"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Grouped kwargs populate flat attrs on MCPServerConfig
# ---------------------------------------------------------------------------


def test_stdio_grouped_kwarg_populates_flat_attrs() -> None:
    sc = MCPServerConfig(
        name="test",
        transport=MCPTransport.STDIO,
        stdio=StdioConfig(command="npx", args=["mcp-server"], env={"A": "1"}, cwd="/srv"),
    )
    assert sc.command == "npx"
    assert sc.args == ["mcp-server"]
    assert sc.env == {"A": "1"}
    assert sc.cwd == "/srv"
    # HTTP attrs should be untouched
    assert sc.url is None
    assert sc.headers is None


def test_http_grouped_kwarg_populates_flat_attrs() -> None:
    sc = MCPServerConfig(
        name="test",
        transport=MCPTransport.SSE,
        http=HttpConfig(url="http://localhost:9000", headers={"X-Api-Key": "secret"}),
    )
    assert sc.url == "http://localhost:9000"
    assert sc.headers == {"X-Api-Key": "secret"}
    # Stdio attrs should be untouched
    assert sc.command is None


def test_both_grouped_kwargs() -> None:
    """Using both stdio= and http= together (unusual but valid for mixed config storage)."""
    sc = MCPServerConfig(
        name="test",
        transport=MCPTransport.STDIO,
        stdio=StdioConfig(command="python"),
        http=HttpConfig(url="http://example.com"),
    )
    assert sc.command == "python"
    assert sc.url == "http://example.com"


# ---------------------------------------------------------------------------
# Flat kwargs still work (backward compat)
# ---------------------------------------------------------------------------


def test_flat_stdio_kwargs_still_work() -> None:
    sc = MCPServerConfig(
        name="flat-stdio",
        transport=MCPTransport.STDIO,
        command="python",
        args=["-m", "myserver"],
        env={"PATH": "/usr/bin"},
        cwd="/app",
    )
    assert sc.command == "python"
    assert sc.args == ["-m", "myserver"]
    assert sc.env == {"PATH": "/usr/bin"}
    assert sc.cwd == "/app"


def test_flat_http_kwargs_still_work() -> None:
    sc = MCPServerConfig(
        name="flat-http",
        transport=MCPTransport.STREAMABLE_HTTP,
        url="http://remote.example.com",
        headers={"Content-Type": "application/json"},
    )
    assert sc.url == "http://remote.example.com"
    assert sc.headers == {"Content-Type": "application/json"}


def test_flat_timeout_kwargs_still_work() -> None:
    sc = MCPServerConfig(
        name="timeouts",
        transport=MCPTransport.SSE,
        url="http://example.com",
        timeout=60.0,
        sse_read_timeout=600.0,
        session_timeout=300.0,
    )
    assert sc.timeout == 60.0
    assert sc.sse_read_timeout == 600.0
    assert sc.session_timeout == 300.0


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------


def test_stdio_conflict_raises_on_command_overlap() -> None:
    with pytest.raises(MCPClientError, match="cannot mix stdio="):
        MCPServerConfig(
            name="conflict",
            transport=MCPTransport.STDIO,
            stdio=StdioConfig(command="python"),
            command="node",
        )


def test_stdio_conflict_raises_on_args_overlap() -> None:
    with pytest.raises(MCPClientError, match="cannot mix stdio="):
        MCPServerConfig(
            name="conflict",
            transport=MCPTransport.STDIO,
            stdio=StdioConfig(command="python"),
            args=["--version"],
        )


def test_http_conflict_raises_on_url_overlap() -> None:
    with pytest.raises(MCPClientError, match="cannot mix http="):
        MCPServerConfig(
            name="conflict",
            transport=MCPTransport.SSE,
            http=HttpConfig(url="http://a.example.com"),
            url="http://b.example.com",
        )


def test_http_conflict_raises_on_headers_overlap() -> None:
    with pytest.raises(MCPClientError, match="cannot mix http="):
        MCPServerConfig(
            name="conflict",
            transport=MCPTransport.SSE,
            http=HttpConfig(url="http://example.com"),
            headers={"X-Token": "tok"},
        )


# ---------------------------------------------------------------------------
# to_dict / from_dict round-trip unchanged
# ---------------------------------------------------------------------------


def test_to_dict_from_dict_roundtrip_stdio() -> None:
    sc = MCPServerConfig(
        name="roundtrip-stdio",
        transport=MCPTransport.STDIO,
        command="python",
        args=["-m", "srv"],
        env={"K": "v"},
        cwd="/tmp",
        timeout=45.0,
        cache_tools=True,
        large_output_tools=["big_tool"],
    )
    d = sc.to_dict()
    sc2 = MCPServerConfig.from_dict(d)
    assert sc2.name == sc.name
    assert sc2.transport == sc.transport
    assert sc2.command == sc.command
    assert sc2.args == sc.args
    assert sc2.env == sc.env
    assert sc2.cwd == sc.cwd
    assert sc2.timeout == sc.timeout
    assert sc2.cache_tools == sc.cache_tools
    assert sc2.large_output_tools == sc.large_output_tools


def test_to_dict_from_dict_roundtrip_http() -> None:
    sc = MCPServerConfig(
        name="roundtrip-http",
        transport=MCPTransport.SSE,
        url="http://example.com/mcp",
        headers={"Authorization": "Bearer x"},
        sse_read_timeout=600.0,
        session_timeout=None,
    )
    d = sc.to_dict()
    sc2 = MCPServerConfig.from_dict(d)
    assert sc2.url == sc.url
    assert sc2.headers == sc.headers
    assert sc2.sse_read_timeout == sc.sse_read_timeout
    assert sc2.session_timeout is None


def test_to_dict_wire_format_unchanged() -> None:
    """Keys emitted by to_dict must match the pre-namespace wire format exactly."""
    sc = MCPServerConfig(
        name="wire",
        transport=MCPTransport.STDIO,
        command="python",
        args=[],
        env=None,
        cwd=None,
        url=None,
        headers=None,
    )
    d = sc.to_dict()
    expected_keys = {
        "name",
        "transport",
        "command",
        "args",
        "env",
        "cwd",
        "url",
        "headers",
        "timeout",
        "sse_read_timeout",
        "cache_tools",
        "session_timeout",
        "large_output_tools",
    }
    assert set(d.keys()) == expected_keys


def test_grouped_kwargs_to_dict_from_dict_roundtrip() -> None:
    """Grouped API produces the same to_dict output as the flat API."""
    sc_flat = MCPServerConfig(
        name="g", transport=MCPTransport.STDIO, command="python", args=["-m", "srv"]
    )
    sc_grouped = MCPServerConfig(
        name="g",
        transport=MCPTransport.STDIO,
        stdio=StdioConfig(command="python", args=["-m", "srv"]),
    )
    assert sc_flat.to_dict() == sc_grouped.to_dict()


# ---------------------------------------------------------------------------
# Read accessor properties
# ---------------------------------------------------------------------------


def test_stdio_config_property_returns_view() -> None:
    sc = MCPServerConfig(
        name="view",
        transport=MCPTransport.STDIO,
        command="uvx",
        args=["mcp-server-fetch"],
        env={"HOME": "/home/user"},
        cwd="/workspace",
    )
    view = sc.stdio_config
    assert view is not None
    assert isinstance(view, StdioConfig)
    assert view.command == "uvx"
    assert view.args == ["mcp-server-fetch"]
    assert view.env == {"HOME": "/home/user"}
    assert view.cwd == "/workspace"


def test_stdio_config_property_none_when_no_command() -> None:
    sc = MCPServerConfig(
        name="http-only",
        transport=MCPTransport.SSE,
        url="http://example.com",
    )
    assert sc.stdio_config is None


def test_http_config_property_returns_view() -> None:
    sc = MCPServerConfig(
        name="view-http",
        transport=MCPTransport.SSE,
        url="http://example.com",
        headers={"X-Key": "val"},
    )
    view = sc.http_config
    assert view is not None
    assert isinstance(view, HttpConfig)
    assert view.url == "http://example.com"
    assert view.headers == {"X-Key": "val"}


def test_http_config_property_none_when_no_url() -> None:
    sc = MCPServerConfig(
        name="stdio-only",
        transport=MCPTransport.STDIO,
        command="python",
    )
    assert sc.http_config is None


def test_grouped_via_accessor_matches_original_grouped_kwarg() -> None:
    stdio_in = StdioConfig(command="node", args=["server.js"], env=None, cwd="/app")
    sc = MCPServerConfig(name="match", transport=MCPTransport.STDIO, stdio=stdio_in)
    view = sc.stdio_config
    assert view == stdio_in
