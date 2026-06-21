"""Tests for ServerEntry namespace kwargs (StdioConfig / HttpConfig).

Covers:
- Grouped kwargs populate flat attrs
- Flat kwargs still work unchanged (backward compat)
- Conflict (grouped + flat) raises MCPConfigError
- to_dict / from_dict (via load_config/save_config) round-trip unchanged
- Read accessor properties return correct views
- load_config still produces correct ServerEntry instances
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from exo.mcp.client import HttpConfig, StdioConfig  # pyright: ignore[reportMissingImports]
from exo_mcp_cli.config import MCPConfigError, ServerEntry, load_config, save_config

# ---------------------------------------------------------------------------
# Grouped kwargs populate flat attrs
# ---------------------------------------------------------------------------


def test_stdio_grouped_kwarg_populates_flat_attrs() -> None:
    entry = ServerEntry(
        name="my-server",
        transport="stdio",
        stdio=StdioConfig(command="npx", args=["mcp-server"], env={"TOKEN": "abc"}, cwd="/tmp"),
    )
    assert entry.command == "npx"
    assert entry.args == ["mcp-server"]
    assert entry.env == {"TOKEN": "abc"}
    assert entry.cwd == "/tmp"
    assert entry.url is None
    assert entry.headers is None


def test_http_grouped_kwarg_populates_flat_attrs() -> None:
    entry = ServerEntry(
        name="remote",
        transport="sse",
        http=HttpConfig(url="http://example.com", headers={"X-Auth": "tok"}),
    )
    assert entry.url == "http://example.com"
    assert entry.headers == {"X-Auth": "tok"}
    assert entry.command is None


def test_both_grouped_kwargs_together() -> None:
    entry = ServerEntry(
        name="mixed",
        transport="stdio",
        stdio=StdioConfig(command="python"),
        http=HttpConfig(url="http://fallback.example.com"),
    )
    assert entry.command == "python"
    assert entry.url == "http://fallback.example.com"


# ---------------------------------------------------------------------------
# Flat kwargs still work (backward compat)
# ---------------------------------------------------------------------------


def test_flat_kwargs_stdio_still_work() -> None:
    entry = ServerEntry(
        name="flat-stdio",
        transport="stdio",
        command="uvx",
        args=["mcp-server-fetch"],
        env={"HOME": "/home/user"},
        cwd="/workspace",
    )
    assert entry.command == "uvx"
    assert entry.args == ["mcp-server-fetch"]
    assert entry.env == {"HOME": "/home/user"}
    assert entry.cwd == "/workspace"


def test_flat_kwargs_http_still_work() -> None:
    entry = ServerEntry(
        name="flat-http",
        transport="streamable_http",
        url="http://remote.example.com/mcp",
        headers={"Content-Type": "application/json"},
    )
    assert entry.url == "http://remote.example.com/mcp"
    assert entry.headers == {"Content-Type": "application/json"}


def test_defaults_unchanged() -> None:
    entry = ServerEntry(name="defaults", transport="stdio", command="python")
    assert entry.args == []
    assert entry.env is None
    assert entry.cwd is None
    assert entry.url is None
    assert entry.headers is None
    assert entry.timeout == 30.0


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------


def test_stdio_conflict_on_command_raises() -> None:
    with pytest.raises(MCPConfigError, match="cannot mix stdio="):
        ServerEntry(
            name="conflict",
            transport="stdio",
            stdio=StdioConfig(command="python"),
            command="node",
        )


def test_stdio_conflict_on_args_raises() -> None:
    with pytest.raises(MCPConfigError, match="cannot mix stdio="):
        ServerEntry(
            name="conflict",
            transport="stdio",
            stdio=StdioConfig(command="python"),
            args=["--extra"],
        )


def test_http_conflict_on_url_raises() -> None:
    with pytest.raises(MCPConfigError, match="cannot mix http="):
        ServerEntry(
            name="conflict",
            transport="sse",
            http=HttpConfig(url="http://a.example.com"),
            url="http://b.example.com",
        )


def test_http_conflict_on_headers_raises() -> None:
    with pytest.raises(MCPConfigError, match="cannot mix http="):
        ServerEntry(
            name="conflict",
            transport="sse",
            http=HttpConfig(url="http://example.com"),
            headers={"X-Token": "extra"},
        )


# ---------------------------------------------------------------------------
# to_dict / save_config + load_config round-trip
# ---------------------------------------------------------------------------


def test_to_dict_flat_stdio() -> None:
    entry = ServerEntry(
        name="srv",
        transport="stdio",
        command="python",
        args=["-m", "srv"],
        env={"K": "v"},
        cwd="/tmp",
        timeout=45.0,
    )
    d = entry.to_dict()
    assert d["transport"] == "stdio"
    assert d["command"] == "python"
    assert d["args"] == ["-m", "srv"]
    assert d["env"] == {"K": "v"}
    assert d["timeout"] == 45.0


def test_to_dict_flat_http() -> None:
    entry = ServerEntry(
        name="srv",
        transport="sse",
        url="http://example.com",
        headers={"X": "Y"},
    )
    d = entry.to_dict()
    assert d["transport"] == "sse"
    assert d["url"] == "http://example.com"
    assert d["headers"] == {"X": "Y"}
    assert "command" not in d


def test_grouped_and_flat_produce_same_to_dict() -> None:
    flat = ServerEntry(name="srv", transport="stdio", command="python", args=["-m", "server"])
    grouped = ServerEntry(
        name="srv",
        transport="stdio",
        stdio=StdioConfig(command="python", args=["-m", "server"]),
    )
    assert flat.to_dict() == grouped.to_dict()


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    entry = ServerEntry(
        name="roundtrip",
        transport="stdio",
        command="node",
        args=["server.js"],
        cwd="/app",
    )
    save_config(path, {"roundtrip": entry})
    loaded = load_config(path)
    assert "roundtrip" in loaded
    e = loaded["roundtrip"]
    assert e.command == "node"
    assert e.args == ["server.js"]
    assert e.cwd == "/app"
    assert e.transport == "stdio"


def test_to_dict_no_extra_keys() -> None:
    """Wire format must not include 'stdio' or 'http' keys — those are init-only."""
    entry = ServerEntry(
        name="wire",
        transport="stdio",
        stdio=StdioConfig(command="python"),
    )
    d = entry.to_dict()
    assert "stdio" not in d
    assert "http" not in d


# ---------------------------------------------------------------------------
# Read accessor properties
# ---------------------------------------------------------------------------


def test_stdio_config_property_returns_view() -> None:
    entry = ServerEntry(
        name="view",
        transport="stdio",
        command="uvx",
        args=["mcp-fetch"],
        env={"HOME": "/home"},
        cwd="/workspace",
    )
    view = entry.stdio_config
    assert view is not None
    assert isinstance(view, StdioConfig)
    assert view.command == "uvx"
    assert view.args == ["mcp-fetch"]
    assert view.env == {"HOME": "/home"}
    assert view.cwd == "/workspace"


def test_stdio_config_property_none_when_no_command() -> None:
    entry = ServerEntry(name="no-cmd", transport="sse", url="http://example.com")
    assert entry.stdio_config is None


def test_http_config_property_returns_view() -> None:
    entry = ServerEntry(
        name="view-http",
        transport="sse",
        url="http://example.com",
        headers={"Authorization": "Bearer tok"},
    )
    view = entry.http_config
    assert view is not None
    assert isinstance(view, HttpConfig)
    assert view.url == "http://example.com"
    assert view.headers == {"Authorization": "Bearer tok"}


def test_http_config_property_none_when_no_url() -> None:
    entry = ServerEntry(name="no-url", transport="stdio", command="python")
    assert entry.http_config is None


def test_grouped_accessor_matches_input() -> None:
    stdio_in = StdioConfig(command="node", args=["srv.js"], env=None, cwd="/app")
    entry = ServerEntry(name="match", transport="stdio", stdio=stdio_in)
    assert entry.stdio_config == stdio_in


# ---------------------------------------------------------------------------
# load_config still works after change
# ---------------------------------------------------------------------------


def test_load_config_plain_json(tmp_path: Path) -> None:
    cfg = {
        "mcpServers": {
            "my-server": {
                "transport": "stdio",
                "command": "python",
                "args": ["-m", "server"],
                "timeout": 60.0,
            },
            "remote-server": {
                "transport": "sse",
                "url": "http://remote.example.com",
            },
        }
    }
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps(cfg))
    servers = load_config(path)
    assert servers["my-server"].command == "python"
    assert servers["my-server"].args == ["-m", "server"]
    assert servers["my-server"].timeout == 60.0
    assert servers["remote-server"].url == "http://remote.example.com"
