"""Tests for MCP execution — env var substitution and config loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from exo.mcp.execution import (  # pyright: ignore[reportMissingImports]
    MCPExecutionError,
    _substitute_recursive,
    load_mcp_client,
    load_mcp_config,
    substitute_env_vars,
)

# ===========================================================================
# substitute_env_vars
# ===========================================================================


class TestSubstituteEnvVars:
    """Tests for env var substitution."""

    def test_simple_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_VAR", "hello")
        assert substitute_env_vars("${MY_VAR}") == "hello"

    def test_multiple_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("A", "1")
        monkeypatch.setenv("B", "2")
        assert substitute_env_vars("${A}-${B}") == "1-2"

    def test_unset_var_becomes_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DOES_NOT_EXIST", raising=False)
        assert substitute_env_vars("prefix-${DOES_NOT_EXIST}-suffix") == "prefix--suffix"

    def test_no_vars(self) -> None:
        assert substitute_env_vars("no vars here") == "no vars here"

    def test_empty_string(self) -> None:
        assert substitute_env_vars("") == ""

    def test_nested_braces_ignored(self) -> None:
        assert substitute_env_vars("${") == "${"


class TestSubstituteRecursive:
    """Tests for recursive substitution in dicts/lists."""

    def test_dict_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("X", "val")
        result = _substitute_recursive({"key": "${X}", "num": 42})
        assert result == {"key": "val", "num": 42}

    def test_nested_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("V", "found")
        result = _substitute_recursive(["${V}", ["inner-${V}"]])
        assert result == ["found", ["inner-found"]]

    def test_passthrough_non_string(self) -> None:
        assert _substitute_recursive(42) == 42
        assert _substitute_recursive(None) is None


# ===========================================================================
# load_mcp_config / load_mcp_client
# ===========================================================================


class TestLoadMCPConfig:
    """Tests for config file loading."""

    def test_load_stdio_server(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PY", "/usr/bin/python")
        cfg_file = tmp_path / "mcp.json"
        cfg_file.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "my-server": {
                            "transport": "stdio",
                            "command": "${PY}",
                            "args": ["-m", "server"],
                        }
                    }
                }
            )
        )
        configs = load_mcp_config(cfg_file)
        assert len(configs) == 1
        assert configs[0].name == "my-server"
        assert configs[0].command == "/usr/bin/python"
        assert configs[0].args == ["-m", "server"]

    def test_load_multiple_servers(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "mcp.json"
        cfg_file.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "a": {"transport": "stdio", "command": "cmd-a"},
                        "b": {"transport": "sse", "url": "http://localhost:8080"},
                    }
                }
            )
        )
        configs = load_mcp_config(cfg_file)
        assert len(configs) == 2
        names = {c.name for c in configs}
        assert names == {"a", "b"}

    def test_env_substitution_in_nested(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("API_KEY", "secret123")
        cfg_file = tmp_path / "mcp.json"
        cfg_file.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "api": {
                            "transport": "sse",
                            "url": "http://host",
                            "headers": {"Authorization": "Bearer ${API_KEY}"},
                        }
                    }
                }
            )
        )
        configs = load_mcp_config(cfg_file)
        assert configs[0].headers == {"Authorization": "Bearer secret123"}

    def test_missing_file(self) -> None:
        with pytest.raises(MCPExecutionError, match="not found"):
            load_mcp_config("/nonexistent/mcp.json")

    def test_invalid_json(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "mcp.json"
        cfg_file.write_text("not json")
        with pytest.raises(MCPExecutionError, match="Failed to parse"):
            load_mcp_config(cfg_file)

    def test_invalid_servers_type(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "mcp.json"
        cfg_file.write_text(json.dumps({"mcpServers": "not-a-dict"}))
        with pytest.raises(MCPExecutionError, match="Expected 'mcpServers'"):
            load_mcp_config(cfg_file)

    def test_empty_servers(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "mcp.json"
        cfg_file.write_text(json.dumps({"mcpServers": {}}))
        configs = load_mcp_config(cfg_file)
        assert configs == []

    def test_defaults(self, tmp_path: Path) -> None:
        """Default transport is stdio, default timeout is 30."""
        cfg_file = tmp_path / "mcp.json"
        cfg_file.write_text(json.dumps({"mcpServers": {"s": {"command": "cmd"}}}))
        configs = load_mcp_config(cfg_file)
        assert configs[0].transport.value == "stdio"
        assert configs[0].timeout == 30.0


class TestLoadMCPClient:
    """Tests for the convenience client loader."""

    def test_creates_client_with_servers(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "mcp.json"
        cfg_file.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "s1": {"transport": "stdio", "command": "cmd1"},
                        "s2": {"transport": "stdio", "command": "cmd2"},
                    }
                }
            )
        )
        client = load_mcp_client(cfg_file)
        assert set(client.server_names) == {"s1", "s2"}


# ===========================================================================
# Integration: config → client
# ===========================================================================


class TestIntegration:
    """Integration tests for the execution module."""

    def test_config_to_client_end_to_end(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Load config with env vars, create client, verify server configs."""
        monkeypatch.setenv("SERVER_CMD", "/usr/local/bin/serve")
        monkeypatch.setenv("SERVER_URL", "http://api.example.com")
        cfg_file = tmp_path / "mcp.json"
        cfg_file.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "local": {
                            "transport": "stdio",
                            "command": "${SERVER_CMD}",
                            "args": ["--port", "8080"],
                        },
                        "remote": {
                            "transport": "sse",
                            "url": "${SERVER_URL}/mcp",
                        },
                    }
                }
            )
        )
        client = load_mcp_client(cfg_file)
        assert set(client.server_names) == {"local", "remote"}
