"""Tests for exo_mcp_cli.config — MCP server config loading and management."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from exo_mcp_cli.config import (
    MCPConfigError,
    ServerEntry,
    add_server,
    default_config_path,
    find_config,
    load_config,
    load_or_empty,
    remove_server,
    save_config,
    substitute_env_vars,
)

# ---------------------------------------------------------------------------
# Helper: write a standard mcp.json
# ---------------------------------------------------------------------------


def _write_mcp_json(path: Path, servers: dict | None = None) -> Path:
    """Write a valid mcp.json to *path* and return it."""
    if servers is None:
        servers = {"test": {"transport": "stdio", "command": "echo", "args": ["hi"]}}
    data = {"mcpServers": servers}
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# ServerEntry defaults
# ---------------------------------------------------------------------------


class TestServerEntryDefaults:
    def test_defaults(self) -> None:
        entry = ServerEntry(name="s1")
        assert entry.transport == "stdio"
        assert entry.command is None
        assert entry.args == []
        assert entry.env is None
        assert entry.cwd is None
        assert entry.url is None
        assert entry.headers is None
        assert entry.timeout == 30.0


# ---------------------------------------------------------------------------
# ServerEntry.validate
# ---------------------------------------------------------------------------


class TestServerEntryValidate:
    def test_valid_stdio(self) -> None:
        entry = ServerEntry(name="ok", transport="stdio", command="python")
        entry.validate()  # should not raise

    def test_stdio_without_command_raises(self) -> None:
        entry = ServerEntry(name="bad", transport="stdio")
        with pytest.raises(MCPConfigError, match="requires 'command'"):
            entry.validate()

    def test_sse_without_url_raises(self) -> None:
        entry = ServerEntry(name="bad-sse", transport="sse")
        with pytest.raises(MCPConfigError, match="requires 'url'"):
            entry.validate()

    def test_valid_sse(self) -> None:
        entry = ServerEntry(name="ok-sse", transport="sse", url="http://localhost:8080")
        entry.validate()  # should not raise

    def test_streamable_http_without_url_raises(self) -> None:
        entry = ServerEntry(name="bad-http", transport="streamable_http")
        with pytest.raises(MCPConfigError, match="requires 'url'"):
            entry.validate()


# ---------------------------------------------------------------------------
# ServerEntry.to_dict
# ---------------------------------------------------------------------------


class TestServerEntryToDict:
    def test_serializes_correctly(self) -> None:
        entry = ServerEntry(
            name="s1",
            transport="stdio",
            command="python",
            args=["-m", "server"],
            env={"KEY": "val"},
        )
        d = entry.to_dict()
        assert d["transport"] == "stdio"
        assert d["command"] == "python"
        assert d["args"] == ["-m", "server"]
        assert d["env"] == {"KEY": "val"}
        # None fields and default timeout omitted
        assert "cwd" not in d
        assert "url" not in d
        assert "headers" not in d
        assert "timeout" not in d

    def test_omits_none_fields(self) -> None:
        entry = ServerEntry(name="minimal", command="echo")
        d = entry.to_dict()
        assert "url" not in d
        assert "headers" not in d
        assert "env" not in d
        assert "cwd" not in d

    def test_includes_nondefault_timeout(self) -> None:
        entry = ServerEntry(name="slow", command="echo", timeout=60.0)
        d = entry.to_dict()
        assert d["timeout"] == 60.0

    def test_roundtrip_via_save_and_load(self, tmp_path: Path) -> None:
        original = ServerEntry(
            name="rt",
            transport="stdio",
            command="python",
            args=["-m", "mymod"],
            env={"A": "1"},
            timeout=45.0,
        )
        cfg_path = tmp_path / "mcp.json"
        save_config(cfg_path, {"rt": original})
        loaded = load_config(cfg_path)
        assert "rt" in loaded
        rt = loaded["rt"]
        assert rt.name == "rt"
        assert rt.transport == "stdio"
        assert rt.command == "python"
        assert rt.args == ["-m", "mymod"]
        assert rt.env == {"A": "1"}
        assert rt.timeout == 45.0


# ---------------------------------------------------------------------------
# substitute_env_vars
# ---------------------------------------------------------------------------


class TestSubstituteEnvVars:
    def test_replaces_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_TOKEN", "abc123")
        assert substitute_env_vars("key=${MY_TOKEN}") == "key=abc123"

    def test_unset_var_replaced_with_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("UNSET_VAR_XYZ", raising=False)
        assert substitute_env_vars("val=${UNSET_VAR_XYZ}end") == "val=end"

    def test_vault_references_untouched(self) -> None:
        result = substitute_env_vars("Bearer ${vault:API_KEY}")
        assert result == "Bearer ${vault:API_KEY}"

    def test_multiple_substitutions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOST", "localhost")
        monkeypatch.setenv("PORT", "8080")
        result = substitute_env_vars("http://${HOST}:${PORT}")
        assert result == "http://localhost:8080"

    def test_no_placeholders(self) -> None:
        assert substitute_env_vars("plain text") == "plain text"

    def test_warn_missing_emits_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.delenv("MISSING_CMD_VAR", raising=False)
        import logging

        with caplog.at_level(logging.WARNING, logger="exo_mcp_cli.config"):
            result = substitute_env_vars("${MISSING_CMD_VAR}", warn_missing=True)
        assert result == ""
        assert "MISSING_CMD_VAR" in caplog.text

    def test_warn_missing_silent_by_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.delenv("MISSING_CMD_VAR", raising=False)
        import logging

        with caplog.at_level(logging.WARNING, logger="exo_mcp_cli.config"):
            result = substitute_env_vars("${MISSING_CMD_VAR}")
        assert result == ""
        assert "MISSING_CMD_VAR" not in caplog.text


# ---------------------------------------------------------------------------
# find_config
# ---------------------------------------------------------------------------


class TestFindConfig:
    def test_finds_mcp_json_in_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Patch _cwd_path (a helper in the module) so the test does not mutate the
        # global process cwd — chdir is not safe under xdist parallel workers.
        cfg = _write_mcp_json(tmp_path / "mcp.json")
        monkeypatch.setattr("exo_mcp_cli.config._cwd", lambda: tmp_path)
        result = find_config()
        assert result is not None
        assert result == cfg

    def test_finds_home_fallback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Use a directory with no mcp.json so the local cwd search fails.
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        monkeypatch.setattr("exo_mcp_cli.config._cwd", lambda: empty_dir)
        # Create the home config dir structure inside tmp_path.
        home_cfg = tmp_path / "home_cfg"
        home_cfg.mkdir()
        _write_mcp_json(home_cfg / "mcp.json")
        monkeypatch.setattr("exo_mcp_cli.config._HOME_CONFIG_DIR", home_cfg)
        result = find_config()
        assert result is not None
        assert result == home_cfg / "mcp.json"

    def test_returns_none_when_no_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Empty dir + patched home dir → no config anywhere.
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        monkeypatch.setattr("exo_mcp_cli.config._cwd", lambda: empty_dir)
        monkeypatch.setattr("exo_mcp_cli.config._HOME_CONFIG_DIR", tmp_path / "nope")
        assert find_config() is None

    def test_explicit_nonexistent_raises(self) -> None:
        with pytest.raises(MCPConfigError, match="Config file not found"):
            find_config("/does/not/exist/mcp.json")

    def test_explicit_valid_path(self, tmp_path: Path) -> None:
        cfg = tmp_path / "custom.json"
        _write_mcp_json(cfg)
        result = find_config(cfg)
        assert result == cfg


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_parses_valid_config(self, tmp_path: Path) -> None:
        cfg = _write_mcp_json(tmp_path / "mcp.json")
        servers = load_config(cfg)
        assert "test" in servers
        assert servers["test"].command == "echo"
        assert servers["test"].args == ["hi"]
        assert servers["test"].transport == "stdio"

    def test_raises_on_invalid_json(self, tmp_path: Path) -> None:
        cfg = tmp_path / "bad.json"
        cfg.write_text("{invalid json!!!", encoding="utf-8")
        with pytest.raises(MCPConfigError, match="Failed to parse"):
            load_config(cfg)

    def test_raises_on_wrong_mcp_servers_type(self, tmp_path: Path) -> None:
        cfg = tmp_path / "bad_type.json"
        cfg.write_text(json.dumps({"mcpServers": "not-a-dict"}), encoding="utf-8")
        with pytest.raises(MCPConfigError, match="Expected 'mcpServers' to be an object"):
            load_config(cfg)

    def test_loads_multiple_servers(self, tmp_path: Path) -> None:
        servers_data = {
            "alpha": {"transport": "stdio", "command": "cmd1"},
            "beta": {"transport": "sse", "url": "http://localhost:3000"},
        }
        cfg = _write_mcp_json(tmp_path / "mcp.json", servers_data)
        servers = load_config(cfg)
        assert len(servers) == 2
        assert servers["alpha"].command == "cmd1"
        assert servers["beta"].url == "http://localhost:3000"

    def test_env_var_substitution_deferred(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Env-var refs are preserved verbatim at load time; resolved at connect time."""
        monkeypatch.setenv("MCP_CMD", "my-tool")
        cfg = tmp_path / "mcp.json"
        data = {"mcpServers": {"s": {"command": "${MCP_CMD}"}}}
        cfg.write_text(json.dumps(data), encoding="utf-8")
        servers = load_config(cfg)
        # Raw ref is preserved — substitution is deferred to _resolve_entry.
        assert servers["s"].command == "${MCP_CMD}"

    def test_empty_mcp_servers_returns_empty(self, tmp_path: Path) -> None:
        cfg = _write_mcp_json(tmp_path / "mcp.json", {})
        servers = load_config(cfg)
        assert servers == {}

    def test_missing_command_env_var_preserved_at_load(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A missing env var in 'command' is preserved verbatim at load time.

        The warn-on-missing check is deferred to _resolve_entry (connection time)
        so the display layer never sees substituted secret values.
        """
        import logging

        monkeypatch.delenv("MISSING_MCP_CMD", raising=False)
        cfg = tmp_path / "mcp.json"
        cfg.write_text(
            '{"mcpServers": {"s": {"transport": "stdio", "command": "${MISSING_MCP_CMD}"}}}',
            encoding="utf-8",
        )
        with caplog.at_level(logging.WARNING, logger="exo_mcp_cli.config"):
            servers = load_config(cfg)
        # Raw ref preserved — no substitution or warning at load time.
        assert servers["s"].command == "${MISSING_MCP_CMD}"
        assert "MISSING_MCP_CMD" not in caplog.text

    def test_missing_url_env_var_preserved_at_load(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A missing env var in 'url' is preserved verbatim at load time."""
        import logging

        monkeypatch.delenv("MISSING_MCP_URL", raising=False)
        cfg = tmp_path / "mcp.json"
        cfg.write_text(
            '{"mcpServers": {"s": {"transport": "sse", "url": "${MISSING_MCP_URL}"}}}',
            encoding="utf-8",
        )
        with caplog.at_level(logging.WARNING, logger="exo_mcp_cli.config"):
            servers = load_config(cfg)
        # Raw ref preserved — no substitution or warning at load time.
        assert servers["s"].url == "${MISSING_MCP_URL}"
        assert "MISSING_MCP_URL" not in caplog.text


# ---------------------------------------------------------------------------
# save_config
# ---------------------------------------------------------------------------


class TestSaveConfig:
    def test_writes_valid_json(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "out.json"
        entry = ServerEntry(name="s1", command="echo", args=["hello"])
        save_config(cfg_path, {"s1": entry})
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        assert "mcpServers" in data
        assert "s1" in data["mcpServers"]
        assert data["mcpServers"]["s1"]["command"] == "echo"

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        deep_path = tmp_path / "a" / "b" / "mcp.json"
        entry = ServerEntry(name="x", command="test")
        save_config(deep_path, {"x": entry})
        assert deep_path.exists()


# ---------------------------------------------------------------------------
# load_or_empty
# ---------------------------------------------------------------------------


class TestLoadOrEmpty:
    def test_returns_empty_for_nonexistent(self, tmp_path: Path) -> None:
        result = load_or_empty(tmp_path / "nope.json")
        assert result == {}

    def test_loads_existing_file(self, tmp_path: Path) -> None:
        cfg = _write_mcp_json(tmp_path / "mcp.json")
        result = load_or_empty(cfg)
        assert "test" in result


# ---------------------------------------------------------------------------
# add_server
# ---------------------------------------------------------------------------


class TestAddServer:
    def test_creates_file_if_not_exists(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "new_mcp.json"
        entry = ServerEntry(name="fresh", command="run")
        add_server(cfg_path, entry)
        assert cfg_path.exists()
        loaded = load_config(cfg_path)
        assert "fresh" in loaded
        assert loaded["fresh"].command == "run"

    def test_updates_existing_entry(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "mcp.json"
        # Add initial entry
        add_server(cfg_path, ServerEntry(name="srv", command="old"))
        # Overwrite with new entry
        add_server(cfg_path, ServerEntry(name="srv", command="new"))
        loaded = load_config(cfg_path)
        assert loaded["srv"].command == "new"

    def test_preserves_other_servers(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "mcp.json"
        add_server(cfg_path, ServerEntry(name="a", command="cmd-a"))
        add_server(cfg_path, ServerEntry(name="b", command="cmd-b"))
        loaded = load_config(cfg_path)
        assert "a" in loaded
        assert "b" in loaded


# ---------------------------------------------------------------------------
# remove_server
# ---------------------------------------------------------------------------


class TestRemoveServer:
    def test_returns_true_for_existing(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "mcp.json"
        add_server(cfg_path, ServerEntry(name="doomed", command="bye"))
        assert remove_server(cfg_path, "doomed") is True

    def test_returns_false_for_missing(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "mcp.json"
        _write_mcp_json(cfg_path)
        assert remove_server(cfg_path, "ghost") is False

    def test_actually_removes(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "mcp.json"
        add_server(cfg_path, ServerEntry(name="rm-me", command="x"))
        remove_server(cfg_path, "rm-me")
        loaded = load_config(cfg_path)
        assert "rm-me" not in loaded

    def test_returns_false_for_nonexistent_file(self, tmp_path: Path) -> None:
        assert remove_server(tmp_path / "missing.json", "anything") is False


# ---------------------------------------------------------------------------
# default_config_path
# ---------------------------------------------------------------------------


class TestDefaultConfigPath:
    def test_returns_cwd_mcp_json(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("exo_mcp_cli.config._cwd", lambda: tmp_path)
        result = default_config_path()
        assert result == tmp_path / "mcp.json"


# ---------------------------------------------------------------------------
# Hints on validate / find_config / load_config / save_config (error DX)
# ---------------------------------------------------------------------------


class TestValidateHints:
    def test_stdio_missing_command_has_hint(self) -> None:
        entry = ServerEntry(name="my-srv", transport="stdio")
        with pytest.raises(MCPConfigError) as exc_info:
            entry.validate()
        assert exc_info.value.hint is not None
        assert "exo-mcp server add" in exc_info.value.hint

    def test_sse_missing_url_has_hint(self) -> None:
        entry = ServerEntry(name="my-srv", transport="sse")
        with pytest.raises(MCPConfigError) as exc_info:
            entry.validate()
        assert exc_info.value.hint is not None
        assert "exo-mcp server add" in exc_info.value.hint


class TestFindConfigHint:
    def test_nonexistent_explicit_path_has_hint(self) -> None:
        with pytest.raises(MCPConfigError) as exc_info:
            find_config("/does/not/exist/mcp.json")
        assert exc_info.value.hint is not None
        assert "--config" in exc_info.value.hint


class TestLoadConfigContext:
    def test_parse_error_carries_config_context(self, tmp_path: Path) -> None:
        cfg = tmp_path / "bad.json"
        cfg.write_text("{not json}", encoding="utf-8")
        with pytest.raises(MCPConfigError) as exc_info:
            load_config(cfg)
        assert exc_info.value.context is not None
        assert "config" in exc_info.value.context
        assert str(cfg) in str(exc_info.value.context["config"])


class TestSaveConfigOSError:
    def test_permission_error_raises_mcp_config_error(self, tmp_path: Path) -> None:
        """An OSError on write is re-raised as MCPConfigError with context."""
        from unittest.mock import patch

        cfg_path = tmp_path / "out.json"
        entry = ServerEntry(name="s1", command="echo")
        with (
            patch("pathlib.Path.write_text", side_effect=OSError("disk full")),
            pytest.raises(MCPConfigError) as exc_info,
        ):
            save_config(cfg_path, {"s1": entry})
        assert "config" in (exc_info.value.context or {})
        assert exc_info.value.hint is not None
