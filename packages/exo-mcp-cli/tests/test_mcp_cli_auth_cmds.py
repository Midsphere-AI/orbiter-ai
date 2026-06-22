"""Tests for exo_mcp_cli.commands.auth — vault secret management CLI commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from exo_mcp_cli.main import app
from exo_mcp_cli.vault import Vault

runner = CliRunner()


# ---------------------------------------------------------------------------
# auth set
# ---------------------------------------------------------------------------


class TestAuthSet:
    def test_set_stores_secret(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXO_MCP_VAULT_KEY", "test-pass")
        test_vault = Vault(vault_path=tmp_path / "v.enc")

        with patch("exo_mcp_cli.main.Vault", return_value=test_vault):
            result = runner.invoke(app, ["auth", "set", "my-key", "secret-val"])

        assert result.exit_code == 0
        assert "my-key" in result.output
        assert "stored" in result.output.lower()
        assert test_vault.get("my-key") == "secret-val"

    def test_set_shows_vault_reference_hint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After setting a secret, the output hints how to reference it."""
        monkeypatch.setenv("EXO_MCP_VAULT_KEY", "test-pass")
        test_vault = Vault(vault_path=tmp_path / "v.enc")

        with patch("exo_mcp_cli.main.Vault", return_value=test_vault):
            result = runner.invoke(app, ["auth", "set", "api-token", "tk-abc"])

        assert result.exit_code == 0
        assert "${vault:api-token}" in result.output

    def test_set_overwrites_existing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXO_MCP_VAULT_KEY", "test-pass")
        test_vault = Vault(vault_path=tmp_path / "v.enc")
        test_vault.set("existing", "old-value")

        with patch("exo_mcp_cli.main.Vault", return_value=test_vault):
            result = runner.invoke(app, ["auth", "set", "existing", "new-value"])

        assert result.exit_code == 0
        assert test_vault.get("existing") == "new-value"

    def test_set_stdin_flag_reads_from_stdin(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--stdin reads the secret from standard input rather than a positional arg."""
        monkeypatch.setenv("EXO_MCP_VAULT_KEY", "test-pass")
        test_vault = Vault(vault_path=tmp_path / "v.enc")

        with patch("exo_mcp_cli.main.Vault", return_value=test_vault):
            result = runner.invoke(
                app,
                ["auth", "set", "piped-key", "--stdin"],
                input="piped-secret\n",
            )

        assert result.exit_code == 0, result.output
        assert "piped-key" in result.output
        assert test_vault.get("piped-key") == "piped-secret"

    def test_set_interactive_prompt_when_no_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Omitting the value arg triggers an interactive hidden prompt."""
        monkeypatch.setenv("EXO_MCP_VAULT_KEY", "test-pass")
        test_vault = Vault(vault_path=tmp_path / "v.enc")

        with patch("exo_mcp_cli.main.Vault", return_value=test_vault):
            # CliRunner.invoke with `input` simulates the interactive prompts:
            # first the hidden prompt, then the confirmation prompt.
            result = runner.invoke(
                app,
                ["auth", "set", "prompt-key"],
                input="prompt-secret\nprompt-secret\n",
            )

        assert result.exit_code == 0, result.output
        assert test_vault.get("prompt-key") == "prompt-secret"

    def test_set_value_arg_not_echoed_to_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The secret value must not appear in the command output regardless of input method."""
        monkeypatch.setenv("EXO_MCP_VAULT_KEY", "test-pass")
        test_vault = Vault(vault_path=tmp_path / "v.enc")

        with patch("exo_mcp_cli.main.Vault", return_value=test_vault):
            result = runner.invoke(app, ["auth", "set", "safe-key", "super-secret"])

        assert result.exit_code == 0
        assert "super-secret" not in result.output


# ---------------------------------------------------------------------------
# auth list
# ---------------------------------------------------------------------------


class TestAuthList:
    def test_list_shows_secret_names(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXO_MCP_VAULT_KEY", "test-pass")
        test_vault = Vault(vault_path=tmp_path / "v.enc")
        test_vault.set("alpha-key", "val1")
        test_vault.set("beta-key", "val2")

        with patch("exo_mcp_cli.main.Vault", return_value=test_vault):
            result = runner.invoke(app, ["auth", "list"])

        assert result.exit_code == 0
        assert "alpha-key" in result.output
        assert "beta-key" in result.output
        # Secret values must NOT appear in list output
        assert "val1" not in result.output
        assert "val2" not in result.output

    def test_list_empty_vault_shows_message(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXO_MCP_VAULT_KEY", "test-pass")
        test_vault = Vault(vault_path=tmp_path / "v.enc")

        with patch("exo_mcp_cli.main.Vault", return_value=test_vault):
            result = runner.invoke(app, ["auth", "list"])

        assert result.exit_code == 0
        assert "No secrets" in result.output

    def test_list_shows_vault_references(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The list output should show the ${vault:NAME} reference syntax."""
        monkeypatch.setenv("EXO_MCP_VAULT_KEY", "test-pass")
        test_vault = Vault(vault_path=tmp_path / "v.enc")
        test_vault.set("db-pass", "pg-secret")

        with patch("exo_mcp_cli.main.Vault", return_value=test_vault):
            result = runner.invoke(app, ["auth", "list"])

        assert result.exit_code == 0
        assert "${vault:db-pass}" in result.output


# ---------------------------------------------------------------------------
# auth remove
# ---------------------------------------------------------------------------


class TestAuthRemove:
    def test_remove_existing_secret(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXO_MCP_VAULT_KEY", "test-pass")
        test_vault = Vault(vault_path=tmp_path / "v.enc")
        test_vault.set("doomed", "byebye")

        with patch("exo_mcp_cli.main.Vault", return_value=test_vault):
            result = runner.invoke(app, ["auth", "remove", "doomed"])

        assert result.exit_code == 0
        assert "doomed" in result.output
        assert "removed" in result.output.lower()
        assert test_vault.get("doomed") is None

    def test_remove_nonexistent_exits_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXO_MCP_VAULT_KEY", "test-pass")
        test_vault = Vault(vault_path=tmp_path / "v.enc")

        with patch("exo_mcp_cli.main.Vault", return_value=test_vault):
            result = runner.invoke(app, ["auth", "remove", "ghost"])

        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_remove_does_not_affect_others(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Removing one secret leaves other secrets intact."""
        monkeypatch.setenv("EXO_MCP_VAULT_KEY", "test-pass")
        test_vault = Vault(vault_path=tmp_path / "v.enc")
        test_vault.set("keep-me", "safe")
        test_vault.set("remove-me", "gone")

        with patch("exo_mcp_cli.main.Vault", return_value=test_vault):
            result = runner.invoke(app, ["auth", "remove", "remove-me"])

        assert result.exit_code == 0
        assert test_vault.get("keep-me") == "safe"
        assert test_vault.get("remove-me") is None
