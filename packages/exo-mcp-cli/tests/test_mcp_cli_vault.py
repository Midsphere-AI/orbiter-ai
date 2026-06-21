"""Tests for exo_mcp_cli.vault — encrypted credential vault."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from exo_mcp_cli.vault import Vault, VaultError, derive_key

# ---------------------------------------------------------------------------
# derive_key
# ---------------------------------------------------------------------------


class TestDeriveKey:
    def test_consistent_output(self) -> None:
        salt = b"0123456789abcdef"
        k1 = derive_key("my-pass", salt)
        k2 = derive_key("my-pass", salt)
        assert k1 == k2

    def test_different_salts(self) -> None:
        k1 = derive_key("same-pass", b"salt_aaaaaaaaaaaa")
        k2 = derive_key("same-pass", b"salt_bbbbbbbbbbbb")
        assert k1 != k2


# ---------------------------------------------------------------------------
# Vault: basic get / set / has / remove
# ---------------------------------------------------------------------------


class TestVaultBasicOps:
    def test_set_and_get_roundtrip(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXO_MCP_VAULT_KEY", "test-passphrase")
        vault = Vault(vault_path=tmp_path / "test.vault")
        vault.set("api_key", "sk-12345")
        assert vault.get("api_key") == "sk-12345"

    def test_get_missing_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXO_MCP_VAULT_KEY", "test-passphrase")
        vault = Vault(vault_path=tmp_path / "test.vault")
        assert vault.get("nonexistent") is None

    def test_has_returns_true_for_existing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXO_MCP_VAULT_KEY", "test-passphrase")
        vault = Vault(vault_path=tmp_path / "test.vault")
        vault.set("secret", "value")
        assert vault.has("secret") is True

    def test_has_returns_false_for_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXO_MCP_VAULT_KEY", "test-passphrase")
        vault = Vault(vault_path=tmp_path / "test.vault")
        assert vault.has("no-such-key") is False

    def test_remove_existing_returns_true(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXO_MCP_VAULT_KEY", "test-passphrase")
        vault = Vault(vault_path=tmp_path / "test.vault")
        vault.set("doomed", "bye")
        assert vault.remove("doomed") is True

    def test_remove_missing_returns_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXO_MCP_VAULT_KEY", "test-passphrase")
        vault = Vault(vault_path=tmp_path / "test.vault")
        assert vault.remove("ghost") is False

    def test_remove_actually_removes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXO_MCP_VAULT_KEY", "test-passphrase")
        vault = Vault(vault_path=tmp_path / "test.vault")
        vault.set("temp", "data")
        vault.remove("temp")
        assert vault.get("temp") is None
        assert vault.has("temp") is False


# ---------------------------------------------------------------------------
# Vault: list_names
# ---------------------------------------------------------------------------


class TestVaultListNames:
    def test_list_names_sorted(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXO_MCP_VAULT_KEY", "test-passphrase")
        vault = Vault(vault_path=tmp_path / "test.vault")
        vault.set("z-key", "1")
        vault.set("a-key", "2")
        vault.set("m-key", "3")
        assert vault.list_names() == ["a-key", "m-key", "z-key"]

    def test_empty_vault_list_names(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXO_MCP_VAULT_KEY", "test-passphrase")
        vault = Vault(vault_path=tmp_path / "test.vault")
        assert vault.list_names() == []


# ---------------------------------------------------------------------------
# Vault: parent directory creation
# ---------------------------------------------------------------------------


class TestVaultDirectoryCreation:
    def test_creates_parent_directories(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXO_MCP_VAULT_KEY", "test-passphrase")
        deep_path = tmp_path / "a" / "b" / "c" / "test.vault"
        vault = Vault(vault_path=deep_path)
        vault.set("key", "value")
        assert deep_path.exists()
        assert vault.get("key") == "value"


# ---------------------------------------------------------------------------
# Vault: wrong passphrase
# ---------------------------------------------------------------------------


class TestVaultWrongPassphrase:
    def test_wrong_passphrase_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        vault_file = tmp_path / "test.vault"
        # Write with one passphrase
        monkeypatch.setenv("EXO_MCP_VAULT_KEY", "correct-passphrase")
        v1 = Vault(vault_path=vault_file)
        v1.set("secret", "value")
        # Read with a different passphrase
        monkeypatch.setenv("EXO_MCP_VAULT_KEY", "wrong-passphrase")
        v2 = Vault(vault_path=vault_file)
        with pytest.raises(VaultError, match="Wrong passphrase"):
            v2.get("secret")


# ---------------------------------------------------------------------------
# Vault: persistence across instances
# ---------------------------------------------------------------------------


class TestVaultPersistence:
    def test_multiple_operations_persist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXO_MCP_VAULT_KEY", "test-passphrase")
        vault = Vault(vault_path=tmp_path / "test.vault")
        vault.set("first", "aaa")
        vault.set("second", "bbb")
        vault.set("third", "ccc")
        assert vault.get("first") == "aaa"
        assert vault.get("second") == "bbb"
        assert vault.get("third") == "ccc"
        assert vault.list_names() == ["first", "second", "third"]

    def test_persists_across_instances(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXO_MCP_VAULT_KEY", "test-passphrase")
        vault_file = tmp_path / "test.vault"
        v1 = Vault(vault_path=vault_file)
        v1.set("alpha", "one")
        v1.set("beta", "two")
        # New instance, same file and passphrase
        v2 = Vault(vault_path=vault_file)
        assert v2.get("alpha") == "one"
        assert v2.get("beta") == "two"


# ---------------------------------------------------------------------------
# Vault: env var passphrase
# ---------------------------------------------------------------------------


class TestVaultEnvKey:
    def test_loads_from_env_var(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXO_MCP_VAULT_KEY", "env-passphrase")
        vault = Vault(vault_path=tmp_path / "test.vault")
        vault.set("from_env", "works")
        assert vault.get("from_env") == "works"

    def test_empty_passphrase_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        vault_file = tmp_path / "test.vault"
        # First create a vault file so that _load needs to decrypt
        monkeypatch.setenv("EXO_MCP_VAULT_KEY", "initial-pass")
        v1 = Vault(vault_path=vault_file)
        v1.set("key", "value")
        # Now try to read with no env var and getpass returning empty string
        monkeypatch.delenv("EXO_MCP_VAULT_KEY", raising=False)
        v2 = Vault(vault_path=vault_file)
        with (
            patch("exo.mcp.vault.getpass.getpass", return_value=""),
            pytest.raises(VaultError, match="cannot be empty"),
        ):
            v2.get("key")


# ---------------------------------------------------------------------------
# Vault: resolve
# ---------------------------------------------------------------------------


class TestVaultResolve:
    def test_resolve_replaces_vault_references(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXO_MCP_VAULT_KEY", "test-passphrase")
        vault = Vault(vault_path=tmp_path / "test.vault")
        vault.set("API_KEY", "sk-abc123")
        result = vault.resolve("Bearer ${vault:API_KEY}")
        assert result == "Bearer sk-abc123"

    def test_resolve_leaves_unknown_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXO_MCP_VAULT_KEY", "test-passphrase")
        vault = Vault(vault_path=tmp_path / "test.vault")
        result = vault.resolve("token=${vault:MISSING}")
        assert result == "token=${vault:MISSING}"

    def test_resolve_multiple_references(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXO_MCP_VAULT_KEY", "test-passphrase")
        vault = Vault(vault_path=tmp_path / "test.vault")
        vault.set("USER", "admin")
        vault.set("PASS", "secret")
        result = vault.resolve("${vault:USER}:${vault:PASS}")
        assert result == "admin:secret"

    def test_resolve_no_references(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXO_MCP_VAULT_KEY", "test-passphrase")
        vault = Vault(vault_path=tmp_path / "test.vault")
        result = vault.resolve("plain text no refs")
        assert result == "plain text no refs"
