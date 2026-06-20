"""Encrypted credential vault for MCP server secrets.

Stores secrets in a Fernet-encrypted file at ``~/.exo-mcp/credentials.vault``.
The encryption key is derived from a passphrase via PBKDF2-HMAC-SHA256
(480,000 iterations).

Passphrase resolution order:
    1. ``EXO_MCP_VAULT_KEY`` environment variable
    2. Interactive prompt via ``getpass.getpass()``

Vault file layout: ``<16-byte salt><Fernet ciphertext>``
"""

from __future__ import annotations

import base64
import getpass
import hashlib
import json
import os
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from exo.types import ExoError  # pyright: ignore[reportMissingImports]

_PBKDF2_ITERATIONS = 480_000
_SALT_LEN = 16
_DEFAULT_VAULT_DIR = Path.home() / ".exo-mcp"
_DEFAULT_VAULT_PATH = _DEFAULT_VAULT_DIR / "credentials.vault"
_ENV_KEY = "EXO_MCP_VAULT_KEY"


class VaultError(ExoError):
    """Raised on vault operation failures."""


def derive_key(passphrase: str, salt: bytes) -> bytes:
    """Derive a 32-byte Fernet key from *passphrase* and *salt*.

    Uses PBKDF2-HMAC-SHA256 with 480k iterations.

    Returns:
        A url-safe base64-encoded 32-byte key suitable for ``Fernet``.
    """
    raw = hashlib.pbkdf2_hmac("sha256", passphrase.encode(), salt, _PBKDF2_ITERATIONS)
    return base64.urlsafe_b64encode(raw)


class Vault:
    """Encrypted credential store backed by a local file.

    Args:
        vault_path: Path to the vault file. Defaults to
            ``~/.exo-mcp/credentials.vault``.
    """

    __slots__ = ("_cache", "_fernet", "_passphrase", "_path", "_salt")

    def __init__(self, vault_path: Path | None = None) -> None:
        self._path = vault_path or _DEFAULT_VAULT_PATH
        self._passphrase: str | None = None
        self._fernet: Fernet | None = None
        self._salt: bytes | None = None
        self._cache: dict[str, str] | None = None

    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------
    # Passphrase & key management
    # ------------------------------------------------------------------

    def _get_passphrase(self) -> str:
        """Resolve the vault passphrase (env var → interactive prompt)."""
        if self._passphrase is not None:
            return self._passphrase
        env = os.environ.get(_ENV_KEY)
        if env:
            self._passphrase = env
            return env
        try:
            pwd = getpass.getpass("Vault passphrase: ")
        except (EOFError, KeyboardInterrupt) as exc:
            raise VaultError("Vault passphrase required") from exc
        if not pwd:
            raise VaultError("Vault passphrase cannot be empty")
        self._passphrase = pwd
        return pwd

    def _get_fernet(self, salt: bytes) -> Fernet:
        """Get or create a Fernet instance for the given salt."""
        if self._fernet is not None and self._salt == salt:
            return self._fernet
        key = derive_key(self._get_passphrase(), salt)
        self._fernet = Fernet(key)
        self._salt = salt
        return self._fernet

    # ------------------------------------------------------------------
    # Load / save
    # ------------------------------------------------------------------

    def _load(self) -> dict[str, str]:
        """Decrypt and parse the vault file. Returns empty dict if no file."""
        if self._cache is not None:
            return self._cache
        if not self._path.exists():
            self._cache = {}
            return self._cache
        raw = self._path.read_bytes()
        if len(raw) < _SALT_LEN + 1:
            raise VaultError(f"Vault file is corrupted: {self._path}")
        salt = raw[:_SALT_LEN]
        ciphertext = raw[_SALT_LEN:]
        fernet = self._get_fernet(salt)
        try:
            plaintext = fernet.decrypt(ciphertext)
        except InvalidToken as exc:
            raise VaultError("Wrong passphrase or corrupted vault") from exc
        try:
            data = json.loads(plaintext)
        except json.JSONDecodeError as exc:
            raise VaultError(f"Vault contents are not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise VaultError("Vault contents must be a JSON object")
        self._cache = data
        return self._cache

    def _save(self, data: dict[str, str]) -> None:
        """Encrypt and write *data* to the vault file."""
        if self._salt is None:
            self._salt = os.urandom(_SALT_LEN)
        fernet = self._get_fernet(self._salt)
        plaintext = json.dumps(data, sort_keys=True).encode()
        ciphertext = fernet.encrypt(plaintext)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_bytes(self._salt + ciphertext)
        self._cache = data

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, name: str) -> str | None:
        """Get a secret by name, or ``None`` if not found."""
        return self._load().get(name)

    def set(self, name: str, value: str) -> None:
        """Store or update a secret."""
        data = self._load()
        data[name] = value
        self._save(data)

    def remove(self, name: str) -> bool:
        """Remove a secret. Returns ``True`` if it existed."""
        data = self._load()
        if name not in data:
            return False
        del data[name]
        self._save(data)
        return True

    def list_names(self) -> list[str]:
        """List all stored secret names (not values)."""
        return sorted(self._load().keys())

    def has(self, name: str) -> bool:
        """Check if a secret exists."""
        return name in self._load()

    def resolve(self, value: str) -> str:
        """Resolve ``${vault:NAME}`` references in a string.

        Unresolved references (missing keys) are left unchanged.
        """
        import re

        def _replace(m: re.Match[str]) -> str:
            secret = self.get(m.group(1))
            return secret if secret is not None else m.group(0)

        return re.sub(r"\$\{vault:([^}]+)\}", _replace, value)
