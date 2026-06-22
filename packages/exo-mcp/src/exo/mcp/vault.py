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
import contextlib
import getpass
import hashlib
import json
import os
import tempfile
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
            raise VaultError(
                "Vault passphrase cannot be empty",
                hint=(
                    "Set EXO_MCP_VAULT_KEY env var to a non-empty passphrase, "
                    "or enter a non-empty passphrase at the prompt."
                ),
            )
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
            raise VaultError(
                f"Vault file is corrupted: {self._path}",
                hint=(
                    f"Delete {self._path} and re-add secrets with vault.set(), "
                    "or restore from a backup."
                ),
            )
        salt = raw[:_SALT_LEN]
        ciphertext = raw[_SALT_LEN:]
        fernet = self._get_fernet(salt)
        try:
            plaintext = fernet.decrypt(ciphertext)
        except InvalidToken as exc:
            raise VaultError(
                "Wrong passphrase or corrupted vault",
                hint=(
                    "Re-run with the correct passphrase (EXO_MCP_VAULT_KEY env var) "
                    f"or delete {self._path} to start over."
                ),
            ) from exc
        try:
            data = json.loads(plaintext)
        except json.JSONDecodeError as exc:
            raise VaultError(f"Vault contents are not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise VaultError("Vault contents must be a JSON object")
        self._cache = data
        return self._cache

    def _save(self, data: dict[str, str], *, new_passphrase: bool = False) -> None:
        """Encrypt and write *data* to the vault file.

        A fresh salt is generated on the very first write and whenever
        ``new_passphrase=True`` is passed (e.g. after a passphrase rotation),
        so that changing the passphrase always re-encrypts under a new salt.
        On routine incremental writes the existing salt is reused to avoid a
        redundant 480k-iter PBKDF2 re-derivation.

        The write is atomic: data is written to a temp file (mode 0o600) in the
        same directory and then renamed into place so a partial write can never
        corrupt the live vault file.  The vault directory is created with mode
        0o700 (owner-only) so no other user can list or read files inside it.
        """
        # Regenerate salt on first-ever write or explicit passphrase change.
        if self._salt is None or new_passphrase:
            self._salt = os.urandom(_SALT_LEN)
        fernet = self._get_fernet(self._salt)
        plaintext = json.dumps(data, sort_keys=True).encode()
        ciphertext = fernet.encrypt(plaintext)

        parent = self._path.parent
        # Create vault directory with owner-only permissions (0o700).
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        # Tighten permissions on the directory in case it already existed with
        # wider permissions (e.g. user created it manually). Best-effort.
        with contextlib.suppress(OSError):
            os.chmod(parent, 0o700)

        payload = self._salt + ciphertext
        # Write atomically: mkstemp gives us an O_CREAT|O_EXCL fd with mode
        # 0o600 (owner read/write only), which prevents other users from ever
        # reading the plaintext during the write window.
        tmp_name: str | None = None
        try:
            fd, tmp_name = tempfile.mkstemp(dir=parent, prefix=".vault_")
            try:
                os.write(fd, payload)
            finally:
                os.close(fd)
            # mkstemp already creates with 0o600 on POSIX, but be explicit.
            os.chmod(tmp_name, 0o600)
            os.replace(tmp_name, self._path)
            tmp_name = None  # replaced successfully, no cleanup needed
        except OSError as exc:
            raise VaultError(f"Failed to write vault to {self._path}: {exc}") from exc
        finally:
            # If replace failed, remove the orphaned temp file.
            if tmp_name is not None:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_name)

        # Tighten the final vault file permissions in case an old vault existed
        # with wider permissions before this write. Best-effort.
        with contextlib.suppress(OSError):
            os.chmod(self._path, 0o600)

        # Only update the in-memory cache after the write succeeds so a failure
        # never leaves the cache diverged from disk.
        self._cache = data

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, name: str) -> str | None:
        """Get a secret by name, or ``None`` if not found."""
        return self._load().get(name)

    def set(self, name: str, value: str) -> None:
        """Store or update a secret."""
        # Build a new dict so that cache is only updated after a successful save.
        new_data = {**self._load(), name: value}
        self._save(new_data)

    def remove(self, name: str) -> bool:
        """Remove a secret. Returns ``True`` if it existed."""
        existing = self._load()
        if name not in existing:
            return False
        new_data = {k: v for k, v in existing.items() if k != name}
        self._save(new_data)
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
