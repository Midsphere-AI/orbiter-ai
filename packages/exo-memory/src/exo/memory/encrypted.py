"""AES-256-GCM encrypted memory store wrapper.

Wraps any ``MemoryStore`` implementation to transparently encrypt/decrypt
the ``content`` field of memory items at rest.

Requires the ``cryptography`` package::

    pip install exo-memory[encryption]
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import (
    AESGCM,  # pyright: ignore[reportMissingImports]
)

from exo.memory.base import (  # pyright: ignore[reportMissingImports]
    ExoMemoryError,
    MemoryCategory,
    MemoryItem,
    MemoryMetadata,
    MemoryStatus,
)

logger = logging.getLogger(__name__)


def derive_key(password: str, salt: bytes | None = None) -> tuple[bytes, bytes]:
    """Derive a 32-byte AES-256 key from a password using PBKDF2-HMAC-SHA256.

    Args:
        password: The password to derive from.
        salt: Optional 16-byte salt. A random one is generated if not provided.

    Returns:
        Tuple of (key, salt) — persist the salt alongside encrypted data.
    """
    if salt is None:
        salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations=480_000)
    return key, salt


class EncryptedMemoryStore:
    """Wrapper that encrypts ``MemoryItem.content`` with AES-256-GCM.

    Delegates all storage operations to an inner ``MemoryStore`` while
    transparently encrypting content on write and decrypting on read.

    Args:
        store: Any object implementing the ``MemoryStore`` protocol.
        key: A 32-byte AES-256 key. Use :func:`derive_key` to create one.
    """

    __slots__ = ("_cipher", "_store")

    def __init__(self, store: Any, key: bytes) -> None:
        if len(key) != 32:
            raise ExoMemoryError(
                f"AES key must be exactly 32 bytes (got {len(key)}).",
                hint="Use derive_key(password) to generate a valid 32-byte key.",
            )
        self._store = store
        self._cipher = AESGCM(key)

    # ------------------------------------------------------------------
    # Encryption helpers
    # ------------------------------------------------------------------

    def _encrypt(self, plaintext: str) -> str:
        """Encrypt plaintext to a hex-encoded nonce+ciphertext string."""
        nonce = os.urandom(12)  # 96-bit nonce for GCM
        ct = self._cipher.encrypt(nonce, plaintext.encode(), None)
        # Store as hex: 24-char nonce + variable ciphertext (includes 16-byte tag)
        return (nonce + ct).hex()

    def _decrypt(self, token: str) -> str:
        """Decrypt a hex-encoded nonce+ciphertext string back to plaintext."""
        raw = bytes.fromhex(token)
        nonce, ct = raw[:12], raw[12:]
        return self._cipher.decrypt(nonce, ct, None).decode()

    # ------------------------------------------------------------------
    # MemoryStore protocol
    # ------------------------------------------------------------------

    async def add(self, item: MemoryItem) -> None:
        """Encrypt item content, then delegate to the inner store."""
        encrypted_item = item.model_copy(update={"content": self._encrypt(item.content)})
        await self._store.add(encrypted_item)

    async def get(self, item_id: str) -> MemoryItem | None:
        """Retrieve and decrypt a memory item by ID."""
        item = await self._store.get(item_id)
        if item is None:
            return None
        try:
            decrypted = self._decrypt(item.content)
        except Exception as exc:
            raise ExoMemoryError(
                f"Failed to decrypt memory item {item_id!r}.",
                context={"item_id": item_id},
                hint=(
                    "Verify the AES key is the same one used when storing this item. "
                    "Data may also be corrupted."
                ),
            ) from exc
        return item.model_copy(update={"content": decrypted})

    async def search(
        self,
        *,
        query: str = "",
        metadata: MemoryMetadata | None = None,
        memory_type: str | None = None,
        category: MemoryCategory | None = None,
        status: MemoryStatus | None = None,
        limit: int = 10,
    ) -> list[MemoryItem]:
        """Search and decrypt results.

        Raises:
            NotImplementedError: If *query* is non-empty.  Keyword search on
                encrypted content is silently broken (the inner store sees
                ciphertext, not plaintext), so we fail loudly instead.
                Metadata, memory_type, category, and status filters still work.
        """
        if query:
            raise ExoMemoryError(
                "EncryptedMemoryStore does not support keyword/semantic search: "
                "the inner store indexes ciphertext, not plaintext.",
                hint="Pass query='' and use metadata/type/status filters instead.",
            )
        items = await self._store.search(
            query="",
            metadata=metadata,
            memory_type=memory_type,
            category=category,
            status=status,
            limit=limit,
        )
        result: list[MemoryItem] = []
        for item in items:
            try:
                decrypted = self._decrypt(item.content)
            except Exception as exc:
                # Log the failure so it is visible — silent drops mask key mismatches
                # and data corruption. We skip the item rather than raising so a
                # partial result is returned (caller can inspect the log).
                logger.warning(
                    "EncryptedMemoryStore: failed to decrypt item id=%r; skipping. "
                    "Check that the AES key matches the one used at write time. error=%s",
                    item.id,
                    exc,
                )
                continue
            result.append(item.model_copy(update={"content": decrypted}))
        return result

    async def clear(
        self,
        *,
        metadata: MemoryMetadata | None = None,
    ) -> int:
        """Delegate clear to the inner store."""
        return await self._store.clear(metadata=metadata)
