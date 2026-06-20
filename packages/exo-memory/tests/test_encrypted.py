"""Tests for AES-256-GCM encrypted memory store wrapper."""

from __future__ import annotations

import os

import pytest

from exo.memory.base import (  # pyright: ignore[reportMissingImports]
    ExoMemoryError,
    MemoryCategory,
    MemoryItem,
    MemoryMetadata,
    MemoryStatus,
    MemoryStore,
)
from exo.memory.encrypted import (  # pyright: ignore[reportMissingImports]
    EncryptedMemoryStore,
    derive_key,
)

# ---------------------------------------------------------------------------
# Minimal in-memory store for testing
# ---------------------------------------------------------------------------


class _InMemoryStore:
    """Minimal MemoryStore for testing the encryption wrapper."""

    def __init__(self) -> None:
        self._items: dict[str, MemoryItem] = {}

    async def add(self, item: MemoryItem) -> None:
        self._items[item.id] = item

    async def get(self, item_id: str) -> MemoryItem | None:
        return self._items.get(item_id)

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
        items = list(self._items.values())
        if memory_type is not None:
            items = [i for i in items if i.memory_type == memory_type]
        if status is not None:
            items = [i for i in items if i.status == status]
        return items[:limit]

    async def clear(self, *, metadata: MemoryMetadata | None = None) -> int:
        count = len(self._items)
        self._items.clear()
        return count


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def aes_key() -> bytes:
    return os.urandom(32)


@pytest.fixture()
def inner_store() -> _InMemoryStore:
    return _InMemoryStore()


@pytest.fixture()
def enc_store(inner_store: _InMemoryStore, aes_key: bytes) -> EncryptedMemoryStore:
    return EncryptedMemoryStore(inner_store, aes_key)


# ---------------------------------------------------------------------------
# derive_key tests
# ---------------------------------------------------------------------------


class TestDeriveKey:
    def test_returns_32_byte_key(self) -> None:
        key, salt = derive_key("my-password")
        assert len(key) == 32
        assert len(salt) == 16

    def test_deterministic_with_same_salt(self) -> None:
        salt = os.urandom(16)
        key1, _ = derive_key("password", salt)
        key2, _ = derive_key("password", salt)
        assert key1 == key2

    def test_different_passwords_different_keys(self) -> None:
        salt = os.urandom(16)
        key1, _ = derive_key("password-a", salt)
        key2, _ = derive_key("password-b", salt)
        assert key1 != key2

    def test_different_salts_different_keys(self) -> None:
        key1, salt1 = derive_key("password")
        key2, salt2 = derive_key("password")
        # Random salts should differ
        assert salt1 != salt2
        assert key1 != key2


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_rejects_short_key(self, inner_store: _InMemoryStore) -> None:
        with pytest.raises(ValueError, match="32 bytes"):
            EncryptedMemoryStore(inner_store, b"too-short")

    def test_rejects_long_key(self, inner_store: _InMemoryStore) -> None:
        with pytest.raises(ValueError, match="32 bytes"):
            EncryptedMemoryStore(inner_store, os.urandom(64))

    def test_accepts_valid_key(self, inner_store: _InMemoryStore) -> None:
        store = EncryptedMemoryStore(inner_store, os.urandom(32))
        assert store is not None


# ---------------------------------------------------------------------------
# Encrypt -> store -> retrieve -> decrypt roundtrip
# ---------------------------------------------------------------------------


class TestRoundtrip:
    @pytest.mark.anyio()
    async def test_add_get_roundtrip(self, enc_store: EncryptedMemoryStore) -> None:
        item = MemoryItem(id="item-1", content="secret data", memory_type="human")
        await enc_store.add(item)
        retrieved = await enc_store.get("item-1")
        assert retrieved is not None
        assert retrieved.content == "secret data"
        assert retrieved.id == "item-1"
        assert retrieved.memory_type == "human"

    @pytest.mark.anyio()
    async def test_content_encrypted_at_rest(
        self, enc_store: EncryptedMemoryStore, inner_store: _InMemoryStore
    ) -> None:
        item = MemoryItem(id="item-2", content="plaintext secret", memory_type="system")
        await enc_store.add(item)
        # The inner store should have encrypted (hex) content, not plaintext
        raw = await inner_store.get("item-2")
        assert raw is not None
        assert raw.content != "plaintext secret"
        # It should be valid hex
        bytes.fromhex(raw.content)

    @pytest.mark.anyio()
    async def test_search_returns_decrypted(self, enc_store: EncryptedMemoryStore) -> None:
        item = MemoryItem(id="s-1", content="searchable secret", memory_type="human")
        await enc_store.add(item)
        results = await enc_store.search(memory_type="human")
        assert len(results) == 1
        assert results[0].content == "searchable secret"

    @pytest.mark.anyio()
    async def test_metadata_preserved(self, enc_store: EncryptedMemoryStore) -> None:
        meta = MemoryMetadata(user_id="u1", session_id="s1")
        item = MemoryItem(
            id="m-1",
            content="meta test",
            memory_type="ai",
            category=MemoryCategory.SEMANTIC,
            metadata=meta,
        )
        await enc_store.add(item)
        retrieved = await enc_store.get("m-1")
        assert retrieved is not None
        assert retrieved.metadata.user_id == "u1"
        assert retrieved.category == MemoryCategory.SEMANTIC

    @pytest.mark.anyio()
    async def test_clear_delegates(self, enc_store: EncryptedMemoryStore) -> None:
        item = MemoryItem(id="c-1", content="to clear", memory_type="human")
        await enc_store.add(item)
        count = await enc_store.clear()
        assert count == 1

    @pytest.mark.anyio()
    async def test_get_nonexistent_returns_none(self, enc_store: EncryptedMemoryStore) -> None:
        result = await enc_store.get("does-not-exist")
        assert result is None


# ---------------------------------------------------------------------------
# Wrong key tests
# ---------------------------------------------------------------------------


class TestWrongKey:
    @pytest.mark.anyio()
    async def test_wrong_key_get_raises(self, inner_store: _InMemoryStore) -> None:
        key_a = os.urandom(32)
        key_b = os.urandom(32)
        store_a = EncryptedMemoryStore(inner_store, key_a)
        store_b = EncryptedMemoryStore(inner_store, key_b)

        item = MemoryItem(id="wk-1", content="encrypted with key A", memory_type="human")
        await store_a.add(item)

        with pytest.raises(ExoMemoryError, match="Failed to decrypt"):
            await store_b.get("wk-1")

    @pytest.mark.anyio()
    async def test_wrong_key_search_skips(self, inner_store: _InMemoryStore) -> None:
        key_a = os.urandom(32)
        key_b = os.urandom(32)
        store_a = EncryptedMemoryStore(inner_store, key_a)
        store_b = EncryptedMemoryStore(inner_store, key_b)

        item = MemoryItem(id="wk-2", content="only key A can read", memory_type="human")
        await store_a.add(item)

        # search with wrong key silently skips undecryptable items
        results = await store_b.search(memory_type="human")
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_conforms_to_memory_store(self, enc_store: EncryptedMemoryStore) -> None:
        assert isinstance(enc_store, MemoryStore)
