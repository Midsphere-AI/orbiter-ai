"""Temporal payload codecs for the Exo distributed integration.

Provides:

- :class:`AESGCMPayloadCodec` — AES-256-GCM encryption at rest.
- :class:`ChainedPayloadCodec` — compose multiple codecs in sequence.
- :class:`BlobStore` — protocol for large-payload (claim-check) storage.
- :class:`InMemoryBlobStore` — dict-backed blob store (tests / single-process).
- :class:`RedisBlobStore` — Redis-backed blob store with optional TTL.
- :class:`ClaimCheckCodec` — offloads oversized payloads to a :class:`BlobStore`.

All codecs implement the :class:`temporalio.converter.PayloadCodec` interface
(async ``encode`` / ``decode`` methods that accept and return
``Sequence[Payload]`` / ``list[Payload]``).
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from exo.distributed.temporal._compat import (  # pyright: ignore[reportMissingImports]
    Payload,
    PayloadCodec,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AESGCMPayloadCodec",
    "BlobStore",
    "ChainedPayloadCodec",
    "ClaimCheckCodec",
    "InMemoryBlobStore",
    "RedisBlobStore",
]

# Encoding tag constants
_ENC_ENCRYPTED = b"binary/encrypted"
_ENC_CLAIM_CHECK = b"binary/claim-check"

# AES-GCM nonce size
_NONCE_SIZE = 12


class AESGCMPayloadCodec(PayloadCodec):
    """AES-256-GCM payload codec for encrypting Temporal payloads at rest.

    Args:
        key: Exactly 32 bytes used as the AES-256 symmetric key.

    Raises:
        ValueError: If ``key`` is not exactly 32 bytes.
    """

    def __init__(self, key: bytes) -> None:
        if len(key) != 32:
            raise ValueError(f"AESGCMPayloadCodec requires exactly 32 bytes, got {len(key)}.")
        self._key = key

    @classmethod
    def from_env(cls, env_var: str = "EXO_TEMPORAL_CODEC_KEY") -> AESGCMPayloadCodec:
        """Construct from a base64-encoded key stored in an environment variable.

        Args:
            env_var: Name of the environment variable (default ``EXO_TEMPORAL_CODEC_KEY``).

        Raises:
            ValueError: If the env var is unset or the decoded key is not 32 bytes.
        """
        import base64

        raw = os.environ.get(env_var)
        if raw is None:
            raise ValueError(
                f"Environment variable {env_var!r} is not set. "
                "Set it to a base64-encoded 32-byte AES-256 key."
            )
        try:
            key = base64.b64decode(raw)
        except Exception as exc:
            raise ValueError(
                f"Environment variable {env_var!r} is not valid base64: {exc}"
            ) from exc
        if len(key) != 32:
            raise ValueError(f"Decoded key from {env_var!r} is {len(key)} bytes; expected 32.")
        return cls(key)

    async def encode(self, payloads: Sequence[Payload]) -> list[Payload]:
        """Encrypt each payload with AES-256-GCM.

        The output payload metadata sets ``encoding`` to ``binary/encrypted``
        and the data is ``nonce (12 bytes) || ciphertext``.
        """
        result: list[Payload] = []
        aesgcm = AESGCM(self._key)
        for payload in payloads:
            data = payload.SerializeToString()
            nonce = os.urandom(_NONCE_SIZE)
            ct = aesgcm.encrypt(nonce, data, None)
            result.append(Payload(metadata={"encoding": _ENC_ENCRYPTED}, data=nonce + ct))
        return result

    async def decode(self, payloads: Sequence[Payload]) -> list[Payload]:
        """Decrypt payloads previously encrypted by :meth:`encode`.

        Payloads whose encoding is not ``binary/encrypted`` are passed through
        unchanged.
        """
        result: list[Payload] = []
        aesgcm = AESGCM(self._key)
        for payload in payloads:
            if payload.metadata.get("encoding") == _ENC_ENCRYPTED:
                nonce = payload.data[:_NONCE_SIZE]
                ct = payload.data[_NONCE_SIZE:]
                pt = aesgcm.decrypt(nonce, ct, None)
                p = Payload()
                p.ParseFromString(pt)
                result.append(p)
            else:
                result.append(payload)
        return result


class ChainedPayloadCodec(PayloadCodec):
    """Compose multiple codecs in series.

    ``encode`` applies codecs in the order given; ``decode`` applies them in
    reverse, correctly unwrapping the layers.

    Args:
        *codecs: One or more :class:`PayloadCodec` instances to chain.
    """

    def __init__(self, *codecs: PayloadCodec) -> None:
        self._codecs = codecs

    async def encode(self, payloads: Sequence[Payload]) -> list[Payload]:
        """Apply all codecs in order (left to right)."""
        result: list[Payload] = list(payloads)
        for codec in self._codecs:
            result = await codec.encode(result)
        return result

    async def decode(self, payloads: Sequence[Payload]) -> list[Payload]:
        """Apply all codecs in reverse order (right to left)."""
        result: list[Payload] = list(payloads)
        for codec in reversed(self._codecs):
            result = await codec.decode(result)
        return result


@runtime_checkable
class BlobStore(Protocol):
    """Protocol for a key/value blob store used by :class:`ClaimCheckCodec`."""

    async def put(self, key: str, data: bytes) -> None:
        """Store *data* under *key*."""
        ...

    async def get(self, key: str) -> bytes:
        """Retrieve data previously stored under *key*."""
        ...

    async def delete(self, key: str) -> None:
        """Remove *key* from the store (no-op if not present)."""
        ...


class InMemoryBlobStore:
    """Dict-backed :class:`BlobStore` for testing or single-process use.

    Not suitable for multi-process deployments — data lives only in memory.
    """

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    async def put(self, key: str, data: bytes) -> None:
        self._store[key] = data

    async def get(self, key: str) -> bytes:
        return self._store[key]

    async def delete(self, key: str) -> None:
        """Remove *key* from the store (no-op if not present)."""
        self._store.pop(key, None)


class RedisBlobStore:
    """Redis-backed :class:`BlobStore` with optional TTL.

    Uses ``redis.asyncio`` (already a hard dependency of ``exo-distributed``).
    The Redis client is created lazily on first use.

    Args:
        redis_url: Redis connection URL, e.g. ``"redis://localhost:6379"``.
        key_prefix: Prefix prepended to every stored key.
        ttl_seconds: If provided, stored keys expire after this many seconds.
    """

    def __init__(
        self,
        redis_url: str,
        *,
        key_prefix: str = "exo:temporal:blob:",
        ttl_seconds: int | None = None,
    ) -> None:
        self._redis_url = redis_url
        self._key_prefix = key_prefix
        self._ttl_seconds = ttl_seconds
        self._client: object | None = None

    def _get_client(self) -> object:
        if self._client is None:
            import redis.asyncio as aioredis

            self._client = aioredis.from_url(self._redis_url, decode_responses=False)
        return self._client

    async def put(self, key: str, data: bytes) -> None:
        client = self._get_client()
        full_key = self._key_prefix + key
        if self._ttl_seconds is not None:
            await client.set(full_key, data, ex=self._ttl_seconds)  # type: ignore[union-attr]
        else:
            await client.set(full_key, data)  # type: ignore[union-attr]

    async def get(self, key: str) -> bytes:
        client = self._get_client()
        full_key = self._key_prefix + key
        value: bytes | None = await client.get(full_key)  # type: ignore[union-attr]
        if value is None:
            raise KeyError(f"Blob not found in Redis: {full_key!r}")
        return value

    async def delete(self, key: str) -> None:
        """Remove *key* from the store (no-op if not present)."""
        client = self._get_client()
        await client.delete(self._key_prefix + key)  # type: ignore[union-attr]

    async def close(self) -> None:
        """Close the underlying Redis connection."""
        if self._client is not None:
            await self._client.aclose()  # type: ignore[union-attr]
            self._client = None


class ClaimCheckCodec(PayloadCodec):
    """Offload oversized payloads to a :class:`BlobStore`.

    Payloads whose serialized size exceeds *threshold_bytes* are stored in the
    blob store and replaced with a lightweight ``binary/claim-check`` token.
    Small payloads pass through unchanged.

    Args:
        blob_store: The backing store (must satisfy :class:`BlobStore` protocol).
        threshold_bytes: Maximum inline payload size before offloading.
        key_namespace: Optional prefix for generated blob keys.
    """

    def __init__(
        self,
        blob_store: BlobStore,
        *,
        threshold_bytes: int = 128 * 1024,
        key_namespace: str = "",
    ) -> None:
        self._store = blob_store
        self._threshold = threshold_bytes
        self._ns = key_namespace

    async def encode(self, payloads: Sequence[Payload]) -> list[Payload]:
        """Offload large payloads; pass small ones through unchanged."""
        result: list[Payload] = []
        for payload in payloads:
            serialized = payload.SerializeToString()
            if len(serialized) > self._threshold:
                key = self._ns + uuid.uuid4().hex
                await self._store.put(key, serialized)
                result.append(Payload(metadata={"encoding": _ENC_CLAIM_CHECK}, data=key.encode()))
            else:
                result.append(payload)
        return result

    async def decode(self, payloads: Sequence[Payload]) -> list[Payload]:
        """Fetch offloaded payloads from the blob store; pass others through.

        After fetching a blob, deletes it from the store to avoid unbounded
        accumulation (claim-check semantics: fetch once, then discard).
        """
        result: list[Payload] = []
        for payload in payloads:
            if payload.metadata.get("encoding") == _ENC_CLAIM_CHECK:
                key = payload.data.decode()
                blob = await self._store.get(key)
                await self._store.delete(key)
                p = Payload()
                p.ParseFromString(blob)
                result.append(p)
            else:
                result.append(payload)
        return result
