"""Tests for Temporal payload codecs and the DataConverter factory.

Covers:
- AESGCMPayloadCodec: round-trip, wrong-key failure, key-length validation, from_env.
- ChainedPayloadCodec: round-trip through two AES codecs; decode reverses order.
- ClaimCheckCodec with InMemoryBlobStore: large payload offloaded, small passes through.
- build_data_converter: returns DataConverter; codec attached correctly.
"""

from __future__ import annotations

import base64
import secrets

import pytest
from cryptography.exceptions import InvalidTag
from temporalio.api.common.v1 import Payload

from exo.distributed.temporal.codec import (
    AESGCMPayloadCodec,
    ChainedPayloadCodec,
    ClaimCheckCodec,
    InMemoryBlobStore,
)
from exo.distributed.temporal.converter import build_data_converter, pydantic_data_converter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_payload(data: bytes, encoding: bytes = b"json/plain") -> Payload:
    """Construct a minimal Payload for testing."""
    return Payload(metadata={"encoding": encoding}, data=data)


def _key() -> bytes:
    """Generate a random 32-byte AES key."""
    return secrets.token_bytes(32)


# ---------------------------------------------------------------------------
# AESGCMPayloadCodec
# ---------------------------------------------------------------------------


async def test_aesgcm_round_trip_single() -> None:
    codec = AESGCMPayloadCodec(_key())
    original = [_make_payload(b"hello world")]
    encrypted = await codec.encode(original)
    assert len(encrypted) == 1
    assert encrypted[0].metadata.get("encoding") == b"binary/encrypted"
    assert encrypted[0].data != b"hello world"

    decrypted = await codec.decode(encrypted)
    assert len(decrypted) == 1
    assert decrypted[0].data == b"hello world"
    assert decrypted[0].metadata.get("encoding") == b"json/plain"


async def test_aesgcm_round_trip_multiple() -> None:
    codec = AESGCMPayloadCodec(_key())
    originals = [_make_payload(f"payload-{i}".encode()) for i in range(5)]
    encrypted = await codec.encode(originals)
    decrypted = await codec.decode(encrypted)
    for orig, dec in zip(originals, decrypted, strict=True):
        assert dec.data == orig.data
        assert dec.metadata.get("encoding") == orig.metadata.get("encoding")


async def test_aesgcm_wrong_key_raises() -> None:
    codec_a = AESGCMPayloadCodec(_key())
    codec_b = AESGCMPayloadCodec(_key())  # different key

    payload = [_make_payload(b"secret")]
    encrypted = await codec_a.encode(payload)

    with pytest.raises(InvalidTag):
        await codec_b.decode(encrypted)


async def test_aesgcm_key_length_validation() -> None:
    with pytest.raises(ValueError, match="32 bytes"):
        AESGCMPayloadCodec(b"too-short")
    with pytest.raises(ValueError, match="32 bytes"):
        AESGCMPayloadCodec(secrets.token_bytes(16))
    with pytest.raises(ValueError, match="32 bytes"):
        AESGCMPayloadCodec(secrets.token_bytes(64))


async def test_aesgcm_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    key = secrets.token_bytes(32)
    monkeypatch.setenv("EXO_TEMPORAL_CODEC_KEY", base64.b64encode(key).decode())
    codec = AESGCMPayloadCodec.from_env()
    assert codec._key == key


async def test_aesgcm_from_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXO_TEMPORAL_CODEC_KEY", raising=False)
    with pytest.raises(ValueError, match="not set"):
        AESGCMPayloadCodec.from_env()


async def test_aesgcm_from_env_wrong_size(monkeypatch: pytest.MonkeyPatch) -> None:
    short_key = secrets.token_bytes(16)
    monkeypatch.setenv("EXO_TEMPORAL_CODEC_KEY", base64.b64encode(short_key).decode())
    with pytest.raises(ValueError, match="16 bytes"):
        AESGCMPayloadCodec.from_env()


async def test_aesgcm_passthrough_non_encrypted() -> None:
    """decode() must leave non-encrypted payloads unchanged."""
    codec = AESGCMPayloadCodec(_key())
    p = _make_payload(b"plain")
    result = await codec.decode([p])
    assert result[0] is p


# ---------------------------------------------------------------------------
# ChainedPayloadCodec
# ---------------------------------------------------------------------------


async def test_chained_round_trip() -> None:
    key_a, key_b = _key(), _key()
    codec = ChainedPayloadCodec(AESGCMPayloadCodec(key_a), AESGCMPayloadCodec(key_b))
    original = [_make_payload(b"chained data")]
    encoded = await codec.encode(original)
    decoded = await codec.decode(encoded)
    assert decoded[0].data == b"chained data"


async def test_chained_order_matters() -> None:
    """Prove that decode reverses the encode order by attempting partial decode."""
    key_a, key_b = _key(), _key()
    codec_a = AESGCMPayloadCodec(key_a)
    codec_b = AESGCMPayloadCodec(key_b)

    # encode: first A, then B  →  output is B(A(payload))
    chained = ChainedPayloadCodec(codec_a, codec_b)
    original = [_make_payload(b"order matters")]
    encoded = await chained.encode(original)

    # Correct decode: reverse order → B.decode then A.decode
    decoded = await chained.decode(encoded)
    assert decoded[0].data == b"order matters"

    # Wrong order (A then B, not reversed) must fail — B.decode(encoded) will
    # fail because the outer wrapper is B-encrypted, not A-encrypted.
    wrong_order = ChainedPayloadCodec(codec_b, codec_a)
    with pytest.raises(InvalidTag):
        await wrong_order.decode(encoded)


async def test_chained_single_codec() -> None:
    codec = ChainedPayloadCodec(AESGCMPayloadCodec(_key()))
    p = [_make_payload(b"solo")]
    assert (await codec.decode(await codec.encode(p)))[0].data == b"solo"


# ---------------------------------------------------------------------------
# ClaimCheckCodec + InMemoryBlobStore
# ---------------------------------------------------------------------------


async def test_claim_check_large_payload_offloaded() -> None:
    store = InMemoryBlobStore()
    threshold = 20  # very small for testing
    codec = ClaimCheckCodec(store, threshold_bytes=threshold)

    large_data = b"x" * 200
    original = [_make_payload(large_data)]
    encoded = await codec.encode(original)

    assert len(encoded) == 1
    enc_payload = encoded[0]
    assert enc_payload.metadata.get("encoding") == b"binary/claim-check"
    # The data should be a short key, not the original large content
    assert len(enc_payload.data) < len(large_data)

    # Decode restores the original
    decoded = await codec.decode(encoded)
    assert decoded[0].data == large_data
    assert decoded[0].metadata.get("encoding") == b"json/plain"


async def test_claim_check_small_payload_passthrough() -> None:
    store = InMemoryBlobStore()
    codec = ClaimCheckCodec(store, threshold_bytes=10000)  # high threshold

    small = _make_payload(b"tiny")
    encoded = await codec.encode([small])
    assert encoded[0] is small  # same object — not wrapped

    decoded = await codec.decode(encoded)
    assert decoded[0] is small


async def test_claim_check_multiple_payloads_mixed() -> None:
    store = InMemoryBlobStore()
    # small payload (b"hi") serializes to ~28 bytes; large (b"z"*100) to ~126 bytes.
    # Pick a threshold that separates them.
    threshold = 50
    codec = ClaimCheckCodec(store, threshold_bytes=threshold)

    small = _make_payload(b"hi")  # ~28 bytes serialized → below threshold
    large = _make_payload(b"z" * 100)  # ~126 bytes serialized → above threshold

    encoded = await codec.encode([small, large])
    assert encoded[0] is small
    assert encoded[1].metadata.get("encoding") == b"binary/claim-check"

    decoded = await codec.decode(encoded)
    assert decoded[0].data == b"hi"
    assert decoded[1].data == b"z" * 100


async def test_claim_check_key_namespace() -> None:
    store = InMemoryBlobStore()
    codec = ClaimCheckCodec(store, threshold_bytes=1, key_namespace="ns:")
    original = [_make_payload(b"ab")]
    encoded = await codec.encode(original)
    key = encoded[0].data.decode()
    assert key.startswith("ns:")
    # The key must actually exist in the store
    assert store._store.get(key) is not None


# ---------------------------------------------------------------------------
# build_data_converter
# ---------------------------------------------------------------------------


def test_build_data_converter_no_codec() -> None:
    conv = build_data_converter()
    assert conv is pydantic_data_converter


def test_build_data_converter_with_codec() -> None:
    from exo.distributed.temporal._compat import DataConverter

    codec = AESGCMPayloadCodec(_key())
    conv = build_data_converter(codec=codec)
    assert isinstance(conv, DataConverter)
    assert conv.payload_codec is codec


def test_build_data_converter_pydantic_converter_class() -> None:
    """The returned converter must use the Pydantic payload converter class."""
    from temporalio.contrib.pydantic import PydanticPayloadConverter

    conv = build_data_converter(codec=AESGCMPayloadCodec(_key()))
    assert isinstance(conv.payload_converter, PydanticPayloadConverter)
