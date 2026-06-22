"""Tests for RedisConfig namespace on Worker — no live Redis required.

Verifies:
- RedisConfig dataclass fields and defaults.
- Worker(redis=RedisConfig(...)) populates flat private attributes correctly.
- Flat kwargs (redis_url=, queue_name=, heartbeat_ttl=) still work unchanged.
- Conflict between redis= and flat counterparts raises ValueError.
- worker.redis_config read accessor returns the correct view.
"""

from __future__ import annotations

import pytest

from exo.distributed import (
    RedisConfig as RedisConfigFromInit,  # pyright: ignore[reportMissingImports]
)
from exo.distributed._redis_config import RedisConfig  # pyright: ignore[reportMissingImports]
from exo.distributed.worker import Worker  # pyright: ignore[reportMissingImports]
from exo.types import ExoError  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# RedisConfig dataclass
# ---------------------------------------------------------------------------


class TestRedisConfig:
    def test_required_url(self) -> None:
        cfg = RedisConfig(url="redis://localhost:6379")
        assert cfg.url == "redis://localhost:6379"

    def test_defaults(self) -> None:
        cfg = RedisConfig(url="redis://localhost:6379")
        assert cfg.queue_name == "exo:tasks"
        assert cfg.heartbeat_ttl == 30

    def test_custom_values(self) -> None:
        cfg = RedisConfig(url="redis://host:1234", queue_name="my:queue", heartbeat_ttl=60)
        assert cfg.url == "redis://host:1234"
        assert cfg.queue_name == "my:queue"
        assert cfg.heartbeat_ttl == 60

    def test_frozen(self) -> None:
        cfg = RedisConfig(url="redis://localhost")
        with pytest.raises((AttributeError, TypeError)):
            cfg.url = "redis://other"  # type: ignore[misc]

    def test_public_export_matches_internal(self) -> None:
        """RedisConfig from __init__ and _redis_config are the same class."""
        assert RedisConfigFromInit is RedisConfig


# ---------------------------------------------------------------------------
# Worker — grouped constructor (redis=RedisConfig(...))
# ---------------------------------------------------------------------------


class TestWorkerGroupedConstructor:
    def test_redis_config_populates_flat_attrs(self) -> None:
        cfg = RedisConfig(url="redis://myhost:6379", queue_name="custom:q", heartbeat_ttl=45)
        w = Worker(redis=cfg)
        assert w._redis_url == "redis://myhost:6379"
        assert w._queue_name == "custom:q"
        assert w._heartbeat_ttl == 45

    def test_redis_config_defaults(self) -> None:
        cfg = RedisConfig(url="redis://localhost")
        w = Worker(redis=cfg)
        assert w._redis_url == "redis://localhost"
        assert w._queue_name == "exo:tasks"
        assert w._heartbeat_ttl == 30

    def test_redis_config_with_other_flat_kwargs(self) -> None:
        """Non-conflicting flat kwargs (concurrency, worker_id) should work alongside redis=."""
        cfg = RedisConfig(url="redis://localhost")
        w = Worker(redis=cfg, concurrency=4, worker_id="my-worker")
        assert w._redis_url == "redis://localhost"
        assert w._concurrency == 4
        assert w.worker_id == "my-worker"


# ---------------------------------------------------------------------------
# Worker — flat constructor (backward compat)
# ---------------------------------------------------------------------------


class TestWorkerFlatConstructor:
    def test_positional_redis_url(self) -> None:
        w = Worker("redis://localhost:6379")
        assert w._redis_url == "redis://localhost:6379"
        assert w._queue_name == "exo:tasks"
        assert w._heartbeat_ttl == 30

    def test_keyword_redis_url(self) -> None:
        w = Worker(redis_url="redis://localhost:6379")
        assert w._redis_url == "redis://localhost:6379"

    def test_all_flat_kwargs(self) -> None:
        w = Worker(
            "redis://host:1234",
            worker_id="my-worker",
            concurrency=4,
            queue_name="custom:q",
            heartbeat_ttl=60,
        )
        assert w.worker_id == "my-worker"
        assert w._concurrency == 4
        assert w._queue_name == "custom:q"
        assert w._heartbeat_ttl == 60

    def test_missing_url_raises(self) -> None:
        with pytest.raises(ExoError, match="Redis URL is required"):
            Worker()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------


class TestWorkerConflictDetection:
    def test_conflict_redis_url(self) -> None:
        cfg = RedisConfig(url="redis://from-config")
        with pytest.raises(ExoError, match="redis_url"):
            Worker("redis://flat-url", redis=cfg)

    def test_conflict_queue_name(self) -> None:
        cfg = RedisConfig(url="redis://localhost")
        with pytest.raises(ExoError, match="queue_name"):
            Worker(redis=cfg, queue_name="other:q")

    def test_conflict_heartbeat_ttl(self) -> None:
        cfg = RedisConfig(url="redis://localhost")
        with pytest.raises(ExoError, match="heartbeat_ttl"):
            Worker(redis=cfg, heartbeat_ttl=99)


# ---------------------------------------------------------------------------
# redis_config read accessor
# ---------------------------------------------------------------------------


class TestWorkerRedisConfigAccessor:
    def test_accessor_after_flat_construction(self) -> None:
        w = Worker("redis://myhost", queue_name="my:q", heartbeat_ttl=45)
        rc = w.redis_config
        assert isinstance(rc, RedisConfig)
        assert rc.url == "redis://myhost"
        assert rc.queue_name == "my:q"
        assert rc.heartbeat_ttl == 45

    def test_accessor_after_grouped_construction(self) -> None:
        cfg = RedisConfig(url="redis://group-host", queue_name="group:q", heartbeat_ttl=20)
        w = Worker(redis=cfg)
        rc = w.redis_config
        assert rc.url == "redis://group-host"
        assert rc.queue_name == "group:q"
        assert rc.heartbeat_ttl == 20

    def test_accessor_returns_frozen_copy(self) -> None:
        """Each call returns a new immutable RedisConfig; mutations don't affect Worker."""
        w = Worker("redis://localhost")
        rc = w.redis_config
        with pytest.raises((AttributeError, TypeError)):
            rc.url = "redis://mutated"  # type: ignore[misc]
        # Original Worker attrs are unchanged
        assert w._redis_url == "redis://localhost"
