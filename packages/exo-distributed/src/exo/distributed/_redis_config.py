"""Shared Redis connection configuration dataclass for distributed components."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RedisConfig:
    """Grouped Redis connection settings shared across distributed components.

    Use this to configure :class:`~exo.distributed.Worker` (and optionally
    :class:`~exo.distributed.broker.TaskBroker`) with a single object instead
    of repeating individual keyword arguments.

    Args:
        url: Redis connection URL (e.g. ``"redis://localhost:6379"``).
        queue_name: Redis Streams queue name (default ``"exo:tasks"``).
        heartbeat_ttl: TTL in seconds for the worker heartbeat key (default 30).

    Example::

        from exo.distributed import Worker, RedisConfig

        cfg = RedisConfig(url="redis://localhost:6379", queue_name="my:queue")
        worker = Worker(redis=cfg)
    """

    url: str
    queue_name: str = "exo:tasks"
    heartbeat_ttl: int = 30
