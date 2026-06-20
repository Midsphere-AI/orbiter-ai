"""Shared Redis connection mixin for distributed components."""

from __future__ import annotations

import logging

import redis.asyncio as aioredis

from exo.types import ExoError  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)


class RedisConnectionMixin:
    """Mixin providing idempotent connect/disconnect/client helpers for Redis.

    Subclasses must call ``super().__init__()`` and set ``self._redis_url``
    before calling :meth:`connect`.
    """

    _redis_url: str
    _redis: aioredis.Redis | None

    def __init__(self) -> None:
        self._redis = None

    async def connect(self) -> None:
        """Open an async Redis connection (idempotent)."""
        if self._redis is None:
            logger.debug("%s connecting to Redis", type(self).__name__)
            self._redis = aioredis.from_url(self._redis_url, decode_responses=True)

    async def disconnect(self) -> None:
        """Close the Redis connection (idempotent)."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
            logger.debug("%s disconnected", type(self).__name__)

    def _client(self) -> aioredis.Redis:
        if self._redis is None:
            msg = f"{type(self).__name__} is not connected. Call connect() first."
            raise ExoError(msg)
        return self._redis
