"""Memory event integration for async processing.

Wraps a MemoryStore with an EventBus to emit events on memory operations,
enabling async processing pipelines (e.g., trigger summarization after
messages accumulate, or extraction after a session ends).
"""

from __future__ import annotations

import logging
from typing import Any

from exo.events import EventBus  # pyright: ignore[reportMissingImports]
from exo.memory.base import (  # pyright: ignore[reportMissingImports]
    MemoryCategory,
    MemoryItem,
    MemoryMetadata,
    MemoryStatus,
)

logger = logging.getLogger(__name__)

# Standard memory event names
MEMORY_ADDED = "memory:added"
MEMORY_SEARCHED = "memory:searched"
MEMORY_CLEARED = "memory:cleared"


class MemoryEventEmitter:
    """Wraps a MemoryStore to emit events on operations.

    Events emitted:
        ``memory:added`` — after ``add(item)``
        ``memory:searched`` — after ``search(...)`` with results
        ``memory:cleared`` — after ``clear(...)`` with count

    Args:
        store: Any MemoryStore-compatible backend.
        bus: EventBus for emitting events.
    """

    __slots__ = ("bus", "store")

    def __init__(self, store: Any, bus: EventBus | None = None) -> None:
        self.store = store
        self.bus = bus or EventBus()

    async def add(self, item: MemoryItem) -> None:
        """Add item and emit ``memory:added`` event."""
        await self.store.add(item)
        logger.debug("memory:added type=%s id=%s", item.memory_type, item.id)
        await self.bus.emit(MEMORY_ADDED, item=item)

    async def get(self, item_id: str) -> MemoryItem | None:
        """Retrieve item by ID (no event emitted)."""
        result: MemoryItem | None = await self.store.get(item_id)
        return result

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
        """Search and emit ``memory:searched`` event."""
        results: list[MemoryItem] = await self.store.search(
            query=query,
            metadata=metadata,
            memory_type=memory_type,
            category=category,
            status=status,
            limit=limit,
        )
        logger.debug("memory:searched query=%r results=%d", query, len(results))
        await self.bus.emit(MEMORY_SEARCHED, results=results, query=query)
        return results

    async def clear(
        self,
        *,
        metadata: MemoryMetadata | None = None,
    ) -> int:
        """Clear and emit ``memory:cleared`` event."""
        count: int = await self.store.clear(metadata=metadata)
        logger.debug("memory:cleared count=%d", count)
        await self.bus.emit(MEMORY_CLEARED, count=count, metadata=metadata)
        return count

    def __repr__(self) -> str:
        return f"MemoryEventEmitter(store={self.store!r})"
