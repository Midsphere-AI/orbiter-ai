"""Unified SearchManager: query across multiple memory stores in parallel."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from exo.memory.base import (  # pyright: ignore[reportMissingImports]
    MemoryCategory,
    MemoryItem,
    MemoryMetadata,
    MemoryStatus,
    MemoryStore,
)

logger = logging.getLogger(__name__)


class SearchManager:
    """Query across multiple :class:`MemoryStore` instances in a single call.

    Results are gathered in parallel, deduplicated by item ID, and sorted by
    ``created_at`` descending (newest first).  A per-store limit prevents any
    single store from dominating the merged result set.
    """

    def __init__(self, stores: Sequence[MemoryStore]) -> None:
        self._stores = list(stores)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        category: MemoryCategory | None = None,
        memory_type: str | None = None,
        status: MemoryStatus | None = None,
        metadata: MemoryMetadata | None = None,
    ) -> list[MemoryItem]:
        """Search all stores and return merged, deduplicated results.

        Each store is queried with *limit* so that no single store can
        return more than *limit* items.  After merging, duplicates (by
        ``item.id``) are removed and the combined list is sorted by
        ``created_at`` descending then trimmed to *limit*.

        Args:
            query: Text query (passed to each backend's search).
            limit: Maximum items per store and in the merged result.
            category: Optional category filter.
            memory_type: Optional memory_type filter (e.g. ``"human"``).
            status: Optional lifecycle status filter.
            metadata: Optional metadata scope filter.
        """
        if not self._stores:
            return []

        tasks = [
            store.search(
                query=query,
                category=category,
                memory_type=memory_type,
                status=status,
                metadata=metadata,
                limit=limit,
            )
            for store in self._stores
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Deduplicate by item ID, keeping the first occurrence.
        seen: dict[str, MemoryItem] = {}
        for result in raw_results:
            if isinstance(result, BaseException):
                logger.warning("Memory store search failed: %s", result)
                continue
            for item in result:
                if item.id not in seen:
                    seen[item.id] = item

        # Sort by created_at descending (newest first).
        merged = sorted(seen.values(), key=lambda m: m.created_at, reverse=True)
        return merged[:limit]

    @property
    def stores(self) -> list[MemoryStore]:
        """Return a copy of the managed store list."""
        return list(self._stores)

    def __repr__(self) -> str:
        return f"SearchManager(stores={len(self._stores)})"
