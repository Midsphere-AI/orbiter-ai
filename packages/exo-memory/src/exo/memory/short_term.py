"""Short-term memory: conversation context with scope filtering and windowing."""

from __future__ import annotations

import logging

from exo.memory.base import (  # pyright: ignore[reportMissingImports]
    AIMemory,
    ExoMemoryError,
    MemoryCategory,
    MemoryItem,
    MemoryMetadata,
    MemoryStatus,
    ToolMemory,
)

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """In-memory conversation store with scope-based filtering and windowing.

    Implements the MemoryStore protocol for managing short-term conversation
    context. Supports three scoping levels (user, session, task), configurable
    message windowing, and tool call/response integrity filtering.

    Attributes:
        scope: Filtering scope — "user", "session", or "task" (default).
        max_rounds: Maximum conversation rounds to keep (0 = unlimited).
    """

    __slots__ = ("_items", "max_rounds", "scope")

    def __init__(
        self,
        *,
        scope: str = "task",
        max_rounds: int = 0,
    ) -> None:
        if scope not in ("user", "session", "task"):
            raise ExoMemoryError(
                f"Invalid scope {scope!r}.",
                context={"scope": scope},
                hint="Pass one of: 'user', 'session', or 'task'.",
            )
        self.scope = scope
        self.max_rounds = max_rounds
        self._items: list[MemoryItem] = []

    # -- MemoryStore protocol --------------------------------------------------

    async def add(self, item: MemoryItem) -> None:
        """Persist a memory item."""
        self._items.append(item)
        logger.debug("added item type=%s id=%s scope=%s", item.memory_type, item.id, self.scope)

    async def get(self, item_id: str) -> MemoryItem | None:
        """Retrieve a memory item by ID."""
        for item in self._items:
            if item.id == item_id:
                return item
        return None

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
        """Search memory with optional filters, scope awareness, and windowing."""
        results = self._filter(
            metadata=metadata,
            memory_type=memory_type,
            category=category,
            status=status,
        )

        # Keyword filter
        if query:
            q = query.lower()
            results = [r for r in results if q in r.content.lower()]

        # Apply windowing
        if self.max_rounds > 0:
            results = self._window(results, self.max_rounds)

        # Ensure tool call integrity
        results = self._filter_incomplete_pairs(results)

        logger.debug("search query=%r results=%d scope=%s", query, len(results[:limit]), self.scope)
        return results[:limit]

    async def clear(
        self,
        *,
        metadata: MemoryMetadata | None = None,
    ) -> int:
        """Remove memory items matching the filter. Returns count removed."""
        if metadata is None:
            count = len(self._items)
            self._items.clear()
            logger.debug("cleared all items count=%d scope=%s", count, self.scope)
            return count

        keep: list[MemoryItem] = []
        removed = 0
        for item in self._items:
            if self._matches_metadata(item.metadata, metadata):
                removed += 1
            else:
                keep.append(item)
        self._items = keep
        logger.debug("cleared filtered items count=%d scope=%s", removed, self.scope)
        return removed

    # -- Filtering helpers -----------------------------------------------------

    def _filter(
        self,
        *,
        metadata: MemoryMetadata | None = None,
        memory_type: str | None = None,
        category: MemoryCategory | None = None,
        status: MemoryStatus | None = None,
    ) -> list[MemoryItem]:
        """Apply scope-based and optional filters."""
        results: list[MemoryItem] = []
        for item in self._items:
            if memory_type and item.memory_type != memory_type:
                continue
            if category is not None and item.category != category:
                continue
            if status and item.status != status:
                continue
            if metadata and not self._matches_metadata(item.metadata, metadata):
                continue
            results.append(item)
        return results

    def _matches_metadata(self, item_meta: MemoryMetadata, filter_meta: MemoryMetadata) -> bool:
        """Check if item metadata matches filter based on current scope."""
        if filter_meta.user_id and item_meta.user_id != filter_meta.user_id:
            return False
        if (
            self.scope in ("session", "task")
            and filter_meta.session_id
            and item_meta.session_id != filter_meta.session_id
        ):
            return False
        if (
            self.scope == "task"
            and filter_meta.task_id
            and item_meta.task_id != filter_meta.task_id
        ):
            return False
        return not (filter_meta.agent_id and item_meta.agent_id != filter_meta.agent_id)

    # -- Windowing -------------------------------------------------------------

    @staticmethod
    def _window(items: list[MemoryItem], max_rounds: int) -> list[MemoryItem]:
        """Keep only the last *max_rounds* conversation rounds.

        A round starts with a human message. System messages are always
        retained. The window is measured from the end of the list.
        """
        # Find positions of human messages (each starts a round)
        human_positions: list[int] = []
        for i, item in enumerate(items):
            if item.memory_type == "human":
                human_positions.append(i)

        if len(human_positions) <= max_rounds:
            return items

        # The window starts at the Nth-from-last human message
        cut_index = human_positions[-max_rounds]

        # Always keep system messages before the cut
        system_msgs = [m for m in items[:cut_index] if m.memory_type == "system"]
        return system_msgs + items[cut_index:]

    # -- Tool call integrity ---------------------------------------------------

    @staticmethod
    def _filter_incomplete_pairs(items: list[MemoryItem]) -> list[MemoryItem]:
        """Remove trailing AI messages with unmatched tool calls.

        If the last AI message contains tool_calls but there are no
        corresponding tool result messages following it, drop those
        dangling messages to prevent LLM confusion.
        """
        if not items:
            return items

        # Walk backwards to find trailing AI messages with pending tool calls
        result = list(items)
        while result:
            last = result[-1]
            # AI message with tool_calls but no following tool results
            if isinstance(last, AIMemory) and last.tool_calls:
                result.pop()
                continue
            # Tool result without a preceding AI message with matching call
            if isinstance(last, ToolMemory):
                # Check if there's a matching AI message earlier
                has_match = False
                for earlier in result[:-1]:
                    if isinstance(earlier, AIMemory) and any(
                        tc.get("id") == last.tool_call_id for tc in earlier.tool_calls
                    ):
                        has_match = True
                        break
                if not has_match:
                    result.pop()
                    continue
            break
        return result

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return (
            f"ShortTermMemory(scope={self.scope!r}, "
            f"max_rounds={self.max_rounds}, items={len(self._items)})"
        )
