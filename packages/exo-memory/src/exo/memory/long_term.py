"""Long-term memory: persistent knowledge across sessions with async LLM extraction.

Provides LongTermMemory (MemoryStore-compatible persistent store) and
MemoryOrchestrator for extracting structured knowledge (user profiles,
agent experiences, facts) from conversation history via LLM.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from exo.memory.base import (  # pyright: ignore[reportMissingImports]
    ExoMemoryError,
    MemoryCategory,
    MemoryItem,
    MemoryMetadata,
    MemoryStatus,
)
from exo.memory.dedup import (  # pyright: ignore[reportMissingImports]
    MemUpdateChecker,
    UpdateDecision,
)

# ---------------------------------------------------------------------------
# Extraction types
# ---------------------------------------------------------------------------


class ExtractionType(StrEnum):
    """Types of knowledge that can be extracted from conversations."""

    USER_PROFILE = "user_profile"
    AGENT_EXPERIENCE = "agent_experience"
    FACTS = "facts"


logger = logging.getLogger(__name__)

_DEFAULT_EXTRACTION_PROMPTS: dict[ExtractionType, str] = {
    ExtractionType.USER_PROFILE: (
        "Extract user preferences, traits, and background from this "
        "conversation. Return structured key-value pairs:\n\n{content}"
    ),
    ExtractionType.AGENT_EXPERIENCE: (
        "Extract lessons learned, successful strategies, and failure "
        "patterns from this agent interaction:\n\n{content}"
    ),
    ExtractionType.FACTS: (
        "Extract verified factual statements and important decisions "
        "from this conversation as a bullet list:\n\n{content}"
    ),
}


# ---------------------------------------------------------------------------
# Extractor protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Extractor(Protocol):
    """Protocol for LLM-powered knowledge extraction.

    Callers provide an implementation wrapping their LLM provider.
    """

    async def extract(self, prompt: str) -> str:
        """Run extraction prompt and return extracted text."""
        ...


# ---------------------------------------------------------------------------
# Processing task
# ---------------------------------------------------------------------------


class MemoryTaskStatus(StrEnum):
    """Lifecycle states for extraction tasks."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(slots=True)
class ExtractionTask:
    """A single knowledge extraction task.

    Attributes:
        task_id: Unique identifier.
        extraction_type: What kind of knowledge to extract.
        source_items: Memory items to extract from.
        status: Current task lifecycle status.
        result: Extracted text (set on completion).
        error: Error message (set on failure).
        metadata: Default metadata to attach to extracted items (set by submit()).
        created_at: ISO-8601 creation timestamp.
        completed_at: ISO-8601 completion timestamp (set on completion/failure).
    """

    extraction_type: ExtractionType
    source_items: list[MemoryItem]
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: MemoryTaskStatus = MemoryTaskStatus.PENDING
    result: str | None = None
    error: str | None = None
    metadata: MemoryMetadata | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    completed_at: str | None = None

    def start(self) -> None:
        """Mark task as running."""
        self.status = MemoryTaskStatus.RUNNING

    def complete(self, result: str) -> None:
        """Mark task as completed with result."""
        self.status = MemoryTaskStatus.COMPLETED
        self.result = result
        self.completed_at = datetime.now(UTC).isoformat()

    def fail(self, error: str) -> None:
        """Mark task as failed with error."""
        self.status = MemoryTaskStatus.FAILED
        self.error = error
        self.completed_at = datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# LongTermMemory
# ---------------------------------------------------------------------------


class LongTermMemory:
    """Persistent memory store for knowledge that spans sessions.

    Implements the MemoryStore protocol. Stores extracted knowledge
    (user profiles, experiences, facts) with deduplication by content hash.

    Attributes:
        namespace: Isolation namespace for multi-tenant usage.
    """

    __slots__ = ("_content_hashes", "_items", "_update_checker", "namespace")

    def __init__(
        self,
        *,
        namespace: str = "default",
        update_checker: MemUpdateChecker | None = None,
    ) -> None:
        self.namespace = namespace
        self._items: dict[str, MemoryItem] = {}
        self._update_checker = update_checker
        # O(1) exact-match dedup: keyed by (memory_type, content) tuples.
        self._content_hashes: set[tuple[str, str]] = set()

    async def add(self, item: MemoryItem) -> None:
        """Persist a memory item, deduplicating by content.

        When an ``update_checker`` is configured, it is consulted to decide
        whether to ADD, SKIP, DELETE existing items, or MERGE contents.
        Without a checker the exact-match dedup behaviour is used (O(1) via
        an internal content-hash set).
        """
        if self._update_checker is not None:
            existing_items = list(self._items.values())
            result = await self._update_checker.check(item, existing_items)
            decision = result.decision

            if decision == UpdateDecision.SKIP:
                logger.debug("checker: skip item type=%s id=%s", item.memory_type, item.id)
                return

            if decision == UpdateDecision.DELETE:
                for did in result.delete_ids:
                    deleted = self._items.pop(did, None)
                    if deleted is not None:
                        self._content_hashes.discard((deleted.memory_type, deleted.content))
                self._items[item.id] = item
                self._content_hashes.add((item.memory_type, item.content))
                logger.debug(
                    "checker: delete ids=%s, added item id=%s namespace=%s",
                    result.delete_ids,
                    item.id,
                    self.namespace,
                )
                return

            if decision == UpdateDecision.MERGE:
                for did in result.delete_ids:
                    deleted = self._items.pop(did, None)
                    if deleted is not None:
                        self._content_hashes.discard((deleted.memory_type, deleted.content))
                merged_content = result.merged_content if result.merged_content else item.content
                merged_item = item.model_copy(update={"content": merged_content})
                self._items[merged_item.id] = merged_item
                self._content_hashes.add((merged_item.memory_type, merged_item.content))
                logger.debug(
                    "checker: merge delete_ids=%s, added merged item id=%s namespace=%s",
                    result.delete_ids,
                    merged_item.id,
                    self.namespace,
                )
                return

            # decision == UpdateDecision.ADD — fall through to normal add
        else:
            # O(1) exact-match dedup via content-hash set
            key = (item.memory_type, item.content)
            if key in self._content_hashes:
                logger.debug("skipping duplicate item type=%s id=%s", item.memory_type, item.id)
                return

        self._items[item.id] = item
        self._content_hashes.add((item.memory_type, item.content))
        logger.debug(
            "added item type=%s id=%s namespace=%s", item.memory_type, item.id, self.namespace
        )

    async def get(self, item_id: str) -> MemoryItem | None:
        """Retrieve a memory item by ID."""
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
        """Search long-term memory with optional filters."""
        results: list[MemoryItem] = []
        for item in self._items.values():
            if memory_type and item.memory_type != memory_type:
                continue
            if category is not None and item.category != category:
                continue
            if status and item.status != status:
                continue
            if metadata and not self._matches_metadata(item.metadata, metadata):
                continue
            if query:
                q = query.lower()
                if q not in item.content.lower():
                    continue
            results.append(item)
        # Sort by creation time (newest first)
        results.sort(key=lambda x: x.created_at, reverse=True)
        logger.debug(
            "search query=%r results=%d namespace=%s", query, len(results[:limit]), self.namespace
        )
        return results[:limit]

    async def clear(
        self,
        *,
        metadata: MemoryMetadata | None = None,
    ) -> int:
        """Remove memory items matching the filter."""
        if metadata is not None and not any(
            [metadata.user_id, metadata.session_id, metadata.task_id, metadata.agent_id]
        ):
            logger.debug("clear() called with all-None metadata fields — returning 0")
            return 0
        if metadata is None:
            count = len(self._items)
            self._items.clear()
            self._content_hashes.clear()
            logger.debug("cleared all items count=%d namespace=%s", count, self.namespace)
            return count

        to_remove = [
            item_id
            for item_id, item in self._items.items()
            if self._matches_metadata(item.metadata, metadata)
        ]
        for item_id in to_remove:
            removed = self._items.pop(item_id)
            self._content_hashes.discard((removed.memory_type, removed.content))
        logger.debug("cleared filtered items count=%d namespace=%s", len(to_remove), self.namespace)
        return len(to_remove)

    @staticmethod
    def _matches_metadata(item_meta: MemoryMetadata, filter_meta: MemoryMetadata) -> bool:
        """Check if item metadata matches filter."""
        if filter_meta.user_id and item_meta.user_id != filter_meta.user_id:
            return False
        if filter_meta.session_id and item_meta.session_id != filter_meta.session_id:
            return False
        if filter_meta.task_id and item_meta.task_id != filter_meta.task_id:
            return False
        return not (filter_meta.agent_id and item_meta.agent_id != filter_meta.agent_id)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"LongTermMemory(namespace={self.namespace!r}, items={len(self._items)})"


# ---------------------------------------------------------------------------
# Orchestrator configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OrchestratorConfig:
    """Configuration for the memory orchestrator.

    Attributes:
        extraction_types: Which knowledge types to extract.
        prompts: Custom prompt overrides per extraction type.
        min_items: Minimum number of items to trigger extraction.
    """

    extraction_types: tuple[ExtractionType, ...] = (
        ExtractionType.USER_PROFILE,
        ExtractionType.AGENT_EXPERIENCE,
        ExtractionType.FACTS,
    )
    prompts: dict[ExtractionType, str] = field(default_factory=dict)
    min_items: int = 3

    def get_prompt(self, extraction_type: ExtractionType) -> str:
        """Get the prompt for an extraction type, falling back to defaults."""
        return self.prompts.get(extraction_type, _DEFAULT_EXTRACTION_PROMPTS[extraction_type])


# ---------------------------------------------------------------------------
# MemoryOrchestrator
# ---------------------------------------------------------------------------


class MemoryOrchestrator:
    """Orchestrates async LLM extraction of structured knowledge from conversations.

    Manages a task queue for extraction jobs, delegates to an Extractor
    (LLM wrapper), and stores results in a LongTermMemory store.

    Attributes:
        store: The long-term memory store to persist extracted knowledge.
        config: Orchestrator configuration.
    """

    __slots__ = ("_tasks", "config", "store")

    def __init__(
        self,
        store: LongTermMemory,
        *,
        config: OrchestratorConfig | None = None,
    ) -> None:
        self.store = store
        self.config = config or OrchestratorConfig()
        self._tasks: dict[str, ExtractionTask] = {}

    def submit(
        self,
        items: Sequence[MemoryItem],
        *,
        extraction_type: ExtractionType | None = None,
        metadata: MemoryMetadata | None = None,
    ) -> list[ExtractionTask]:
        """Submit extraction tasks for the given memory items.

        Creates one task per configured extraction type (or a single task
        if *extraction_type* is specified).

        Returns:
            List of created ExtractionTask objects.
        """
        types = (extraction_type,) if extraction_type else self.config.extraction_types
        tasks: list[ExtractionTask] = []
        for etype in types:
            task = ExtractionTask(
                extraction_type=etype,
                source_items=list(items),
                metadata=metadata,
            )
            self._tasks[task.task_id] = task
            tasks.append(task)
        logger.info("long-term extraction triggered: items=%d types=%d", len(items), len(tasks))
        return tasks

    async def process(
        self,
        task_id: str,
        extractor: Any,
        *,
        metadata: MemoryMetadata | None = None,
    ) -> ExtractionTask:
        """Process a single extraction task.

        Runs the extractor on the task's source items, stores the result
        in long-term memory, and updates the task status.

        Args:
            task_id: ID of the task to process.
            extractor: Object implementing the Extractor protocol.
            metadata: Optional metadata to attach to extracted memory items.

        Returns:
            The updated ExtractionTask.

        Raises:
            KeyError: If no task with the given ID exists.
        """
        task = self._tasks.get(task_id)
        if task is None:
            raise ExoMemoryError(
                f"No extraction task with id {task_id!r}.",
                context={"task_id": task_id},
                hint="Check the task_id returned by submit(). Use list_tasks() to inspect known tasks.",
            )

        task.start()
        logger.debug("processing extraction task=%s type=%s", task.task_id, task.extraction_type)
        try:
            content = _format_extraction_items(task.source_items)
            prompt = self.config.get_prompt(task.extraction_type).format(content=content)
            result = await extractor.extract(prompt)
            task.complete(result)
            logger.debug("extraction completed task=%s", task.task_id)

            # Store extracted knowledge.
            # Precedence: explicit metadata arg > task.metadata > empty default.
            effective_metadata = metadata or task.metadata or MemoryMetadata()
            item = MemoryItem(
                content=result,
                memory_type=task.extraction_type.value,
                metadata=effective_metadata,
            )
            await self.store.add(item)
        except Exception as exc:
            task.fail(str(exc))
            logger.error("extraction failed task=%s: %s", task.task_id, exc, exc_info=True)
        finally:
            # Evict the terminal task entry to prevent unbounded growth of
            # _tasks.  Callers that need the result already received the
            # returned ExtractionTask object.
            self._tasks.pop(task.task_id, None)

        return task

    async def process_all(
        self,
        extractor: Any,
        *,
        metadata: MemoryMetadata | None = None,
    ) -> list[ExtractionTask]:
        """Process all pending tasks.

        Returns:
            List of processed ExtractionTask objects.
        """
        pending = [t for t in self._tasks.values() if t.status == MemoryTaskStatus.PENDING]
        results: list[ExtractionTask] = []
        for task in pending:
            result = await self.process(task.task_id, extractor, metadata=metadata)
            results.append(result)
        return results

    def get_task(self, task_id: str) -> ExtractionTask | None:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def list_tasks(self, *, status: MemoryTaskStatus | None = None) -> list[ExtractionTask]:
        """List tasks, optionally filtered by status."""
        if status is None:
            return list(self._tasks.values())
        return [t for t in self._tasks.values() if t.status == status]

    def __repr__(self) -> str:
        return f"MemoryOrchestrator(store={self.store!r}, tasks={len(self._tasks)})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_extraction_items(items: Sequence[MemoryItem]) -> str:
    """Format memory items into a readable string for extraction prompts."""
    lines: list[str] = []
    for item in items:
        role = item.memory_type.upper()
        lines.append(f"[{role}]: {item.content}")
    return "\n".join(lines)
