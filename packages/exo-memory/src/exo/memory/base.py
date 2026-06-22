"""Memory store protocol and typed memory item hierarchy.

Provides pluggable memory storage with scoped metadata and status lifecycle.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from exo.types import ExoError  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)


class ExoMemoryError(ExoError):
    """Base exception for memory operations."""


# ---------------------------------------------------------------------------
# Status lifecycle
# ---------------------------------------------------------------------------


class MemoryStatus(StrEnum):
    """Memory item status lifecycle: DRAFT -> ACCEPTED -> DISCARD."""

    DRAFT = "draft"
    ACCEPTED = "accepted"
    DISCARD = "discard"


class MemoryCategory(StrEnum):
    """Taxonomy for classifying memory knowledge types.

    Orthogonal to ``memory_type`` (which tracks conversation roles like
    human/ai/tool/system). Categories classify *what kind of knowledge*
    a memory item represents.
    """

    USER_PROFILE = "user_profile"
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    VARIABLE = "variable"
    SUMMARY = "summary"
    CONVERSATION = "conversation"


_VALID_TRANSITIONS: dict[MemoryStatus, set[MemoryStatus]] = {
    MemoryStatus.DRAFT: {MemoryStatus.ACCEPTED, MemoryStatus.DISCARD},
    MemoryStatus.ACCEPTED: {MemoryStatus.DISCARD},
    MemoryStatus.DISCARD: set(),
}


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class MemoryMetadata(BaseModel):
    """Scoping metadata for memory items.

    All fields are optional — set the ones relevant to your use case.
    """

    model_config = {"frozen": True}

    user_id: str | None = None
    session_id: str | None = None
    task_id: str | None = None
    agent_id: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Memory item hierarchy
# ---------------------------------------------------------------------------


class MemoryItem(BaseModel):
    """Base class for all memory entries.

    Attributes:
        id: Unique identifier (auto-generated UUID).
        content: The stored content.
        memory_type: Discriminator for subtype dispatch.
        status: Current lifecycle status.
        metadata: Scoping information.
        created_at: ISO-8601 creation timestamp.
        updated_at: ISO-8601 last-update timestamp.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    content: str
    memory_type: str
    category: MemoryCategory | None = None
    status: MemoryStatus = MemoryStatus.ACCEPTED
    metadata: MemoryMetadata = Field(default_factory=MemoryMetadata)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    def transition(self, new_status: MemoryStatus) -> MemoryItem:
        """Transition to a new status and return self for chaining.

        Mutates the model in-place (MemoryItem is intentionally non-frozen
        because status evolves over its lifetime).  Callers that need an
        independent copy should call ``model_copy()`` first.

        Raises:
            ExoMemoryError: If the transition is invalid.
        """
        allowed = _VALID_TRANSITIONS.get(self.status, set())
        if new_status not in allowed:
            msg = f"Cannot transition from {self.status!r} to {new_status!r}"
            raise ExoMemoryError(msg)
        old_status = self.status
        self.status = new_status
        self.updated_at = datetime.now(UTC).isoformat()
        logger.debug("item %s transitioned %s -> %s", self.id, old_status, new_status)
        return self


class SystemMemory(MemoryItem):
    """System/initialization memory (e.g. system prompts)."""

    memory_type: str = "system"


class HumanMemory(MemoryItem):
    """User/human message memory."""

    memory_type: str = "human"


class AIMemory(MemoryItem):
    """AI/assistant response memory."""

    memory_type: str = "ai"
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)


class ToolMemory(MemoryItem):
    """Tool execution result memory."""

    memory_type: str = "tool"
    tool_call_id: str = ""
    tool_name: str = ""
    is_error: bool = False


# ---------------------------------------------------------------------------
# MemoryStore protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class MemoryStore(Protocol):
    """Protocol for pluggable memory backends.

    Implementations must support add, get, search, and clear operations.
    All methods are async to support both in-memory and remote backends.
    """

    async def add(self, item: MemoryItem) -> None:
        """Persist a memory item."""
        ...

    async def get(self, item_id: str) -> MemoryItem | None:
        """Retrieve a memory item by ID. Returns None if not found."""
        ...

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
        """Search memory items with optional filters.

        Default implementation performs keyword search over ``self._items``
        when the backing store exposes that attribute.  Concrete backends
        should override this with their native search.
        """
        items: list[MemoryItem] = list(getattr(self, "_items", []))
        if query:
            q = query.lower()
            items = [i for i in items if q in i.content.lower()]
        if memory_type:
            items = [i for i in items if i.memory_type == memory_type]
        if category is not None:
            items = [i for i in items if i.category == category]
        if status:
            items = [i for i in items if i.status == status]
        if metadata:
            if metadata.user_id:
                items = [i for i in items if i.metadata.user_id == metadata.user_id]
            if metadata.session_id:
                items = [i for i in items if i.metadata.session_id == metadata.session_id]
            if metadata.task_id:
                items = [i for i in items if i.metadata.task_id == metadata.task_id]
            if metadata.agent_id:
                items = [i for i in items if i.metadata.agent_id == metadata.agent_id]
        return items[:limit]

    async def clear(
        self,
        *,
        metadata: MemoryMetadata | None = None,
    ) -> int:
        """Remove memory items matching the filter. Returns count removed."""
        ...



# ---------------------------------------------------------------------------
# AgentMemory — composite of short-term and long-term stores
# ---------------------------------------------------------------------------


@dataclass
class AgentMemory:
    """Composite memory object bundling short-term and long-term stores.

    Pass an ``AgentMemory`` instance to ``Agent(memory=...)`` to configure
    which backends are used for conversation history and persistent facts.

    Attributes:
        short_term: Fast in-process store for the current conversation.
        long_term: Persistent store for cross-session knowledge.
    """

    short_term: MemoryStore
    long_term: MemoryStore
