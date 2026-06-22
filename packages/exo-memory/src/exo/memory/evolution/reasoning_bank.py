"""ReasoningBank — structured memory evolution with semantic dedup and recall.

Stores memories as structured entries (title/description/content) and provides
query-based recall using embeddings or keyword fallback.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import Any

from exo.memory.base import MemoryItem  # pyright: ignore[reportMissingImports]
from exo.memory.evolution import MemoryEvolutionStrategy  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReasoningEntry:
    """A structured memory entry in the ReasoningBank.

    Attributes:
        title: Short identifier for the memory.
        description: Summary of what this memory covers.
        content: Full memory content.
        item_id: ID of the corresponding MemoryItem.
    """

    title: str
    description: str
    content: str
    item_id: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _parse_entry(item: MemoryItem) -> ReasoningEntry:
    """Parse a MemoryItem's content into a ReasoningEntry.

    Supports JSON format: ``{"title": "...", "description": "...", "content": "..."}``.
    Falls back to using the full content as the entry's content field.
    """
    try:
        data = json.loads(item.content)
        if isinstance(data, dict):
            return ReasoningEntry(
                title=data.get("title", ""),
                description=data.get("description", ""),
                content=data.get("content", item.content),
                item_id=item.id,
            )
    except (json.JSONDecodeError, TypeError):
        pass
    return ReasoningEntry(
        title="",
        description="",
        content=item.content,
        item_id=item.id,
    )


def _entry_text(entry: ReasoningEntry) -> str:
    """Combine entry fields into a single string for comparison/embedding."""
    parts: list[str] = []
    if entry.title:
        parts.append(entry.title)
    if entry.description:
        parts.append(entry.description)
    parts.append(entry.content)
    return " ".join(parts)


def _keyword_similarity(a: str, b: str) -> float:
    """Simple keyword overlap (Jaccard) similarity between two strings."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    union = words_a | words_b
    return len(words_a & words_b) / len(union)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class ReasoningBankStrategy(MemoryEvolutionStrategy):
    """Structured memory evolution with semantic deduplication and recall.

    Stores memories as ReasoningEntry objects (title/description/content).
    Deduplicates entries by semantic similarity when an embeddings provider
    is available, falling back to keyword matching.

    Args:
        embeddings: Optional embeddings provider with
            ``async embed(text) -> list[float]``.  Falls back to keyword
            matching when ``None``.
        similarity_threshold: Cosine similarity threshold for deduplication.
            Default ``0.85``.
    """

    name: str = "reasoning_bank"

    def __init__(
        self,
        *,
        embeddings: Any | None = None,
        similarity_threshold: float = 0.85,
    ) -> None:
        self._embeddings = embeddings
        self._similarity_threshold = similarity_threshold
        self._entries: list[ReasoningEntry] = []
        # Text-content → vector cache (shared across evolve() calls)
        self._vector_cache: dict[str, list[float]] = {}
        # item_id → entry-text cache to avoid recomputing _entry_text() per call
        self._item_text_cache: dict[str, str] = {}

    # -- internals ----------------------------------------------------------

    async def _get_embedding(self, text: str) -> list[float] | None:
        """Get embedding for *text* (cached).  Returns ``None`` if no provider."""
        if self._embeddings is None:
            return None
        if text in self._vector_cache:
            return self._vector_cache[text]
        vec: list[float] = await self._embeddings.embed(text)
        self._vector_cache[text] = vec
        return vec

    async def _similarity(self, text_a: str, text_b: str) -> float:
        """Compute similarity between two texts."""
        if self._embeddings is not None:
            vec_a = await self._get_embedding(text_a)
            vec_b = await self._get_embedding(text_b)
            if vec_a is not None and vec_b is not None:
                return _cosine_similarity(vec_a, vec_b)
        return _keyword_similarity(text_a, text_b)

    # -- public API ---------------------------------------------------------

    async def evolve(
        self,
        items: list[MemoryItem],
        *,
        context: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Deduplicate items by semantic similarity.

        When two items exceed ``similarity_threshold``, the entry with more
        content is kept and the shorter one is dropped.

        Args:
            items: Input memory items to evolve.
            context: Optional context dict.

        Returns:
            Deduplicated list of memory items.
        """
        if not items:
            return []

        entries = [_parse_entry(item) for item in items]
        item_map = {item.id: item for item in items}

        keep = [True] * len(entries)

        for i in range(len(entries)):
            if not keep[i]:
                continue
            # Cache entry text by item_id to avoid recomputing across evolve() calls.
            item_id_i = entries[i].item_id
            if item_id_i not in self._item_text_cache:
                self._item_text_cache[item_id_i] = _entry_text(entries[i])
            text_i = self._item_text_cache[item_id_i]
            for j in range(i + 1, len(entries)):
                if not keep[j]:
                    continue
                item_id_j = entries[j].item_id
                if item_id_j not in self._item_text_cache:
                    self._item_text_cache[item_id_j] = _entry_text(entries[j])
                text_j = self._item_text_cache[item_id_j]
                sim = await self._similarity(text_i, text_j)
                if sim >= self._similarity_threshold:
                    # Keep the longer / more detailed entry
                    if len(text_j) > len(text_i):
                        keep[i] = False
                        break
                    else:
                        keep[j] = False

        self._entries = [entries[i] for i in range(len(entries)) if keep[i]]
        return [item_map[entries[i].item_id] for i in range(len(entries)) if keep[i]]

    async def recall(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[ReasoningEntry]:
        """Retrieve relevant entries by semantic search.

        Uses embeddings when available, falls back to keyword matching
        (case-insensitive word overlap across title, description, content).

        Args:
            query: The search query.
            top_k: Maximum number of entries to return.  Default ``5``.

        Returns:
            Most relevant entries sorted by relevance (highest first).
        """
        if not self._entries:
            return []

        scored: list[tuple[float, ReasoningEntry]] = []
        for entry in self._entries:
            text = _entry_text(entry)
            sim = await self._similarity(query, text)
            scored.append((sim, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]


__all__ = ["ReasoningBankStrategy", "ReasoningEntry"]
