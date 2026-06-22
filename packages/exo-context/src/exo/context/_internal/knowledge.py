"""KnowledgeStore — in-memory artifact indexing with chunking and search.

Provides auto-indexing of artifacts written to a :class:`Workspace`.  Artifacts
are chunked into overlapping segments and stored for keyword-based search.  A
workspace can optionally attach a ``KnowledgeStore`` so that every write/update
auto-indexes the content, and every delete removes it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from exo.observability.logging import get_logger  # pyright: ignore[reportMissingImports]
from exo.types import ExoError  # pyright: ignore[reportMissingImports]

_log = get_logger(__name__)


class KnowledgeError(ExoError):
    """Raised for knowledge store operation errors."""


# ── Chunk ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Chunk:
    """A segment of an artifact's content."""

    artifact_name: str
    index: int
    content: str
    start: int
    end: int


# ── Search result ───────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A single search hit with relevance score."""

    chunk: Chunk
    score: float


# ── Chunker ─────────────────────────────────────────────────────────


def chunk_text(
    text: str,
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[str]:
    """Split *text* into overlapping fixed-stride segments.

    Slides a window of *chunk_size* characters over the text with a step of
    ``chunk_size - chunk_overlap`` characters.  Splitting is strictly by
    character offset — no paragraph or line-boundary detection is performed.
    """
    if chunk_size <= 0:
        raise KnowledgeError(
            "chunk_size must be positive.",
            hint="chunk_size must be >= 1 (e.g. chunk_size=512).",
        )
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise KnowledgeError(
            "chunk_overlap must be in [0, chunk_size).",
            hint=f"chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size ({chunk_size}).",
        )

    if len(text) <= chunk_size:
        return [text] if text else []

    chunks: list[str] = []
    pos = 0
    step = chunk_size - chunk_overlap

    while pos < len(text):
        end = min(pos + chunk_size, len(text))
        segment = text[pos:end]
        chunks.append(segment)
        if end >= len(text):
            break
        pos += step

    return chunks


# ── KnowledgeStore ──────────────────────────────────────────────────


class KnowledgeStore:
    """In-memory artifact index with chunking and keyword search.

    Parameters
    ----------
    chunk_size:
        Maximum characters per chunk.
    chunk_overlap:
        Overlap between consecutive chunks.
    """

    __slots__ = ("_chunk_overlap", "_chunk_size", "_chunks")

    def __init__(
        self,
        *,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._chunks: dict[str, list[Chunk]] = {}

    # ── Properties ───────────────────────────────────────────────────

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        return self._chunk_overlap

    # ── Index operations ─────────────────────────────────────────────

    def add(self, name: str, content: str) -> list[Chunk]:
        """Index an artifact's content.  Re-indexes if already present."""
        if not name:
            raise KnowledgeError(
                "artifact name is required.",
                hint="Pass a non-empty string as the artifact name.",
            )

        segments = chunk_text(
            content,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        chunks = [
            Chunk(
                artifact_name=name,
                index=i,
                content=seg,
                start=i * (self._chunk_size - self._chunk_overlap),
                end=min(
                    i * (self._chunk_size - self._chunk_overlap) + len(seg),
                    len(content),
                ),
            )
            for i, seg in enumerate(segments)
        ]
        self._chunks[name] = chunks
        _log.debug("indexed artifact %r: %d chunks from %d chars", name, len(chunks), len(content))
        return chunks

    def remove(self, name: str) -> bool:
        """Remove an artifact from the index.  Returns True if removed."""
        removed = self._chunks.pop(name, None) is not None
        if removed:
            _log.debug("removed artifact %r from index", name)
        return removed

    def get(self, name: str) -> list[Chunk]:
        """Get all chunks for an artifact.  Returns empty list if missing."""
        return list(self._chunks.get(name, []))

    def get_range(self, name: str, start: int, end: int) -> list[Chunk]:
        """Get chunks within a character range [start, end) for an artifact."""
        chunks = self._chunks.get(name, [])
        return [c for c in chunks if c.end > start and c.start < end]

    # ── Search ───────────────────────────────────────────────────────

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Keyword search across all indexed artifacts.

        Scores each chunk using ``sum(log(1 + tf))`` for each query term that
        appears in the chunk (raw term frequency, **not** normalised by chunk
        length).  Returns up to *top_k* results sorted by descending score.
        """
        if not query.strip():
            return []

        terms = _tokenize(query)
        if not terms:
            return []

        results: list[SearchResult] = []
        for chunks in self._chunks.values():
            for chunk in chunks:
                score = _score_chunk(chunk.content, terms)
                if score > 0:
                    results.append(SearchResult(chunk=chunk, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        top = results[:top_k]
        _log.debug("search query=%r found %d results (top_k=%d)", query, len(results), top_k)
        return top

    # ── Introspection ────────────────────────────────────────────────

    @property
    def artifact_names(self) -> list[str]:
        return list(self._chunks.keys())

    def total_chunks(self) -> int:
        return sum(len(cs) for cs in self._chunks.values())

    def __len__(self) -> int:
        return len(self._chunks)

    def __repr__(self) -> str:
        return (
            f"KnowledgeStore(artifacts={len(self)}, "
            f"chunks={self.total_chunks()}, "
            f"chunk_size={self._chunk_size})"
        )


# ── Scoring helpers ─────────────────────────────────────────────────

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenization."""
    return [m.group().lower() for m in _WORD_RE.finditer(text)]


def _score_chunk(content: str, query_terms: list[str]) -> float:
    """Score a chunk: ``sum(log(1 + tf))`` for each matching query term.

    ``tf`` is the raw count of the term in *content* (not normalised by chunk length).
    """
    tokens = _tokenize(content)
    if not tokens:
        return 0.0

    freq: dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1

    score = 0.0
    for term in query_terms:
        tf = freq.get(term, 0)
        if tf > 0:
            score += math.log(1 + tf)

    return score
