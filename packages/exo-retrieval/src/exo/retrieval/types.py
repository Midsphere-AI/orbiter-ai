"""Core retrieval types: documents, chunks, results, and errors."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from exo.types import ExoError


class Document(BaseModel):
    """A document stored in a retrieval system.

    Attributes:
        id: Unique identifier for the document.
        content: Full text content of the document.
        metadata: Arbitrary metadata (e.g., source, author, tags).
        embedding: Optional pre-computed embedding vector.
    """

    id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None


class Chunk(BaseModel):
    """An immutable slice of a document for retrieval.

    Attributes:
        document_id: ID of the parent document.
        index: Position of this chunk within the document (0-based).
        content: Text content of the chunk.
        start: Character offset where the chunk begins in the document.
        end: Character offset where the chunk ends in the document.
        metadata: Arbitrary metadata inherited or derived from the document.
    """

    model_config = {"frozen": True}

    document_id: str
    index: int
    content: str
    start: int
    end: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """A scored chunk returned from a retrieval query.

    Attributes:
        chunk: The matched chunk.
        score: Similarity or relevance score (higher is better).
        metadata: Additional metadata from the retrieval process.
    """

    model_config = {"frozen": True}

    chunk: Chunk
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalError(ExoError):
    """Raised when a retrieval operation fails.

    Attributes:
        operation: The operation that failed (e.g., "embed", "search", "index").
        details: Additional context about the failure.
    """

    def __init__(
        self,
        message: str,
        *,
        operation: str = "",
        details: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        hint: str | None = None,
        doc: str | None = None,
    ) -> None:
        # Merge operation/details into context for structured ExoError rendering,
        # while keeping self.operation and self.details for backward compatibility.
        merged_context: dict[str, Any] = {}
        if operation:
            merged_context["operation"] = operation
        if details:
            merged_context.update(details)
        if context:
            merged_context.update(context)
        super().__init__(message, context=merged_context or None, hint=hint, doc=doc)
        self.operation = operation
        self.details = details or {}
