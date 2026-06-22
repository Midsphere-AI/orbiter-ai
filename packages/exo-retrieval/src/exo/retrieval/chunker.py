"""Text chunking strategies for splitting documents into retrieval-friendly chunks.

A ``Chunker`` takes a ``Document`` and produces a list of ``Chunk`` objects,
each preserving character offsets into the original document content.

Strategies:
- ``CharacterChunker``: fixed character-count windows with configurable overlap.
- ``ParagraphChunker``: splits at paragraph boundaries, respecting a size limit.
- ``TokenChunker``: splits by token count (requires optional ``tiktoken``).
"""

from __future__ import annotations

import abc
import logging
import re

from exo.retrieval.types import (  # pyright: ignore[reportMissingImports]
    Chunk,
    Document,
    RetrievalError,
)

logger = logging.getLogger(__name__)


class Chunker(abc.ABC):
    """Abstract base class for text chunkers.

    Subclasses must implement ``chunk`` to split a document into chunks.
    """

    @abc.abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into chunks.

        Args:
            document: The document to split.

        Returns:
            A list of ``Chunk`` objects with character offsets preserved.
        """


class CharacterChunker(Chunker):
    """Splits text by character count with configurable overlap.

    Args:
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise RetrievalError(
                "chunk_size must be positive",
                hint="chunk_size must be a positive integer; chunk_overlap must be less than chunk_size.",
            )
        if chunk_overlap < 0:
            raise RetrievalError(
                "chunk_overlap must be non-negative",
                hint="chunk_size must be a positive integer; chunk_overlap must be less than chunk_size.",
            )
        if chunk_overlap >= chunk_size:
            raise RetrievalError(
                "chunk_overlap must be less than chunk_size",
                hint="chunk_size must be a positive integer; chunk_overlap must be less than chunk_size.",
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into fixed-size character chunks with overlap.

        Args:
            document: The document to split.

        Returns:
            A list of ``Chunk`` objects.
        """
        text = document.content
        if not text:
            return []

        chunks: list[Chunk] = []
        start = 0
        index = 0
        step = self.chunk_size - self.chunk_overlap

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(
                Chunk(
                    document_id=document.id,
                    index=index,
                    content=text[start:end],
                    start=start,
                    end=end,
                    metadata=dict(document.metadata),
                )
            )
            index += 1
            start += step

            # Avoid creating a tiny trailing chunk that's entirely within overlap
            if start < len(text) and start + self.chunk_overlap >= len(text):
                # Remaining text is smaller than overlap — already covered
                break

        return chunks


class ParagraphChunker(Chunker):
    """Splits text at paragraph boundaries, respecting a maximum chunk size.

    Paragraphs are delimited by one or more blank lines. If a single paragraph
    exceeds ``chunk_size``, it is included as its own chunk (not split mid-word).

    Args:
        chunk_size: Maximum number of characters per chunk.
    """

    def __init__(self, chunk_size: int = 1000) -> None:
        if chunk_size <= 0:
            raise RetrievalError(
                "chunk_size must be positive",
                hint="chunk_size must be a positive integer.",
            )
        self.chunk_size = chunk_size

    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document at paragraph boundaries.

        Args:
            document: The document to split.

        Returns:
            A list of ``Chunk`` objects.
        """
        text = document.content
        if not text:
            return []

        # Split on blank lines (one or more empty lines)
        paragraphs = re.split(r"\n\s*\n", text)

        chunks: list[Chunk] = []
        current_parts: list[str] = []
        current_start: int | None = None
        current_end: int = 0
        current_len = 0
        para_positions: list[tuple[str, int]] = []

        # Find each paragraph's start position in the original text
        search_from = 0
        for para in paragraphs:
            idx = text.find(para, search_from)
            para_positions.append((para, idx))
            search_from = idx + len(para)

        index = 0
        for para, para_start in para_positions:
            para_end = para_start + len(para)
            sep = "\n\n" if current_parts else ""
            would_be = current_len + len(sep) + len(para)

            if current_parts and would_be > self.chunk_size:
                # Flush current buffer — use tracked end, not re-joined length
                content = "\n\n".join(current_parts)
                assert current_start is not None
                chunks.append(
                    Chunk(
                        document_id=document.id,
                        index=index,
                        content=content,
                        start=current_start,
                        end=current_end,
                        metadata=dict(document.metadata),
                    )
                )
                index += 1
                current_parts = [para]
                current_start = para_start
                current_end = para_end
                current_len = len(para)
            else:
                if not current_parts:
                    current_start = para_start
                current_parts.append(para)
                current_end = para_end
                current_len = current_len + len(sep) + len(para)

        # Flush remaining
        if current_parts:
            content = "\n\n".join(current_parts)
            assert current_start is not None
            chunks.append(
                Chunk(
                    document_id=document.id,
                    index=index,
                    content=content,
                    start=current_start,
                    end=current_end,
                    metadata=dict(document.metadata),
                )
            )

        return chunks


class TokenChunker(Chunker):
    """Splits text by token count using ``tiktoken``.

    Requires the optional ``tiktoken`` package. Falls back to a simple
    whitespace tokenizer if ``tiktoken`` is not available.

    Args:
        chunk_size: Maximum number of tokens per chunk.
        chunk_overlap: Number of overlapping tokens between consecutive chunks.
        encoding: The tiktoken encoding name (default ``"cl100k_base"``).
    """

    def __init__(
        self,
        chunk_size: int = 200,
        chunk_overlap: int = 20,
        encoding: str = "cl100k_base",
    ) -> None:
        if chunk_size <= 0:
            raise RetrievalError(
                "chunk_size must be positive",
                hint="chunk_size must be a positive integer; chunk_overlap must be less than chunk_size.",
            )
        if chunk_overlap < 0:
            raise RetrievalError(
                "chunk_overlap must be non-negative",
                hint="chunk_size must be a positive integer; chunk_overlap must be less than chunk_size.",
            )
        if chunk_overlap >= chunk_size:
            raise RetrievalError(
                "chunk_overlap must be less than chunk_size",
                hint="chunk_size must be a positive integer; chunk_overlap must be less than chunk_size.",
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._encoding_name = encoding
        self._encoder: _TokenEncoder | None = None

    def _get_encoder(self) -> _TokenEncoder:
        """Lazily initialise the token encoder."""
        if self._encoder is None:
            try:
                import tiktoken  # pyright: ignore[reportMissingImports]

                enc = tiktoken.get_encoding(self._encoding_name)
                self._encoder = _TiktokenEncoder(enc)
            except ModuleNotFoundError:
                logger.warning(
                    "tiktoken is not installed; TokenChunker is using a whitespace fallback"
                    " tokenizer. Install tiktoken for accurate token-based chunking:"
                    " pip install tiktoken"
                )
                self._encoder = _WhitespaceEncoder()
        return self._encoder

    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into token-count-based chunks.

        Args:
            document: The document to split.

        Returns:
            A list of ``Chunk`` objects with character offsets preserved.
        """
        text = document.content
        if not text or not text.strip():
            return []

        encoder = self._get_encoder()
        token_spans = encoder.encode_with_offsets(text)

        if not token_spans:
            return []

        chunks: list[Chunk] = []
        index = 0
        step = self.chunk_size - self.chunk_overlap
        i = 0

        while i < len(token_spans):
            window = token_spans[i : i + self.chunk_size]
            start_char = window[0][1]
            end_char = window[-1][2]
            content = text[start_char:end_char]

            # Strip leading whitespace from chunk boundaries (token encoders
            # like tiktoken may attach leading spaces to tokens, causing
            # non-first chunks to start with whitespace).
            lstripped = content.lstrip()
            if lstripped:
                start_char += len(content) - len(lstripped)
                content = lstripped

            chunks.append(
                Chunk(
                    document_id=document.id,
                    index=index,
                    content=content,
                    start=start_char,
                    end=end_char,
                    metadata=dict(document.metadata),
                )
            )
            index += 1
            i += step

            # Avoid tiny trailing chunk entirely within overlap
            if i < len(token_spans) and i + self.chunk_overlap >= len(token_spans):
                break

        return chunks


# ---------------------------------------------------------------------------
# Internal token encoder abstraction
# ---------------------------------------------------------------------------


class _TokenEncoder(abc.ABC):
    """Internal ABC for token encoders."""

    @abc.abstractmethod
    def encode_with_offsets(self, text: str) -> list[tuple[int, int, int]]:
        """Encode text and return (token_id, start_char, end_char) triples."""


class _TiktokenEncoder(_TokenEncoder):
    """Encoder backed by tiktoken."""

    def __init__(self, enc: object) -> None:
        self._enc = enc

    def encode_with_offsets(self, text: str) -> list[tuple[int, int, int]]:
        enc = self._enc
        tokens: list[int] = enc.encode(text)  # type: ignore[union-attr]
        spans: list[tuple[int, int, int]] = []
        offset = 0
        for tok_id in tokens:
            decoded: str = enc.decode([tok_id])  # type: ignore[union-attr]
            # Find the decoded text in the original starting from offset
            idx = text.find(decoded, offset)
            if idx == -1:
                # Token decode did not round-trip cleanly (e.g. byte-level BPE
                # tokens that form invalid UTF-8 substrings in isolation).
                # Fall back to current offset so the span is contiguous.
                logger.warning(
                    "_TiktokenEncoder: decoded token %r not found in text at offset %d;"
                    " falling back to current offset. Chunk character offsets may be"
                    " slightly inaccurate near this position.",
                    decoded,
                    offset,
                )
                idx = offset
            end = idx + len(decoded)
            spans.append((tok_id, idx, end))
            offset = end
        return spans


class _WhitespaceEncoder(_TokenEncoder):
    """Fallback encoder that splits on whitespace boundaries."""

    def encode_with_offsets(self, text: str) -> list[tuple[int, int, int]]:
        spans: list[tuple[int, int, int]] = []
        for match in re.finditer(r"\S+", text):
            spans.append((0, match.start(), match.end()))
        return spans
