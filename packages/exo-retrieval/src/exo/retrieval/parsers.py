"""Document parsers for extracting text from common file formats.

A ``Parser`` takes a file source (string content, bytes, or file path) and
produces a ``Document`` with extracted text content.

Parsers:
- ``TextParser``: passthrough for plain text.
- ``MarkdownParser``: strips formatting, preserves structure.
- ``JSONParser``: flattens JSON to readable text with key paths.
- ``PDFParser``: extracts text from PDF files (requires optional ``pymupdf``).
"""

from __future__ import annotations

import abc
import json
import re
import uuid
from pathlib import Path
from typing import Any

from exo.retrieval.types import Document, RetrievalError  # pyright: ignore[reportMissingImports]


class Parser(abc.ABC):
    """Abstract base class for document parsers.

    Subclasses must implement ``parse`` to extract text from a source.
    """

    @abc.abstractmethod
    def parse(self, source: str | bytes | Path) -> Document:
        """Parse a source into a Document.

        Args:
            source: A string (text content or file path), bytes, or Path object.

        Returns:
            A ``Document`` with extracted text content.
        """


def _read_source(source: str | bytes | Path) -> tuple[str, dict[str, Any]]:
    """Read source content and build metadata.

    Returns:
        A tuple of (text_content, metadata).
    """
    metadata: dict[str, Any] = {}
    if isinstance(source, Path):
        metadata["source"] = str(source)
        return source.read_text(encoding="utf-8"), metadata
    if isinstance(source, bytes):
        return source.decode("utf-8"), metadata
    return source, metadata


def _read_source_bytes(source: str | bytes | Path) -> tuple[bytes, dict[str, Any]]:
    """Read source as bytes and build metadata.

    Returns:
        A tuple of (raw_bytes, metadata).
    """
    metadata: dict[str, Any] = {}
    if isinstance(source, Path):
        metadata["source"] = str(source)
        return source.read_bytes(), metadata
    if isinstance(source, bytes):
        return source, metadata
    return source.encode("utf-8"), metadata


class TextParser(Parser):
    """Passthrough parser for plain text.

    Returns the input text as-is in a Document.
    """

    def parse(self, source: str | bytes | Path) -> Document:
        """Parse plain text into a Document.

        Args:
            source: Plain text as a string, bytes, or file path.

        Returns:
            A ``Document`` with the text content unchanged.
        """
        text, metadata = _read_source(source)
        metadata["format"] = "text"
        return Document(id=str(uuid.uuid4()), content=text, metadata=metadata)


class MarkdownParser(Parser):
    """Parser that strips Markdown formatting while preserving structure.

    Headings are converted to plain text lines, links retain their display
    text, and inline formatting (bold, italic, code) is removed.
    """

    _PATTERNS: list[tuple[re.Pattern[str], str]] = [
        # Images: ![alt](url) -> alt
        (re.compile(r"!\[([^\]]*)\]\([^)]*\)"), r"\1"),
        # Links: [text](url) -> text
        (re.compile(r"\[([^\]]*)\]\([^)]*\)"), r"\1"),
        # Bold/italic: ***text***, **text**, *text*, ___text___, __text__, _text_
        (re.compile(r"\*{3}(.+?)\*{3}"), r"\1"),
        (re.compile(r"\*{2}(.+?)\*{2}"), r"\1"),
        (re.compile(r"\*(.+?)\*"), r"\1"),
        (re.compile(r"_{3}(.+?)_{3}"), r"\1"),
        (re.compile(r"_{2}(.+?)_{2}"), r"\1"),
        (re.compile(r"(?<!\w)_(.+?)_(?!\w)"), r"\1"),
        # Strikethrough: ~~text~~ -> text
        (re.compile(r"~~(.+?)~~"), r"\1"),
        # Inline code: `code` -> code
        (re.compile(r"`(.+?)`"), r"\1"),
        # Headings: # Heading -> Heading
        (re.compile(r"^#{1,6}\s+", re.MULTILINE), ""),
        # Blockquotes: > text -> text
        (re.compile(r"^>\s?", re.MULTILINE), ""),
        # Unordered list markers: - item, * item, + item -> item
        (re.compile(r"^[\s]*[-*+]\s+", re.MULTILINE), ""),
        # Ordered list markers: 1. item -> item
        (re.compile(r"^[\s]*\d+\.\s+", re.MULTILINE), ""),
        # Horizontal rules
        (re.compile(r"^[-*_]{3,}\s*$", re.MULTILINE), ""),
        # HTML tags
        (re.compile(r"<[^>]+>"), ""),
    ]

    def parse(self, source: str | bytes | Path) -> Document:
        """Parse Markdown into a Document with formatting stripped.

        Args:
            source: Markdown text as a string, bytes, or file path.

        Returns:
            A ``Document`` with Markdown formatting removed.
        """
        text, metadata = _read_source(source)
        metadata["format"] = "markdown"

        # Remove fenced code block markers but keep content
        text = re.sub(r"^```[^\n]*\n", "", text, flags=re.MULTILINE)
        text = re.sub(r"^```\s*$", "", text, flags=re.MULTILINE)

        for pattern, replacement in self._PATTERNS:
            text = pattern.sub(replacement, text)

        # Collapse multiple blank lines into one
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        return Document(id=str(uuid.uuid4()), content=text, metadata=metadata)


_JSON_MAX_DEPTH = 64  # default recursion cap for JSONParser._flatten


class JSONParser(Parser):
    """Parser that flattens JSON into readable text with key paths.

    Nested structures are represented with dot-separated paths, and arrays
    use bracket notation. For example::

        {"user": {"name": "Alice", "tags": ["a", "b"]}}

    becomes::

        user.name: Alice
        user.tags[0]: a
        user.tags[1]: b

    Args:
        max_depth: Maximum nesting depth before a ``RetrievalError`` is raised
            (default 64).  Protects against ``RecursionError`` on pathological
            inputs.
    """

    def __init__(self, max_depth: int = _JSON_MAX_DEPTH) -> None:
        self._max_depth = max_depth

    def parse(self, source: str | bytes | Path) -> Document:
        """Parse JSON into a Document with flattened key paths.

        Args:
            source: JSON text as a string, bytes, or file path.

        Returns:
            A ``Document`` with flattened key-value text.

        Raises:
            RetrievalError: If the JSON is nested deeper than *max_depth*.
        """
        text, metadata = _read_source(source)
        metadata["format"] = "json"

        data = json.loads(text)
        lines = list(self._flatten(data, "", 0))
        content = "\n".join(lines)

        return Document(id=str(uuid.uuid4()), content=content, metadata=metadata)

    def _flatten(self, obj: Any, prefix: str, depth: int) -> list[str]:
        """Recursively flatten a JSON object into key-path: value lines.

        Args:
            obj: Current JSON value.
            prefix: Dot-path prefix accumulated so far.
            depth: Current recursion depth (0-based).

        Raises:
            RetrievalError: If *depth* exceeds ``self._max_depth``.
        """
        if depth > self._max_depth:
            raise RetrievalError(
                f"JSON nesting depth exceeds maximum of {self._max_depth}.",
                context={"max_depth": self._max_depth, "prefix": prefix},
                hint=(
                    f"Reduce nesting in the input JSON or increase max_depth"
                    f" (current limit: {self._max_depth})."
                ),
            )
        lines: list[str] = []
        if isinstance(obj, dict):
            for key, value in obj.items():
                path = f"{prefix}.{key}" if prefix else key
                lines.extend(self._flatten(value, path, depth + 1))
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                path = f"{prefix}[{i}]"
                lines.extend(self._flatten(value, path, depth + 1))
        else:
            display = str(obj) if obj is not None else "null"
            lines.append(f"{prefix}: {display}")
        return lines


class PDFParser(Parser):
    """Parser that extracts text from PDF files.

    Requires the optional ``pymupdf`` package. Install with::

        pip install pymupdf
    """

    def parse(self, source: str | bytes | Path) -> Document:
        """Parse a PDF into a Document with extracted text.

        Args:
            source: PDF content as bytes, a file path string, or a Path object.

        Returns:
            A ``Document`` with text extracted from all pages.

        Raises:
            ImportError: If ``pymupdf`` is not installed.
        """
        try:
            import pymupdf  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise ImportError(
                "pymupdf is required for PDF parsing. Install it with: pip install pymupdf"
            ) from exc

        raw_bytes, metadata = _read_source_bytes(source)
        metadata["format"] = "pdf"

        doc = pymupdf.open(stream=raw_bytes, filetype="pdf")
        pages: list[str] = []
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            if page_text.strip():
                pages.append(page_text)
            metadata.setdefault("page_count", 0)
            metadata["page_count"] = page_num + 1
        doc.close()

        content = "\n\n".join(pages)
        return Document(id=str(uuid.uuid4()), content=content, metadata=metadata)
