"""Tests for Chunker ABC, CharacterChunker, ParagraphChunker, and TokenChunker."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from exo.retrieval.chunker import (
    CharacterChunker,
    Chunker,
    ParagraphChunker,
    TokenChunker,
    _WhitespaceEncoder,
)
from exo.retrieval.types import Chunk, Document, RetrievalError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(
    content: str = "hello world",
    doc_id: str = "doc-1",
    metadata: dict[str, Any] | None = None,
) -> Document:
    """Build a Document with sensible defaults."""
    return Document(id=doc_id, content=content, metadata=metadata or {})


# ---------------------------------------------------------------------------
# Chunker ABC
# ---------------------------------------------------------------------------


class TestChunkerABC:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            Chunker()  # type: ignore[abstract]

    def test_concrete_subclass(self) -> None:
        class Dummy(Chunker):
            def chunk(self, document: Document) -> list[Chunk]:
                return []

        d = Dummy()
        assert isinstance(d, Chunker)


# ---------------------------------------------------------------------------
# CharacterChunker
# ---------------------------------------------------------------------------


class TestCharacterChunker:
    def test_empty_document(self) -> None:
        chunker = CharacterChunker(chunk_size=10, chunk_overlap=0)
        result = chunker.chunk(_doc(content=""))
        assert result == []

    def test_single_chunk(self) -> None:
        chunker = CharacterChunker(chunk_size=100, chunk_overlap=0)
        doc = _doc(content="short text")
        result = chunker.chunk(doc)
        assert len(result) == 1
        assert result[0].content == "short text"
        assert result[0].start == 0
        assert result[0].end == 10
        assert result[0].index == 0
        assert result[0].document_id == "doc-1"

    def test_multiple_chunks_no_overlap(self) -> None:
        chunker = CharacterChunker(chunk_size=5, chunk_overlap=0)
        doc = _doc(content="abcdefghij")  # 10 chars
        result = chunker.chunk(doc)
        assert len(result) == 2
        assert result[0].content == "abcde"
        assert result[0].start == 0
        assert result[0].end == 5
        assert result[1].content == "fghij"
        assert result[1].start == 5
        assert result[1].end == 10

    def test_overlap(self) -> None:
        chunker = CharacterChunker(chunk_size=6, chunk_overlap=2)
        doc = _doc(content="abcdefghijkl")  # 12 chars, step=4
        result = chunker.chunk(doc)
        assert len(result) == 3
        assert result[0].content == "abcdef"
        assert result[0].start == 0
        assert result[1].content == "efghij"
        assert result[1].start == 4
        assert result[2].content == "ijkl"
        assert result[2].start == 8

    def test_preserves_offsets(self) -> None:
        chunker = CharacterChunker(chunk_size=5, chunk_overlap=0)
        text = "hello world"
        doc = _doc(content=text)
        result = chunker.chunk(doc)
        # Verify we can reconstruct from offsets
        for c in result:
            assert text[c.start : c.end] == c.content

    def test_preserves_metadata(self) -> None:
        chunker = CharacterChunker(chunk_size=100)
        doc = _doc(content="text", metadata={"source": "test.txt"})
        result = chunker.chunk(doc)
        assert result[0].metadata == {"source": "test.txt"}

    def test_indexes_are_sequential(self) -> None:
        chunker = CharacterChunker(chunk_size=3, chunk_overlap=0)
        doc = _doc(content="abcdefghi")
        result = chunker.chunk(doc)
        assert [c.index for c in result] == [0, 1, 2]

    def test_invalid_chunk_size(self) -> None:
        with pytest.raises(RetrievalError, match="chunk_size must be positive"):
            CharacterChunker(chunk_size=0)

    def test_invalid_overlap_negative(self) -> None:
        with pytest.raises(RetrievalError, match="chunk_overlap must be non-negative"):
            CharacterChunker(chunk_size=10, chunk_overlap=-1)

    def test_invalid_overlap_too_large(self) -> None:
        with pytest.raises(RetrievalError, match="chunk_overlap must be less than"):
            CharacterChunker(chunk_size=10, chunk_overlap=10)

    def test_exact_size_document(self) -> None:
        chunker = CharacterChunker(chunk_size=5, chunk_overlap=0)
        doc = _doc(content="abcde")
        result = chunker.chunk(doc)
        assert len(result) == 1
        assert result[0].content == "abcde"


# ---------------------------------------------------------------------------
# ParagraphChunker
# ---------------------------------------------------------------------------


class TestParagraphChunker:
    def test_empty_document(self) -> None:
        chunker = ParagraphChunker(chunk_size=100)
        result = chunker.chunk(_doc(content=""))
        assert result == []

    def test_single_paragraph(self) -> None:
        chunker = ParagraphChunker(chunk_size=100)
        doc = _doc(content="A single paragraph with no breaks.")
        result = chunker.chunk(doc)
        assert len(result) == 1
        assert result[0].content == "A single paragraph with no breaks."
        assert result[0].start == 0

    def test_multiple_paragraphs_fit_in_one_chunk(self) -> None:
        chunker = ParagraphChunker(chunk_size=100)
        doc = _doc(content="First paragraph.\n\nSecond paragraph.")
        result = chunker.chunk(doc)
        assert len(result) == 1
        assert result[0].content == "First paragraph.\n\nSecond paragraph."

    def test_paragraphs_split_when_exceeding_size(self) -> None:
        chunker = ParagraphChunker(chunk_size=20)
        text = "First paragraph.\n\nSecond paragraph."
        doc = _doc(content=text)
        result = chunker.chunk(doc)
        assert len(result) == 2
        assert result[0].content == "First paragraph."
        assert result[1].content == "Second paragraph."

    def test_preserves_offsets(self) -> None:
        chunker = ParagraphChunker(chunk_size=20)
        text = "Hello.\n\nWorld.\n\nEnd."
        doc = _doc(content=text)
        result = chunker.chunk(doc)
        for c in result:
            assert text[c.start : c.end] == c.content

    def test_oversized_paragraph_kept_intact(self) -> None:
        """A single paragraph larger than chunk_size is not split."""
        chunker = ParagraphChunker(chunk_size=5)
        doc = _doc(content="This is a very long paragraph.")
        result = chunker.chunk(doc)
        assert len(result) == 1
        assert result[0].content == "This is a very long paragraph."

    def test_multiple_blank_lines_as_separator(self) -> None:
        chunker = ParagraphChunker(chunk_size=10)
        text = "Part A.\n\n\n\nPart B."
        doc = _doc(content=text)
        result = chunker.chunk(doc)
        assert len(result) == 2
        assert result[0].content == "Part A."
        assert result[1].content == "Part B."

    def test_preserves_metadata(self) -> None:
        chunker = ParagraphChunker(chunk_size=100)
        doc = _doc(content="text", metadata={"key": "val"})
        result = chunker.chunk(doc)
        assert result[0].metadata == {"key": "val"}

    def test_indexes_are_sequential(self) -> None:
        chunker = ParagraphChunker(chunk_size=10)
        doc = _doc(content="AAA.\n\nBBB.\n\nCCC.")
        result = chunker.chunk(doc)
        assert [c.index for c in result] == list(range(len(result)))

    def test_invalid_chunk_size(self) -> None:
        with pytest.raises(RetrievalError, match="chunk_size must be positive"):
            ParagraphChunker(chunk_size=0)

    def test_three_paragraphs_two_fit(self) -> None:
        """Two small paragraphs fit, third starts a new chunk."""
        chunker = ParagraphChunker(chunk_size=25)
        text = "Hello.\n\nWorld.\n\nThis is a longer paragraph."
        doc = _doc(content=text)
        result = chunker.chunk(doc)
        assert len(result) == 2
        assert result[0].content == "Hello.\n\nWorld."
        assert result[1].content == "This is a longer paragraph."


# ---------------------------------------------------------------------------
# TokenChunker
# ---------------------------------------------------------------------------


class TestTokenChunker:
    def test_empty_document(self) -> None:
        chunker = TokenChunker(chunk_size=10, chunk_overlap=0)
        result = chunker.chunk(_doc(content=""))
        assert result == []

    def test_single_chunk_whitespace_fallback(self) -> None:
        """Without tiktoken, falls back to whitespace tokenizer."""
        chunker = TokenChunker(chunk_size=100, chunk_overlap=0)
        doc = _doc(content="hello world foo")
        result = chunker.chunk(doc)
        assert len(result) == 1
        assert result[0].content == "hello world foo"
        assert result[0].start == 0

    def test_multiple_chunks_whitespace_fallback(self) -> None:
        chunker = TokenChunker(chunk_size=2, chunk_overlap=0)
        doc = _doc(content="a b c d")
        result = chunker.chunk(doc)
        assert len(result) == 2
        assert result[0].content == "a b"
        assert result[1].content == "c d"

    def test_overlap_whitespace_fallback(self) -> None:
        chunker = TokenChunker(chunk_size=3, chunk_overlap=1)
        doc = _doc(content="a b c d e")
        result = chunker.chunk(doc)
        # step = 2, tokens: a b c d e
        # chunk0: a b c (tokens 0-2)
        # chunk1: c d e (tokens 2-4)
        assert len(result) == 2
        assert "a" in result[0].content
        assert "e" in result[1].content

    def test_preserves_offsets_whitespace_fallback(self) -> None:
        chunker = TokenChunker(chunk_size=2, chunk_overlap=0)
        text = "hello world foo bar"
        doc = _doc(content=text)
        result = chunker.chunk(doc)
        for c in result:
            assert text[c.start : c.end] == c.content

    def test_preserves_metadata(self) -> None:
        chunker = TokenChunker(chunk_size=100)
        doc = _doc(content="text", metadata={"k": "v"})
        result = chunker.chunk(doc)
        assert result[0].metadata == {"k": "v"}

    def test_indexes_are_sequential(self) -> None:
        chunker = TokenChunker(chunk_size=2, chunk_overlap=0)
        doc = _doc(content="a b c d e f")
        result = chunker.chunk(doc)
        assert [c.index for c in result] == list(range(len(result)))

    def test_invalid_chunk_size(self) -> None:
        with pytest.raises(RetrievalError, match="chunk_size must be positive"):
            TokenChunker(chunk_size=0)

    def test_invalid_overlap_negative(self) -> None:
        with pytest.raises(RetrievalError, match="chunk_overlap must be non-negative"):
            TokenChunker(chunk_size=10, chunk_overlap=-1)

    def test_invalid_overlap_too_large(self) -> None:
        with pytest.raises(RetrievalError, match="chunk_overlap must be less than"):
            TokenChunker(chunk_size=10, chunk_overlap=10)

    def test_uses_tiktoken_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify tiktoken encoder is used when importable."""
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1, 2, 3]
        mock_enc.decode.side_effect = lambda toks: {1: "hello", 2: " ", 3: "world"}[toks[0]]

        mock_tiktoken = MagicMock()
        mock_tiktoken.get_encoding.return_value = mock_enc

        import sys

        sys.modules["tiktoken"] = mock_tiktoken
        try:
            chunker = TokenChunker(chunk_size=100, chunk_overlap=0)
            chunker._encoder = None  # Reset cached encoder
            doc = _doc(content="hello world")
            result = chunker.chunk(doc)
            assert len(result) == 1
            mock_tiktoken.get_encoding.assert_called_once_with("cl100k_base")
        finally:
            del sys.modules["tiktoken"]

    def test_whitespace_only_document(self) -> None:
        chunker = TokenChunker(chunk_size=5, chunk_overlap=0)
        doc = _doc(content="   ")
        result = chunker.chunk(doc)
        assert result == []


# ---------------------------------------------------------------------------
# WhitespaceEncoder
# ---------------------------------------------------------------------------


class TestWhitespaceEncoder:
    def test_basic_encoding(self) -> None:
        enc = _WhitespaceEncoder()
        spans = enc.encode_with_offsets("hello world")
        assert len(spans) == 2
        assert spans[0] == (0, 0, 5)  # "hello"
        assert spans[1] == (0, 6, 11)  # "world"

    def test_empty_string(self) -> None:
        enc = _WhitespaceEncoder()
        spans = enc.encode_with_offsets("")
        assert spans == []

    def test_multiple_spaces(self) -> None:
        enc = _WhitespaceEncoder()
        spans = enc.encode_with_offsets("a   b")
        assert len(spans) == 2
        assert spans[0] == (0, 0, 1)
        assert spans[1] == (0, 4, 5)
