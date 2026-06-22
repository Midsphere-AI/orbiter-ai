"""Tests for Parser ABC, TextParser, MarkdownParser, JSONParser, and PDFParser."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from exo.retrieval.parsers import (
    JSONParser,
    MarkdownParser,
    Parser,
    PDFParser,
    TextParser,
)
from exo.retrieval.types import Document

# ---------------------------------------------------------------------------
# Parser ABC
# ---------------------------------------------------------------------------


class TestParserABC:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            Parser()  # type: ignore[abstract]

    def test_concrete_subclass(self) -> None:
        class Dummy(Parser):
            def parse(self, source: str | bytes | Path) -> Document:
                return Document(id="1", content="")

        d = Dummy()
        assert isinstance(d, Parser)


# ---------------------------------------------------------------------------
# TextParser
# ---------------------------------------------------------------------------


class TestTextParser:
    def test_string_input(self) -> None:
        parser = TextParser()
        doc = parser.parse("hello world")
        assert doc.content == "hello world"
        assert doc.metadata["format"] == "text"

    def test_bytes_input(self) -> None:
        parser = TextParser()
        doc = parser.parse(b"hello bytes")
        assert doc.content == "hello bytes"
        assert doc.metadata["format"] == "text"

    def test_path_input(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("file content", encoding="utf-8")
        parser = TextParser()
        doc = parser.parse(f)
        assert doc.content == "file content"
        assert doc.metadata["format"] == "text"
        assert doc.metadata["source"] == str(f)

    def test_empty_string(self) -> None:
        parser = TextParser()
        doc = parser.parse("")
        assert doc.content == ""

    def test_generates_unique_ids(self) -> None:
        parser = TextParser()
        doc1 = parser.parse("a")
        doc2 = parser.parse("b")
        assert doc1.id != doc2.id


# ---------------------------------------------------------------------------
# MarkdownParser
# ---------------------------------------------------------------------------


class TestMarkdownParser:
    def test_strips_headings(self) -> None:
        parser = MarkdownParser()
        doc = parser.parse("# Title\n\n## Subtitle\n\nBody text.")
        assert "# " not in doc.content
        assert "Title" in doc.content
        assert "Subtitle" in doc.content
        assert "Body text." in doc.content

    def test_strips_bold_and_italic(self) -> None:
        parser = MarkdownParser()
        doc = parser.parse("This is **bold** and *italic* text.")
        assert doc.content == "This is bold and italic text."

    def test_strips_bold_italic_combined(self) -> None:
        parser = MarkdownParser()
        doc = parser.parse("This is ***bold italic*** text.")
        assert doc.content == "This is bold italic text."

    def test_strips_strikethrough(self) -> None:
        parser = MarkdownParser()
        doc = parser.parse("This is ~~deleted~~ text.")
        assert doc.content == "This is deleted text."

    def test_strips_inline_code(self) -> None:
        parser = MarkdownParser()
        doc = parser.parse("Use `print()` to output.")
        assert doc.content == "Use print() to output."

    def test_strips_links(self) -> None:
        parser = MarkdownParser()
        doc = parser.parse("Visit [Exo](https://exo.ai) now.")
        assert doc.content == "Visit Exo now."
        assert "https://" not in doc.content

    def test_strips_images(self) -> None:
        parser = MarkdownParser()
        doc = parser.parse("![alt text](image.png)")
        assert doc.content == "alt text"

    def test_strips_blockquotes(self) -> None:
        parser = MarkdownParser()
        doc = parser.parse("> This is a quote.")
        assert doc.content == "This is a quote."

    def test_strips_unordered_list_markers(self) -> None:
        parser = MarkdownParser()
        doc = parser.parse("- Item 1\n- Item 2\n* Item 3")
        assert "Item 1" in doc.content
        assert "Item 2" in doc.content
        assert "Item 3" in doc.content
        assert "- " not in doc.content
        assert "* " not in doc.content

    def test_strips_ordered_list_markers(self) -> None:
        parser = MarkdownParser()
        doc = parser.parse("1. First\n2. Second")
        assert "First" in doc.content
        assert "Second" in doc.content
        assert "1. " not in doc.content

    def test_strips_fenced_code_blocks(self) -> None:
        parser = MarkdownParser()
        doc = parser.parse("```python\nprint('hello')\n```")
        assert "print('hello')" in doc.content
        assert "```" not in doc.content

    def test_strips_html_tags(self) -> None:
        parser = MarkdownParser()
        doc = parser.parse("Text with <strong>HTML</strong> inside.")
        assert doc.content == "Text with HTML inside."

    def test_collapses_blank_lines(self) -> None:
        parser = MarkdownParser()
        doc = parser.parse("Line 1\n\n\n\n\nLine 2")
        assert doc.content == "Line 1\n\nLine 2"

    def test_bytes_input(self) -> None:
        parser = MarkdownParser()
        doc = parser.parse(b"# Hello\n\nWorld")
        assert "Hello" in doc.content
        assert doc.metadata["format"] == "markdown"

    def test_path_input(self, tmp_path: Path) -> None:
        f = tmp_path / "test.md"
        f.write_text("# Test\n\nContent.", encoding="utf-8")
        parser = MarkdownParser()
        doc = parser.parse(f)
        assert "Test" in doc.content
        assert doc.metadata["source"] == str(f)

    def test_preserves_text_structure(self) -> None:
        md = "# Title\n\nParagraph one.\n\nParagraph two."
        parser = MarkdownParser()
        doc = parser.parse(md)
        assert "Title" in doc.content
        assert "Paragraph one." in doc.content
        assert "Paragraph two." in doc.content

    def test_empty_input(self) -> None:
        parser = MarkdownParser()
        doc = parser.parse("")
        assert doc.content == ""


# ---------------------------------------------------------------------------
# JSONParser
# ---------------------------------------------------------------------------


class TestJSONParser:
    def test_flat_object(self) -> None:
        parser = JSONParser()
        doc = parser.parse('{"name": "Alice", "age": 30}')
        assert "name: Alice" in doc.content
        assert "age: 30" in doc.content
        assert doc.metadata["format"] == "json"

    def test_nested_object(self) -> None:
        parser = JSONParser()
        data = json.dumps({"user": {"name": "Bob", "email": "bob@test.com"}})
        doc = parser.parse(data)
        assert "user.name: Bob" in doc.content
        assert "user.email: bob@test.com" in doc.content

    def test_array(self) -> None:
        parser = JSONParser()
        data = json.dumps({"tags": ["a", "b", "c"]})
        doc = parser.parse(data)
        assert "tags[0]: a" in doc.content
        assert "tags[1]: b" in doc.content
        assert "tags[2]: c" in doc.content

    def test_nested_array_of_objects(self) -> None:
        parser = JSONParser()
        data = json.dumps({"users": [{"name": "Alice"}, {"name": "Bob"}]})
        doc = parser.parse(data)
        assert "users[0].name: Alice" in doc.content
        assert "users[1].name: Bob" in doc.content

    def test_null_value(self) -> None:
        parser = JSONParser()
        doc = parser.parse('{"key": null}')
        assert "key: null" in doc.content

    def test_boolean_values(self) -> None:
        parser = JSONParser()
        doc = parser.parse('{"active": true, "deleted": false}')
        assert "active: True" in doc.content
        assert "deleted: False" in doc.content

    def test_top_level_array(self) -> None:
        parser = JSONParser()
        doc = parser.parse("[1, 2, 3]")
        assert "[0]: 1" in doc.content
        assert "[1]: 2" in doc.content
        assert "[2]: 3" in doc.content

    def test_deeply_nested(self) -> None:
        parser = JSONParser()
        data = json.dumps({"a": {"b": {"c": "deep"}}})
        doc = parser.parse(data)
        assert "a.b.c: deep" in doc.content

    def test_empty_object(self) -> None:
        parser = JSONParser()
        doc = parser.parse("{}")
        assert doc.content == ""

    def test_bytes_input(self) -> None:
        parser = JSONParser()
        doc = parser.parse(b'{"key": "value"}')
        assert "key: value" in doc.content

    def test_path_input(self, tmp_path: Path) -> None:
        f = tmp_path / "test.json"
        f.write_text('{"name": "test"}', encoding="utf-8")
        parser = JSONParser()
        doc = parser.parse(f)
        assert "name: test" in doc.content
        assert doc.metadata["source"] == str(f)

    def test_invalid_json_raises(self) -> None:
        parser = JSONParser()
        with pytest.raises(json.JSONDecodeError):
            parser.parse("not json")

    def test_max_depth_exceeded_raises(self) -> None:
        """Nesting deeper than max_depth must raise RetrievalError, not RecursionError."""
        from exo.retrieval.types import RetrievalError

        parser = JSONParser(max_depth=3)
        # Build JSON nested 4 levels deep: {"a": {"b": {"c": {"d": 1}}}}
        data = json.dumps({"a": {"b": {"c": {"d": 1}}}})
        with pytest.raises(RetrievalError, match="nesting depth exceeds"):
            parser.parse(data)

    def test_max_depth_at_limit_succeeds(self) -> None:
        """JSON nested exactly at max_depth must succeed."""
        parser = JSONParser(max_depth=3)
        # {"a": {"b": {"c": 1}}} — 3 levels deep (a.b.c is at depth 3 when counting from 0)
        data = json.dumps({"a": {"b": {"c": 1}}})
        doc = parser.parse(data)
        assert "a.b.c: 1" in doc.content

    def test_default_max_depth_handles_moderately_deep(self) -> None:
        """Default max_depth=64 must handle up to 64 nesting levels without error."""
        parser = JSONParser()  # default max_depth=64
        # Build a chain 10 levels deep — well within limit
        obj: dict = {}
        inner = obj
        for i in range(10):
            inner[f"k{i}"] = {}
            inner = inner[f"k{i}"]  # type: ignore[assignment]
        inner["leaf"] = "value"  # type: ignore[index]
        doc = parser.parse(json.dumps(obj))
        assert "value" in doc.content

    def test_custom_max_depth_default_is_backward_compatible(self) -> None:
        """JSONParser() with no args must behave identically to old code on normal inputs."""
        parser = JSONParser()
        data = json.dumps({"user": {"name": "Alice", "tags": ["a", "b"]}})
        doc = parser.parse(data)
        assert "user.name: Alice" in doc.content
        assert "user.tags[0]: a" in doc.content


# ---------------------------------------------------------------------------
# PDFParser
# ---------------------------------------------------------------------------


class TestPDFParser:
    def test_import_error_when_pymupdf_missing(self) -> None:
        parser = PDFParser()
        with patch.dict("sys.modules", {"pymupdf": None}):
            with pytest.raises(ImportError, match="pymupdf is required"):
                parser.parse(b"fake pdf bytes")

    def test_parse_pdf_bytes(self) -> None:
        """Test PDF parsing with a mocked pymupdf module."""
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Page 1 text"

        mock_doc = MagicMock()
        mock_doc.__iter__ = lambda self: iter([mock_page])
        mock_doc.__enter__ = lambda self: self
        mock_doc.__exit__ = lambda self, *a: None

        mock_pymupdf = MagicMock()
        mock_pymupdf.open.return_value = mock_doc

        with patch.dict("sys.modules", {"pymupdf": mock_pymupdf}):
            parser = PDFParser()
            doc = parser.parse(b"fake pdf bytes")

        assert doc.content == "Page 1 text"
        assert doc.metadata["format"] == "pdf"
        assert doc.metadata["page_count"] == 1

    def test_parse_multiple_pages(self) -> None:
        """Test PDF with multiple pages."""
        page1 = MagicMock()
        page1.get_text.return_value = "First page"
        page2 = MagicMock()
        page2.get_text.return_value = "Second page"

        mock_doc = MagicMock()
        mock_doc.__iter__ = lambda self: iter([page1, page2])

        mock_pymupdf = MagicMock()
        mock_pymupdf.open.return_value = mock_doc

        with patch.dict("sys.modules", {"pymupdf": mock_pymupdf}):
            parser = PDFParser()
            doc = parser.parse(b"fake pdf")

        assert "First page" in doc.content
        assert "Second page" in doc.content
        assert doc.metadata["page_count"] == 2

    def test_parse_skips_blank_pages(self) -> None:
        """Blank pages should not contribute content but still count."""
        page1 = MagicMock()
        page1.get_text.return_value = "Content"
        page2 = MagicMock()
        page2.get_text.return_value = "   \n  "

        mock_doc = MagicMock()
        mock_doc.__iter__ = lambda self: iter([page1, page2])

        mock_pymupdf = MagicMock()
        mock_pymupdf.open.return_value = mock_doc

        with patch.dict("sys.modules", {"pymupdf": mock_pymupdf}):
            parser = PDFParser()
            doc = parser.parse(b"fake pdf")

        assert doc.content == "Content"
        assert doc.metadata["page_count"] == 2

    def test_parse_from_path(self, tmp_path: Path) -> None:
        """Test PDF parsing from a file path."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"fake pdf content")

        mock_page = MagicMock()
        mock_page.get_text.return_value = "PDF text"

        mock_doc = MagicMock()
        mock_doc.__iter__ = lambda self: iter([mock_page])

        mock_pymupdf = MagicMock()
        mock_pymupdf.open.return_value = mock_doc

        with patch.dict("sys.modules", {"pymupdf": mock_pymupdf}):
            parser = PDFParser()
            doc = parser.parse(pdf_file)

        assert doc.content == "PDF text"
        assert doc.metadata["source"] == str(pdf_file)
        assert doc.metadata["format"] == "pdf"
