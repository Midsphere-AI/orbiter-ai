"""Tests for exo.eval.scorers — rule-based scorers."""

from __future__ import annotations

import json
from typing import Any

import pytest

from exo.eval.scorers import (  # pyright: ignore[reportMissingImports]
    FormatValidationScorer,
    OutputCompletenessScorer,
    OutputCorrectnessScorer,
    OutputLengthScorer,
    SchemaValidationScorer,
)

# ---------------------------------------------------------------------------
# FormatValidationScorer
# ---------------------------------------------------------------------------


class TestFormatValidationScorerInit:
    def test_default_json(self) -> None:
        s = FormatValidationScorer()
        assert s._format == "json"
        assert s._name == "format_json"

    def test_custom_name(self) -> None:
        s = FormatValidationScorer("xml", name="my_xml")
        assert s._name == "my_xml"

    def test_unsupported_format(self) -> None:
        with pytest.raises(ValueError, match="Unsupported format"):
            FormatValidationScorer("html")


class TestFormatJSON:
    async def test_valid_json(self) -> None:
        s = FormatValidationScorer("json")
        sr = await s.score("c1", None, '{"key": "value"}')
        assert sr.score == 1.0
        assert sr.details["format"] == "json"

    async def test_invalid_json(self) -> None:
        s = FormatValidationScorer("json")
        sr = await s.score("c1", None, "not json")
        assert sr.score == 0.0
        assert "error" in sr.details

    async def test_json_array(self) -> None:
        s = FormatValidationScorer("json")
        sr = await s.score("c1", None, "[1, 2, 3]")
        assert sr.score == 1.0


class TestFormatXML:
    async def test_valid_xml(self) -> None:
        s = FormatValidationScorer("xml")
        sr = await s.score("c1", None, "<root><child>text</child></root>")
        assert sr.score == 1.0

    async def test_invalid_xml(self) -> None:
        s = FormatValidationScorer("xml")
        sr = await s.score("c1", None, "not xml at all")
        assert sr.score == 0.0

    async def test_malformed_xml(self) -> None:
        s = FormatValidationScorer("xml")
        sr = await s.score("c1", None, "<root><unclosed>")
        assert sr.score == 0.0


class TestFormatMarkdown:
    async def test_with_heading(self) -> None:
        s = FormatValidationScorer("markdown")
        sr = await s.score("c1", None, "# Hello World\nSome content")
        assert sr.score == 1.0

    async def test_with_list(self) -> None:
        s = FormatValidationScorer("markdown")
        sr = await s.score("c1", None, "- item 1\n- item 2")
        assert sr.score == 1.0

    async def test_with_bold(self) -> None:
        s = FormatValidationScorer("markdown")
        sr = await s.score("c1", None, "This is **bold** text")
        assert sr.score == 1.0

    async def test_plain_text(self) -> None:
        s = FormatValidationScorer("markdown")
        sr = await s.score("c1", None, "Just plain text with no formatting")
        assert sr.score == 0.0


class TestFormatCSV:
    async def test_valid_csv(self) -> None:
        s = FormatValidationScorer("csv")
        sr = await s.score("c1", None, "name,age\nAlice,30\nBob,25")
        assert sr.score == 1.0
        assert sr.details.get("delimiter") == ","

    async def test_tab_delimited(self) -> None:
        s = FormatValidationScorer("csv")
        sr = await s.score("c1", None, "name\tage\nAlice\t30")
        assert sr.score == 1.0
        assert sr.details.get("delimiter") == "\t"

    async def test_too_few_rows(self) -> None:
        s = FormatValidationScorer("csv")
        sr = await s.score("c1", None, "just one row")
        assert sr.score == 0.0

    async def test_inconsistent_columns(self) -> None:
        s = FormatValidationScorer("csv")
        sr = await s.score("c1", None, "a,b,c\nd,e")
        assert sr.score == 0.0


class TestFormatNoneOutput:
    async def test_none_output(self) -> None:
        s = FormatValidationScorer("json")
        sr = await s.score("c1", None, None)
        assert sr.score == 0.0


# ---------------------------------------------------------------------------
# SchemaValidationScorer
# ---------------------------------------------------------------------------


class TestSchemaValidationScorer:
    async def test_valid_object(self) -> None:
        schema: dict[str, Any] = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }
        s = SchemaValidationScorer(schema)
        sr = await s.score("c1", None, json.dumps({"name": "Alice", "age": 30}))
        assert sr.score == 1.0

    async def test_missing_required(self) -> None:
        schema: dict[str, Any] = {"type": "object", "required": ["name"]}
        s = SchemaValidationScorer(schema)
        sr = await s.score("c1", None, json.dumps({"age": 30}))
        assert sr.score == 0.0
        assert any("name" in e for e in sr.details["errors"])

    async def test_wrong_type(self) -> None:
        schema: dict[str, Any] = {"type": "array"}
        s = SchemaValidationScorer(schema)
        sr = await s.score("c1", None, json.dumps({"key": "value"}))
        assert sr.score == 0.0

    async def test_invalid_json_input(self) -> None:
        schema: dict[str, Any] = {"type": "object"}
        s = SchemaValidationScorer(schema)
        sr = await s.score("c1", None, "not json")
        assert sr.score == 0.0
        assert "Invalid JSON" in sr.details["error"]

    async def test_nested_property_validation(self) -> None:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "required": ["city"],
                },
            },
        }
        s = SchemaValidationScorer(schema)
        sr = await s.score("c1", None, json.dumps({"address": {}}))
        assert sr.score == 0.0
        assert any("city" in e for e in sr.details["errors"])

    async def test_custom_name(self) -> None:
        s = SchemaValidationScorer({"type": "object"}, name="my_schema")
        sr = await s.score("c1", None, "{}")
        assert sr.scorer_name == "my_schema"


# ---------------------------------------------------------------------------
# OutputCorrectnessScorer
# ---------------------------------------------------------------------------


class TestOutputCorrectnessScorer:
    async def test_exact_match(self) -> None:
        s = OutputCorrectnessScorer(ground_truth="Hello World")
        sr = await s.score("c1", None, "Hello World")
        assert sr.score == 1.0

    async def test_normalized_match(self) -> None:
        s = OutputCorrectnessScorer(ground_truth="hello  world")
        sr = await s.score("c1", None, "  HELLO   WORLD  ")
        assert sr.score == 1.0

    async def test_no_match(self) -> None:
        s = OutputCorrectnessScorer(ground_truth="expected")
        sr = await s.score("c1", None, "something else")
        assert sr.score == 0.0

    async def test_normalize_off(self) -> None:
        s = OutputCorrectnessScorer(ground_truth="Hello", normalize=False)
        sr = await s.score("c1", None, "hello")
        assert sr.score == 0.0

    async def test_keywords_all_found(self) -> None:
        s = OutputCorrectnessScorer(keywords=["python", "code"])
        sr = await s.score("c1", None, "Here is some Python code example")
        assert sr.score == 1.0
        assert sr.details["missing"] == []

    async def test_keywords_partial(self) -> None:
        s = OutputCorrectnessScorer(keywords=["python", "java", "rust"])
        sr = await s.score("c1", None, "I love Python and Rust")
        assert sr.score == pytest.approx(2 / 3)
        assert "java" in sr.details["missing"]

    async def test_keywords_none_found(self) -> None:
        s = OutputCorrectnessScorer(keywords=["alpha", "beta"])
        sr = await s.score("c1", None, "nothing here")
        assert sr.score == 0.0

    async def test_no_criteria(self) -> None:
        s = OutputCorrectnessScorer()
        sr = await s.score("c1", None, "anything")
        assert sr.score == 0.0
        assert "error" in sr.details


# ---------------------------------------------------------------------------
# OutputLengthScorer
# ---------------------------------------------------------------------------


class TestOutputLengthScorer:
    async def test_within_range(self) -> None:
        s = OutputLengthScorer(min_length=5, max_length=20)
        sr = await s.score("c1", None, "Hello World")
        assert sr.score == 1.0

    async def test_too_short(self) -> None:
        s = OutputLengthScorer(min_length=100)
        sr = await s.score("c1", None, "short")
        assert sr.score == 0.0

    async def test_too_long(self) -> None:
        s = OutputLengthScorer(max_length=5)
        sr = await s.score("c1", None, "this is too long")
        assert sr.score == 0.0

    async def test_empty_output(self) -> None:
        s = OutputLengthScorer(min_length=1)
        sr = await s.score("c1", None, "")
        assert sr.score == 0.0

    async def test_details(self) -> None:
        s = OutputLengthScorer(min_length=1, max_length=100)
        sr = await s.score("c1", None, "test")
        assert sr.details["length"] == 4
        assert sr.details["min"] == 1
        assert sr.details["max"] == 100


# ---------------------------------------------------------------------------
# OutputCompletenessScorer
# ---------------------------------------------------------------------------


class TestOutputCompletenessScorer:
    async def test_all_sections_present(self) -> None:
        s = OutputCompletenessScorer(["introduction", "conclusion"])
        sr = await s.score("c1", None, "Introduction paragraph.\n\nConclusion paragraph.")
        assert sr.score == 1.0

    async def test_some_missing(self) -> None:
        s = OutputCompletenessScorer(["intro", "body", "conclusion"])
        sr = await s.score("c1", None, "Here is the intro and body.")
        assert sr.score == pytest.approx(2 / 3)
        assert "conclusion" in sr.details["missing"]

    async def test_none_present(self) -> None:
        s = OutputCompletenessScorer(["alpha", "beta"])
        sr = await s.score("c1", None, "nothing relevant here")
        assert sr.score == 0.0

    async def test_empty_sections(self) -> None:
        s = OutputCompletenessScorer([])
        sr = await s.score("c1", None, "anything")
        assert sr.score == 0.0

    async def test_case_insensitive(self) -> None:
        s = OutputCompletenessScorer(["Summary"])
        sr = await s.score("c1", None, "Here is the summary of findings.")
        assert sr.score == 1.0
