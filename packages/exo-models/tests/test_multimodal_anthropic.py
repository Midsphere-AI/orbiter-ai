"""Tests for Anthropic multimodal content conversion helpers."""

from __future__ import annotations

import pytest

from exo.models.anthropic import _build_messages, _content_blocks_to_anthropic
from exo.models.types import ModelError
from exo.types import (
    AudioBlock,
    DocumentBlock,
    ImageDataBlock,
    ImageURLBlock,
    TextBlock,
    ToolResult,
    UserMessage,
    VideoBlock,
)


class TestContentBlocksToAnthropic:
    def test_text_block(self) -> None:
        parts = _content_blocks_to_anthropic([TextBlock(text="hello")])
        assert parts == [{"type": "text", "text": "hello"}]

    def test_image_url_block(self) -> None:
        parts = _content_blocks_to_anthropic([ImageURLBlock(url="https://example.com/img.png")])
        assert parts == [
            {"type": "image", "source": {"type": "url", "url": "https://example.com/img.png"}}
        ]

    def test_image_data_block(self) -> None:
        parts = _content_blocks_to_anthropic(
            [ImageDataBlock(data="abc123", media_type="image/png")]
        )
        assert parts == [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "abc123",
                },
            }
        ]

    def test_document_block_without_title(self) -> None:
        parts = _content_blocks_to_anthropic([DocumentBlock(data="pdfdata")])
        assert len(parts) == 1
        assert parts[0]["type"] == "document"
        assert parts[0]["source"]["type"] == "base64"
        assert "title" not in parts[0]

    def test_document_block_with_title(self) -> None:
        parts = _content_blocks_to_anthropic([DocumentBlock(data="pdfdata", title="Report")])
        assert parts[0]["title"] == "Report"

    def test_audio_block_raises(self) -> None:
        # Anthropic does not support audio — must raise ModelError, not silently drop.
        with pytest.raises(ModelError, match="audio"):
            _content_blocks_to_anthropic([AudioBlock(data="audiodata")])

    def test_video_block_raises(self) -> None:
        # Anthropic does not support video — must raise ModelError, not silently drop.
        with pytest.raises(ModelError, match="video"):
            _content_blocks_to_anthropic([VideoBlock(url="gs://bucket/v.mp4")])

    def test_multiple_blocks(self) -> None:
        parts = _content_blocks_to_anthropic(
            [
                TextBlock(text="describe"),
                ImageDataBlock(data="base64img", media_type="image/jpeg"),
            ]
        )
        assert len(parts) == 2
        assert parts[0]["type"] == "text"
        assert parts[1]["type"] == "image"


class TestBuildMessagesMultimodal:
    def test_user_message_string_unchanged(self) -> None:
        _system, msgs = _build_messages([UserMessage(content="hello")])
        assert msgs == [{"role": "user", "content": "hello"}]

    def test_user_message_with_blocks(self) -> None:
        blocks = [TextBlock(text="hi"), ImageURLBlock(url="https://x.com/a.png")]
        _sys, msgs = _build_messages([UserMessage(content=blocks)])
        assert msgs[0]["role"] == "user"
        assert isinstance(msgs[0]["content"], list)
        assert len(msgs[0]["content"]) == 2

    def test_tool_result_string_content(self) -> None:
        tr = ToolResult(tool_call_id="id1", tool_name="tool", content="result")
        _, msgs = _build_messages([tr])
        assert msgs[0]["role"] == "user"
        block = msgs[0]["content"][0]
        assert block["type"] == "tool_result"
        assert block["content"] == "result"

    def test_tool_result_with_image_blocks(self) -> None:
        blocks = [ImageDataBlock(data="imgdata", media_type="image/png")]
        tr = ToolResult(tool_call_id="id1", tool_name="dalle", content=blocks)
        _, msgs = _build_messages([tr])
        assert msgs[0]["role"] == "user"
        tool_block = msgs[0]["content"][0]
        assert tool_block["type"] == "tool_result"
        assert isinstance(tool_block["content"], list)
        assert tool_block["content"][0]["type"] == "image"

    def test_tool_result_error(self) -> None:
        tr = ToolResult(tool_call_id="id1", tool_name="tool", error="boom")
        _, msgs = _build_messages([tr])
        tool_block = msgs[0]["content"][0]
        assert tool_block["content"] == "boom"
        assert tool_block["is_error"] is True
