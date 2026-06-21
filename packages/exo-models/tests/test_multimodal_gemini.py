"""Tests for Gemini/Vertex multimodal content conversion helpers."""

from __future__ import annotations

import pytest

from exo.models._media import _guess_mime_from_url, content_blocks_to_google
from exo.models.gemini import _to_google_contents
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


class TestContentBlocksToGoogle:
    def test_text_block(self) -> None:
        parts = content_blocks_to_google([TextBlock(text="hello")])
        assert parts == [{"text": "hello"}]

    def test_image_url_block_https(self) -> None:
        parts = content_blocks_to_google([ImageURLBlock(url="https://example.com/img.jpg")])
        assert len(parts) == 1
        assert "file_data" in parts[0]
        assert parts[0]["file_data"]["file_uri"] == "https://example.com/img.jpg"
        assert parts[0]["file_data"]["mime_type"] == "image/jpeg"

    def test_image_url_block_gs(self) -> None:
        parts = content_blocks_to_google([ImageURLBlock(url="gs://bucket/image.png")])
        assert parts[0]["file_data"]["mime_type"] == "image/png"

    def test_image_url_block_data_uri(self) -> None:
        parts = content_blocks_to_google([ImageURLBlock(url="data:image/png;base64,abc123")])
        assert "inline_data" in parts[0]
        assert parts[0]["inline_data"]["mime_type"] == "image/png"
        assert parts[0]["inline_data"]["data"] == "abc123"

    def test_image_data_block(self) -> None:
        parts = content_blocks_to_google([ImageDataBlock(data="imgdata", media_type="image/jpeg")])
        assert parts == [{"inline_data": {"mime_type": "image/jpeg", "data": "imgdata"}}]

    def test_audio_block(self) -> None:
        parts = content_blocks_to_google([AudioBlock(data="audiodata", format="mp3")])
        assert parts == [{"inline_data": {"mime_type": "audio/mp3", "data": "audiodata"}}]

    def test_video_block_url(self) -> None:
        parts = content_blocks_to_google([VideoBlock(url="gs://bucket/video.mp4")])
        assert "file_data" in parts[0]
        assert parts[0]["file_data"]["file_uri"] == "gs://bucket/video.mp4"

    def test_video_block_data(self) -> None:
        parts = content_blocks_to_google([VideoBlock(data="videodata", media_type="video/mp4")])
        assert "inline_data" in parts[0]
        assert parts[0]["inline_data"]["data"] == "videodata"

    def test_video_block_empty_raises(self) -> None:
        # A VideoBlock with neither url nor data must raise ModelError (not silently drop).
        with pytest.raises(ModelError, match="VideoBlock has neither url nor data"):
            content_blocks_to_google([VideoBlock()])

    def test_document_block(self) -> None:
        parts = content_blocks_to_google([DocumentBlock(data="pdfdata")])
        assert parts == [{"inline_data": {"mime_type": "application/pdf", "data": "pdfdata"}}]

    def test_multiple_blocks(self) -> None:
        parts = content_blocks_to_google(
            [TextBlock(text="look"), ImageDataBlock(data="img", media_type="image/png")]
        )
        assert len(parts) == 2
        assert "text" in parts[0]
        assert "inline_data" in parts[1]


class TestGuessMimeFromUrl:
    def test_jpeg(self) -> None:
        assert _guess_mime_from_url("https://x.com/photo.jpg") == "image/jpeg"
        assert _guess_mime_from_url("https://x.com/photo.jpeg") == "image/jpeg"

    def test_png(self) -> None:
        assert _guess_mime_from_url("gs://bucket/img.png") == "image/png"

    def test_mp4(self) -> None:
        assert _guess_mime_from_url("https://x.com/video.mp4") == "video/mp4"

    def test_pdf(self) -> None:
        assert _guess_mime_from_url("https://x.com/doc.pdf") == "application/pdf"

    def test_unknown_fallback(self) -> None:
        assert _guess_mime_from_url("https://x.com/file.xyz") == "application/octet-stream"

    def test_strips_query_params(self) -> None:
        assert _guess_mime_from_url("https://x.com/img.png?token=abc") == "image/png"


class TestGeminiToGoogleContentsMultimodal:
    def test_user_message_string(self) -> None:
        from exo.types import UserMessage

        contents, _sys = _to_google_contents([UserMessage(content="hello")])
        assert contents[0]["parts"] == [{"text": "hello"}]

    def test_user_message_with_blocks(self) -> None:
        blocks = [TextBlock(text="describe"), ImageDataBlock(data="img", media_type="image/jpeg")]
        contents, _ = _to_google_contents([UserMessage(content=blocks)])
        assert len(contents[0]["parts"]) == 2
        assert "text" in contents[0]["parts"][0]
        assert "inline_data" in contents[0]["parts"][1]

    def test_tool_result_string_content(self) -> None:
        tr = ToolResult(tool_call_id="id1", tool_name="search", content="found it")
        contents, _ = _to_google_contents([tr])
        assert contents[0]["role"] == "user"
        fn_part = contents[0]["parts"][0]
        assert "function_response" in fn_part
        assert fn_part["function_response"]["response"]["content"] == "found it"

    def test_tool_result_with_media_blocks(self) -> None:
        blocks = [ImageDataBlock(data="imgdata", media_type="image/png")]
        tr = ToolResult(tool_call_id="id1", tool_name="imagen", content=blocks)
        contents, _ = _to_google_contents([tr])
        parts = contents[0]["parts"]
        # function_response part + media parts
        assert len(parts) >= 2
        assert "function_response" in parts[0]
        assert "inline_data" in parts[1]
