"""Tests for built-in media generation tools (DALL-E 3, Imagen 3, Veo 2)."""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exo.tool import ToolError
from exo.types import ImageDataBlock, ImageURLBlock, VideoBlock


class TestDalleGenerateImage:
    async def test_returns_image_url_blocks(self) -> None:
        from exo.models.media_tools import dalle_generate_image

        mock_image = MagicMock()
        mock_image.url = "https://oaidalleapiprodscus.blob.core.windows.net/private/img.png"

        mock_response = MagicMock()
        mock_response.data = [mock_image]

        mock_client = AsyncMock()
        mock_client.images.generate = AsyncMock(return_value=mock_response)
        # Support async-CM: __aenter__ returns mock_client, __aexit__ is a no-op
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        # Patch openai.AsyncOpenAI since it's imported lazily inside the function
        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI.return_value = mock_client
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            result = await dalle_generate_image.execute(
                prompt="a cat on a rainbow",
                size="1024x1024",
                quality="standard",
                style="vivid",
            )

        assert len(result) == 1
        assert isinstance(result[0], ImageURLBlock)
        assert "oaidalleapiprodscus" in result[0].url

    async def test_raises_tool_error_on_api_failure(self) -> None:
        from exo.models.media_tools import dalle_generate_image

        mock_client = AsyncMock()
        mock_client.images.generate = AsyncMock(side_effect=Exception("API error"))
        # Support async-CM: __aenter__ returns mock_client, __aexit__ is a no-op
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI.return_value = mock_client
        with (
            patch.dict("sys.modules", {"openai": mock_openai_module}),
            pytest.raises(ToolError, match="DALL-E 3 generation failed"),
        ):
            await dalle_generate_image.execute(prompt="test")

    async def test_skips_images_without_url(self) -> None:
        from exo.models.media_tools import dalle_generate_image

        mock_image = MagicMock()
        mock_image.url = None

        mock_response = MagicMock()
        mock_response.data = [mock_image]

        mock_client = AsyncMock()
        mock_client.images.generate = AsyncMock(return_value=mock_response)
        # Support async-CM: __aenter__ returns mock_client, __aexit__ is a no-op
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI.return_value = mock_client
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            result = await dalle_generate_image.execute(prompt="test")

        assert result == []


class TestImagenGenerateImage:
    async def test_returns_image_data_blocks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from exo.models.media_tools import imagen_generate_image

        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        raw_bytes = b"\x89PNG\r\nfake_png_bytes"
        mock_image = MagicMock()
        mock_image.image.image_bytes = raw_bytes

        mock_response = MagicMock()
        mock_response.generated_images = [mock_image]

        mock_aio = AsyncMock()
        mock_aio.models.generate_images = AsyncMock(return_value=mock_response)

        mock_client = MagicMock()
        mock_client.aio = mock_aio

        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client

        mock_google = MagicMock()
        mock_google.genai = mock_genai

        with patch.dict("sys.modules", {"google": mock_google, "google.genai": mock_genai}):
            result = await imagen_generate_image.execute(
                prompt="a mountain at sunset",
                aspect_ratio="1:1",
                number_of_images=1,
            )

        assert len(result) == 1
        assert isinstance(result[0], ImageDataBlock)
        assert result[0].media_type == "image/png"
        decoded = base64.b64decode(result[0].data)
        assert decoded == raw_bytes

    async def test_raises_when_no_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from exo.models.media_tools import imagen_generate_image

        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        with pytest.raises(ToolError, match="GOOGLE_API_KEY"):
            await imagen_generate_image.execute(prompt="test")


class TestVeoGenerateVideo:
    async def test_raises_when_no_project(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from exo.models.media_tools import veo_generate_video

        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

        with pytest.raises(ToolError, match="GOOGLE_CLOUD_PROJECT"):
            await veo_generate_video.execute(prompt="test")

    async def test_returns_video_blocks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from exo.models.media_tools import veo_generate_video

        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-project")
        monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")

        mock_video = MagicMock()
        mock_video.video.uri = "gs://my-bucket/generated_video.mp4"

        mock_result = MagicMock()
        mock_result.generated_videos = [mock_video]

        mock_operation = MagicMock()
        mock_operation.done = True
        mock_operation.result = mock_result

        mock_aio = AsyncMock()
        mock_aio.models.generate_videos = AsyncMock(return_value=mock_operation)

        mock_client = MagicMock()
        mock_client.aio = mock_aio

        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client

        mock_google = MagicMock()
        mock_google.genai = mock_genai

        with patch.dict("sys.modules", {"google": mock_google, "google.genai": mock_genai}):
            result = await veo_generate_video.execute(
                prompt="a sunset over the ocean",
                duration_seconds=5,
                aspect_ratio="16:9",
            )

        assert len(result) == 1
        assert isinstance(result[0], VideoBlock)
        assert result[0].url == "gs://my-bucket/generated_video.mp4"
        assert result[0].media_type == "video/mp4"
