"""Built-in media generation tools: DALL-E 3, Imagen 3, Veo 2.

These tools can be added directly to any Agent and allow LLMs to generate
images and videos using their respective APIs.

Usage::

    from exo.models import dalle_generate_image, imagen_generate_image

    agent = Agent(
        name="artist",
        model="openai:gpt-4o",
        tools=[dalle_generate_image, imagen_generate_image],
    )
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import time

from exo.tool import ToolError, tool
from exo.types import ImageDataBlock, ImageURLBlock, VideoBlock

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DALL-E 3 image generation
# ---------------------------------------------------------------------------


@tool
async def dalle_generate_image(
    prompt: str,
    size: str = "1024x1024",
    quality: str = "standard",
    style: str = "vivid",
) -> list[ImageURLBlock]:
    """Generate an image using OpenAI DALL-E 3.

    Args:
        prompt: Text description of the image to generate.
        size: Image dimensions — "1024x1024", "1792x1024", or "1024x1792".
        quality: Image quality — "standard" or "hd".
        style: Generation style — "vivid" or "natural".

    Returns:
        List of ImageURLBlock objects with the generated image URLs.
    """
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise ToolError("openai package is required for dalle_generate_image") from exc

    # Use async-CM to ensure the underlying httpx client is closed on exit.
    async with AsyncOpenAI(timeout=30.0, max_retries=2) as client:
        try:
            response = await client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=size,  # type: ignore[arg-type]
                quality=quality,  # type: ignore[arg-type]
                style=style,  # type: ignore[arg-type]
                n=1,
            )
        except Exception as exc:
            raise ToolError(f"DALL-E 3 generation failed: {exc}") from exc

        blocks: list[ImageURLBlock] = []
        for img in response.data:
            if img.url:
                blocks.append(ImageURLBlock(url=img.url))
    return blocks


# ---------------------------------------------------------------------------
# Imagen 3 image generation
# ---------------------------------------------------------------------------


@tool
async def imagen_generate_image(
    prompt: str,
    aspect_ratio: str = "1:1",
    number_of_images: int = 1,
) -> list[ImageDataBlock]:
    """Generate images using Google Imagen 3 via the Gemini API.

    Requires ``GOOGLE_API_KEY`` environment variable.

    Args:
        prompt: Text description of the image to generate.
        aspect_ratio: Aspect ratio — "1:1", "16:9", "9:16", "4:3", or "3:4".
        number_of_images: Number of images to generate (1-4).

    Returns:
        List of ImageDataBlock objects with base64-encoded PNG image data.
    """
    try:
        from google import genai
    except ImportError as exc:
        raise ToolError("google-genai package is required for imagen_generate_image") from exc

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ToolError("GOOGLE_API_KEY environment variable is required for imagen_generate_image")

    # genai.Client has a sync close() but no async CM; use try/finally to avoid fd leaks.
    client = genai.Client(api_key=api_key)
    try:
        try:
            response = await asyncio.wait_for(
                client.aio.models.generate_images(
                    model="imagen-3.0-generate-002",
                    prompt=prompt,
                    config={
                        "number_of_images": number_of_images,
                        "aspect_ratio": aspect_ratio,
                    },
                ),
                timeout=60.0,
            )
        except TimeoutError as exc:
            raise ToolError(
                "Imagen 3 generation timed out after 60s — check Vertex AI quota and try again."
            ) from exc
        except Exception as exc:
            raise ToolError(f"Imagen 3 generation failed: {exc}") from exc

        blocks: list[ImageDataBlock] = []
        for img in response.generated_images:
            image_bytes = getattr(img.image, "image_bytes", None)
            if image_bytes:
                data = base64.b64encode(image_bytes).decode("utf-8")
                blocks.append(ImageDataBlock(data=data, media_type="image/png"))
    finally:
        client.close()
    return blocks


# ---------------------------------------------------------------------------
# Veo 2 video generation
# ---------------------------------------------------------------------------


@tool
async def veo_generate_video(
    prompt: str,
    duration_seconds: int = 5,
    aspect_ratio: str = "16:9",
) -> list[VideoBlock]:
    """Generate a video using Google Veo 2 via Vertex AI.

    Requires ``GOOGLE_CLOUD_PROJECT`` and ``GOOGLE_CLOUD_LOCATION``
    environment variables.

    Args:
        prompt: Text description of the video to generate.
        duration_seconds: Video duration in seconds (5-8).
        aspect_ratio: Aspect ratio — "16:9" or "9:16".

    Returns:
        List of VideoBlock objects with URLs to the generated videos.
    """
    try:
        from google import genai
    except ImportError as exc:
        raise ToolError("google-genai package is required for veo_generate_video") from exc

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    if not project:
        raise ToolError(
            "GOOGLE_CLOUD_PROJECT environment variable is required for veo_generate_video"
        )

    # genai.Client has a sync close() but no async CM; use try/finally to avoid fd leaks.
    client = genai.Client(vertexai=True, project=project, location=location)
    try:
        try:
            operation = await asyncio.wait_for(
                client.aio.models.generate_videos(
                    model="veo-2.0-generate-001",
                    prompt=prompt,
                    config={
                        "duration_seconds": duration_seconds,
                        "aspect_ratio": aspect_ratio,
                        "number_of_videos": 1,
                    },
                ),
                timeout=30.0,
            )
        except TimeoutError as exc:
            raise ToolError(
                "Veo 2 video generation request timed out after 30s — check network connectivity."
            ) from exc
        except Exception as exc:
            raise ToolError(f"Veo 2 generation failed: {exc}") from exc

        # Poll until operation is complete — guard against indefinite hangs.
        _poll_timeout = 300.0  # 5 minutes total
        _poll_interval = 5.0
        _deadline = time.monotonic() + _poll_timeout
        try:
            while not operation.done:
                if time.monotonic() >= _deadline:
                    raise ToolError(
                        f"Veo 2 operation timed out after {int(_poll_timeout)}s — "
                        "check the Vertex AI console for job status.",
                    )
                await asyncio.sleep(_poll_interval)
                operation = await client.aio.operations.get(operation)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"Veo 2 polling failed: {exc}") from exc

        blocks: list[VideoBlock] = []
        for video in operation.result.generated_videos:
            video_uri = getattr(video.video, "uri", None)
            if video_uri:
                blocks.append(VideoBlock(url=video_uri, media_type="video/mp4"))
    finally:
        client.close()
    return blocks
