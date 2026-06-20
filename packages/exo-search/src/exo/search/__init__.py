"""Exo Search — AI-powered search engine with deep research and citations.

Usage:
    from exo.search import search, search_with_details

    answer = await search("What is quantum computing?")
    result = await search_with_details("latest AI news", mode="balanced")
"""

from __future__ import annotations

try:
    from importlib.metadata import version

    __version__: str = version("exo-search")
except Exception:
    __version__ = "0.1.0"

from .config import SearchConfig
from .conversation import ConversationManager
from .pipeline import run_search_pipeline, stream_search_pipeline
from .types import PipelineEvent, ResearchMode, SearchResponse


async def search(
    query: str,
    mode: str = "balanced",
    chat_history: list[tuple[str, str]] | None = None,
    config: SearchConfig | None = None,
) -> str:
    """Run a search query and return the answer with citations."""
    result = await run_search_pipeline(
        query=query,
        chat_history=chat_history,
        mode=mode,
        config=config,
    )
    return result.answer


def stream(
    query: str,
    mode: str = "balanced",
    chat_history: list[tuple[str, str]] | None = None,
    config: SearchConfig | None = None,
):
    """Stream the search pipeline, yielding events as they happen.

    Yields PipelineEvent for stage transitions, exo StreamEvent for
    real-time text/tool events, and a final SearchResponse.

    Usage::

        async for event in stream("quantum computing", mode="quality"):
            if isinstance(event, PipelineEvent):
                print(f"[{event.stage}] {event.status}")
            elif isinstance(event, TextEvent):
                print(event.text, end="", flush=True)
            elif isinstance(event, SearchResponse):
                print(event.sources)
    """
    return stream_search_pipeline(
        query=query,
        chat_history=chat_history,
        mode=mode,
        config=config,
    )


async def search_with_details(
    query: str,
    mode: str = "balanced",
    chat_history: list[tuple[str, str]] | None = None,
    config: SearchConfig | None = None,
) -> SearchResponse:
    """Run a search query and return full response with sources and suggestions."""
    return await run_search_pipeline(
        query=query,
        chat_history=chat_history,
        mode=mode,
        config=config,
    )


__all__: list[str] = [
    "ConversationManager",
    "PipelineEvent",
    "ResearchMode",
    "SearchConfig",
    "SearchResponse",
    "search",
    "search_with_details",
    "stream",
]
