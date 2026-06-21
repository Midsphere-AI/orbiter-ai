"""Tests for exo-search correctness fixes:

1. _collected_results ContextVar race — concurrent searches no longer share state.
2. stream() async generator — async for works correctly over stream().
"""

from __future__ import annotations

import asyncio
import inspect

from exo.search import stream
from exo.search.tools.searxng import (
    _collected_results_var,
    clear_collected_results,
    get_collected_results,
)

# ---------------------------------------------------------------------------
# 1. _collected_results ContextVar — per-task isolation
# ---------------------------------------------------------------------------


class TestCollectedResultsContextVar:
    """Verify that concurrent searches each maintain independent result lists."""

    def test_clear_sets_empty_list_in_context(self) -> None:
        """clear_collected_results() resets the context-local list to empty."""
        _collected_results_var.set([{"url": "https://example.com", "title": "pre-existing"}])
        clear_collected_results()
        assert get_collected_results() == []

    def test_get_returns_copy(self) -> None:
        """get_collected_results() returns a snapshot copy, not the live list."""
        clear_collected_results()
        results = get_collected_results()
        # Mutating the returned list must not affect the context variable
        results.append({"url": "https://mutated.com", "title": "mutated"})
        assert get_collected_results() == []

    async def test_concurrent_searches_have_independent_collectors(self) -> None:
        """Two concurrent tasks see separate collected result lists.

        Task A populates its own list; Task B's list stays empty.
        The ContextVar ensures no cross-contamination.
        """
        task_a_results: list[dict] = []
        task_b_results: list[dict] = []

        async def task_a() -> None:
            clear_collected_results()
            # Simulate a search tool appending results
            _collected_results_var.set(
                [*get_collected_results(), {"url": "https://a.com", "title": "Result A"}]
            )
            # Small delay so task_b runs concurrently
            await asyncio.sleep(0)
            task_a_results.extend(get_collected_results())

        async def task_b() -> None:
            clear_collected_results()
            # task_b appends nothing — its list should stay empty
            await asyncio.sleep(0)
            task_b_results.extend(get_collected_results())

        await asyncio.gather(task_a(), task_b())

        # Task A saw its own result
        assert len(task_a_results) == 1
        assert task_a_results[0]["url"] == "https://a.com"
        # Task B saw an empty list (no contamination from task_a)
        assert task_b_results == []

    async def test_clear_does_not_affect_sibling_task(self) -> None:
        """clear_collected_results() in one task does not wipe another task's list."""
        barrier = asyncio.Event()

        async def populate_then_wait() -> list[dict]:
            clear_collected_results()
            _collected_results_var.set(
                [{"url": "https://keeper.com", "title": "Keep me"}]
            )
            await barrier.wait()
            return get_collected_results()

        async def clear_and_signal() -> None:
            clear_collected_results()
            barrier.set()

        keep_task = asyncio.ensure_future(populate_then_wait())
        clear_task = asyncio.ensure_future(clear_and_signal())
        kept, _ = await asyncio.gather(keep_task, clear_task)

        # populate_then_wait's results should survive the sibling's clear()
        assert len(kept) == 1
        assert kept[0]["url"] == "https://keeper.com"


# ---------------------------------------------------------------------------
# 2. stream() — must be an async generator (supports async for)
# ---------------------------------------------------------------------------


class TestStreamAsyncGenerator:
    def test_stream_is_async_generator_function(self) -> None:
        """stream() must be declared as async def with yield — an async gen function."""
        assert inspect.isasyncgenfunction(stream), (
            "stream() must be an async generator function so that "
            "'async for event in stream(q)' works without awaiting first"
        )

    def test_calling_stream_returns_async_iterator(self) -> None:
        """stream('q') must return an object supporting the async iterator protocol."""
        result = stream("test query")
        assert hasattr(result, "__aiter__"), "stream() return value must have __aiter__"
        assert hasattr(result, "__anext__"), "stream() return value must have __anext__"

    def test_stream_result_is_not_a_coroutine(self) -> None:
        """stream('q') must NOT return a coroutine (which would require await)."""
        result = stream("test query")
        assert not asyncio.iscoroutine(result), (
            "stream() must not return a bare coroutine — "
            "callers use 'async for', not 'await'"
        )
        # Clean up the unawaited generator
        result.aclose()
