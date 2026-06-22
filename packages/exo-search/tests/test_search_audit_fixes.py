"""Tests for exo-search audit-finding fixes.

Covers the specific bugs identified in the audit:
1. _multi_search mutates the results list in-place so parallel workers see additions.
2. configure_search_keys is ContextVar-isolated — concurrent tasks don't share keys.
6. Empty-URL results are filtered out in _multi_search.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from exo.search.tools.searxng import (
    _collected_results_var,
    _multi_search,
    _search_keys_var,
    clear_collected_results,
    configure_search_keys,
    get_collected_results,
)

# ---------------------------------------------------------------------------
# Finding 1: _multi_search must extend the existing list IN PLACE so that
#             results are visible to the parent context (e.g. parallel_research).
# ---------------------------------------------------------------------------


class TestMultiSearchInPlaceMutation:
    """_multi_search extends the existing ContextVar list rather than replacing it."""

    async def test_results_accumulate_across_calls(self) -> None:
        """Successive _multi_search calls accumulate in the same list."""
        clear_collected_results()

        fake_results_a = [{"url": "https://a.com", "title": "A", "content": "A content"}]
        fake_results_b = [{"url": "https://b.com", "title": "B", "content": "B content"}]

        with patch(
            "exo.search.tools.searxng._search", side_effect=[fake_results_a, fake_results_b]
        ):
            await _multi_search(["query1", "query2"])

        collected = get_collected_results()
        urls = {r["url"] for r in collected}
        assert "https://a.com" in urls
        assert "https://b.com" in urls

    async def test_child_task_results_visible_in_parent(self) -> None:
        """Results added by a child asyncio task are visible to the parent context.

        This is the core fix for Finding 1: because we extend the list IN PLACE
        rather than replacing the ContextVar, the parent's reference to the list
        object sees the child's additions.
        """
        clear_collected_results()
        # Grab a reference to the list BEFORE spawning the child task
        parent_list = _collected_results_var.get()
        assert parent_list is not None  # clear_collected_results set an empty list

        fake_result = [{"url": "https://child.com", "title": "Child", "content": "child"}]

        async def child_task() -> None:
            with patch("exo.search.tools.searxng._search", return_value=fake_result):
                await _multi_search(["query"])

        await asyncio.create_task(child_task())

        # The parent's list reference must now contain the child's result
        collected = get_collected_results()
        urls = {r["url"] for r in collected}
        assert "https://child.com" in urls, (
            "Child task results not visible in parent — _multi_search may have called "
            "_collected_results_var.set() (replace) instead of .extend() (in-place)"
        )


# ---------------------------------------------------------------------------
# Finding 2: configure_search_keys must be ContextVar-isolated per task.
# ---------------------------------------------------------------------------


class TestConfigureSearchKeysContextVarIsolation:
    """configure_search_keys stores keys per-task, not in a shared global."""

    async def test_concurrent_tasks_have_independent_keys(self) -> None:
        """Two concurrent tasks each see only their own search keys."""
        task_a_key: list[str] = []
        task_b_key: list[str] = []

        async def task_a() -> None:
            configure_search_keys(serper_api_key="key-for-A")
            await asyncio.sleep(0)
            task_a_key.append(_search_keys_var.get() or {})  # type: ignore[arg-type]

        async def task_b() -> None:
            configure_search_keys(serper_api_key="key-for-B")
            await asyncio.sleep(0)
            task_b_key.append(_search_keys_var.get() or {})  # type: ignore[arg-type]

        await asyncio.gather(task_a(), task_b())

        assert task_a_key[0].get("serper") == "key-for-A"
        assert task_b_key[0].get("serper") == "key-for-B"
        # Neither key leaked into the other task
        assert task_a_key[0].get("serper") != "key-for-B"
        assert task_b_key[0].get("serper") != "key-for-A"

    def test_configure_sets_only_provided_keys(self) -> None:
        """configure_search_keys omits empty/falsy values from the stored dict."""
        configure_search_keys(serper_api_key="skey")
        keys = _search_keys_var.get() or {}
        assert keys.get("serper") == "skey"
        assert "jina" not in keys
        assert "searxng_url" not in keys

    def test_configure_empty_call_stores_empty_dict(self) -> None:
        """configure_search_keys() with no args stores an empty dict."""
        configure_search_keys()
        keys = _search_keys_var.get()
        assert keys == {}


# ---------------------------------------------------------------------------
# Finding 6: Empty-URL results must be filtered out in _multi_search.
# ---------------------------------------------------------------------------


class TestEmptyUrlFilter:
    """_multi_search must not add results with an empty or missing URL."""

    async def test_empty_url_results_are_skipped(self) -> None:
        """Results with url='' or missing url key are excluded from the collector."""
        clear_collected_results()

        fake_results = [
            {"url": "", "title": "No URL", "content": "bad"},
            {"url": "https://good.com", "title": "Good", "content": "valid"},
            {"title": "Missing URL key", "content": "also bad"},  # no 'url' key
        ]

        with patch("exo.search.tools.searxng._search", return_value=fake_results):
            await _multi_search(["query"])

        collected = get_collected_results()
        assert len(collected) == 1, f"Expected 1 result (the one with a URL), got {collected}"
        assert collected[0]["url"] == "https://good.com"
