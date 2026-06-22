"""Search tools matching Exo Search's tool interfaces.

Uses Serper API (``SERPER_API_KEY``) by default for fast Google Search.
Falls back to a local SearXNG instance when ``SEARXNG_URL`` is set
without a Serper key.

Tools write their raw results to a per-task context variable so the
researcher pipeline can retrieve them after the agent run completes.
Concurrent searches each hold their own result list, avoiding races.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextvars import ContextVar
from urllib.parse import quote_plus

from exo import tool
from exo.observability.logging import get_logger  # pyright: ignore[reportMissingImports]

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Per-task result collector — stored in a ContextVar so concurrent searches
# each see an independent list; no shared mutable state between runs.
# ---------------------------------------------------------------------------

_collected_results_var: ContextVar[list[dict] | None] = ContextVar(
    "_collected_results_var", default=None
)


def _get_results_list() -> list[dict]:
    """Return the current context's result list, initialising it if absent."""
    lst = _collected_results_var.get()
    if lst is None:
        lst = []
        _collected_results_var.set(lst)
    return lst


def get_collected_results() -> list[dict]:
    """Return all results collected during the current research run."""
    return list(_get_results_list())


def clear_collected_results() -> None:
    """Reset the result collector for a new research run (per-task context)."""
    _collected_results_var.set([])


# ---------------------------------------------------------------------------
# Search backend config — stored in a ContextVar so concurrent pipeline calls
# don't cross-contaminate each other's API keys.
# ---------------------------------------------------------------------------

_search_keys_var: ContextVar[dict[str, str] | None] = ContextVar("_search_keys_var", default=None)


def configure_search_keys(
    serper_api_key: str = "",
    jina_api_key: str = "",
    searxng_url: str = "",
) -> None:
    """Set search backend API keys for the current task context (falls back to env vars)."""
    keys: dict[str, str] = {}
    if serper_api_key:
        keys["serper"] = serper_api_key
    if jina_api_key:
        keys["jina"] = jina_api_key
    if searxng_url:
        keys["searxng_url"] = searxng_url
    _search_keys_var.set(keys)


def _get_search_keys() -> dict[str, str]:
    """Return the search-key dict for the current task context."""
    return _search_keys_var.get() or {}


# ---------------------------------------------------------------------------
# SearXNG query helper
# ---------------------------------------------------------------------------


_MAX_RETRIES = 3
_RETRY_DELAY = 2  # seconds between retries


def _available_providers() -> list[tuple[str, str]]:
    """Return list of (provider_name, key_or_url) for all configured search backends."""
    keys = _get_search_keys()
    providers: list[tuple[str, str]] = []
    serper_key = keys.get("serper") or os.environ.get("SERPER_API_KEY", "")
    if serper_key:
        providers.append(("serper", serper_key))
    jina_key = keys.get("jina") or os.environ.get("JINA_API_KEY", "")
    if jina_key:
        providers.append(("jina", jina_key))
    # Only include SearXNG if explicitly configured (not the default localhost)
    searxng_url = keys.get("searxng_url") or os.environ.get("SEARXNG_URL", "")
    if searxng_url and "localhost" not in searxng_url and "127.0.0.1" not in searxng_url:
        providers.append(("searxng", searxng_url))
    elif searxng_url and not providers:
        # Use SearXNG as last resort only if no other providers available
        providers.append(("searxng", searxng_url))
    return providers


def _search_single_provider(
    provider_name: str,
    provider_key: str,
    query: str,
    categories: str = "general",
    engines: str = "",
    num_results: int = 10,
    timeout: int = 15,
) -> list[dict]:
    """Call a single search provider by name and return results."""
    if provider_name == "serper":
        from .serper import serper_search

        return serper_search(query, categories, engines, num_results, timeout, api_key=provider_key)
    if provider_name == "jina":
        from .jina import jina_search

        return jina_search(query, categories, engines, num_results, timeout, api_key=provider_key)
    if provider_name == "searxng":
        return _searxng_search(query, categories, engines, num_results, timeout, provider_key)
    _log.warning("unknown search provider: %s", provider_name)
    return []


def _merge_results(batches: list[list[dict]], num_results: int) -> list[dict]:
    """Merge multiple result batches by URL, keeping the entry with longer content."""
    seen: dict[str, dict] = {}
    order: list[str] = []
    for batch in batches:
        for r in batch:
            url = r.get("url", "")
            if not url:
                continue
            if url in seen:
                # Keep the result with longer content
                if len(r.get("content", "")) > len(seen[url].get("content", "")):
                    seen[url] = r
            else:
                seen[url] = r
                order.append(url)
    return [seen[u] for u in order[:num_results]]


def _search(
    query: str,
    categories: str = "general",
    engines: str = "",
    num_results: int = 10,
    timeout: int = 15,
) -> list[dict]:
    """Dispatch to all configured providers, merging results.

    When multiple providers are available, queries all of them in parallel
    using a thread pool, then merges results by URL (keeping the entry with
    longer ``content`` when duplicates occur).

    Reads API keys from module-level ``_search_keys`` (set via
    ``configure_search_keys``), falling back to environment variables.
    """
    providers = _available_providers()

    if not providers:
        # No provider configured — try SearXNG at default URL as last resort
        _log.debug("search no providers configured, trying default SearXNG")
        return _searxng_search(query, categories, engines, num_results, timeout, "")

    if len(providers) == 1:
        name, key = providers[0]
        _log.debug("search backend=%s query=%r", name, query)
        return _search_single_provider(name, key, query, categories, engines, num_results, timeout)

    # Multiple providers — fan out in parallel.
    # NOTE: _search() is called from asyncio.to_thread() at the call sites above,
    # so we are already on a worker thread here.  Spinning up a *nested*
    # ThreadPoolExecutor would create N_queries x N_providers threads.
    # Instead, run each provider call sequentially within this worker thread;
    # the concurrency across *queries* is still provided by the outer
    # asyncio.to_thread() gather in _multi_search / search_and_collect.
    _log.debug(
        "search multi-provider fan-out providers=%s query=%r",
        [p[0] for p in providers],
        query,
    )
    batches: list[list[dict]] = []
    for name, key in providers:
        try:
            results = _search_single_provider(
                name, key, query, categories, engines, num_results, timeout
            )
            _log.debug("provider %s returned %d results", name, len(results))
            batches.append(results)
        except Exception as exc:
            _log.warning("provider %s failed: %s", name, exc)

    merged = _merge_results(batches, num_results)
    _log.debug("multi-provider merged=%d results", len(merged))
    return merged


def _searxng_search(
    query: str,
    categories: str = "general",
    engines: str = "",
    num_results: int = 10,
    timeout: int = 15,
    searxng_url: str = "",
) -> list[dict]:
    """Execute a search against SearXNG with retry on empty results.

    Retries up to _MAX_RETRIES times when engines are suspended/rate-limited,
    giving them time to recover between attempts.
    """
    import time
    import urllib.request

    base_url = searxng_url or os.environ.get("SEARXNG_URL", "http://localhost:8888")
    url = f"{base_url}/search?q={quote_plus(query)}&format=json&categories={quote_plus(categories)}"
    if engines:
        url += f"&engines={quote_plus(engines)}"

    for attempt in range(_MAX_RETRIES):
        _log.debug("searxng attempt=%d/%d query=%r", attempt + 1, _MAX_RETRIES, query)
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
        except Exception as exc:
            _log.warning("searxng attempt %d failed: %s", attempt + 1, exc)
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAY * (attempt + 1))
                continue
            _log.warning("searxng all retries exhausted for query=%r", query)
            return []

        results = []
        for item in data.get("results", [])[:num_results]:
            results.append(
                {
                    "title": item.get("title", "Untitled"),
                    "url": item.get("url", ""),
                    "content": item.get("content", "") or item.get("title", ""),
                }
            )

        if results or attempt == _MAX_RETRIES - 1:
            return results

        # Engines likely suspended — wait and retry
        time.sleep(_RETRY_DELAY * (attempt + 1))

    return []


async def _multi_search(
    queries: list[str],
    categories: str = "general",
    engines: str = "",
    num_results: int = 10,
) -> str:
    """Run multiple queries in parallel, collect results, and return formatted text."""
    queries = queries[:3]

    tasks = [asyncio.to_thread(_search, q, categories, engines, num_results) for q in queries]
    all_results = await asyncio.gather(*tasks)

    # Flatten and deduplicate by URL (skip results with empty URLs)
    seen_urls: set[str] = set()
    unique_results: list[dict] = []
    for batch in all_results:
        for r in batch:
            if r.get("url") and r["url"] not in seen_urls:
                seen_urls.add(r["url"])
                unique_results.append(r)

    _log.debug("multi_search queries=%s total_results=%d", queries, len(unique_results))

    # Side-effect: collect results for the pipeline (per-task context variable).
    # Mutate the existing list IN PLACE so that parent tasks (e.g. parallel_research
    # workers that share the same ContextVar snapshot) also see the additions.
    _get_results_list().extend(unique_results)

    if not unique_results:
        return "No results found."

    # Format as readable text for the LLM
    lines = []
    for i, r in enumerate(unique_results, 1):
        lines.append(f"[{i}] {r['title']} | {r['url']} | {r['content']}")
    return "\n".join(lines)


async def search_and_collect(
    queries: list[str], categories: str = "general", engines: str = ""
) -> list[dict]:
    """Search and return raw structured results (for pipeline use, not a tool)."""
    queries = queries[:5]
    tasks = [asyncio.to_thread(_search, q, categories, engines) for q in queries]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    seen_urls: set[str] = set()
    unique: list[dict] = []
    for i, batch in enumerate(raw_results):
        if isinstance(batch, BaseException):
            _log.warning("search query %d/%d failed: %s", i + 1, len(queries), batch)
            continue
        for r in batch:
            if r["url"] not in seen_urls:
                seen_urls.add(r["url"])
                unique.append(r)
    _log.debug("search_and_collect queries=%d results=%d", len(queries), len(unique))
    return unique


@tool
async def web_search(queries: list[str]) -> str:
    """Perform web searches for up to 3 queries in parallel.

    Args:
        queries: An array of search queries to perform web searches for.
    """
    return await _multi_search(queries, categories="general")


@tool
async def academic_search(queries: list[str]) -> str:
    """Perform academic searches for scholarly articles and research. Up to 3 queries.

    Args:
        queries: List of academic search queries.
    """
    return await _multi_search(queries, categories="science", engines="arxiv,google scholar,pubmed")


@tool
async def social_search(queries: list[str]) -> str:
    """Perform social media searches for discussions and trends. Up to 3 queries.

    Args:
        queries: List of social search queries.
    """
    return await _multi_search(queries, categories="social media", engines="reddit")
