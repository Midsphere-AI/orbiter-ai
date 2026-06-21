"""Web page content extraction helpers.

Fetch and extract readable text content from web pages, stripping
navigation, scripts, and other non-content elements.

The public ``@tool``-decorated entry point is ``scrape_url``.  All other
functions in this module are internal pipeline helpers.
"""

from __future__ import annotations

import asyncio
import html
import json
import os
import re
from urllib.parse import quote, urlparse

from exo import tool
from exo.observability.logging import get_logger  # pyright: ignore[reportMissingImports]

_log = get_logger(__name__)

_MAX_CHARS = 10_000

_CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# PDF extraction helpers
# ---------------------------------------------------------------------------


def _is_pdf_url(url: str) -> bool:
    """Check if a URL likely points to a PDF document."""
    parsed = urlparse(url)
    path_lower = parsed.path.lower()
    return path_lower.endswith(".pdf") or "/pdf/" in path_lower


def _fetch_pdf(url: str, max_chars: int = _MAX_CHARS) -> str:
    """Download a PDF and extract text via PyMuPDF (fitz).

    Returns ``""`` if pymupdf is not installed or on any failure so callers
    can fall back to other extraction methods.
    """
    try:
        import fitz
    except ImportError:
        _log.debug("pymupdf not installed, skipping PDF extraction for %s", url)
        return ""

    import urllib.request

    _log.debug("fetch_pdf url=%r", url)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _CHROME_UA})
        with urllib.request.urlopen(req, timeout=30) as resp:
            pdf_bytes = resp.read()
    except Exception as exc:
        _log.warning("pdf download failed for %s: %s", url, exc)
        return ""

    # Verify PDF magic header
    if pdf_bytes[:5] != b"%PDF-":
        _log.debug("response is not a PDF (missing %%PDF- header) for %s", url)
        return ""

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        _log.warning("pdf open failed for %s: %s", url, exc)
        return ""

    parts: list[str] = []
    total = 0
    char_limit = max_chars * 2  # read extra then truncate for better boundaries
    try:
        for page in doc:
            page_text = page.get_text()
            parts.append(page_text)
            total += len(page_text)
            if total > char_limit:
                break
    except Exception as exc:
        _log.warning("pdf text extraction failed for %s: %s", url, exc)
        return ""
    finally:
        doc.close()

    text = "\n".join(parts).strip()
    if not text:
        return ""

    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n... [truncated]"
    _log.debug("pdf extracted %d chars from %s", len(text), url)
    return text


def _smart_truncate(text: str, query: str, max_chars: int) -> tuple[str, float]:
    """Truncate content keeping sections most relevant to the query.

    Instead of cutting at a fixed character count, splits the content into
    sections (by headings or paragraph breaks), scores each by keyword
    relevance to the query, and keeps the most relevant sections up to
    the character budget. Always keeps the first section (usually intro).

    Returns:
        A tuple of ``(truncated_text, omission_ratio)`` where
        ``omission_ratio`` is ``0.0`` when nothing was omitted and
        approaches ``1.0`` when most content was dropped.
    """
    if len(text) <= max_chars:
        return text, 0.0

    original_len = len(text)

    # Extract query keywords for relevance scoring
    query_words = {w.lower() for w in re.split(r"\W+", query) if len(w) > 2}
    if not query_words:
        truncated = text[:max_chars] + "\n\n... [truncated]"
        return truncated, 1.0 - max_chars / original_len

    # Split by markdown headings or double newlines
    section_pattern = re.compile(r"(?:^|\n)(?=#{1,4}\s|\n\n)", re.MULTILINE)
    sections = section_pattern.split(text)
    sections = [s.strip() for s in sections if s.strip()]
    if not sections:
        truncated = text[:max_chars] + "\n\n... [truncated]"
        return truncated, 1.0 - max_chars / original_len

    # Score each section by keyword overlap
    scored: list[tuple[float, int, str]] = []
    for idx, section in enumerate(sections):
        section_lower = section.lower()
        hits = sum(1 for w in query_words if w in section_lower)
        score = hits / len(query_words) if query_words else 0
        # Boost first section (usually intro/summary)
        if idx == 0:
            score += 1.0
        scored.append((score, idx, section))

    # Sort by score descending, keep original order for output
    scored.sort(key=lambda x: -x[0])

    kept_indices: set[int] = set()
    kept_chars = 0
    for _score, idx, section in scored:
        if kept_chars + len(section) > max_chars:
            # Try to fit a partial section if it's the first one
            remaining = max_chars - kept_chars
            if remaining > 200 and idx not in kept_indices:
                kept_indices.add(idx)
                kept_chars += remaining  # will truncate this section
            break
        kept_indices.add(idx)
        kept_chars += len(section)

    # Reassemble in original order
    output_parts: list[str] = []
    omitted_chars = 0
    for idx, section in enumerate(sections):
        if idx in kept_indices:
            # Truncate last section if over budget
            if len("\n\n".join(output_parts)) + len(section) > max_chars:
                remaining = max_chars - len("\n\n".join(output_parts))
                if remaining > 200:
                    output_parts.append(section[:remaining] + "...")
            else:
                output_parts.append(section)
        else:
            omitted_chars += len(section)

    result = "\n\n".join(output_parts)
    if omitted_chars > 0:
        result += f"\n\n[...{omitted_chars} chars omitted from less relevant sections]"

    omission_ratio = omitted_chars / original_len if original_len > 0 else 0.0
    return result, omission_ratio


# ---------------------------------------------------------------------------
# Alternative fetch for heavily-truncated pages
# ---------------------------------------------------------------------------

_SIMPLE_UA = "Mozilla/5.0 (compatible; ExoSearch/1.0)"


def _try_alternative_fetch(url: str, query: str, max_chars: int = _MAX_CHARS) -> str:
    """Try alternative strategies to fetch content when the primary fetch was truncated.

    Strategy A: For Archive.org URLs, try the raw djvu text endpoint.
    Strategy B: Re-fetch with a simpler user-agent for lighter page versions.

    Returns ``""`` if nothing works.
    """
    import urllib.request

    # Strategy A: Archive.org djvu text
    parsed = urlparse(url)
    if "archive.org" in parsed.netloc and "/details/" in parsed.path:
        # Extract item ID from /details/ITEM_ID
        parts = parsed.path.rstrip("/").split("/")
        try:
            idx = parts.index("details")
            item_id = parts[idx + 1]
        except (ValueError, IndexError):
            item_id = ""
        if item_id:
            djvu_url = f"https://archive.org/stream/{item_id}/{item_id}_djvu.txt"
            _log.debug("trying archive.org djvu text: %s", djvu_url)
            try:
                req = urllib.request.Request(djvu_url, headers={"User-Agent": _SIMPLE_UA})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    text = resp.read().decode("utf-8", errors="replace")
                if text and len(text) > 100:
                    if len(text) > max_chars:
                        text, _ = _smart_truncate(text, query, max_chars)
                    return text
            except Exception as exc:
                _log.debug("archive.org djvu fetch failed: %s", exc)

    # Strategy B: Simpler user-agent for lighter page versions
    safe_url = quote(url, safe=":/?#[]@!$&'()*+,;=-._~%")
    try:
        req = urllib.request.Request(safe_url, headers={"User-Agent": _SIMPLE_UA})
        with urllib.request.urlopen(req, timeout=5) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        _log.debug("simple UA fetch failed for %s: %s", url, exc)
        return ""

    # Strip HTML like _fetch_page_fallback
    text = re.sub(r"<script[^>]*>.*?</script>", "", raw, flags=re.S)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text or len(text) < 100:
        return ""
    if len(text) > max_chars:
        text, _ = _smart_truncate(text, query, max_chars)
    return text


def _fetch_via_jina(
    url: str,
    jina_url: str,
    max_chars: int = _MAX_CHARS,
    api_key: str = "",
    query: str = "",
) -> str:
    """Fetch page content as clean markdown via Jina Reader.

    When ``api_key`` is set, uses the Jina Cloud Reader API (``r.jina.ai``)
    for faster fetches with image captioning. Otherwise falls back to a
    self-hosted Jina Reader instance at ``jina_url``.
    """
    _log.debug("fetch url=%r method=%s", url, "jina_cloud" if api_key else "jina_self_hosted")
    if api_key:
        from .jina import jina_reader_fetch

        return jina_reader_fetch(url, api_key, max_chars)

    import urllib.request

    reader_url = f"{jina_url}/{url}"
    req = urllib.request.Request(reader_url, headers={"X-Respond-With": "markdown"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        _log.warning("jina reader failed for %s: %s", url, exc)
        return ""
    if len(text) > max_chars:
        if query:
            text, omission_ratio = _smart_truncate(text, query, max_chars)
            # If heavy truncation, try alternative fetch for fuller content
            if omission_ratio > 0.3 and query:
                alt = _try_alternative_fetch(url, query, max_chars)
                if alt and len(alt) > len(text) * 0.5:
                    _log.debug(
                        "using alternative fetch for %s (omission=%.1f%%, alt=%d chars)",
                        url,
                        omission_ratio * 100,
                        len(alt),
                    )
                    text = alt
        else:
            text = text[:max_chars] + "\n\n... [truncated]"
    return text


def _fetch_page_fallback(url: str) -> str:
    """Fetch a web page via direct HTTP and extract text with regex stripping."""
    _log.debug("fetch fallback url=%r", url)
    import ssl
    import urllib.request

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"Error: unsupported scheme '{parsed.scheme}'. Use http or https."

    safe_url = quote(url, safe=":/?#[]@!$&'()*+,;=-._~%")
    req = urllib.request.Request(safe_url, headers={"User-Agent": _CHROME_UA})
    content_type = ""
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw_bytes = resp.read()
    except Exception as exc:
        # Retry with unverified SSL on certificate errors (common on .gov.in sites)
        is_ssl = "CERTIFICATE_VERIFY_FAILED" in str(exc) or "SSL" in type(exc).__name__
        if not is_ssl:
            _log.warning("fetch fallback failed for %s: %s", url, exc)
            return f"Error fetching {url}: {exc}"
        _log.debug("fetch SSL verify failed, retrying without verification: %s", url)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        try:
            req2 = urllib.request.Request(safe_url, headers={"User-Agent": _CHROME_UA})
            with urllib.request.urlopen(req2, timeout=30, context=ctx) as resp:
                content_type = resp.headers.get("Content-Type", "")
                raw_bytes = resp.read()
        except Exception as exc2:
            _log.warning("fetch fallback failed for %s: %s", url, exc2)
            return f"Error fetching {url}: {exc2}"

    # Detect PDF from Content-Type or magic header and delegate to PDF extractor
    is_pdf = "pdf" in content_type.lower() or raw_bytes[:5] == b"%PDF-"
    if is_pdf:
        _log.debug("fallback detected PDF for %s, delegating to _fetch_pdf", url)
        pdf_text = _fetch_pdf(url)
        if pdf_text:
            return pdf_text

    raw = raw_bytes.decode("utf-8", errors="replace")

    text = re.sub(r"<script[^>]*>.*?</script>", "", raw, flags=re.S)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.S)
    text = re.sub(r"<nav[^>]*>.*?</nav>", "", text, flags=re.S)
    text = re.sub(r"<footer[^>]*>.*?</footer>", "", text, flags=re.S)
    text = re.sub(r"<header[^>]*>.*?</header>", "", text, flags=re.S)

    article_match = re.search(r"<article[^>]*>(.*?)</article>", text, flags=re.S)
    main_match = re.search(r"<main[^>]*>(.*?)</main>", text, flags=re.S)
    if article_match:
        text = article_match.group(1)
    elif main_match:
        text = main_match.group(1)

    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) > _MAX_CHARS:
        text = text[:_MAX_CHARS] + "... [truncated]"
    return text


def _fetch_page(url: str) -> str:
    """Fetch a page, trying PDF extraction then Jina Cloud / self-hosted Reader then fallback."""
    # Try PDF extraction first for likely PDF URLs
    if _is_pdf_url(url):
        pdf_text = _fetch_pdf(url)
        if pdf_text:
            return pdf_text

    jina_api_key = os.environ.get("JINA_API_KEY", "")
    if jina_api_key:
        result = _fetch_via_jina(url, "", api_key=jina_api_key)
        if result:
            return result
    jina_url = os.environ.get("JINA_READER_URL", "")
    if jina_url:
        result = _fetch_via_jina(url, jina_url)
        if result:
            return result
    return _fetch_page_fallback(url)


async def fetch_page_content(url: str) -> str:
    """Fetch a web page and return its extracted text content.

    Strips scripts, styles, navigation, headers, and footers.  Prefers
    content from <article> or <main> tags when available.

    Args:
        url: Fully-qualified URL to fetch (e.g. https://example.com).
    """
    return await asyncio.to_thread(_fetch_page, url)


async def fetch_multiple_pages(urls_json: str) -> str:
    """Fetch multiple web pages concurrently and return their combined content.

    Args:
        urls_json: JSON array of URL strings to fetch (max 3).
    """
    try:
        urls = json.loads(urls_json)
    except (json.JSONDecodeError, TypeError):
        return "Error: urls_json must be a valid JSON array of URL strings."

    if not isinstance(urls, list):
        return "Error: urls_json must be a JSON array."

    # Limit to 3 concurrent fetches.
    urls = urls[:3]

    tasks = [asyncio.to_thread(_fetch_page, url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    sections: list[str] = []
    for url, result in zip(urls, results, strict=True):
        sections.append(f"=== {url} ===")
        if isinstance(result, Exception):
            sections.append(f"Error fetching {url}: {result}")
        else:
            sections.append(result)
        sections.append("")

    return "\n".join(sections).strip()


@tool
async def scrape_url(urls: list[str]) -> str:
    """Scrape and extract content from up to 3 URLs.

    Args:
        urls: A list of URLs to scrape content from.
    """
    urls = urls[:3]
    tasks = [asyncio.to_thread(_fetch_page, url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    sections = []
    for url, result in zip(urls, results, strict=True):
        if isinstance(result, Exception):
            sections.append(f"Error fetching {url}: {result}")
        else:
            sections.append(f"=== {url} ===\n{result}")
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Pipeline helper — enrich SearchResults with full page content
# ---------------------------------------------------------------------------


async def enrich_results(
    results: list,
    jina_url: str,
    max_results: int = 5,
    jina_api_key: str = "",
    query: str = "",
    max_chars: int = _MAX_CHARS,
) -> list:
    """Scrape full page content for the top results via Jina Reader.

    Returns a new list with enriched content for the first ``max_results``
    items and original snippets for the rest. Skips results already marked
    as ``enriched`` (e.g., from Jina Search). Falls back gracefully when
    Jina Reader is unavailable or a page fails to fetch.
    """
    from ..types import SearchResult

    if (not jina_url and not jina_api_key) or not results:
        return results

    to_enrich = results[:max_results]
    _log.debug("enrich targets=%d/%d", min(max_results, len(results)), len(results))

    async def _timed_fetch(url: str) -> str:
        # Route PDF URLs directly to PDF extractor instead of Jina
        if _is_pdf_url(url):
            pdf_text = await asyncio.wait_for(
                asyncio.to_thread(_fetch_pdf, url, max_chars),
                timeout=15.0,
            )
            if pdf_text:
                return pdf_text
        return await asyncio.wait_for(
            asyncio.to_thread(
                _fetch_via_jina,
                url,
                jina_url,
                max_chars=max_chars,
                api_key=jina_api_key,
                query=query,
            ),
            timeout=8.0,
        )

    enriched: list[SearchResult] = []
    tasks_with_indices: list[tuple[int, asyncio.Task]] = []  # type: ignore[type-arg]
    for i, r in enumerate(to_enrich):
        if r.enriched:
            enriched.append(r)
        else:
            enriched.append(r)  # placeholder
            tasks_with_indices.append((i, asyncio.ensure_future(_timed_fetch(r.url))))

    if tasks_with_indices:
        task_results = await asyncio.gather(
            *[t for _, t in tasks_with_indices],
            return_exceptions=True,
        )
        # Track indices that got thin content for re-fetch
        thin_indices: list[int] = []
        for (idx, _), content in zip(tasks_with_indices, task_results, strict=True):
            r = to_enrich[idx]
            if isinstance(content, str) and content:
                enriched[idx] = SearchResult(
                    title=r.title,
                    url=r.url,
                    content=content,
                    enriched=True,
                )
                if len(content) < 500 and query:
                    thin_indices.append(idx)

        # Re-fetch thin results via alternative strategies
        if thin_indices:
            _log.debug("re-fetching %d thin results via alternative strategy", len(thin_indices))
            alt_tasks = [
                asyncio.wait_for(
                    asyncio.to_thread(_try_alternative_fetch, to_enrich[idx].url, query, max_chars),
                    timeout=5.0,
                )
                for idx in thin_indices
            ]
            alt_results = await asyncio.gather(*alt_tasks, return_exceptions=True)
            for idx, alt in zip(thin_indices, alt_results, strict=True):
                if isinstance(alt, str) and alt and len(alt) > len(enriched[idx].content):
                    r = to_enrich[idx]
                    enriched[idx] = SearchResult(
                        title=r.title,
                        url=r.url,
                        content=alt,
                        enriched=True,
                    )

    enriched_count = sum(1 for r in enriched[: len(to_enrich)] if r.enriched)
    _log.info("enriched %d/%d pages", enriched_count, len(to_enrich))

    enriched.extend(results[max_results:])
    return enriched
