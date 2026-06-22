"""Shared resilience primitives for the framework's I/O paths.

Downstream packages (models, mcp, distributed, retrieval) call into transient,
network-bound services. This module gives them one place to get retry-with-
backoff and cancellation-aware error handling, so every package degrades the
same way and surfaces an actionable error on exhaustion. See
``error-dx-charter.md``.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable

from exo.types import ExoError

from .errors import unwrap_exception_group


def is_cancellation(exc: BaseException) -> bool:
    """True if *exc* is (or wraps) an ``asyncio.CancelledError``.

    Cancellation may arrive bare or nested inside an ``ExceptionGroup`` from a
    ``TaskGroup``/``gather``. Use this before a broad ``except`` decides to
    convert an error — cancellation must always be re-raised, never swallowed.
    """
    exc = unwrap_exception_group(exc)
    return isinstance(exc, asyncio.CancelledError)


async def retry_async[T](
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    jitter: float = 0.25,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> T:
    """Call ``fn()`` with exponential backoff + jitter on transient failures.

    Retries up to *attempts* times on exceptions matching *retry_on* (default:
    any ``Exception`` — note this never includes ``asyncio.CancelledError``,
    which is a ``BaseException`` and is always re-raised immediately). The last
    failure is re-raised once attempts are exhausted, so callers should wrap it
    in an actionable ``ExoError`` with ``from exc``.

    Delay before retry *n* (0-indexed) is ``min(max_delay, base_delay * 2**n)``
    plus up to ``jitter`` fraction of random spread to avoid thundering herds.
    """
    if attempts < 1:
        raise ExoError(
            f"retry_async called with attempts={attempts}.",
            hint="attempts must be >= 1.",
        )

    last_exc: BaseException | None = None
    for attempt in range(attempts):
        try:
            return await fn()
        except asyncio.CancelledError:
            raise  # Never retry cancellation — propagate immediately.
        except retry_on as exc:
            last_exc = exc
            if attempt == attempts - 1:
                break
            if on_retry is not None:
                on_retry(attempt, exc)
            delay = min(max_delay, base_delay * (2**attempt))
            if jitter:
                delay += delay * jitter * random.random()
            await asyncio.sleep(delay)

    assert last_exc is not None  # loop ran >= 1 time and never returned
    raise last_exc
