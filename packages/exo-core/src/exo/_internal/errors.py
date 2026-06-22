"""Helpers for keeping async failures legible at developer-facing boundaries.

Because Exo is purely async, a single failure often reaches a developer wrapped
in an ``ExceptionGroup`` (from ``asyncio.gather`` / ``TaskGroup``) buried under
asyncio-internal frames. These helpers collapse that noise down to the real
cause so error messages stay readable. See ``error-dx-charter.md``.
"""

from __future__ import annotations


def unwrap_exception_group(exc: BaseException) -> BaseException:
    """Collapse single-child ``ExceptionGroup``s down to the real exception.

    ``asyncio.gather(...)`` and ``TaskGroup`` wrap a lone failure in an
    ``ExceptionGroup``, which surfaces to developers as confusing nested noise.
    When a group holds exactly one leaf exception, return that leaf (recursively,
    so nested single-child groups fully unwrap). Multi-error groups are returned
    untouched — nothing is silently dropped.
    """
    seen: set[int] = set()
    while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
        if id(exc) in seen:  # paranoia: never loop on a self-referential group
            break
        seen.add(id(exc))
        exc = exc.exceptions[0]
    return exc


def describe_exception_group(eg: BaseExceptionGroup) -> str:
    """Render a multi-error group as one compact, readable line.

    e.g. ``"3 errors: TimeoutError: provider timed out; ValueError: bad id; ..."``.
    Used when a group genuinely holds multiple failures and we want a legible
    summary instead of a flattened traceback wall.
    """
    leaves: list[BaseException] = []
    _flatten(eg, leaves)
    shown = leaves[:3]
    rendered = "; ".join(f"{type(e).__name__}: {e}" for e in shown)
    if len(leaves) > len(shown):
        rendered += f"; … (+{len(leaves) - len(shown)} more)"
    return f"{len(leaves)} errors: {rendered}"


def _flatten(exc: BaseException, out: list[BaseException]) -> None:
    if isinstance(exc, BaseExceptionGroup):
        for sub in exc.exceptions:
            _flatten(sub, out)
    else:
        out.append(exc)
