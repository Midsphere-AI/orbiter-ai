"""Error taxonomy for exo-search.

All public errors in this package subclass ``ExoError`` so callers see a
consistent, structured error with context and actionable hints rather than
raw ``ValueError`` or unwrapped ``ExceptionGroup`` noise.
"""

from __future__ import annotations

from exo.types import ExoError  # pyright: ignore[reportMissingImports]


class SearchError(ExoError):
    """Raised when a search pipeline stage fails.

    Carries the stage name, mode, and query so the developer can pinpoint
    which part of the pipeline threw without reading asyncio internals.
    """


class SearchConfigError(ExoError):
    """Raised when ``SearchConfig`` receives invalid or conflicting arguments."""
