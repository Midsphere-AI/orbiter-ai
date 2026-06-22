"""Temporal failure conversion and retry-classification for Exo errors.

This module maps Exo's ``ExoError`` hierarchy onto Temporal's retryability model
so that deterministic / permanent failures are NOT retried while transient ones are.

Retry semantics summary
-----------------------
- **Non-retryable (permanent)**: configuration errors, schema/validation errors,
  parse errors, guard aborts, tool definition errors, quota-agnostic model errors
  (context_length / invalid_request / authentication / permission), and standard
  Python programming errors (ValueError, TypeError, KeyError).
- **Retryable (transient)**: rate-limit / unknown-code ModelError, generic
  ExoError, BackgroundTaskError, network / timeout errors, and anything else not
  explicitly classified as permanent. Unknown exceptions default to retryable=True
  (matching Temporal's built-in default) to avoid silently dropping work.

Usage::

    from exo.distributed.temporal.failure import (  # pyright: ignore[reportMissingImports]
        is_retryable,
        to_application_error,
        NON_RETRYABLE_ERROR_TYPES,
    )

    # In a RetryPolicy:
    from temporalio.common import RetryPolicy
    policy = RetryPolicy(non_retryable_error_types=list(NON_RETRYABLE_ERROR_TYPES))

    # In an activity exception handler:
    raise to_application_error(exc)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from exo.distributed.temporal._compat import (  # pyright: ignore[reportMissingImports]
    HAS_TEMPORAL,
    ApplicationError,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = [
    "NON_RETRYABLE_ERROR_TYPES",
    "is_retryable",
    "to_application_error",
]

# ---------------------------------------------------------------------------
# Non-retryable type names
# ---------------------------------------------------------------------------

#: Bare class names of error types that are deterministic / permanent.
#: Feed this directly into ``RetryPolicy(non_retryable_error_types=...)``.
#: Temporal matches these strings against the ``type`` field on
#: ``ApplicationError`` — they must be exact class names (no module path).
NON_RETRYABLE_ERROR_TYPES: tuple[str, ...] = (
    # --- Exo core — deterministic errors ---
    "ConfigError",
    "ToolError",
    "MaxToolCallsExceeded",
    "GuardAbortError",
    "TaskLoopAbort",
    "RegistryError",
    "LoaderError",
    # --- Exo _internal — structural / parse errors ---
    "OutputParseError",
    "ExpressionError",
    "GraphError",
    "BranchError",
    "LoopError",
    # --- Standard Python programming errors ---
    "ValueError",
    "TypeError",
    "KeyError",
    # --- Pydantic schema validation ---
    "ValidationError",
)


# ---------------------------------------------------------------------------
# Permanent type registry (for isinstance checks)
# ---------------------------------------------------------------------------


def _build_permanent_types() -> tuple[type[BaseException], ...]:
    """Collect the concrete types that map to non-retryable=True.

    This is called once at module import time.  If optional packages are
    missing the corresponding types are simply omitted rather than crashing.
    """
    from exo._internal.background import BackgroundTaskError  # noqa: F401 (import check)
    from exo._internal.branch_node import BranchError  # pyright: ignore[reportMissingImports]
    from exo._internal.expression import ExpressionError  # pyright: ignore[reportMissingImports]
    from exo._internal.graph import GraphError  # pyright: ignore[reportMissingImports]
    from exo._internal.loop_node import LoopError  # pyright: ignore[reportMissingImports]
    from exo._internal.output_parser import (  # pyright: ignore[reportMissingImports]
        OutputParseError,
    )
    from exo.agent import TaskLoopAbort  # pyright: ignore[reportMissingImports]
    from exo.config import ConfigError  # pyright: ignore[reportMissingImports]
    from exo.loader import LoaderError  # pyright: ignore[reportMissingImports]
    from exo.ptc import MaxToolCallsExceeded  # pyright: ignore[reportMissingImports]
    from exo.rail import GuardAbortError  # pyright: ignore[reportMissingImports]
    from exo.registry import RegistryError  # pyright: ignore[reportMissingImports]
    from exo.tool import ToolError  # pyright: ignore[reportMissingImports]

    try:
        from pydantic import ValidationError

        _pydantic_errors: tuple[type[BaseException], ...] = (ValidationError,)
    except ImportError:  # pragma: no cover
        _pydantic_errors = ()

    base: tuple[type[BaseException], ...] = (
        ConfigError,
        ToolError,
        MaxToolCallsExceeded,
        GuardAbortError,
        TaskLoopAbort,
        RegistryError,
        LoaderError,
        OutputParseError,
        ExpressionError,
        GraphError,
        BranchError,
        LoopError,
        ValueError,
        TypeError,
        KeyError,
    )
    return base + _pydantic_errors


_PERMANENT_TYPES: tuple[type[BaseException], ...] = _build_permanent_types()

# ModelError codes that make the error non-retryable even though ModelError
# itself is not in _PERMANENT_TYPES.
_NON_RETRYABLE_MODEL_CODES: frozenset[str] = frozenset(
    {"context_length", "invalid_request", "authentication", "permission"}
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_retryable(exc: BaseException) -> bool:
    """Return ``True`` if *exc* should be retried by Temporal, ``False`` if not.

    Classification rules (evaluated in order):

    1. If ``exo.models.types.ModelError`` is importable and *exc* is one:
       - code in ``{"context_length", "invalid_request", "authentication",
         "permission"}`` → **not** retryable.
       - code ``"rate_limit"`` or unknown → **retryable**.
    2. If *exc* is an instance of any type in ``_PERMANENT_TYPES``
       (ConfigError, ToolError, ValueError, ValidationError, …) → **not** retryable.
    3. Everything else (generic ExoError, BackgroundTaskError, network errors,
       unknown exceptions) → **retryable** (matches Temporal's default behaviour).
    """
    # --- ModelError (optional exo-models package) ---
    try:
        from exo.models.types import (  # pyright: ignore[reportMissingImports]
            ModelError,  # type: ignore[import-not-found]
        )

        if isinstance(exc, ModelError):
            code: str = getattr(exc, "code", "")
            return code not in _NON_RETRYABLE_MODEL_CODES
    except ImportError:
        pass  # exo-models not installed; skip ModelError classification

    # --- Permanent Exo / builtin / pydantic types; default to retryable ---
    return not isinstance(exc, _PERMANENT_TYPES)


def to_application_error(
    exc: BaseException,
    *,
    non_retryable: bool | None = None,
) -> ApplicationError:  # type: ignore[type-arg]
    """Wrap *exc* as a Temporal ``ApplicationError`` suitable for re-raising.

    Parameters
    ----------
    exc:
        The original exception to wrap.
    non_retryable:
        Override the retry classification.  When ``None`` (default) the value
        is derived from :func:`is_retryable`.

    Returns
    -------
    ApplicationError
        A Temporal application error whose ``type`` is the bare class name of
        *exc*, whose ``non_retryable`` flag reflects retry classification, and
        whose ``details`` carry any structured Exo context (``context``, ``hint``,
        ``doc``) when *exc* is an :class:`~exo.types.ExoError`.

    Raises
    ------
    RuntimeError
        If ``temporalio`` is not installed.  Install it with
        ``pip install exo-distributed[temporal]``.
    """
    if not HAS_TEMPORAL or ApplicationError is None:
        raise RuntimeError(
            "temporalio is not installed. Install it with: pip install exo-distributed[temporal]"
        )

    _non_retryable: bool = non_retryable if non_retryable is not None else not is_retryable(exc)

    # Collect structured details from ExoError fields.
    details: list[object] = []
    try:
        from exo.types import ExoError  # pyright: ignore[reportMissingImports]

        if isinstance(exc, ExoError):
            if exc.context:
                details.append({"context": exc.context})
            if exc.hint:
                details.append({"hint": exc.hint})
            if exc.doc:
                details.append({"doc": exc.doc})
    except ImportError:  # pragma: no cover
        pass

    return ApplicationError(
        str(exc),
        *details,
        type=type(exc).__name__,
        non_retryable=_non_retryable,
    )
