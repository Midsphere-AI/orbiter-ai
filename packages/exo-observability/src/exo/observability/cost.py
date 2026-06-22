"""Cost estimation and tracking for LLM usage.

Provides a price table of common models, cost calculation from token usage,
a stateful tracker that accumulates costs across multiple calls, and a
helper to stamp cost attributes onto the current OpenTelemetry span.

Usage::

    from exo.observability.cost import estimate_cost, CostTracker, stamp_cost_on_current_span

    result = estimate_cost("openai:gpt-4o", input_tokens=1000, output_tokens=500)
    print(result.total_usd)

    tracker = CostTracker()
    tracker.add("anthropic:claude-sonnet-4-6", usage)
    print(tracker.total_usd)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Price table
# ---------------------------------------------------------------------------

# Prices are indicative public list prices (USD per 1 million tokens).
# They may change — use register_price() to override at runtime.
# Sources: provider pricing pages as of 2025-06. All keys are lowercase.
PRICE_TABLE: dict[str, tuple[float, float]] = {
    # --- OpenAI ---
    "openai:gpt-4o": (2.50, 10.00),
    "openai:gpt-4o-mini": (0.15, 0.60),
    "openai:gpt-4.5": (75.00, 150.00),
    "openai:gpt-4-turbo": (10.00, 30.00),
    "openai:gpt-3.5-turbo": (0.50, 1.50),
    "openai:o1": (15.00, 60.00),
    "openai:o1-mini": (3.00, 12.00),
    "openai:o1-pro": (150.00, 600.00),
    "openai:o3": (10.00, 40.00),
    "openai:o3-mini": (1.10, 4.40),
    "openai:o4-mini": (1.10, 4.40),
    # --- Anthropic ---
    "anthropic:claude-opus-4": (15.00, 75.00),
    "anthropic:claude-opus-3": (15.00, 75.00),
    "anthropic:claude-sonnet-4-6": (3.00, 15.00),
    "anthropic:claude-sonnet-4-5": (3.00, 15.00),
    "anthropic:claude-sonnet-3-5": (3.00, 15.00),
    "anthropic:claude-haiku-3-5": (0.80, 4.00),
    "anthropic:claude-haiku-3": (0.25, 1.25),
    # --- Google ---
    "google:gemini-2.5-flash": (0.15, 0.60),
    "google:gemini-2.5-pro": (1.25, 10.00),
    "google:gemini-2.0-flash": (0.10, 0.40),
    "google:gemini-1.5-flash": (0.075, 0.30),
    "google:gemini-1.5-pro": (1.25, 5.00),
}


# ---------------------------------------------------------------------------
# Usage protocol — matches exo.core's Usage model without importing it
# ---------------------------------------------------------------------------


@runtime_checkable
class _UsageLike(Protocol):
    input_tokens: int
    output_tokens: int


# ---------------------------------------------------------------------------
# CostResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CostResult:
    """The cost breakdown for a single LLM call.

    Args:
        input_cost_usd: Cost attributed to input (prompt) tokens.
        output_cost_usd: Cost attributed to output (completion) tokens.
        total_usd: Sum of input and output costs.
        model: The model string used for lookup (as provided).
        priced: False when the model was not found in the price table — all
            cost fields will be 0.0 in that case.
        input_tokens: Raw input token count (mirrors the value passed to
            :func:`estimate_cost`).
        output_tokens: Raw output token count.
    """

    input_cost_usd: float
    output_cost_usd: float
    total_usd: float
    model: str
    priced: bool
    input_tokens: int = 0
    output_tokens: int = 0


# ---------------------------------------------------------------------------
# Internal lookup helper
# ---------------------------------------------------------------------------


def _lookup_price(model: str) -> tuple[float, float] | None:
    """Return ``(input_per_1m, output_per_1m)`` or ``None`` if unknown.

    Tries, in order:
    1. Exact match (case-insensitive).
    2. Match on the part after the first ``":"`` (bare model name).
    """
    lower = model.lower()
    # 1. Exact case-insensitive match
    for key, prices in PRICE_TABLE.items():
        if key.lower() == lower:
            return prices
    # 2. Bare model name (strip provider prefix from the lookup key)
    bare = lower.split(":", 1)[-1]
    for key, prices in PRICE_TABLE.items():
        if key.split(":", 1)[-1].lower() == bare:
            return prices
    return None


# ---------------------------------------------------------------------------
# Public estimation functions
# ---------------------------------------------------------------------------


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> CostResult:
    """Estimate the USD cost for a single LLM call.

    Args:
        model: Model string in ``"provider:model"`` format (or bare model name).
        input_tokens: Number of prompt/input tokens.
        output_tokens: Number of completion/output tokens.

    Returns:
        A :class:`CostResult`.  If the model is not in the price table,
        ``priced=False`` and all cost fields are 0.0 — no exception is raised.
    """
    prices = _lookup_price(model)
    if prices is None:
        _log.debug("No price entry for model %r — cost will be zero", model)
        return CostResult(
            input_cost_usd=0.0,
            output_cost_usd=0.0,
            total_usd=0.0,
            model=model,
            priced=False,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    input_per_1m, output_per_1m = prices
    input_cost = input_tokens * input_per_1m / 1_000_000
    output_cost = output_tokens * output_per_1m / 1_000_000
    return CostResult(
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        total_usd=input_cost + output_cost,
        model=model,
        priced=True,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def estimate_cost_from_usage(model: str, usage: _UsageLike) -> CostResult:
    """Estimate cost from a :class:`~exo.types.Usage` (or any duck-typed equivalent).

    Args:
        model: Model string in ``"provider:model"`` format.
        usage: Any object exposing ``input_tokens`` and ``output_tokens`` ints.

    Returns:
        A :class:`CostResult`.
    """
    return estimate_cost(model, usage.input_tokens, usage.output_tokens)


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------


class CostTracker:
    """Accumulates cost across multiple LLM calls.

    Thread-safety is not guaranteed — use a single tracker per async task or
    add external locking when sharing across threads.

    Example::

        tracker = CostTracker()
        for model, usage in calls:
            tracker.add(model, usage)
        print(f"Total: ${tracker.total_usd:.6f}")
        print(tracker.by_model)
    """

    def __init__(self) -> None:
        self._total_usd: float = 0.0
        self._by_model: dict[str, float] = {}
        self._input_tokens: int = 0
        self._output_tokens: int = 0

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, model: str, usage: _UsageLike) -> CostResult:
        """Record a single LLM call and return its :class:`CostResult`.

        Args:
            model: The model string.
            usage: A :class:`~exo.types.Usage` or any object with
                ``input_tokens`` / ``output_tokens`` attributes.

        Returns:
            The :class:`CostResult` for this call (also reflected in
            accumulated totals).
        """
        result = estimate_cost_from_usage(model, usage)
        self._total_usd += result.total_usd
        # Normalize key to lowercase so case-insensitive _lookup_price matches
        # consistently against _by_model entries.
        model_key = model.lower()
        self._by_model[model_key] = self._by_model.get(model_key, 0.0) + result.total_usd
        self._input_tokens += usage.input_tokens
        self._output_tokens += usage.output_tokens
        return result

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._total_usd = 0.0
        self._by_model.clear()
        self._input_tokens = 0
        self._output_tokens = 0

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def total_usd(self) -> float:
        """Total accumulated cost in USD."""
        return self._total_usd

    @property
    def by_model(self) -> dict[str, float]:
        """Per-model accumulated cost (USD), keyed by model string."""
        return dict(self._by_model)

    @property
    def input_tokens(self) -> int:
        """Total accumulated input tokens."""
        return self._input_tokens

    @property
    def output_tokens(self) -> int:
        """Total accumulated output tokens."""
        return self._output_tokens

    def __repr__(self) -> str:
        return (
            f"CostTracker(total_usd={self._total_usd:.6f}, "
            f"models={list(self._by_model)}, "
            f"input_tokens={self._input_tokens}, "
            f"output_tokens={self._output_tokens})"
        )


# ---------------------------------------------------------------------------
# OTel span stamping
# ---------------------------------------------------------------------------


def stamp_cost_on_current_span(result: CostResult) -> None:
    """Set cost attributes on the active OpenTelemetry span (no-op if absent).

    Sets the following span attributes (from :mod:`exo.observability.semconv`):

    - ``exo.cost.input_tokens``  — integer input token count implied by cost
    - ``exo.cost.output_tokens`` — integer output token count implied by cost
    - ``exo.cost.total_usd``     — total USD cost (float)

    This function is a no-op when:
    - ``opentelemetry`` is not installed.
    - There is no active span (the span is a ``NonRecordingSpan``).
    - ``result.priced`` is ``False``.

    Args:
        result: A :class:`CostResult` from :func:`estimate_cost` or
            :meth:`CostTracker.add`.
    """
    if not result.priced:
        return
    try:
        import opentelemetry.trace as _trace

        from exo.observability import semconv  # pyright: ignore[reportMissingImports]

        span = _trace.get_current_span()
        if not span.is_recording():
            return
        span.set_attribute(semconv.COST_INPUT_TOKENS, result.input_tokens)
        span.set_attribute(semconv.COST_OUTPUT_TOKENS, result.output_tokens)
        span.set_attribute(semconv.COST_TOTAL_USD, result.total_usd)
    except ImportError:
        _log.debug("stamp_cost_on_current_span: opentelemetry not installed, skipping")


# ---------------------------------------------------------------------------
# Price registration
# ---------------------------------------------------------------------------


def register_price(model: str, input_per_1m: float, output_per_1m: float) -> None:
    """Register or override a model's price in the global :data:`PRICE_TABLE`.

    Args:
        model: Model string in ``"provider:model"`` format (or bare name).
        input_per_1m: Input token price in USD per 1 million tokens.
        output_per_1m: Output token price in USD per 1 million tokens.

    Example::

        register_price("openai:gpt-5", input_per_1m=5.0, output_per_1m=20.0)
    """
    PRICE_TABLE[model] = (input_per_1m, output_per_1m)
    _log.debug(
        "Registered price for %r: $%.4f/$%.4f per 1M tokens", model, input_per_1m, output_per_1m
    )
