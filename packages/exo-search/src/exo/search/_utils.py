"""Shared utilities for exo-search agents."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def resolve_provider(model: str):
    """Resolve an LLM provider for the given model string.

    Returns the provider instance, or ``None`` when resolution fails
    (e.g. missing API key, unknown model).  Callers should pass ``None``
    directly to ``run()`` — it will fall back to environment-based
    auto-detection.

    Args:
        model: Model string in ``"provider:model"`` format.
    """
    try:
        from exo.models import get_provider

        return get_provider(model)
    except Exception as exc:
        logger.warning("provider resolution failed for %s: %s", model, exc)
        return None
