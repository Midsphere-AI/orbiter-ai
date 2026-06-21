"""Backward-compatible re-exports of the canonical trajectory types.

The authoritative definitions live in ``exo.eval.trajectory``.  This module
re-exports everything so that existing code importing from ``exo.train.trajectory``
continues to work without modification.
"""

from exo.eval.trajectory import (  # pyright: ignore[reportMissingImports]
    DefaultStrategy,
    TrajectoryDataset,
    TrajectoryError,
    TrajectoryExtractor,
    TrajectoryItem,
    TrajectoryStrategy,
)

__all__: list[str] = [
    "DefaultStrategy",
    "TrajectoryDataset",
    "TrajectoryError",
    "TrajectoryExtractor",
    "TrajectoryItem",
    "TrajectoryStrategy",
]
