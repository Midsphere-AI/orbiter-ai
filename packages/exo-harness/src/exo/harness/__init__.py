"""Exo Harness: composable orchestration for agent runs."""

from importlib.metadata import PackageNotFoundError, version
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

try:
    __version__: str = version("exo-harness")
except PackageNotFoundError:
    __version__ = "0.1.0"

from exo.harness.base import Harness, HarnessContext, HarnessError, HarnessNode
from exo.harness.checkpoint import CheckpointAdapter
from exo.harness.middleware import CostTrackingMiddleware, Middleware, TimeoutMiddleware
from exo.harness.parallel import SubAgentError
from exo.harness.types import (
    HarnessCheckpoint,
    HarnessEvent,
    SessionState,
    SubAgentResult,
    SubAgentStatus,
    SubAgentTask,
)

__all__: list[str] = [
    "CheckpointAdapter",
    "CostTrackingMiddleware",
    "Harness",
    "HarnessCheckpoint",
    "HarnessContext",
    "HarnessError",
    "HarnessEvent",
    "HarnessNode",
    "Middleware",
    "SessionState",
    "SubAgentError",
    "SubAgentResult",
    "SubAgentStatus",
    "SubAgentTask",
    "TimeoutMiddleware",
]
