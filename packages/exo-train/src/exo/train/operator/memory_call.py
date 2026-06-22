"""MemoryCallOperator — wraps memory operations with tunable config."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

from exo.train.operator.base import (  # pyright: ignore[reportMissingImports]
    Operator,
    TunableKind,
    TunableSpec,
)


@dataclass(frozen=True, slots=True)
class MemoryCallTrace:
    """Execution trace entry for a single memory operation."""

    operator_name: str
    enabled: bool
    max_retries: int
    kwargs: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MemoryCallOperator(Operator):
    """Operator that wraps memory operations.

    Exposes ``enabled`` and ``max_retries`` as tunable parameters so
    that an optimizer can control whether memory is used and how
    resilient the retrieval is.

    The ``memory_fn`` callable receives keyword arguments passed to
    :meth:`execute`.  If the operator is disabled (``enabled=False``),
    :meth:`execute` returns ``None`` without calling ``memory_fn``.

    Args:
        op_name: Unique identifier for this operator.
        memory_fn: Async callable that performs the memory operation.
        enabled: Whether the memory operator is active.
        max_retries: Maximum retry attempts on failure.
    """

    def __init__(
        self,
        op_name: str,
        memory_fn: Any,
        *,
        enabled: bool = True,
        max_retries: int = 1,
        max_traces: int = 1000,
    ) -> None:
        if max_retries < 1:
            msg = f"max_retries must be >= 1, got {max_retries}"
            raise ValueError(msg)
        self._name = op_name
        self._memory_fn = memory_fn
        self._enabled = enabled
        self._max_retries = max_retries
        self._max_traces = max_traces
        self._traces: deque[MemoryCallTrace] = deque(maxlen=max_traces)

    @property
    def name(self) -> str:
        return self._name

    @property
    def traces(self) -> list[MemoryCallTrace]:
        """Return recorded execution traces (newest last)."""
        return list(self._traces)

    async def execute(self, **kwargs: Any) -> Any:
        """Execute the memory function with retry logic.

        Returns ``None`` immediately if the operator is disabled.
        Otherwise retries up to ``max_retries`` times on failure.
        """
        start = time.monotonic()
        error: str | None = None
        result: Any = None

        if not self._enabled:
            self._append_trace(
                MemoryCallTrace(
                    operator_name=self._name,
                    enabled=False,
                    max_retries=self._max_retries,
                    kwargs=kwargs,
                    result=None,
                    duration_ms=(time.monotonic() - start) * 1000,
                    timestamp=time.time(),
                )
            )
            return None

        last_exc: Exception | None = None
        for _attempt in range(self._max_retries):
            try:
                result = await self._memory_fn(**kwargs)
                error = None
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                error = str(exc)
        else:
            # All retries exhausted
            duration_ms = (time.monotonic() - start) * 1000
            self._append_trace(
                MemoryCallTrace(
                    operator_name=self._name,
                    enabled=self._enabled,
                    max_retries=self._max_retries,
                    kwargs=kwargs,
                    result=None,
                    error=error,
                    duration_ms=duration_ms,
                    timestamp=time.time(),
                )
            )
            if last_exc is not None:
                raise last_exc
            return None

        duration_ms = (time.monotonic() - start) * 1000
        self._append_trace(
            MemoryCallTrace(
                operator_name=self._name,
                enabled=self._enabled,
                max_retries=self._max_retries,
                kwargs=kwargs,
                result=result,
                error=error,
                duration_ms=duration_ms,
                timestamp=time.time(),
            )
        )
        return result

    def _append_trace(self, trace: MemoryCallTrace) -> None:
        """Append *trace* to the ring buffer; deque(maxlen=...) evicts oldest O(1)."""
        self._traces.append(trace)

    def get_tunables(self) -> list[TunableSpec]:
        return [
            TunableSpec(
                name="enabled",
                kind=TunableKind.DISCRETE,
                current_value=self._enabled,
                constraints={"choices": [True, False]},
            ),
            TunableSpec(
                name="max_retries",
                kind=TunableKind.DISCRETE,
                current_value=self._max_retries,
                constraints={"min": 1, "max": 10},
            ),
        ]

    def get_state(self) -> dict[str, Any]:
        return {
            "enabled": self._enabled,
            "max_retries": self._max_retries,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        max_retries = state["max_retries"]
        if max_retries < 1:
            msg = f"max_retries must be >= 1, got {max_retries}"
            raise ValueError(msg)
        self._enabled = state["enabled"]
        self._max_retries = max_retries
