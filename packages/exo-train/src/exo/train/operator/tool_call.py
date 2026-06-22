"""ToolCallOperator — wraps a tool invocation with a tunable description."""

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
class ToolCallTrace:
    """Execution trace entry for a single tool call."""

    operator_name: str
    tool_description: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ToolCallOperator(Operator):
    """Operator that wraps a tool invocation.

    Exposes ``tool_description`` as a tunable parameter so that an
    optimizer can refine how the tool is presented to the LLM.

    The ``tool_fn`` callable receives keyword arguments passed to
    :meth:`execute`.

    Args:
        op_name: Unique identifier for this operator.
        tool_fn: Async callable that executes the tool.
        tool_description: Initial description of the tool (tunable).
    """

    def __init__(
        self,
        op_name: str,
        tool_fn: Any,
        *,
        tool_description: str = "",
        max_traces: int = 1000,
    ) -> None:
        self._name = op_name
        self._tool_fn = tool_fn
        self._tool_description = tool_description
        self._max_traces = max_traces
        self._traces: deque[ToolCallTrace] = deque(maxlen=max_traces)

    @property
    def name(self) -> str:
        return self._name

    @property
    def traces(self) -> list[ToolCallTrace]:
        """Return recorded execution traces (newest last)."""
        return list(self._traces)

    async def execute(self, **kwargs: Any) -> Any:
        """Execute the wrapped tool function.

        All keyword arguments are forwarded to ``tool_fn``.
        """
        start = time.monotonic()
        error: str | None = None
        result: Any = None
        try:
            result = await self._tool_fn(**kwargs)
            return result
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            duration_ms = (time.monotonic() - start) * 1000
            # deque(maxlen=...) evicts the oldest entry automatically — O(1).
            self._traces.append(
                ToolCallTrace(
                    operator_name=self._name,
                    tool_description=self._tool_description,
                    kwargs=kwargs,
                    result=result,
                    error=error,
                    duration_ms=duration_ms,
                    timestamp=time.time(),
                )
            )

    def get_tunables(self) -> list[TunableSpec]:
        return [
            TunableSpec(
                name="tool_description",
                kind=TunableKind.TEXT,
                current_value=self._tool_description,
            ),
        ]

    def get_state(self) -> dict[str, Any]:
        return {"tool_description": self._tool_description}

    def load_state(self, state: dict[str, Any]) -> None:
        self._tool_description = state["tool_description"]
