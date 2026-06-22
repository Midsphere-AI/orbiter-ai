"""LLMCallOperator — wraps an LLM call with tunable prompts."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

from exo.train.operator.base import (  # pyright: ignore[reportMissingImports]
    Operator,
    TunableKind,
    TunableSpec,
)


@dataclass(frozen=True, slots=True)
class LLMCallTrace:
    """Execution trace entry for a single LLM call."""

    operator_name: str
    system_prompt: str
    user_prompt: str
    result: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LLMCallOperator(Operator):
    """Operator that wraps an LLM call.

    Exposes ``system_prompt`` and ``user_prompt`` as tunable parameters
    so that an optimizer can iteratively refine them.

    The ``llm_fn`` callable receives keyword arguments including
    ``system_prompt`` and ``user_prompt``, plus any extra kwargs passed
    to :meth:`execute`.

    Args:
        op_name: Unique identifier for this operator.
        llm_fn: Async callable that performs the LLM call.
        system_prompt: Initial system prompt.
        user_prompt: Initial user prompt template.
    """

    def __init__(
        self,
        op_name: str,
        llm_fn: Any,
        *,
        system_prompt: str = "",
        user_prompt: str = "",
        max_traces: int = 1000,
    ) -> None:
        self._name = op_name
        self._llm_fn = llm_fn
        self._system_prompt = system_prompt
        self._user_prompt = user_prompt
        self._max_traces = max_traces
        self._traces: deque[LLMCallTrace] = deque(maxlen=max_traces)

    @property
    def name(self) -> str:
        return self._name

    @property
    def traces(self) -> list[LLMCallTrace]:
        """Return recorded execution traces (newest last)."""
        return list(self._traces)

    async def execute(self, **kwargs: Any) -> Any:
        """Call the LLM function with current prompts.

        Keyword arguments are forwarded to ``llm_fn`` alongside the
        current ``system_prompt`` and ``user_prompt``.
        """
        start = time.monotonic()
        error: str | None = None
        result: Any = None
        try:
            result = await self._llm_fn(
                system_prompt=self._system_prompt,
                user_prompt=self._user_prompt,
                **kwargs,
            )
            return result
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            duration_ms = (time.monotonic() - start) * 1000
            # deque(maxlen=...) evicts the oldest entry automatically — O(1).
            self._traces.append(
                LLMCallTrace(
                    operator_name=self._name,
                    system_prompt=self._system_prompt,
                    user_prompt=self._user_prompt,
                    result=result,
                    error=error,
                    duration_ms=duration_ms,
                    timestamp=time.time(),
                )
            )

    def get_tunables(self) -> list[TunableSpec]:
        return [
            TunableSpec(
                name="system_prompt",
                kind=TunableKind.PROMPT,
                current_value=self._system_prompt,
            ),
            TunableSpec(
                name="user_prompt",
                kind=TunableKind.PROMPT,
                current_value=self._user_prompt,
            ),
        ]

    def get_state(self) -> dict[str, Any]:
        return {
            "system_prompt": self._system_prompt,
            "user_prompt": self._user_prompt,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self._system_prompt = state["system_prompt"]
        self._user_prompt = state["user_prompt"]
