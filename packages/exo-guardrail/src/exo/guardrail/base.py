"""BaseGuardrail: hook-based guardrail that integrates with Agent's HookManager."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from exo.guardrail.types import (  # pyright: ignore[reportMissingImports]
    GuardrailBackend,
    GuardrailError,
    GuardrailResult,
    RiskLevel,
)
from exo.hooks import Hook, HookPoint

if TYPE_CHECKING:
    from exo.agent import Agent


# Risk levels that trigger automatic blocking.
_BLOCKING_LEVELS = frozenset({RiskLevel.HIGH, RiskLevel.CRITICAL})


class BaseGuardrail:
    """A guardrail that registers itself as hooks on an Agent's HookManager.

    Args:
        backend: Detection backend.  Must not be ``None`` — passing ``None``
            raises ``ValueError`` at construction time to prevent silent
            no-op guardrails from giving false assurance.
        events: Hook point names (e.g. ``["pre_llm_call"]``) to monitor.
            Only these events will have hooks registered.

    Example::

        backend = MyDetectionBackend()
        guard = BaseGuardrail(backend=backend, events=["pre_llm_call"])
        guard.attach(agent)
        # ... agent runs, guardrail hooks fire automatically
        guard.detach(agent)
    """

    def __init__(
        self,
        backend: GuardrailBackend | None = None,
        events: list[str] | None = None,
    ) -> None:
        if backend is None:
            raise ValueError(
                "BaseGuardrail requires a backend. "
                "Passing backend=None creates a silent no-op guardrail that gives false "
                "assurance of protection. Supply a GuardrailBackend instance instead."
            )
        self.backend = backend
        self.events = events or []
        # Track hooks per agent so detach can remove exactly the right ones.
        self._hooks: dict[int, dict[HookPoint, Hook]] = {}

    def _resolve_hook_points(self) -> list[HookPoint]:
        """Convert string event names to HookPoint enum values."""
        points: list[HookPoint] = []
        for name in self.events:
            try:
                points.append(HookPoint(name))
            except ValueError:
                msg = f"Unknown hook point: {name!r}"
                raise ValueError(msg) from None
        return points

    def attach(self, agent: Agent) -> None:
        """Register guardrail hooks on an agent's hook_manager.

        Each event in ``self.events`` gets an async hook that calls
        :meth:`detect` and raises :class:`GuardrailError` when the
        risk level is HIGH or CRITICAL.

        Existing hooks on the agent are not disturbed — guardrail hooks
        are appended to the list.

        Args:
            agent: The agent to attach to.
        """
        agent_id = id(agent)
        if agent_id in self._hooks:
            return  # Already attached.

        points = self._resolve_hook_points()
        registered: dict[HookPoint, Hook] = {}

        for point in points:

            async def _hook(__point: HookPoint = point, **data: Any) -> None:
                result = await self.detect(__point.value, **data)
                if not result.is_safe and result.risk_level in _BLOCKING_LEVELS:
                    raise GuardrailError(
                        f"Guardrail blocked at {__point.value}: "
                        f"{result.risk_type or 'unknown risk'}",
                        risk_level=result.risk_level,
                        risk_type=result.risk_type,
                        details=result.details,
                    )

            agent.hook_manager.add(point, _hook)
            registered[point] = _hook

        self._hooks[agent_id] = registered

    def detach(self, agent: Agent) -> None:
        """Remove previously registered guardrail hooks from an agent.

        Args:
            agent: The agent to detach from.
        """
        agent_id = id(agent)
        registered = self._hooks.pop(agent_id, None)
        if registered is None:
            return
        for point, hook in registered.items():
            agent.hook_manager.remove(point, hook)

    async def detect(self, event: str, **data: Any) -> GuardrailResult:
        """Run the backend analysis and return a guardrail result.

        Args:
            event: The hook point name that triggered detection.
            **data: Keyword arguments from the hook invocation.

        Returns:
            A :class:`GuardrailResult` indicating whether the data is safe.
        """
        assessment = await self.backend.analyze({"event": event, **data})

        if not assessment.has_risk:
            return GuardrailResult.safe()

        return GuardrailResult.block(
            risk_level=assessment.risk_level,
            risk_type=assessment.risk_type or "unknown",
            details=assessment.details,
        )
