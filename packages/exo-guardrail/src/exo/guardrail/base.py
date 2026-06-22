"""BaseGuardrail: hook-based guardrail that integrates with Agent's HookManager."""

from __future__ import annotations

import logging
import weakref
from typing import TYPE_CHECKING, Any

from exo.guardrail.types import (  # pyright: ignore[reportMissingImports]
    GuardrailBackend,
    GuardrailConfigError,
    GuardrailError,
    GuardrailResult,
    RiskLevel,
)
from exo.hooks import Hook, HookPoint

if TYPE_CHECKING:
    from exo.agent import Agent

logger = logging.getLogger(__name__)


# Risk levels that trigger automatic blocking.
_BLOCKING_LEVELS = frozenset({RiskLevel.HIGH, RiskLevel.CRITICAL})


_VALID_HOOK_POINT_VALUES: frozenset[str] = frozenset(hp.value for hp in HookPoint)


class BaseGuardrail:
    """A guardrail that registers itself as hooks on an Agent's HookManager.

    Args:
        backend: Detection backend.  Must not be ``None`` — passing ``None``
            raises ``ValueError`` at construction time to prevent silent
            no-op guardrails from giving false assurance.
        events: Hook points to monitor.  Each entry may be a
            :class:`~exo.hooks.HookPoint` enum member (recommended —
            enables auto-complete and catches typos at construction time)
            or a plain string for backward compatibility.  Only these
            events will have hooks registered.

    Example::

        from exo.hooks import HookPoint

        backend = MyDetectionBackend()
        # Preferred: use HookPoint enum for discoverability
        guard = BaseGuardrail(backend=backend, events=[HookPoint.PRE_LLM_CALL])
        # Plain strings still work for backward compatibility
        guard = BaseGuardrail(backend=backend, events=["pre_llm_call"])
        guard.attach(agent)
        # ... agent runs, guardrail hooks fire automatically
        guard.detach(agent)
    """

    def __init__(
        self,
        backend: GuardrailBackend | None = None,
        events: list[HookPoint | str] | None = None,
    ) -> None:
        if backend is None:
            raise GuardrailConfigError(
                "BaseGuardrail requires a backend. "
                "Passing backend=None creates a silent no-op guardrail that gives false "
                "assurance of protection.",
                hint="Supply a GuardrailBackend instance (e.g. PatternBackend or "
                "LLMGuardrailBackend) as the backend= argument.",
            )
        self.backend = backend

        # Normalize each entry to its string value and validate eagerly so
        # typos surface immediately rather than silently at attach() time.
        normalized: list[str] = []
        for entry in events or []:
            value = entry.value if isinstance(entry, HookPoint) else entry
            if value not in _VALID_HOOK_POINT_VALUES:
                valid = ", ".join(sorted(_VALID_HOOK_POINT_VALUES))
                raise GuardrailConfigError(
                    f"Unknown hook point: {value!r}.",
                    context={"hook_point": value},
                    hint=(
                        f"Valid hook points are: {valid}. "
                        "Use HookPoint.<NAME> for auto-complete, e.g. HookPoint.PRE_LLM_CALL."
                    ),
                )
            normalized.append(value)
        self.events: list[str] = normalized

        # Track hooks per agent so detach can remove exactly the right ones.
        # WeakKeyDictionary prevents stale entries from id-reuse after GC.
        self._hooks: weakref.WeakKeyDictionary[Any, dict[HookPoint, Hook]] = (
            weakref.WeakKeyDictionary()
        )

    def _resolve_hook_points(self) -> list[HookPoint]:
        """Convert string event names to HookPoint enum values."""
        points: list[HookPoint] = []
        for name in self.events:
            try:
                points.append(HookPoint(name))
            except ValueError:
                raise GuardrailConfigError(
                    f"Unknown hook point: {name!r}",
                    context={"hook_point": name},
                    hint="Use HookPoint.<NAME> for validated hook points.",
                ) from None
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
        if agent in self._hooks:
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
                        context={
                            "hook": __point.value,
                            "risk_level": result.risk_level,
                            "risk_type": result.risk_type or "unknown",
                        },
                        hint=(
                            "Inspect the 'details' attribute for matched patterns. "
                            "Adjust guardrail thresholds or whitelist the input if it "
                            "is a false positive."
                        ),
                    )
                elif not result.is_safe:
                    logger.warning(
                        "Guardrail detected sub-blocking risk at %s: level=%s type=%s",
                        __point.value,
                        result.risk_level,
                        result.risk_type or "unknown",
                    )

            agent.hook_manager.add(point, _hook)
            registered[point] = _hook

        self._hooks[agent] = registered

    def detach(self, agent: Agent) -> None:
        """Remove previously registered guardrail hooks from an agent.

        Args:
            agent: The agent to detach from.
        """
        registered = self._hooks.pop(agent, None)
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
