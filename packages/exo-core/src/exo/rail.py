"""Guard ABC with priority ordering and retry capability.

Guards are structured lifecycle guards that attach to agent hook points.
Each guard has a priority (lower runs first) and returns a ``GuardAction``
to control execution flow: continue, skip, retry, or abort.
"""

from __future__ import annotations

import abc
import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from exo.hooks import Hook, HookPoint
from exo.types import ExoError

if TYPE_CHECKING:
    from exo.rail_types import GuardContext, InvokeInputs, ModelCallInputs, ToolCallInputs


class GuardAction(enum.StrEnum):
    """Action returned by a guard to control execution flow.

    Attributes:
        CONTINUE: Proceed to the next guard (or the guarded operation).
        SKIP: Skip the guarded operation entirely.
        RETRY: Retry the guarded operation (see ``RetryRequest``).
        ABORT: Abort the agent run immediately.
    """

    CONTINUE = "continue"
    SKIP = "skip"
    RETRY = "retry"
    ABORT = "abort"


# Deprecated alias: use GuardAction
RailAction = GuardAction


@dataclass(frozen=True)
class RetryRequest:
    """Parameters for a RETRY action.

    Attach an instance to the guard's return context when returning
    ``GuardAction.RETRY`` so the caller knows how to retry.

    Args:
        delay: Seconds to wait before retrying.
        max_retries: Maximum number of retry attempts.
        reason: Human-readable explanation for the retry.
    """

    delay: float = 0.0
    max_retries: int = 1
    reason: str = ""


class GuardAbortError(ExoError):
    """Raised when a guard returns ``GuardAction.ABORT``.

    Args:
        guard_name: Name of the guard that triggered the abort.
        reason: Human-readable reason for the abort.
    """

    _label: str = "Guard"

    def __init__(self, guard_name: str, reason: str = "") -> None:
        self.guard_name = guard_name
        # Keep rail_name as an alias attribute for backward compat
        self.rail_name = guard_name
        self.reason = reason
        msg = f"{self._label} '{guard_name}' aborted"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class RailAbortError(GuardAbortError):
    """Deprecated: use GuardAbortError instead.

    Subclass of GuardAbortError that preserves the ``Rail '...' aborted``
    message format for backward compatibility with existing code that
    catches or stringifies ``RailAbortError`` instances.
    """

    _label: str = "Rail"


class Guard(abc.ABC):
    """Abstract base class for agent guards.

    A guard is a lifecycle guard that inspects and optionally alters
    agent execution at specific hook points.  Guards are run in
    ascending ``priority`` order (lower numbers run first).

    Args:
        name: Unique identifier for this guard.
        priority: Execution order (lower = earlier). Defaults to 50.

    Example:
        >>> class BlockDangerousTool(Guard):
        ...     async def handle(self, ctx):
        ...         if ctx.inputs.tool_name == "rm_rf":
        ...             return GuardAction.ABORT
        ...         return GuardAction.CONTINUE
    """

    def __init__(self, name: str, *, priority: int = 50) -> None:
        self.name = name
        self.priority = priority

    @abc.abstractmethod
    async def handle(self, ctx: GuardContext) -> GuardAction | None:
        """Inspect and optionally act on a lifecycle event.

        Args:
            ctx: The guard context containing agent, event, typed
                inputs, and cross-guard extra dict.

        Returns:
            A ``GuardAction`` indicating what should happen next,
            or ``None`` (treated as ``CONTINUE``).
        """
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, priority={self.priority})"


# Deprecated alias: use Guard
Rail = Guard


def _build_inputs(
    event: HookPoint, data: dict[str, Any]
) -> InvokeInputs | ModelCallInputs | ToolCallInputs:
    """Build the appropriate typed inputs model from raw hook data.

    Args:
        event: The lifecycle hook point (determines which model to use).
        data: Keyword arguments originally passed to the hook.

    Returns:
        A typed inputs model matching the event kind.
    """
    from exo.rail_types import InvokeInputs, ModelCallInputs, ToolCallInputs

    if event in (HookPoint.PRE_TOOL_CALL, HookPoint.POST_TOOL_CALL):
        return ToolCallInputs(
            tool_name=data.get("tool_name", ""),
            arguments=data.get("arguments", {}),
            result=data.get("result"),
            metadata=data.get("metadata"),
        )
    if event in (HookPoint.PRE_LLM_CALL, HookPoint.POST_LLM_CALL):
        return ModelCallInputs(
            messages=data.get("messages", []),
            tools=data.get("tools"),
            response=data.get("response"),
            usage=data.get("usage"),
        )
    return InvokeInputs(
        input=data.get("input", ""),
        messages=data.get("messages"),
        result=data.get("result"),
    )


class GuardManager:
    """Manages guards with priority ordering and cross-guard state.

    Guards are run in ascending priority order (lower numbers first).
    A shared ``extra`` dict is passed through all guards within a single
    invocation, enabling cross-guard coordination.

    The manager can be registered as a single hook on
    :class:`~exo.hooks.HookManager` via :meth:`hook_for`.

    Example:
        >>> manager = GuardManager()
        >>> manager.add(my_safety_guard)
        >>> action = await manager.run(HookPoint.PRE_LLM_CALL, agent=agent, messages=msgs)
    """

    def __init__(self) -> None:
        self._guards: list[Guard] = []

    @property
    def _rails(self) -> list[Guard]:
        """Deprecated: internal alias for ``_guards`` for backward compatibility."""
        return self._guards

    def add(self, guard: Guard) -> None:
        """Add a guard to the manager.

        Args:
            guard: The guard instance to add.
        """
        self._guards.append(guard)

    def remove(self, guard: Guard) -> None:
        """Remove a guard from the manager.

        Args:
            guard: The guard instance to remove.

        Raises:
            ValueError: If the guard is not in the manager.
        """
        self._guards.remove(guard)

    def clear(self) -> None:
        """Remove all guards."""
        self._guards.clear()

    async def run(self, event: HookPoint, **data: Any) -> GuardAction:
        """Run all guards for an event in priority order.

        Builds a :class:`~exo.rail_types.GuardContext` from the keyword
        arguments, then executes each guard sorted by ascending priority.
        Returns the first non-CONTINUE action, or CONTINUE if all guards pass.

        The ``extra`` dict on the context is shared across all guards in
        the same invocation, allowing cross-guard state sharing.

        Args:
            event: The lifecycle hook point.
            **data: Keyword arguments (``agent`` plus event-specific data).

        Returns:
            The resulting :class:`GuardAction`.
        """
        from exo.rail_types import GuardContext

        agent = data.get("agent")
        inputs = _build_inputs(event, data)
        extra: dict[str, Any] = {}
        ctx = GuardContext(agent=agent, event=event, inputs=inputs, extra=extra)

        for guard in sorted(self._guards, key=lambda r: r.priority):
            action = await guard.handle(ctx)
            if action is not None and action != GuardAction.CONTINUE:
                return action

        return GuardAction.CONTINUE

    def hook_for(self, event: HookPoint) -> Hook:
        """Create a hook callable for a specific lifecycle event.

        The returned async callable can be registered on a
        :class:`~exo.hooks.HookManager`.  If any guard returns
        :attr:`GuardAction.ABORT`, a :class:`GuardAbortError` is raised
        so that the agent run is halted.

        Args:
            event: The lifecycle hook point.

        Returns:
            An async callable compatible with the Hook type.
        """

        async def _hook(**hook_data: Any) -> None:
            action = await self.run(event, **hook_data)
            if action == GuardAction.ABORT:
                raise RailAbortError("RailManager", reason=f"Rail aborted at {event.value}")

        return _hook


# Deprecated alias: use GuardManager
RailManager = GuardManager
