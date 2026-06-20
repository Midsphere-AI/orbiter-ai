"""Async event bus for decoupled communication."""

from __future__ import annotations

import contextlib
from collections import defaultdict
from collections.abc import Callable, Coroutine
from typing import Any

EventHandler = Callable[..., Coroutine[Any, Any, None]]
"""Type alias for async event handler functions."""


class EventBus:
    """A simple async event bus.

    Handlers are called sequentially in registration order when an event
    is emitted. Event names are plain strings.

    .. note::
        **Naming divergence (I-14):** ``EventBus`` uses ``on``/``off``/``emit``
        while ``HookManager`` (``exo.hooks``) uses ``add``/``remove``/``run``.
        These are intentionally separate APIs — do **not** rename the public
        methods, as doing so would break existing callers.
    """

    def __init__(self) -> None:
        self._handlers: defaultdict[str, list[EventHandler]] = defaultdict(list)

    def on(self, event: str, handler: EventHandler) -> None:
        """Subscribe a handler to an event.

        Args:
            event: The event name to listen for.
            handler: Async callable to invoke when the event fires.
        """
        self._handlers[event].append(handler)

    def off(self, event: str, handler: EventHandler) -> None:
        """Unsubscribe a handler from an event.

        Silently does nothing if the handler is not registered.

        Args:
            event: The event name.
            handler: The handler to remove (first occurrence only).
        """
        with contextlib.suppress(ValueError):
            self._handlers[event].remove(handler)

    async def emit(self, event: str, **data: Any) -> None:
        """Emit an event, calling all registered handlers sequentially.

        Handlers are called in registration order.  If a handler raises
        an exception it propagates immediately, aborting any subsequent
        handlers for that event — identical to ``HookManager.run()``
        behavior.  Callers that need isolation must wrap individual
        handlers in ``try/except`` before registering them.

        Args:
            event: The event name to emit.
            **data: Keyword arguments passed to each handler.
        """
        for handler in self._handlers[event]:
            await handler(**data)

    def has_handlers(self, event: str) -> bool:
        """Check whether any handlers are registered for an event.

        Args:
            event: The event name to check.

        Returns:
            True if at least one handler is registered.
        """
        return len(self._handlers[event]) > 0

    def clear(self) -> None:
        """Remove all handlers for all events."""
        self._handlers.clear()
