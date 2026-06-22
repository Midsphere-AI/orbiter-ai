"""Cooperative cancellation token for distributed task execution."""

from __future__ import annotations

import asyncio


class CancellationToken:
    """Token checked cooperatively between agent execution steps.

    Workers set :attr:`cancelled` to ``True`` when a cancel signal is
    received for the active task.  The agent execution loop checks this
    token between steps and stops early if cancellation was requested.

    In addition to the synchronous :attr:`cancelled` poll, callers can
    ``await`` :meth:`wait` to be woken the moment cancellation is requested —
    used by the Temporal executor to propagate a cancel to the running
    workflow handle without busy-polling.
    """

    def __init__(self) -> None:
        self._cancelled = False
        # Lazily created so the token can be constructed outside a running loop.
        self._event: asyncio.Event | None = None

    @property
    def cancelled(self) -> bool:
        """Whether cancellation has been requested."""
        return self._cancelled

    def cancel(self) -> None:
        """Request cancellation."""
        self._cancelled = True
        if self._event is not None:
            self._event.set()

    async def wait(self) -> None:
        """Block until cancellation is requested.

        Returns immediately if the token is already cancelled.
        """
        if self._cancelled:
            return
        if self._event is None:
            self._event = asyncio.Event()
            # Cover a cancel() that landed between the check and event creation.
            if self._cancelled:
                self._event.set()
        await self._event.wait()
