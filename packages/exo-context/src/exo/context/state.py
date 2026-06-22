"""Hierarchical key-value state with parent inheritance.

ContextState supports:
- Parent-child state inheritance: child reads from parent if key is missing locally.
- Write isolation: set/update only affects local state.
- to_dict() merges parent + local (local wins).
- local_dict() returns only local entries.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from exo.types import ExoError  # pyright: ignore[reportMissingImports]


class ContextStateKeyError(ExoError):
    """Raised when a key is missing from :class:`ContextState`."""


class ContextState:
    """Hierarchical key-value state with parent chain lookup.

    Reads search local data first, then walk up the parent chain.
    Writes always target local data only.
    """

    __slots__ = ("_data", "_parent")

    def __init__(
        self,
        initial: dict[str, Any] | None = None,
        *,
        parent: ContextState | None = None,
    ) -> None:
        self._data: dict[str, Any] = dict(initial) if initial else {}
        self._parent: ContextState | None = parent

    # ── Read ──────────────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        """Get value by key, searching local then parent chain."""
        if key in self._data:
            return self._data[key]
        if self._parent is not None:
            return self._parent.get(key, default)
        return default

    def __getitem__(self, key: str) -> Any:
        if key in self._data:
            return self._data[key]
        if self._parent is not None:
            return self._parent[key]
        raise ContextStateKeyError(
            f"Key {key!r} not found in context state.",
            hint="Use .get(key) to return a default instead of raising, or check 'key in state' first.",
        )

    def __contains__(self, key: object) -> bool:
        if key in self._data:
            return True
        if self._parent is not None:
            return key in self._parent
        return False

    # ── Write (local only) ────────────────────────────────────────────

    def set(self, key: str, value: Any) -> None:
        """Set a value in local state only."""
        self._data[key] = value

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def update(self, mapping: dict[str, Any] | ContextState | None = None, **kwargs: Any) -> None:
        """Batch-update local state from a dict, another ContextState, or kwargs."""
        if mapping is not None:
            if isinstance(mapping, ContextState):
                self._data.update(mapping._data)
            else:
                self._data.update(mapping)
        if kwargs:
            self._data.update(kwargs)

    def delete(self, key: str) -> None:
        """Delete a key from local state. Raises KeyError if not in local."""
        del self._data[key]

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def pop(self, key: str, *args: Any) -> Any:
        """Remove and return value from local state."""
        return self._data.pop(key, *args)

    def clear(self) -> None:
        """Clear all local state. Does not affect parent."""
        self._data.clear()

    # ── Introspection ─────────────────────────────────────────────────

    def local_dict(self) -> dict[str, Any]:
        """Return a copy of local-only data (no parent)."""
        return self._data.copy()

    def to_dict(self) -> dict[str, Any]:
        """Return merged dict: parent values overridden by local values."""
        if self._parent is not None:
            merged = self._parent.to_dict()
            merged.update(self._data)
            return merged
        return self._data.copy()

    @property
    def parent(self) -> ContextState | None:
        """Return the parent state, if any."""
        return self._parent

    def keys(self) -> set[str]:
        """All accessible keys (local + parent chain)."""
        result = set(self._data.keys())
        if self._parent is not None:
            result |= self._parent.keys()
        return result

    def __len__(self) -> int:
        return len(self.keys())

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __bool__(self) -> bool:
        return bool(self._data) or (self._parent is not None and bool(self._parent))

    # ── Representation ────────────────────────────────────────────────

    def __repr__(self) -> str:
        local_n = len(self._data)
        total_n = len(self)
        parent_info = f", inherited={total_n - local_n}" if self._parent else ""
        return f"ContextState(local={local_n}{parent_info})"
