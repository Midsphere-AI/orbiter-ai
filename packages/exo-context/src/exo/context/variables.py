"""Dynamic variable registry with nested path resolution.

Provides a registry for dynamic variables that can be resolved from
context state using dot-separated paths like ``user.name`` or
``session.active``.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from exo.types import ExoError  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)

from exo.context.state import ContextState  # pyright: ignore[reportMissingImports]

# Pre-compiled template placeholder pattern for resolve_template hot path.
_TEMPLATE_RE = re.compile(r"\$\{([^}]+)\}")


class VariableResolveError(ExoError):
    """Raised when a variable path cannot be resolved."""


class DynamicVariableRegistry:
    """Registry of named variable resolvers with nested path support.

    Variables are registered as dot-separated paths (e.g., ``user.name``)
    and resolved from a :class:`ContextState` or a flat dict.

    Usage::

        reg = DynamicVariableRegistry()
        reg.register("user.name", lambda state: state.get("user_name", "anon"))
        value = reg.resolve("user.name", state)
    """

    __slots__ = ("_resolvers",)

    def __init__(self) -> None:
        self._resolvers: dict[str, Any] = {}

    def register(self, path: str, resolver: Any = None) -> Any:
        """Register a resolver for *path*.

        *resolver* can be:
        - A callable ``(state) -> value``
        - A static value
        - ``None`` — returns a decorator for the callable form

        Parameters
        ----------
        path:
            Dot-separated variable path (e.g. ``"user.name"``).
        resolver:
            Callable or static value. If ``None``, acts as a decorator.
        """
        if resolver is not None:
            self._resolvers[path] = resolver
            return resolver

        def decorator(fn: Any) -> Any:
            self._resolvers[path] = fn
            return fn

        return decorator

    def resolve(self, path: str, state: ContextState | dict[str, Any]) -> Any:
        """Resolve a variable path to its value.

        Resolution order:
        1. Exact match in registered resolvers (callable invoked with state)
        2. Nested path lookup in state (e.g. ``"a.b"`` → ``state["a"]["b"]``)

        Raises :class:`VariableResolveError` if the path cannot be resolved.
        """
        # 1. Check registered resolvers
        if path in self._resolvers:
            resolver = self._resolvers[path]
            if callable(resolver):
                logger.debug("resolving variable %r via registered callable", path)
                return resolver(state)
            return resolver

        # 2. Try nested path lookup in state
        logger.debug("resolving variable %r via nested state lookup", path)
        return self._resolve_nested(path, state)

    def _resolve_nested(self, path: str, state: ContextState | dict[str, Any]) -> Any:
        """Walk a dot-separated path through nested dicts/state."""
        parts = path.split(".")
        current: Any = state

        for part in parts:
            if isinstance(current, ContextState):
                if part not in current:
                    raise VariableResolveError(
                        f"Variable path {path!r} not found at segment {part!r}.",
                        hint=(
                            "Register a resolver via DynamicVariableRegistry.register() "
                            "or ensure the state contains the key path."
                        ),
                    )
                current = current.get(part)
            elif isinstance(current, dict):
                if part not in current:
                    raise VariableResolveError(
                        f"Variable path {path!r} not found at segment {part!r}.",
                        hint=(
                            "Register a resolver via DynamicVariableRegistry.register() "
                            "or ensure the state contains the key path."
                        ),
                    )
                current = current[part]
            else:
                # Try getattr for object access
                if hasattr(current, part):
                    current = getattr(current, part)
                else:
                    raise VariableResolveError(
                        f"Variable path {path!r} not found at segment {part!r}.",
                        hint=(
                            "Register a resolver via DynamicVariableRegistry.register() "
                            "or ensure the state contains the key path."
                        ),
                    )
        return current

    def has(self, path: str) -> bool:
        """Check if a resolver is registered for *path*."""
        return path in self._resolvers

    def list_all(self) -> list[str]:
        """Return all registered variable paths."""
        return list(self._resolvers.keys())

    def resolve_template(self, template: str, state: ContextState | dict[str, Any]) -> str:
        """Resolve ``${path}`` placeholders in a template string.

        Unresolvable variables are left as-is.
        """

        def _replace(match: re.Match[str]) -> str:
            path = match.group(1)
            try:
                value = self.resolve(path, state)
                return str(value)
            except VariableResolveError:
                return match.group(0)

        return _TEMPLATE_RE.sub(_replace, template)

    def __repr__(self) -> str:
        return f"DynamicVariableRegistry(variables={len(self._resolvers)})"
