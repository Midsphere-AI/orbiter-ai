"""Plugin system for extending Exo CLI functionality.

Plugins can hook into CLI lifecycle events (startup, pre-run, post-run,
shutdown) and register additional commands or modify agent configuration.

Discovery sources:

1. **Entry points** — packages declaring the ``exo.plugins`` group
   in their metadata (e.g. ``[project.entry-points."exo.plugins"]``).
2. **Python files** — ``.py`` files in a plugins directory, each
   exporting a ``plugin`` attribute conforming to :class:`PluginSpec`.

Usage::

    manager = PluginManager()
    manager.load_entrypoints()
    manager.load_directory("/path/to/plugins")
    await manager.startup(app_context)
    # ... run agent ...
    await manager.shutdown()
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import logging
import sys
from collections.abc import Callable, Coroutine
from enum import StrEnum
from pathlib import Path
from typing import Any

from exo.types import ExoError  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hook points
# ---------------------------------------------------------------------------

HookFn = Callable[..., Coroutine[Any, Any, None]]
"""Async callable invoked at a plugin hook point."""


class PluginHook(StrEnum):
    """Lifecycle hook points for CLI plugins."""

    STARTUP = "startup"
    SHUTDOWN = "shutdown"
    PRE_RUN = "pre_run"
    POST_RUN = "post_run"


# ---------------------------------------------------------------------------
# Plugin spec
# ---------------------------------------------------------------------------


class PluginError(ExoError):
    """Raised when plugin loading or lifecycle fails."""


class PluginSpec:
    """Describes a single plugin with optional lifecycle hooks.

    Parameters:
        name: Unique plugin identifier.
        hooks: Mapping of :class:`PluginHook` → async callable.
        version: Optional version string.
        description: One-line description.
    """

    __slots__ = ("_description", "_hooks", "_name", "_version")

    def __init__(
        self,
        *,
        name: str,
        hooks: dict[PluginHook, HookFn] | None = None,
        version: str = "0.0.0",
        description: str = "",
    ) -> None:
        self._name = name
        self._hooks = dict(hooks) if hooks else {}
        self._version = version
        self._description = description

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def description(self) -> str:
        return self._description

    @property
    def hooks(self) -> dict[PluginHook, HookFn]:
        return dict(self._hooks)

    def __repr__(self) -> str:
        return f"PluginSpec(name={self._name!r}, version={self._version!r})"


# ---------------------------------------------------------------------------
# Plugin manager
# ---------------------------------------------------------------------------

_ENTRYPOINT_GROUP = "exo.plugins"


class PluginManager:
    """Discovers, loads, and manages CLI plugin lifecycle.

    Plugins are identified by name — duplicates are rejected.
    """

    __slots__ = ("_plugins",)

    def __init__(self) -> None:
        self._plugins: dict[str, PluginSpec] = {}

    # -- query ---------------------------------------------------------------

    @property
    def plugins(self) -> dict[str, PluginSpec]:
        """Mapping of name → loaded plugin specs (copy)."""
        return dict(self._plugins)

    def get(self, name: str) -> PluginSpec | None:
        """Look up a plugin by name."""
        return self._plugins.get(name)

    # -- registration --------------------------------------------------------

    def register(self, spec: PluginSpec) -> None:
        """Register a plugin spec.

        Raises:
            PluginError: If a plugin with the same name is already registered.
        """
        if spec.name in self._plugins:
            raise PluginError(f"Duplicate plugin: {spec.name!r}")
        self._plugins[spec.name] = spec
        logger.debug("Registered plugin: %s (%s)", spec.name, spec.version)

    # -- discovery: entry points ---------------------------------------------

    def load_entrypoints(self) -> int:
        """Discover and load plugins from ``exo.plugins`` entry points.

        Each entry point should reference a :class:`PluginSpec` instance
        or a callable returning one.

        Returns:
            Number of plugins successfully loaded.
        """
        loaded = 0
        eps = importlib.metadata.entry_points()
        # Filter for our group (Python 3.12+ returns SelectableGroups)
        group = (
            eps.select(group=_ENTRYPOINT_GROUP)
            if hasattr(eps, "select")
            else eps.get(_ENTRYPOINT_GROUP, [])
        )

        for ep in group:
            try:
                obj = ep.load()
                spec = obj() if callable(obj) and not isinstance(obj, PluginSpec) else obj
                if not isinstance(spec, PluginSpec):
                    logger.warning("Entry point %r did not yield a PluginSpec", ep.name)
                    continue
                self.register(spec)
                loaded += 1
            except PluginError:
                raise
            except Exception:
                logger.warning("Failed to load plugin entry point %r", ep.name, exc_info=True)
        return loaded

    # -- discovery: directory ------------------------------------------------

    def load_directory(self, directory: str | Path) -> int:
        """Load plugins from ``.py`` files in *directory*.

        Each file must expose a module-level ``plugin`` attribute that is
        a :class:`PluginSpec` instance.

        Returns:
            Number of plugins successfully loaded.

        Raises:
            PluginError: If the directory doesn't exist.
        """
        base = Path(directory)
        if not base.is_dir():
            raise PluginError(f"Plugin directory not found: {base}")

        loaded = 0
        for path in sorted(base.iterdir()):
            if not path.is_file() or path.suffix != ".py" or path.name.startswith("_"):
                continue
            try:
                spec = _load_plugin_file(path)
                if spec is not None:
                    self.register(spec)
                    loaded += 1
            except PluginError:
                raise
            except Exception:
                logger.warning("Failed to load plugin from %s", path, exc_info=True)
        return loaded

    # -- lifecycle -----------------------------------------------------------

    async def run_hook(self, hook: PluginHook, **kwargs: Any) -> None:
        """Run a lifecycle hook across all registered plugins.

        Plugins are invoked in registration order.  Errors from a plugin
        hook propagate immediately.
        """
        for spec in self._plugins.values():
            fn = spec.hooks.get(hook)
            if fn is not None:
                await fn(**kwargs)

    async def startup(self, **kwargs: Any) -> None:
        """Run the startup hook on all plugins."""
        await self.run_hook(PluginHook.STARTUP, **kwargs)

    async def shutdown(self, **kwargs: Any) -> None:
        """Run the shutdown hook on all plugins."""
        await self.run_hook(PluginHook.SHUTDOWN, **kwargs)


# ---------------------------------------------------------------------------
# File loader helper
# ---------------------------------------------------------------------------


def _load_plugin_file(path: Path) -> PluginSpec | None:
    """Import a Python file and extract its ``plugin`` attribute."""
    module_name = f"_exo_plugin_{path.stem}"
    file_spec = importlib.util.spec_from_file_location(module_name, path)
    if file_spec is None or file_spec.loader is None:
        logger.warning("Cannot create module spec for %s", path)
        return None

    module = importlib.util.module_from_spec(file_spec)
    sys.modules[module_name] = module
    try:
        file_spec.loader.exec_module(module)
    except Exception:
        del sys.modules[module_name]
        raise

    plugin = getattr(module, "plugin", None)
    if plugin is None:
        del sys.modules[module_name]
        return None

    if not isinstance(plugin, PluginSpec):
        del sys.modules[module_name]
        logger.warning("'plugin' in %s is not a PluginSpec (got %s)", path, type(plugin).__name__)
        return None

    return plugin
