"""Skill source watchers for Exo hot-reloading."""

try:
    from importlib.metadata import version

    __version__: str = version("exo-skills")
except Exception:
    __version__ = "0.1.0"

from exo_skills.watchers.github import GitHubPollingWatcher
from exo_skills.watchers.local import LocalFileWatcher

__all__: list[str] = ["GitHubPollingWatcher", "LocalFileWatcher"]
