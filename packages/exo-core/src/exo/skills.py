"""Multi-source skill registry for loading skills from local paths and GitHub.

Skills are markdown files with YAML front-matter containing metadata
(name, description, tool_list, type, active). The registry loads skills
from local directories and GitHub repositories, caching remote repos
at ``~/.exo/skills/``.

Usage::

    reg = SkillRegistry()
    reg.register_source("/path/to/skills")
    reg.register_source("https://github.com/user/repo/tree/main/skills")
    skills = reg.load_all()
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import subprocess
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from exo.types import ExoError

if TYPE_CHECKING:
    from exo.agent import Agent
    from exo.events import EventBus
    from exo.tool import Tool

logger = logging.getLogger(__name__)

# Branch names may contain slashes (e.g. feat/my-branch).  We capture the
# branch as *everything* up to the first "/" that precedes an optional
# subdirectory segment.  To allow subdirectories we split on "/tree/" and then
# use a greedy match that consumes as little as possible once we detect that a
# subdir is present.  The regex keeps the existing behaviour for branch-only
# URLs and adds multi-slash branch support.
_GITHUB_RE = re.compile(
    r"https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/.]+?)(?:\.git)?"
    r"(?:/tree/(?P<branch>[^?#]+?)(?:/(?P<subdir>[^?#]+))?)?"
    r"(?:[?#].*)?$"
)

DEFAULT_CACHE_DIR = Path.home() / ".exo" / "skills"
SKILL_FILENAMES = {"skill.md", "SKILL.md"}


class SkillError(ExoError):
    """Raised for skill loading or registry errors."""


class ConflictStrategy(StrEnum):
    """How to handle duplicate skill names across sources."""

    KEEP_FIRST = "keep_first"
    KEEP_LAST = "keep_last"
    RAISE = "raise"


class Skill:
    """A loaded skill with metadata and usage content.

    Args:
        name: Unique skill name.
        description: Human-readable description.
        usage: Markdown body content (usage instructions).
        tool_list: Mapping of tool names to action lists.
        skill_type: Skill type — empty string for tool skills, ``"agent"`` for agent skills.
        active: Whether the skill starts active.
        path: Filesystem path to the source skill.md file.
    """

    __slots__ = ("active", "description", "name", "path", "skill_type", "tool_list", "usage")

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        usage: str = "",
        tool_list: dict[str, list[str]] | None = None,
        skill_type: str = "",
        active: bool = True,
        path: str = "",
    ) -> None:
        self.name = name
        self.description = description
        self.usage = usage
        self.tool_list: dict[str, list[str]] = tool_list or {}
        self.skill_type = skill_type
        self.active = active
        self.path = path

    def __repr__(self) -> str:
        return f"Skill(name={self.name!r}, type={self.skill_type!r}, active={self.active})"


def parse_github_url(url: str) -> dict[str, str] | None:
    """Parse a GitHub URL into owner, repo, branch, and subdirectory.

    Returns:
        Dict with keys ``owner``, ``repo``, ``branch``, ``subdir`` or
        ``None`` if the URL is not a valid GitHub URL.
    """
    m = _GITHUB_RE.match(url)
    if not m:
        return None
    return {
        "owner": m.group("owner"),
        "repo": m.group("repo"),
        "branch": m.group("branch") or "main",
        "subdir": m.group("subdir") or "",
    }


def extract_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Extract YAML front-matter and body from a markdown skill file.

    Args:
        text: Full text content of a skill.md file.

    Returns:
        Tuple of (front-matter dict, body string). Front-matter keys
        are lowercased. The ``tool_list`` value is JSON-parsed if present.

    Note:
        This parser handles **single-line** ``key: value`` pairs only.
        Multi-line YAML values (block scalars, flow sequences that span
        lines, etc.) are *not* supported.  The value is everything after
        the first ``:`` on the line, stripped.  Use a proper YAML parser
        (e.g. ``yaml.safe_load``) if you need richer front-matter.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    end = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end < 0:
        return {}, text

    meta: dict[str, Any] = {}
    for line in lines[1:end]:
        colon = line.find(":")
        if colon < 0:
            continue
        key = line[:colon].strip().lower()
        val = line[colon + 1 :].strip()
        if key == "tool_list":
            try:
                meta[key] = json.loads(val)
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning(
                    "Malformed tool_list in front-matter — expected JSON object, got %r: %s",
                    val,
                    exc,
                )
                meta[key] = {}
        elif key == "active":
            meta[key] = val.lower() == "true"
        else:
            meta[key] = val

    body = "\n".join(lines[end + 1 :]).strip()
    return meta, body


def _clone_github(parsed: dict[str, str], cache_dir: Path) -> Path:
    """Shallow-clone a GitHub repo into the cache directory.

    Returns:
        Path to the clone (or subdirectory within it).
    """
    owner = parsed["owner"]
    repo = parsed["repo"]
    branch = parsed["branch"]
    subdir = parsed["subdir"]

    # Validate the branch name before it reaches ``git clone``: a value
    # starting with ``-`` would be parsed by git as a flag, and only plain
    # ref characters are ever legitimate here.
    if branch.startswith("-") or not re.fullmatch(r"[A-Za-z0-9._/-]+", branch):
        raise SkillError(
            f"Invalid branch name {branch!r}",
            context={"owner": owner, "repo": repo, "branch": branch},
            hint="Branch names may only contain letters, digits, '.', '_', '/', '-'.",
        )

    clone_dir = cache_dir / owner / repo / branch
    if not clone_dir.exists():
        clone_dir.parent.mkdir(parents=True, exist_ok=True)
        clone_url = f"https://github.com/{owner}/{repo}.git"
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", f"--branch={branch}", clone_url, str(clone_dir)],
                check=True,
                capture_output=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired as exc:
            # Partial clone directory may exist — remove it so the next attempt
            # does not short-circuit to a corrupt/empty clone.
            shutil.rmtree(clone_dir, ignore_errors=True)
            raise SkillError(
                f"git clone timed out for {owner}/{repo}@{branch}",
                context={"url": clone_url, "branch": branch},
                hint="The clone exceeded 60 seconds. Check your network connection.",
            ) from exc
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
            # Remove any partial clone directory left by a failed git clone.
            shutil.rmtree(clone_dir, ignore_errors=True)
            raise SkillError(
                f"Failed to clone {owner}/{repo}@{branch}",
                context={"url": clone_url, "branch": branch},
                hint="Ensure git is installed and the repository URL and branch exist.",
            ) from exc

    if subdir:
        target = clone_dir / subdir
        # Guard against path traversal: a crafted ``subdir`` (e.g. with ``..``)
        # must not resolve outside the clone root.
        if not target.resolve().is_relative_to(clone_dir.resolve()):
            raise SkillError(
                f"Subdirectory '{subdir}' escapes the repository clone",
                context={"owner": owner, "repo": repo, "branch": branch, "subdir": subdir},
                hint="Use a subdirectory path inside the repository.",
            )
        if not target.is_dir():
            raise SkillError(
                f"Subdirectory '{subdir}' not found in {owner}/{repo}@{branch}",
                context={"owner": owner, "repo": repo, "branch": branch, "subdir": subdir},
                hint=f"Check that '{subdir}' exists in the {branch} branch of {owner}/{repo}.",
            )
        return target
    return clone_dir


def _collect_skills(root: Path) -> dict[str, Skill]:
    """Walk a directory tree and collect skills from skill.md files."""
    skills: dict[str, Skill] = {}
    if not root.is_dir():
        return skills

    skill_files = sorted(
        {f for name in SKILL_FILENAMES for f in root.rglob(name)}
    )
    for skill_file in skill_files:
        if not skill_file.is_file():
            continue
        try:
            text = skill_file.read_text(encoding="utf-8")
            meta, body = extract_front_matter(text)
            if not meta.get("name"):
                logger.warning(
                    "Skill file %s has no 'name' front-matter field; using directory name %r",
                    skill_file,
                    skill_file.parent.name,
                )
            name = meta.get("name") or skill_file.parent.name
            skill = Skill(
                name=name,
                description=meta.get("description", meta.get("desc", "")),
                usage=body,
                tool_list=meta.get("tool_list") if isinstance(meta.get("tool_list"), dict) else {},
                skill_type=meta.get("type", ""),
                active=meta.get("active", True) if isinstance(meta.get("active"), bool) else True,
                path=str(skill_file),
            )
            if name in skills:
                logger.warning(
                    "Duplicate skill name %r within the same source scan: "
                    "existing path %s, new path %s — last one wins",
                    name,
                    skills[name].path,
                    skill_file,
                )
            skills[name] = skill
        except Exception as exc:
            logger.warning(
                "Failed to load skill file %s — skipping: %s",
                skill_file,
                exc,
                exc_info=True,
            )
            continue
    return skills


class SkillRegistry:
    """Multi-source skill registry.

    Loads skills from local filesystem paths and GitHub repository URLs.
    Remote repositories are shallow-cloned and cached at *cache_dir*.

    Args:
        conflict: Strategy for handling duplicate skill names.
        cache_dir: Directory for caching GitHub clones.
    """

    def __init__(
        self,
        *,
        conflict: ConflictStrategy | str = ConflictStrategy.KEEP_FIRST,
        cache_dir: Path | None = None,
    ) -> None:
        self._conflict = ConflictStrategy(conflict)
        self._cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self._sources: list[str] = []
        self._skills: dict[str, Skill] = {}

    @property
    def skills(self) -> dict[str, Skill]:
        """All loaded skills keyed by name."""
        return dict(self._skills)

    def register_source(self, source: str) -> None:
        """Add a skill source (local path or GitHub URL)."""
        self._sources.append(source)

    def load_all(self) -> dict[str, Skill]:
        """Load skills from all registered sources.

        Returns:
            Dict mapping skill name to Skill.

        Raises:
            SkillError: On conflict when strategy is ``raise``.

        Note:
            This method performs **blocking** I/O (including a ``subprocess``
            call for GitHub sources).  Call :meth:`aload_all` from async code
            to avoid blocking the event loop.
        """
        self._skills.clear()
        for source in self._sources:
            parsed = parse_github_url(source)
            if parsed:
                root = _clone_github(parsed, self._cache_dir)
            else:
                root = Path(source).expanduser().resolve()
                if not root.is_dir():
                    raise SkillError(
                        f"Skill source directory not found: {root}",
                        context={"path": str(root)},
                        hint="Create the directory or pass a valid GitHub URL (https://github.com/...) instead of a local path.",
                    )
            new_skills = _collect_skills(root)
            for name, skill in new_skills.items():
                self._merge(name, skill)
        return dict(self._skills)

    async def aload_all(self) -> dict[str, Skill]:
        """Async version of :meth:`load_all` — offloads blocking I/O to a thread.

        Equivalent to ``load_all()`` but safe to ``await`` from an async
        context because the blocking ``subprocess`` clone is executed in a
        thread-pool executor rather than on the event loop.

        Returns:
            Dict mapping skill name to Skill.

        Raises:
            SkillError: On conflict when strategy is ``raise``.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load_all)

    def _merge(self, name: str, skill: Skill) -> None:
        """Merge a skill into the registry, applying conflict strategy."""
        if name not in self._skills:
            self._skills[name] = skill
            return
        if self._conflict == ConflictStrategy.KEEP_FIRST:
            return
        if self._conflict == ConflictStrategy.KEEP_LAST:
            self._skills[name] = skill
            return
        existing = self._skills[name]
        raise SkillError(
            f"Duplicate skill '{name}' (conflict_strategy=raise)",
            context={
                "skill": name,
                "existing_path": existing.path,
                "new_path": skill.path,
            },
            hint="Rename one of the skill definitions or use conflict='keep_first'/'keep_last' to resolve automatically.",
        )

    def get(self, name: str) -> Skill:
        """Retrieve a skill by name.

        Raises:
            SkillError: If the skill is not found.
        """
        if name not in self._skills:
            raise SkillError(
                f"Skill '{name}' not found",
                context={"name": name},
                hint=f"Call registry.list_names() to see available skills. Loaded: {list(self._skills)[:5]}",
            )
        return self._skills[name]

    def search(
        self,
        *,
        query: str = "",
        skill_type: str | None = None,
        active_only: bool = False,
    ) -> list[Skill]:
        """Search skills by text query, type, and active status.

        Args:
            query: Case-insensitive substring to match against name or description.
            skill_type: Filter by skill type (e.g. ``"agent"``).
            active_only: If ``True``, only return active skills.

        Returns:
            List of matching skills.
        """
        results: list[Skill] = []
        q = query.lower()
        for skill in self._skills.values():
            if active_only and not skill.active:
                continue
            if skill_type is not None and skill.skill_type != skill_type:
                continue
            if q and q not in skill.name.lower() and q not in skill.description.lower():
                continue
            results.append(skill)
        return results

    def list_names(self) -> list[str]:
        """Return all loaded skill names."""
        return list(self._skills.keys())

    def add_skill(self, skill: Skill) -> None:
        """Add a skill to the registry, applying the conflict strategy.

        Raises:
            SkillError: If a duplicate name is found and strategy is ``raise``.
        """
        self._merge(skill.name, skill)

    def remove_skill(self, name: str) -> Skill | None:
        """Remove and return a skill by name, or ``None`` if not present."""
        return self._skills.pop(name, None)

    def update_skill(self, skill: Skill) -> None:
        """Replace an existing skill entry (unconditional overwrite)."""
        self._skills[skill.name] = skill


# ---------------------------------------------------------------------------
# Hot-reload abstractions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkillChangeEvent:
    """Represents a single skill change detected by a watcher."""

    kind: Literal["added", "modified", "removed"]
    skill_name: str
    skill: Skill | None  # None only for kind="removed"
    source_path: str


class SkillWatcher(ABC):
    """Base class for skill source watchers.

    Implementations monitor a skill source (local directory, GitHub repo,
    etc.) and yield batches of change events.  Each batch represents one
    "settle" cycle — filesystem events are debounced, polling results are
    diffed.
    """

    @abstractmethod
    async def watch(self) -> AsyncGenerator[list[SkillChangeEvent]]:
        """Yield batches of change events.  Blocks between batches."""
        ...  # pragma: no cover
        yield  # make type-checkers treat this as an async generator

    @abstractmethod
    async def stop(self) -> None:
        """Signal the watcher to shut down.  Must cause ``watch()`` to return."""
        ...  # pragma: no cover


@runtime_checkable
class ToolResolver(Protocol):
    """Maps skill metadata to actual Tool implementations."""

    def resolve(self, skill: Skill) -> list[Tool]: ...


class DictToolResolver:
    """Simple resolver that maps skill names to Tool objects via a dict."""

    def __init__(self, tool_map: dict[str, Tool | list[Tool]]) -> None:
        self._map = tool_map

    def resolve(self, skill: Skill) -> list[Tool]:
        entry = self._map.get(skill.name)
        if entry is None:
            return []
        if isinstance(entry, list):
            return entry
        return [entry]


def _skill_fingerprint(skill: Skill) -> tuple[str, str, str, str, str, bool]:
    """Return a comparable tuple for change detection (Skill has no __eq__).

    ``tool_list`` is serialised with ``sort_keys=True`` so that insertion-order
    differences in the dict do not cause spurious "modified" events.
    """
    return (
        skill.name,
        skill.description,
        skill.usage,
        json.dumps(skill.tool_list, sort_keys=True),
        skill.skill_type,
        skill.active,
    )


class SkillSyncManager:
    """Orchestrates skill watchers and pushes changes to bound agents.

    Args:
        registry: The :class:`SkillRegistry` whose state is kept in sync.
        tool_resolver: Maps skills to Tool objects.  Accepts a
            :class:`ToolResolver` instance or a plain
            ``dict[str, Tool | list[Tool]]``.
        event_bus: Optional :class:`~exo.events.EventBus` for emitting
            change events.
        instructions_builder: Optional callable that rebuilds agent
            instructions from the current list of active skills.
    """

    _MAX_BACKOFF: float = 60.0

    def __init__(
        self,
        registry: SkillRegistry,
        tool_resolver: ToolResolver | dict[str, Tool | list[Tool]],
        *,
        event_bus: EventBus | None = None,
        instructions_builder: Callable[[list[Skill]], str] | None = None,
    ) -> None:
        self._registry = registry
        if isinstance(tool_resolver, dict):
            self._resolver: ToolResolver = DictToolResolver(tool_resolver)
        else:
            self._resolver = tool_resolver
        self._bus = event_bus
        self._instructions_builder = instructions_builder

        self._watchers: list[SkillWatcher] = []
        self._agents: set[Agent] = set()
        self._tasks: list[asyncio.Task[None]] = []
        self._skill_tools: dict[str, list[str]] = {}

    # -- public API ---------------------------------------------------------

    def add_watcher(self, watcher: SkillWatcher) -> None:
        """Register a watcher.  Must be called before :meth:`start`."""
        self._watchers.append(watcher)

    def bind_agent(self, agent: Agent) -> None:
        """Bind an agent to receive skill change updates."""
        self._agents.add(agent)

    def unbind_agent(self, agent: Agent) -> None:
        """Stop pushing updates to an agent."""
        self._agents.discard(agent)

    async def start(self) -> None:
        """Start all watchers as background ``asyncio.Task`` instances."""
        if self._tasks:
            raise RuntimeError(
                "SkillSyncManager is already running; call stop() before start()."
            )
        for watcher in self._watchers:
            task = asyncio.create_task(self._run_watcher(watcher))
            self._tasks.append(task)

    async def stop(self) -> None:
        """Stop all watchers, cancel tasks, and await cleanup."""
        for watcher in self._watchers:
            try:
                await watcher.stop()
            except Exception:
                logger.warning("Error stopping watcher %s", watcher, exc_info=True)

        for task in self._tasks:
            task.cancel()

        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.warning("Error awaiting watcher task %s", task, exc_info=True)

        self._tasks.clear()

    async def __aenter__(self) -> SkillSyncManager:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    # -- internal -----------------------------------------------------------

    async def _run_watcher(self, watcher: SkillWatcher) -> None:
        """Run a single watcher with exponential backoff on errors."""
        backoff = 1.0
        while True:
            try:
                async for batch in watcher.watch():
                    await self._process_batch(batch)
                return  # watch() finished normally
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.error(
                    "Watcher %s raised an error, retrying in %.1fs",
                    watcher,
                    backoff,
                    exc_info=True,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._MAX_BACKOFF)

    async def _process_batch(self, batch: list[SkillChangeEvent]) -> None:
        """Apply a batch of change events to registry and bound agents."""
        for event in batch:
            if event.kind == "added":
                await self._handle_added(event)
            elif event.kind == "modified":
                await self._handle_modified(event)
            elif event.kind == "removed":
                await self._handle_removed(event)

        # Rebuild instructions for all bound agents after the batch
        if self._instructions_builder is not None:
            active_skills = list(self._registry.skills.values())
            new_instructions = self._instructions_builder(active_skills)
            for agent in self._agents:
                agent.instructions = new_instructions

    async def _handle_added(self, event: SkillChangeEvent) -> None:
        assert event.skill is not None
        self._registry.add_skill(event.skill)

        tools = self._resolver.resolve(event.skill)
        tool_names: list[str] = []
        for t in tools:
            for agent in self._agents:
                try:
                    await agent.add_tool(t)
                except Exception:
                    logger.warning(
                        "Failed to add tool '%s' to agent '%s' (skill '%s')",
                        t.name,
                        agent.name,
                        event.skill_name,
                        exc_info=True,
                    )
            tool_names.append(t.name)
        self._skill_tools[event.skill_name] = tool_names

        if self._bus is not None:
            await self._bus.emit("skill:added", skill=event.skill)

    async def _handle_modified(self, event: SkillChangeEvent) -> None:
        assert event.skill is not None
        old_skill = (
            self._registry.get(event.skill_name)
            if event.skill_name in self._registry._skills
            else None
        )

        # Remove old tools
        old_tool_names = self._skill_tools.pop(event.skill_name, [])
        for name in old_tool_names:
            for agent in self._agents:
                try:
                    agent.remove_tool(name)
                except Exception:
                    logger.warning(
                        "Failed to remove tool '%s' from agent '%s' (skill '%s')",
                        name,
                        agent.name,
                        event.skill_name,
                        exc_info=True,
                    )

        # Update registry and add new tools
        self._registry.update_skill(event.skill)
        tools = self._resolver.resolve(event.skill)
        tool_names: list[str] = []
        for t in tools:
            for agent in self._agents:
                try:
                    await agent.add_tool(t)
                except Exception:
                    logger.warning(
                        "Failed to add tool '%s' to agent '%s' (skill '%s')",
                        t.name,
                        agent.name,
                        event.skill_name,
                        exc_info=True,
                    )
            tool_names.append(t.name)
        self._skill_tools[event.skill_name] = tool_names

        if self._bus is not None:
            await self._bus.emit("skill:modified", old_skill=old_skill, new_skill=event.skill)

    async def _handle_removed(self, event: SkillChangeEvent) -> None:
        old_skill = self._registry.remove_skill(event.skill_name)

        old_tool_names = self._skill_tools.pop(event.skill_name, [])
        for name in old_tool_names:
            for agent in self._agents:
                try:
                    agent.remove_tool(name)
                except Exception:
                    logger.warning(
                        "Failed to remove tool '%s' from agent '%s' (skill '%s')",
                        name,
                        agent.name,
                        event.skill_name,
                        exc_info=True,
                    )

        if self._bus is not None:
            await self._bus.emit("skill:removed", skill=old_skill or event.skill)
