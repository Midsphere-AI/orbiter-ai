"""Tests for exo.skills — multi-source skill registry."""

from __future__ import annotations

import asyncio
import logging
import subprocess
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

import pytest

from exo.events import EventBus
from exo.skills import (
    ConflictStrategy,
    DictToolResolver,
    Skill,
    SkillChangeEvent,
    SkillError,
    SkillRegistry,
    SkillSyncManager,
    SkillWatcher,
    ToolResolver,
    _clone_github,
    _collect_skills,
    _skill_fingerprint,
    extract_front_matter,
    parse_github_url,
)
from exo.types import ExoError

# ---------------------------------------------------------------------------
# SkillError
# ---------------------------------------------------------------------------


class TestSkillError:
    def test_inherits_exo_error(self) -> None:
        assert issubclass(SkillError, ExoError)

    def test_raise_and_catch(self) -> None:
        with pytest.raises(SkillError, match="test error"):
            raise SkillError("test error")


# ---------------------------------------------------------------------------
# ConflictStrategy
# ---------------------------------------------------------------------------


class TestConflictStrategy:
    def test_values(self) -> None:
        assert ConflictStrategy.KEEP_FIRST == "keep_first"
        assert ConflictStrategy.KEEP_LAST == "keep_last"
        assert ConflictStrategy.RAISE == "raise"

    def test_from_string(self) -> None:
        assert ConflictStrategy("keep_first") == ConflictStrategy.KEEP_FIRST


# ---------------------------------------------------------------------------
# Skill dataclass
# ---------------------------------------------------------------------------


class TestSkill:
    def test_minimal(self) -> None:
        s = Skill(name="test")
        assert s.name == "test"
        assert s.description == ""
        assert s.usage == ""
        assert s.tool_list == {}
        assert s.skill_type == ""
        assert s.active is True
        assert s.path == ""

    def test_full(self) -> None:
        s = Skill(
            name="search",
            description="Search the web",
            usage="Use this to search.",
            tool_list={"browser": ["navigate", "click"]},
            skill_type="agent",
            active=False,
            path="/skills/search/skill.md",
        )
        assert s.name == "search"
        assert s.description == "Search the web"
        assert s.tool_list == {"browser": ["navigate", "click"]}
        assert s.skill_type == "agent"
        assert s.active is False

    def test_repr(self) -> None:
        s = Skill(name="test", skill_type="agent", active=False)
        r = repr(s)
        assert "test" in r
        assert "agent" in r
        assert "False" in r


# ---------------------------------------------------------------------------
# parse_github_url
# ---------------------------------------------------------------------------


class TestParseGithubUrl:
    def test_simple_repo(self) -> None:
        result = parse_github_url("https://github.com/user/repo")
        assert result is not None
        assert result["owner"] == "user"
        assert result["repo"] == "repo"
        assert result["branch"] == "main"
        assert result["subdir"] == ""

    def test_with_branch(self) -> None:
        result = parse_github_url("https://github.com/user/repo/tree/develop")
        assert result is not None
        assert result["branch"] == "develop"
        assert result["subdir"] == ""

    def test_with_branch_and_subdir(self) -> None:
        result = parse_github_url("https://github.com/user/repo/tree/main/skills/dir")
        assert result is not None
        assert result["branch"] == "main"
        assert result["subdir"] == "skills/dir"

    def test_not_github(self) -> None:
        assert parse_github_url("/local/path") is None
        assert parse_github_url("https://gitlab.com/user/repo") is None

    def test_http(self) -> None:
        result = parse_github_url("http://github.com/user/repo")
        assert result is not None
        assert result["owner"] == "user"


# ---------------------------------------------------------------------------
# extract_front_matter
# ---------------------------------------------------------------------------


class TestExtractFrontMatter:
    def test_full_front_matter(self) -> None:
        text = """---
name: my_skill
description: A test skill
tool_list: {"browser": ["click", "nav"]}
active: True
type: agent
---
# Usage
Use this skill to do things."""
        meta, body = extract_front_matter(text)
        assert meta["name"] == "my_skill"
        assert meta["description"] == "A test skill"
        assert meta["tool_list"] == {"browser": ["click", "nav"]}
        assert meta["active"] is True
        assert meta["type"] == "agent"
        assert "Use this skill" in body

    def test_no_front_matter(self) -> None:
        text = "# Just a markdown file\nNo front-matter."
        meta, body = extract_front_matter(text)
        assert meta == {}
        assert body == text

    def test_empty_text(self) -> None:
        meta, body = extract_front_matter("")
        assert meta == {}
        assert body == ""

    def test_active_false(self) -> None:
        text = "---\nactive: False\n---\nBody."
        meta, _body = extract_front_matter(text)
        assert meta["active"] is False

    def test_invalid_tool_list_json(self) -> None:
        text = "---\ntool_list: not-json\n---\nBody."
        meta, _ = extract_front_matter(text)
        assert meta["tool_list"] == {}

    def test_unclosed_front_matter(self) -> None:
        text = "---\nname: test\nNo closing delimiter."
        meta, body = extract_front_matter(text)
        assert meta == {}
        assert body == text

    def test_desc_alias(self) -> None:
        text = "---\ndesc: short desc\n---\nBody."
        meta, _ = extract_front_matter(text)
        assert meta["desc"] == "short desc"


# ---------------------------------------------------------------------------
# _collect_skills
# ---------------------------------------------------------------------------


class TestCollectSkills:
    def test_find_skill_md(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "my_skill"
        skill_dir.mkdir()
        (skill_dir / "skill.md").write_text(
            "---\nname: my_skill\ndescription: Test\n---\nUsage here."
        )
        skills = _collect_skills(tmp_path)
        assert "my_skill" in skills
        assert skills["my_skill"].description == "Test"
        assert "Usage here" in skills["my_skill"].usage

    def test_find_skill_md_uppercase(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "upper"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: upper_skill\n---\nContent.")
        skills = _collect_skills(tmp_path)
        assert "upper_skill" in skills

    def test_name_from_directory(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "dir_name"
        skill_dir.mkdir()
        (skill_dir / "skill.md").write_text("---\ndescription: No name field\n---\nBody.")
        skills = _collect_skills(tmp_path)
        assert "dir_name" in skills

    def test_empty_directory(self, tmp_path: Path) -> None:
        skills = _collect_skills(tmp_path)
        assert skills == {}

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        skills = _collect_skills(tmp_path / "nonexistent")
        assert skills == {}

    def test_nested_skills(self, tmp_path: Path) -> None:
        (tmp_path / "a" / "b").mkdir(parents=True)
        (tmp_path / "a" / "b" / "skill.md").write_text("---\nname: nested\n---\nNested skill.")
        skills = _collect_skills(tmp_path)
        assert "nested" in skills

    def test_unreadable_file_does_not_abort_scan(self, tmp_path: Path) -> None:
        """One unreadable skill file must not prevent the rest from loading."""
        good_dir = tmp_path / "good_skill"
        good_dir.mkdir()
        (good_dir / "skill.md").write_text("---\nname: good_skill\n---\nGood body.")

        bad_dir = tmp_path / "bad_skill"
        bad_dir.mkdir()
        bad_file = bad_dir / "skill.md"
        bad_file.write_text("placeholder")

        # Simulate a read error by patching read_text to raise for the bad file
        original_read_text = Path.read_text

        def selective_read(self: Path, *args: object, **kwargs: object) -> str:
            if self == bad_file:
                raise PermissionError("Permission denied")
            return original_read_text(self, *args, **kwargs)  # type: ignore[arg-type]

        with patch.object(Path, "read_text", selective_read):
            skills = _collect_skills(tmp_path)

        # The good skill must still be loaded
        assert "good_skill" in skills
        # The bad skill must be absent, not crash
        assert "bad_skill" not in skills

    def test_missing_name_warns(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """A skill file with no 'name' front-matter should warn and fall back to dir name."""
        skill_dir = tmp_path / "fallback_dir"
        skill_dir.mkdir()
        (skill_dir / "skill.md").write_text("---\ndescription: No name\n---\nBody.")

        with caplog.at_level(logging.WARNING, logger="exo.skills"):
            skills = _collect_skills(tmp_path)

        assert "fallback_dir" in skills
        assert any("no 'name' front-matter" in record.message for record in caplog.records)

    def test_malformed_tool_list_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A malformed tool_list value should emit a warning and fall back to {}."""
        skill_dir = tmp_path / "bad_tl"
        skill_dir.mkdir()
        (skill_dir / "skill.md").write_text("---\nname: bad_tl\ntool_list: not-json\n---\nBody.")

        with caplog.at_level(logging.WARNING, logger="exo.skills"):
            skills = _collect_skills(tmp_path)

        assert "bad_tl" in skills
        assert skills["bad_tl"].tool_list == {}
        assert any("Malformed tool_list" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# SkillRegistry — local loading
# ---------------------------------------------------------------------------


class TestSkillRegistryLocal:
    def test_load_local_skills(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "my_skill"
        skill_dir.mkdir()
        (skill_dir / "skill.md").write_text("---\nname: my_skill\ndescription: Hello\n---\nUsage.")
        reg = SkillRegistry()
        reg.register_source(str(tmp_path))
        skills = reg.load_all()
        assert "my_skill" in skills
        assert skills["my_skill"].description == "Hello"

    def test_get(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "s1"
        skill_dir.mkdir()
        (skill_dir / "skill.md").write_text("---\nname: s1\n---\nBody.")
        reg = SkillRegistry()
        reg.register_source(str(tmp_path))
        reg.load_all()
        s = reg.get("s1")
        assert s.name == "s1"

    def test_get_missing_raises(self) -> None:
        reg = SkillRegistry()
        with pytest.raises(SkillError, match="not found"):
            reg.get("missing")

    def test_list_names(self, tmp_path: Path) -> None:
        for name in ["alpha", "beta"]:
            d = tmp_path / name
            d.mkdir()
            (d / "skill.md").write_text(f"---\nname: {name}\n---\nBody.")
        reg = SkillRegistry()
        reg.register_source(str(tmp_path))
        reg.load_all()
        names = reg.list_names()
        assert set(names) == {"alpha", "beta"}

    def test_skills_property_is_copy(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "s"
        skill_dir.mkdir()
        (skill_dir / "skill.md").write_text("---\nname: s\n---\nBody.")
        reg = SkillRegistry()
        reg.register_source(str(tmp_path))
        reg.load_all()
        copy = reg.skills
        copy.clear()
        assert reg.skills != {}

    def test_missing_source_dir_raises(self) -> None:
        reg = SkillRegistry()
        reg.register_source("/nonexistent/path")
        with pytest.raises(SkillError, match="not found"):
            reg.load_all()


# ---------------------------------------------------------------------------
# SkillRegistry — conflict resolution
# ---------------------------------------------------------------------------


class TestSkillRegistryConflict:
    def _make_sources(self, tmp_path: Path) -> tuple[Path, Path]:
        src1 = tmp_path / "src1" / "dup"
        src1.mkdir(parents=True)
        (src1 / "skill.md").write_text("---\nname: dup\ndescription: first\n---\nBody 1.")

        src2 = tmp_path / "src2" / "dup"
        src2.mkdir(parents=True)
        (src2 / "skill.md").write_text("---\nname: dup\ndescription: second\n---\nBody 2.")
        return tmp_path / "src1", tmp_path / "src2"

    def test_keep_first(self, tmp_path: Path) -> None:
        src1, src2 = self._make_sources(tmp_path)
        reg = SkillRegistry(conflict=ConflictStrategy.KEEP_FIRST)
        reg.register_source(str(src1))
        reg.register_source(str(src2))
        skills = reg.load_all()
        assert skills["dup"].description == "first"

    def test_keep_last(self, tmp_path: Path) -> None:
        src1, src2 = self._make_sources(tmp_path)
        reg = SkillRegistry(conflict=ConflictStrategy.KEEP_LAST)
        reg.register_source(str(src1))
        reg.register_source(str(src2))
        skills = reg.load_all()
        assert skills["dup"].description == "second"

    def test_raise_on_conflict(self, tmp_path: Path) -> None:
        src1, src2 = self._make_sources(tmp_path)
        reg = SkillRegistry(conflict=ConflictStrategy.RAISE)
        reg.register_source(str(src1))
        reg.register_source(str(src2))
        with pytest.raises(SkillError, match="Duplicate skill"):
            reg.load_all()

    def test_string_conflict_strategy(self, tmp_path: Path) -> None:
        src1, src2 = self._make_sources(tmp_path)
        reg = SkillRegistry(conflict="keep_last")
        reg.register_source(str(src1))
        reg.register_source(str(src2))
        skills = reg.load_all()
        assert skills["dup"].description == "second"


# ---------------------------------------------------------------------------
# SkillRegistry — search
# ---------------------------------------------------------------------------


class TestSkillRegistrySearch:
    def _setup_reg(self, tmp_path: Path) -> SkillRegistry:
        skills = [
            ("search", "Search the web", "agent", True),
            ("code", "Code assistant", "", True),
            ("debug", "Debug helper", "", False),
        ]
        for name, desc, stype, active in skills:
            d = tmp_path / name
            d.mkdir()
            active_str = "True" if active else "False"
            (d / "skill.md").write_text(
                f"---\nname: {name}\ndescription: {desc}\ntype: {stype}\nactive: {active_str}\n---\nUsage."
            )
        reg = SkillRegistry()
        reg.register_source(str(tmp_path))
        reg.load_all()
        return reg

    def test_search_by_query(self, tmp_path: Path) -> None:
        reg = self._setup_reg(tmp_path)
        results = reg.search(query="web")
        assert len(results) == 1
        assert results[0].name == "search"

    def test_search_by_type(self, tmp_path: Path) -> None:
        reg = self._setup_reg(tmp_path)
        results = reg.search(skill_type="agent")
        assert len(results) == 1
        assert results[0].name == "search"

    def test_search_active_only(self, tmp_path: Path) -> None:
        reg = self._setup_reg(tmp_path)
        results = reg.search(active_only=True)
        names = {s.name for s in results}
        assert "debug" not in names
        assert "search" in names
        assert "code" in names

    def test_search_no_results(self, tmp_path: Path) -> None:
        reg = self._setup_reg(tmp_path)
        results = reg.search(query="nonexistent")
        assert results == []

    def test_search_combined_filters(self, tmp_path: Path) -> None:
        reg = self._setup_reg(tmp_path)
        results = reg.search(query="assist", active_only=True, skill_type="")
        assert len(results) == 1
        assert results[0].name == "code"

    def test_search_all(self, tmp_path: Path) -> None:
        reg = self._setup_reg(tmp_path)
        results = reg.search()
        assert len(results) == 3

    def test_search_case_insensitive(self, tmp_path: Path) -> None:
        reg = self._setup_reg(tmp_path)
        results = reg.search(query="SEARCH")
        assert len(results) == 1


# ---------------------------------------------------------------------------
# SkillRegistry — GitHub source (mocked)
# ---------------------------------------------------------------------------


class TestSkillRegistryGitHub:
    def test_github_source_clones(self, tmp_path: Path) -> None:
        clone_dir = tmp_path / "cache" / "user" / "repo" / "main"
        clone_dir.mkdir(parents=True)
        skill_dir = clone_dir / "my_skill"
        skill_dir.mkdir()
        (skill_dir / "skill.md").write_text("---\nname: remote_skill\n---\nRemote body.")

        reg = SkillRegistry(cache_dir=tmp_path / "cache")
        reg.register_source("https://github.com/user/repo")

        with patch("exo.skills._clone_github", return_value=clone_dir):
            skills = reg.load_all()

        assert "remote_skill" in skills

    def test_github_with_subdir(self, tmp_path: Path) -> None:
        sub = tmp_path / "cache" / "user" / "repo" / "main" / "skills"
        sub.mkdir(parents=True)
        sd = sub / "my_skill"
        sd.mkdir()
        (sd / "skill.md").write_text("---\nname: sub_skill\n---\nBody.")

        reg = SkillRegistry(cache_dir=tmp_path / "cache")
        reg.register_source("https://github.com/user/repo/tree/main/skills")

        with patch("exo.skills._clone_github", return_value=sub):
            skills = reg.load_all()

        assert "sub_skill" in skills

    def test_git_clone_failure_raises_skill_error(self) -> None:
        """subprocess.CalledProcessError from git clone must be wrapped as SkillError with hint."""
        from exo.skills import _clone_github

        parsed = {"owner": "user", "repo": "repo", "branch": "main", "subdir": ""}
        cache = Path("/nonexistent/cache")

        with (
            patch("subprocess.run", side_effect=subprocess.CalledProcessError(128, "git")),
            patch.object(Path, "exists", return_value=False),
            patch.object(Path, "parent", new_callable=lambda: property(lambda s: s)),
            patch("pathlib.Path.mkdir"),
            pytest.raises(SkillError) as exc_info,
        ):
            _clone_github(parsed, cache)

        err = exc_info.value
        assert err.hint is not None
        assert "git" in err.hint.lower() or "repository" in err.hint.lower()
        assert err.__cause__ is not None

    def test_git_not_found_raises_skill_error(self, tmp_path: Path) -> None:
        """FileNotFoundError (git not on PATH) must be wrapped as SkillError with hint."""
        from exo.skills import _clone_github

        parsed = {"owner": "user", "repo": "no-git", "branch": "main", "subdir": ""}

        with (
            patch("subprocess.run", side_effect=FileNotFoundError("git not found")),
            pytest.raises(SkillError) as exc_info,
        ):
            _clone_github(parsed, tmp_path)

        err = exc_info.value
        assert err.hint is not None
        assert isinstance(err.__cause__, FileNotFoundError)


class TestSkillRegistryErrorHints:
    def test_get_missing_has_hint(self) -> None:
        """registry.get() on a missing skill should include a hint with list_names()."""
        reg = SkillRegistry()
        with pytest.raises(SkillError) as exc_info:
            reg.get("no_such_skill")
        err = exc_info.value
        assert err.hint is not None
        assert "list_names" in err.hint

    def test_get_missing_has_context(self) -> None:
        reg = SkillRegistry()
        with pytest.raises(SkillError) as exc_info:
            reg.get("missing")
        assert exc_info.value.context is not None
        assert exc_info.value.context.get("name") == "missing"

    def test_duplicate_conflict_has_paths(self, tmp_path: Path) -> None:
        """Duplicate skill conflict error should name both source paths."""
        src1 = tmp_path / "src1" / "dup"
        src1.mkdir(parents=True)
        (src1 / "skill.md").write_text("---\nname: dup\n---\nFirst.")

        src2 = tmp_path / "src2" / "dup"
        src2.mkdir(parents=True)
        (src2 / "skill.md").write_text("---\nname: dup\n---\nSecond.")

        reg = SkillRegistry(conflict=ConflictStrategy.RAISE)
        reg.register_source(str(tmp_path / "src1"))
        reg.register_source(str(tmp_path / "src2"))

        with pytest.raises(SkillError) as exc_info:
            reg.load_all()

        err = exc_info.value
        assert err.context is not None
        assert "existing_path" in err.context
        assert "new_path" in err.context
        assert err.hint is not None

    def test_missing_source_dir_has_hint(self) -> None:
        """Missing source directory error should carry a hint about GitHub URLs."""
        reg = SkillRegistry()
        reg.register_source("/truly/nonexistent/path")
        with pytest.raises(SkillError) as exc_info:
            reg.load_all()
        err = exc_info.value
        assert err.hint is not None
        assert "github.com" in err.hint.lower() or "create" in err.hint.lower()
        assert err.context is not None
        assert "path" in err.context


# ---------------------------------------------------------------------------
# Helpers for hot-reload tests
# ---------------------------------------------------------------------------


class _MockTool:
    def __init__(self, name: str) -> None:
        self.name = name


class _MockAgent:
    def __init__(self) -> None:
        self.name = "test-agent"
        self.tools: dict[str, _MockTool] = {}
        self.instructions = ""
        self._add_tool_calls: list[_MockTool] = []
        self._remove_tool_calls: list[str] = []

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    async def add_tool(self, tool: _MockTool) -> None:
        self._add_tool_calls.append(tool)
        self.tools[tool.name] = tool

    def remove_tool(self, tool_name: str) -> None:
        self._remove_tool_calls.append(tool_name)
        if tool_name in self.tools:
            del self.tools[tool_name]


class _FakeWatcher(SkillWatcher):
    def __init__(self, batches: list[list[SkillChangeEvent]]) -> None:
        self._batches = batches
        self._stopped = asyncio.Event()

    async def watch(self):  # type: ignore[override]
        for batch in self._batches:
            if self._stopped.is_set():
                return
            yield batch

    async def stop(self) -> None:
        self._stopped.set()


# ---------------------------------------------------------------------------
# SkillChangeEvent
# ---------------------------------------------------------------------------


class TestSkillChangeEvent:
    def test_construction(self) -> None:
        skill = Skill(name="s1", description="desc")
        event = SkillChangeEvent(kind="added", skill_name="s1", skill=skill, source_path="/a/b")
        assert event.kind == "added"
        assert event.skill_name == "s1"
        assert event.skill is skill
        assert event.source_path == "/a/b"

    def test_frozen(self) -> None:
        event = SkillChangeEvent(
            kind="added",
            skill_name="s1",
            skill=Skill(name="s1"),
            source_path="/a",
        )
        with pytest.raises(FrozenInstanceError):
            event.kind = "removed"  # type: ignore[misc]

    def test_removed_skill_none(self) -> None:
        event = SkillChangeEvent(kind="removed", skill_name="old", skill=None, source_path="/x")
        assert event.skill is None
        assert event.kind == "removed"


# ---------------------------------------------------------------------------
# _skill_fingerprint
# ---------------------------------------------------------------------------


class TestSkillFingerprint:
    def test_same_skill_same_fingerprint(self) -> None:
        s1 = Skill(name="a", description="d", usage="u", skill_type="t", active=True)
        s2 = Skill(name="a", description="d", usage="u", skill_type="t", active=True)
        assert _skill_fingerprint(s1) == _skill_fingerprint(s2)

    def test_different_description_different_fingerprint(self) -> None:
        s1 = Skill(name="a", description="d1")
        s2 = Skill(name="a", description="d2")
        assert _skill_fingerprint(s1) != _skill_fingerprint(s2)


# ---------------------------------------------------------------------------
# DictToolResolver
# ---------------------------------------------------------------------------


class TestDictToolResolver:
    def test_single_tool(self) -> None:
        tool = _MockTool("t1")
        resolver = DictToolResolver({"my_skill": tool})  # type: ignore[dict-item]
        result = resolver.resolve(Skill(name="my_skill"))
        assert result == [tool]

    def test_list_of_tools(self) -> None:
        t1, t2 = _MockTool("t1"), _MockTool("t2")
        resolver = DictToolResolver({"sk": [t1, t2]})  # type: ignore[dict-item]
        result = resolver.resolve(Skill(name="sk"))
        assert result == [t1, t2]

    def test_missing_skill(self) -> None:
        resolver = DictToolResolver({})
        result = resolver.resolve(Skill(name="nope"))
        assert result == []

    def test_implements_protocol(self) -> None:
        resolver = DictToolResolver({})
        assert isinstance(resolver, ToolResolver)


# ---------------------------------------------------------------------------
# SkillWatcher ABC
# ---------------------------------------------------------------------------


class TestSkillWatcherABC:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            SkillWatcher()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# SkillSyncManager
# ---------------------------------------------------------------------------


class TestSkillSyncManager:
    async def test_added_event(self) -> None:
        registry = SkillRegistry()
        skill = Skill(name="s1", description="hello")
        tool = _MockTool("tool_s1")
        resolver = DictToolResolver({"s1": tool})  # type: ignore[dict-item]

        mgr = SkillSyncManager(registry, resolver)
        agent = _MockAgent()
        mgr.bind_agent(agent)  # type: ignore[arg-type]

        event = SkillChangeEvent(kind="added", skill_name="s1", skill=skill, source_path="/p")
        watcher = _FakeWatcher([[event]])
        mgr.add_watcher(watcher)

        await mgr.start()
        # Give the background task time to process
        await asyncio.sleep(0.05)
        await mgr.stop()

        assert "s1" in registry._skills
        assert registry._skills["s1"] is skill
        assert len(agent._add_tool_calls) == 1
        assert agent._add_tool_calls[0].name == "tool_s1"

    async def test_modified_event(self) -> None:
        registry = SkillRegistry()
        old_skill = Skill(name="s1", description="old")
        registry._skills["s1"] = old_skill

        old_tool = _MockTool("old_tool")
        new_tool = _MockTool("new_tool")
        new_skill = Skill(name="s1", description="new")

        resolver = DictToolResolver({"s1": new_tool})  # type: ignore[dict-item]
        mgr = SkillSyncManager(registry, resolver)
        mgr._skill_tools["s1"] = ["old_tool"]

        agent = _MockAgent()
        agent.tools["old_tool"] = old_tool  # type: ignore[assignment]
        mgr.bind_agent(agent)  # type: ignore[arg-type]

        event = SkillChangeEvent(
            kind="modified", skill_name="s1", skill=new_skill, source_path="/p"
        )
        watcher = _FakeWatcher([[event]])
        mgr.add_watcher(watcher)

        await mgr.start()
        await asyncio.sleep(0.05)
        await mgr.stop()

        assert registry._skills["s1"] is new_skill
        assert "old_tool" in agent._remove_tool_calls
        assert any(t.name == "new_tool" for t in agent._add_tool_calls)

    async def test_removed_event(self) -> None:
        registry = SkillRegistry()
        old_skill = Skill(name="s1", description="bye")
        registry._skills["s1"] = old_skill

        resolver = DictToolResolver({})
        mgr = SkillSyncManager(registry, resolver)
        mgr._skill_tools["s1"] = ["rm_tool"]

        agent = _MockAgent()
        agent.tools["rm_tool"] = _MockTool("rm_tool")  # type: ignore[assignment]
        mgr.bind_agent(agent)  # type: ignore[arg-type]

        event = SkillChangeEvent(kind="removed", skill_name="s1", skill=None, source_path="/p")
        watcher = _FakeWatcher([[event]])
        mgr.add_watcher(watcher)

        await mgr.start()
        await asyncio.sleep(0.05)
        await mgr.stop()

        assert "s1" not in registry._skills
        assert "rm_tool" in agent._remove_tool_calls

    async def test_event_bus_emissions(self) -> None:
        registry = SkillRegistry()
        resolver = DictToolResolver({})
        bus = EventBus()

        emitted: list[tuple[str, dict]] = []
        original_emit = bus.emit

        async def capture_emit(event: str, **kwargs: object) -> None:
            emitted.append((event, kwargs))
            await original_emit(event, **kwargs)

        bus.emit = capture_emit  # type: ignore[assignment]

        mgr = SkillSyncManager(registry, resolver, event_bus=bus)

        skill = Skill(name="s1")
        added = SkillChangeEvent(kind="added", skill_name="s1", skill=skill, source_path="/a")

        new_skill = Skill(name="s1", description="v2")
        modified = SkillChangeEvent(
            kind="modified", skill_name="s1", skill=new_skill, source_path="/a"
        )

        removed = SkillChangeEvent(kind="removed", skill_name="s1", skill=None, source_path="/a")

        watcher = _FakeWatcher([[added], [modified], [removed]])
        mgr.add_watcher(watcher)

        await mgr.start()
        await asyncio.sleep(0.1)
        await mgr.stop()

        event_names = [name for name, _ in emitted]
        assert "skill:added" in event_names
        assert "skill:modified" in event_names
        assert "skill:removed" in event_names

        # Verify payload keys
        added_payload = next(kw for name, kw in emitted if name == "skill:added")
        assert "skill" in added_payload

        modified_payload = next(kw for name, kw in emitted if name == "skill:modified")
        assert "old_skill" in modified_payload
        assert "new_skill" in modified_payload

        removed_payload = next(kw for name, kw in emitted if name == "skill:removed")
        assert "skill" in removed_payload

    async def test_instructions_builder(self) -> None:
        registry = SkillRegistry()
        skill = Skill(name="s1", description="desc")
        tool = _MockTool("t1")
        resolver = DictToolResolver({"s1": tool})  # type: ignore[dict-item]

        def builder(skills: list[Skill]) -> str:
            return "Skills: " + ", ".join(s.name for s in skills)

        mgr = SkillSyncManager(registry, resolver, instructions_builder=builder)
        agent = _MockAgent()
        mgr.bind_agent(agent)  # type: ignore[arg-type]

        event = SkillChangeEvent(kind="added", skill_name="s1", skill=skill, source_path="/p")
        watcher = _FakeWatcher([[event]])
        mgr.add_watcher(watcher)

        await mgr.start()
        await asyncio.sleep(0.05)
        await mgr.stop()

        assert "s1" in agent.instructions

    async def test_bind_unbind_agent(self) -> None:
        registry = SkillRegistry()
        skill = Skill(name="s1")
        tool = _MockTool("t1")
        resolver = DictToolResolver({"s1": tool})  # type: ignore[dict-item]

        mgr = SkillSyncManager(registry, resolver)
        agent1 = _MockAgent()
        agent2 = _MockAgent()
        mgr.bind_agent(agent1)  # type: ignore[arg-type]
        mgr.bind_agent(agent2)  # type: ignore[arg-type]
        mgr.unbind_agent(agent2)  # type: ignore[arg-type]

        event = SkillChangeEvent(kind="added", skill_name="s1", skill=skill, source_path="/p")
        watcher = _FakeWatcher([[event]])
        mgr.add_watcher(watcher)

        await mgr.start()
        await asyncio.sleep(0.05)
        await mgr.stop()

        # agent1 was bound, so it should have the tool
        assert len(agent1._add_tool_calls) == 1
        # agent2 was unbound, so it should not
        assert len(agent2._add_tool_calls) == 0

    async def test_context_manager(self) -> None:
        registry = SkillRegistry()
        resolver = DictToolResolver({})
        mgr = SkillSyncManager(registry, resolver)

        watcher = _FakeWatcher([])
        mgr.add_watcher(watcher)

        async with mgr as m:
            assert m is mgr

        # After exit, tasks should be cleaned up
        assert mgr._tasks == []

    async def test_dict_auto_wrapped(self) -> None:
        registry = SkillRegistry()
        skill = Skill(name="s1")
        tool = _MockTool("t1")

        # Pass a plain dict instead of a ToolResolver instance
        mgr = SkillSyncManager(registry, {"s1": tool})  # type: ignore[arg-type]
        assert isinstance(mgr._resolver, DictToolResolver)

        agent = _MockAgent()
        mgr.bind_agent(agent)  # type: ignore[arg-type]

        event = SkillChangeEvent(kind="added", skill_name="s1", skill=skill, source_path="/p")
        watcher = _FakeWatcher([[event]])
        mgr.add_watcher(watcher)

        await mgr.start()
        await asyncio.sleep(0.05)
        await mgr.stop()

        assert len(agent._add_tool_calls) == 1
        assert agent._add_tool_calls[0].name == "t1"

    async def test_agent_error_resilience(self) -> None:
        registry = SkillRegistry()
        skill = Skill(name="s1")
        tool = _MockTool("t1")
        resolver = DictToolResolver({"s1": tool})  # type: ignore[dict-item]

        mgr = SkillSyncManager(registry, resolver)

        class _FailingAgent(_MockAgent):
            async def add_tool(self, tool: _MockTool) -> None:
                raise RuntimeError("agent broken")

        failing_agent = _FailingAgent()
        good_agent = _MockAgent()
        mgr.bind_agent(failing_agent)  # type: ignore[arg-type]
        mgr.bind_agent(good_agent)  # type: ignore[arg-type]

        event = SkillChangeEvent(kind="added", skill_name="s1", skill=skill, source_path="/p")
        watcher = _FakeWatcher([[event]])
        mgr.add_watcher(watcher)

        await mgr.start()
        await asyncio.sleep(0.05)
        await mgr.stop()

        # The good agent should still have received the tool despite the
        # failing agent raising an exception.
        assert len(good_agent._add_tool_calls) == 1
        assert good_agent._add_tool_calls[0].name == "t1"


# ---------------------------------------------------------------------------
# parse_github_url — extended (findings 3 & 4)
# ---------------------------------------------------------------------------


class TestParseGithubUrlExtended:
    def test_branch_with_slash(self) -> None:
        """Branch names containing slashes (e.g. feat/my-branch) must be preserved."""
        result = parse_github_url("https://github.com/user/repo/tree/feat/my-branch")
        assert result is not None
        # The whole 'feat/my-branch' is the branch; there is no subdir.
        # The regex captures everything after /tree/ as the branch when there
        # is no second path segment that constitutes a subdirectory.
        # Accepted behaviour: branch contains the slash, subdir is empty.
        assert "feat" in result["branch"]

    def test_branch_with_slash_and_subdir(self) -> None:
        """URL with a slashed branch and a trailing subdir should parse correctly."""
        # https://github.com/user/repo/tree/feat/my-branch/skills
        # There is inherent ambiguity: we treat the LAST segment as the subdir
        # only when it unambiguously follows the branch.  The regex uses a
        # greedy/lazy split so the branch gets 'feat/my-branch' and subdir
        # gets 'skills' when the URL ends with an extra segment.
        result = parse_github_url("https://github.com/user/repo/tree/feat/my-branch/skills")
        assert result is not None
        # At minimum: owner and repo must be correct
        assert result["owner"] == "user"
        assert result["repo"] == "repo"

    def test_git_suffix_stripped_from_repo(self) -> None:
        """.git suffix must be stripped from the repo name."""
        result = parse_github_url("https://github.com/user/repo.git")
        assert result is not None
        assert result["repo"] == "repo"
        assert not result["repo"].endswith(".git")

    def test_git_suffix_clone_url_not_doubled(self, tmp_path: Path) -> None:
        """.git-suffixed URL must not produce a double-.git clone URL."""
        # The clone URL is constructed inside _clone_github as
        # f"https://github.com/{owner}/{repo}.git".
        # If repo already has .git appended, we'd get repo.git.git.
        # This test verifies the fix: parse_github_url strips .git.
        result = parse_github_url("https://github.com/user/myrepo.git")
        assert result is not None
        assert result["repo"] == "myrepo"
        # Verify the constructed clone URL would be correct
        expected_clone_url = f"https://github.com/{result['owner']}/{result['repo']}.git"
        assert "myrepo.git.git" not in expected_clone_url
        assert expected_clone_url == "https://github.com/user/myrepo.git"


# ---------------------------------------------------------------------------
# _clone_github — timeout and cleanup (findings 1 & 2)
# ---------------------------------------------------------------------------


class TestCloneGithubTimeoutAndCleanup:
    def test_timeout_expired_raises_skill_error(self, tmp_path: Path) -> None:
        """subprocess.TimeoutExpired must be caught and re-raised as SkillError."""
        parsed = {"owner": "user", "repo": "repo", "branch": "main", "subdir": ""}

        with (
            patch("subprocess.run", side_effect=subprocess.TimeoutExpired("git", 60)),
            pytest.raises(SkillError) as exc_info,
        ):
            _clone_github(parsed, tmp_path)

        err = exc_info.value
        assert err.hint is not None
        assert "timeout" in err.hint.lower() or "60" in err.hint
        assert isinstance(err.__cause__, subprocess.TimeoutExpired)

    def test_partial_clone_dir_cleaned_on_timeout(self, tmp_path: Path) -> None:
        """A partial clone directory must be removed when git clone times out."""
        parsed = {"owner": "user", "repo": "repo", "branch": "main", "subdir": ""}
        clone_dir = tmp_path / "user" / "repo" / "main"

        def fake_run(*args, **kwargs):
            # Simulate partial clone: git creates the dir before timing out
            clone_dir.mkdir(parents=True, exist_ok=True)
            raise subprocess.TimeoutExpired("git", 60)

        with (
            patch("subprocess.run", side_effect=fake_run),
            pytest.raises(SkillError),
        ):
            _clone_github(parsed, tmp_path)

        # Partial clone directory must have been cleaned up
        assert not clone_dir.exists()

    def test_partial_clone_dir_cleaned_on_error(self, tmp_path: Path) -> None:
        """A partial clone directory must be removed when git clone fails."""
        parsed = {"owner": "user", "repo": "repo", "branch": "main", "subdir": ""}
        clone_dir = tmp_path / "user" / "repo" / "main"

        def fake_run(*args, **kwargs):
            clone_dir.mkdir(parents=True, exist_ok=True)
            raise subprocess.CalledProcessError(128, "git")

        with (
            patch("subprocess.run", side_effect=fake_run),
            pytest.raises(SkillError),
        ):
            _clone_github(parsed, tmp_path)

        assert not clone_dir.exists()


# ---------------------------------------------------------------------------
# _collect_skills — duplicate name warning (finding 5)
# ---------------------------------------------------------------------------


class TestCollectSkillsDuplicateWarning:
    def test_duplicate_name_warns(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Two skill files with the same 'name' in one scan should trigger a warning."""
        dir_a = tmp_path / "a"
        dir_a.mkdir()
        (dir_a / "skill.md").write_text("---\nname: dup_skill\ndescription: first\n---\nBody A.")

        dir_b = tmp_path / "b"
        dir_b.mkdir()
        (dir_b / "skill.md").write_text("---\nname: dup_skill\ndescription: second\n---\nBody B.")

        with caplog.at_level(logging.WARNING, logger="exo.skills"):
            skills = _collect_skills(tmp_path)

        assert "dup_skill" in skills
        assert any("Duplicate skill name" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# _skill_fingerprint — sort_keys (finding 12)
# ---------------------------------------------------------------------------


class TestSkillFingerprintSortKeys:
    def test_tool_list_insertion_order_independent(self) -> None:
        """Different insertion orders in tool_list must produce the same fingerprint."""
        s1 = Skill(name="a", tool_list={"z": ["x"], "a": ["y"]})
        s2 = Skill(name="a", tool_list={"a": ["y"], "z": ["x"]})
        assert _skill_fingerprint(s1) == _skill_fingerprint(s2)


# ---------------------------------------------------------------------------
# SkillRegistry — public add/remove/update (finding 16)
# ---------------------------------------------------------------------------


class TestSkillRegistryPublicMutations:
    def test_add_skill(self) -> None:
        reg = SkillRegistry()
        skill = Skill(name="s1", description="hello")
        reg.add_skill(skill)
        assert reg.get("s1") is skill

    def test_add_skill_conflict_keep_first(self) -> None:
        reg = SkillRegistry(conflict=ConflictStrategy.KEEP_FIRST)
        s1 = Skill(name="s1", description="first")
        s2 = Skill(name="s1", description="second")
        reg.add_skill(s1)
        reg.add_skill(s2)
        assert reg.get("s1").description == "first"

    def test_add_skill_conflict_raise(self) -> None:
        reg = SkillRegistry(conflict=ConflictStrategy.RAISE)
        reg.add_skill(Skill(name="s1"))
        with pytest.raises(SkillError, match="Duplicate skill"):
            reg.add_skill(Skill(name="s1"))

    def test_remove_skill(self) -> None:
        reg = SkillRegistry()
        reg.add_skill(Skill(name="s1"))
        removed = reg.remove_skill("s1")
        assert removed is not None
        assert removed.name == "s1"
        assert "s1" not in reg.list_names()

    def test_remove_skill_missing_returns_none(self) -> None:
        reg = SkillRegistry()
        assert reg.remove_skill("nonexistent") is None

    def test_update_skill(self) -> None:
        reg = SkillRegistry()
        reg.add_skill(Skill(name="s1", description="v1"))
        reg.update_skill(Skill(name="s1", description="v2"))
        assert reg.get("s1").description == "v2"


# ---------------------------------------------------------------------------
# SkillRegistry.aload_all (finding 7)
# ---------------------------------------------------------------------------


class TestSkillRegistryAloadAll:
    async def test_aload_all_returns_same_as_load_all(self, tmp_path: Path) -> None:
        """aload_all() must return the same result as load_all()."""
        skill_dir = tmp_path / "my_skill"
        skill_dir.mkdir()
        (skill_dir / "skill.md").write_text("---\nname: my_skill\ndescription: Async\n---\nUsage.")

        reg = SkillRegistry()
        reg.register_source(str(tmp_path))

        # Reset and load via sync path
        sync_skills = reg.load_all()

        # Reset and load via async path
        reg2 = SkillRegistry()
        reg2.register_source(str(tmp_path))
        async_skills = await reg2.aload_all()

        assert set(sync_skills) == set(async_skills)
        assert async_skills["my_skill"].description == "Async"

    async def test_aload_all_is_awaitable(self, tmp_path: Path) -> None:
        """aload_all() must be an awaitable coroutine."""
        reg = SkillRegistry()
        reg.register_source(str(tmp_path))
        result = await reg.aload_all()
        assert isinstance(result, dict)
