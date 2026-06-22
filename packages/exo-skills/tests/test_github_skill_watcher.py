"""Tests for GitHubPollingWatcher — GitHub skill polling via git pull."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exo.skills import Skill, SkillError
from exo_skills.watchers.github import GitHubPollingWatcher, _diff_snapshots

# ---------------------------------------------------------------------------
# _diff_snapshots helper (same logic as local, but duplicated in github.py)
# ---------------------------------------------------------------------------


class TestGitHubDiffSnapshots:
    def test_empty_both(self) -> None:
        assert _diff_snapshots({}, {}, "https://example.com") == []

    def test_added(self) -> None:
        new = {"s1": Skill(name="s1")}
        events = _diff_snapshots({}, new, "https://example.com")
        assert len(events) == 1
        assert events[0].kind == "added"

    def test_removed(self) -> None:
        old = {"s1": Skill(name="s1")}
        events = _diff_snapshots(old, {}, "https://example.com")
        assert len(events) == 1
        assert events[0].kind == "removed"

    def test_modified(self) -> None:
        old = {"s1": Skill(name="s1", description="v1")}
        new = {"s1": Skill(name="s1", description="v2")}
        events = _diff_snapshots(old, new, "https://example.com")
        assert len(events) == 1
        assert events[0].kind == "modified"


# ---------------------------------------------------------------------------
# GitHubPollingWatcher
# ---------------------------------------------------------------------------


class TestGitHubPollingWatcher:
    def test_invalid_url_raises(self) -> None:
        with pytest.raises(SkillError, match="Invalid GitHub URL"):
            GitHubPollingWatcher("/not/a/github/url")

    def test_invalid_url_has_hint(self) -> None:
        """Invalid GitHub URL error should include the expected format as a hint."""
        with pytest.raises(SkillError) as exc_info:
            GitHubPollingWatcher("/not/a/github/url")
        err = exc_info.value
        assert err.hint is not None
        assert "github.com" in err.hint

    def test_invalid_url_has_context(self) -> None:
        """Invalid GitHub URL error should include the bad URL in context."""
        bad_url = "/not/a/github/url"
        with pytest.raises(SkillError) as exc_info:
            GitHubPollingWatcher(bad_url)
        err = exc_info.value
        assert err.context is not None
        assert err.context.get("source_url") == bad_url

    def test_valid_url_parses(self) -> None:
        watcher = GitHubPollingWatcher(
            "https://github.com/acme/skills/tree/main/agents",
            poll_interval=10.0,
        )
        assert watcher._parsed["owner"] == "acme"
        assert watcher._parsed["repo"] == "skills"
        assert watcher._parsed["branch"] == "main"
        assert watcher._parsed["subdir"] == "agents"

    async def test_stop_terminates_watch(self, tmp_path: Path) -> None:
        """stop() should cause watch() to return without blocking."""
        url = "https://github.com/user/repo"
        clone_dir = tmp_path / "user" / "repo" / "main"
        clone_dir.mkdir(parents=True)

        watcher = GitHubPollingWatcher(url, poll_interval=0.1, cache_dir=tmp_path)

        with patch("exo_skills.watchers.github._clone_github", return_value=clone_dir):

            async def stop_soon() -> None:
                await asyncio.sleep(0.05)
                await watcher.stop()

            task = asyncio.create_task(stop_soon())

            batches = []
            async for b in watcher.watch():
                batches.append(b)

            await task
            assert batches == []

    async def test_detects_added_skill_after_pull(self, tmp_path: Path) -> None:
        """After a git pull adds a new skill, the watcher should yield an 'added' event."""
        url = "https://github.com/user/repo"
        clone_dir = tmp_path / "user" / "repo" / "main"
        clone_dir.mkdir(parents=True)

        watcher = GitHubPollingWatcher(url, poll_interval=0.1, cache_dir=tmp_path)

        call_count = 0

        async def fake_subprocess(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # On first pull, "create" a skill file
            if call_count == 1:
                skill_dir = clone_dir / "new_skill"
                skill_dir.mkdir(exist_ok=True)
                (skill_dir / "skill.md").write_text(
                    "---\nname: new_skill\ndescription: Added by pull\n---\nBody."
                )
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"Already up to date.", b""))
            return mock_proc

        with (
            patch("exo_skills.watchers.github._clone_github", return_value=clone_dir),
            patch(
                "exo_skills.watchers.github.asyncio.create_subprocess_exec",
                side_effect=fake_subprocess,
            ),
        ):
            batch = None
            async for b in watcher.watch():
                batch = b
                await watcher.stop()
                break

        assert batch is not None
        assert any(e.kind == "added" and e.skill_name == "new_skill" for e in batch)

    async def test_no_changes_yields_nothing(self, tmp_path: Path) -> None:
        """If git pull results in no skill changes, no batch is yielded."""
        url = "https://github.com/user/repo"
        clone_dir = tmp_path / "user" / "repo" / "main"
        clone_dir.mkdir(parents=True)

        watcher = GitHubPollingWatcher(url, poll_interval=0.1, cache_dir=tmp_path)

        pull_count = 0

        async def fake_subprocess(*args, **kwargs):
            nonlocal pull_count
            pull_count += 1
            # Stop after 2 polls with no changes
            if pull_count >= 2:
                await watcher.stop()
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"Already up to date.", b""))
            return mock_proc

        with (
            patch("exo_skills.watchers.github._clone_github", return_value=clone_dir),
            patch(
                "exo_skills.watchers.github.asyncio.create_subprocess_exec",
                side_effect=fake_subprocess,
            ),
        ):
            batches = []
            async for b in watcher.watch():
                batches.append(b)

        assert batches == []

    async def test_git_pull_failure_continues(self, tmp_path: Path) -> None:
        """A failing git pull should log a warning and continue to the next poll."""
        url = "https://github.com/user/repo"
        clone_dir = tmp_path / "user" / "repo" / "main"
        clone_dir.mkdir(parents=True)

        watcher = GitHubPollingWatcher(url, poll_interval=0.1, cache_dir=tmp_path)

        call_count = 0

        async def fake_subprocess(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_proc = MagicMock()
            if call_count == 1:
                # First pull fails
                mock_proc.returncode = 1
                mock_proc.communicate = AsyncMock(return_value=(b"", b"error: merge conflict"))
            else:
                # Second pull succeeds — stop
                mock_proc.returncode = 0
                mock_proc.communicate = AsyncMock(return_value=(b"Already up to date.", b""))
                await watcher.stop()
            return mock_proc

        with (
            patch("exo_skills.watchers.github._clone_github", return_value=clone_dir),
            patch(
                "exo_skills.watchers.github.asyncio.create_subprocess_exec",
                side_effect=fake_subprocess,
            ),
        ):
            batches = []
            async for b in watcher.watch():
                batches.append(b)

        # Should have survived the failure and polled again
        assert call_count >= 2
