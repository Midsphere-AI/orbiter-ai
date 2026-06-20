# exo-skills

> Skill source watchers for Exo hot-reloading.

`exo-skills` supplies the watcher backends that keep Exo's skill registry in sync with external sources at runtime. Drop skill markdown files into a local directory or a GitHub repository and the matching watcher streams `SkillChangeEvent` batches as files are added, modified, or removed — with no restart required.

## Installation

```bash
pip install exo-skills
# or
uv add exo-skills
```

## Quick start

```python
from exo_skills import LocalFileWatcher, GitHubPollingWatcher

# Watch a local directory for skill file changes
watcher = LocalFileWatcher("./skills", debounce_ms=500)
async for batch in watcher.watch():
    for event in batch:
        print(event.kind, event.skill_name)  # "added" | "modified" | "removed"
    # call watcher.stop() when done

# Watch a GitHub repository, polling every 60 seconds
watcher = GitHubPollingWatcher(
    "https://github.com/acme/skills/tree/main/agents",
    poll_interval=60.0,
)
async for batch in watcher.watch():
    for event in batch:
        print(event.kind, event.skill_name, event.skill)
```

## What's inside

- **`LocalFileWatcher`** — watches a local directory with `watchfiles`; debounces filesystem events and diffs skill snapshots before yielding change batches
- **`GitHubPollingWatcher`** — shallow-clones a GitHub repo on first use, then runs `git pull --ff-only` on a configurable interval and yields only changed skills

## Part of [Exo](https://github.com/midsphere-ai/exo)

Get the full framework with `pip install exo-ai`.

---

MIT © Midsphere AI
