# exo-skills — Error DX & Resilience Audit

> Source files audited:
> - `packages/exo-core/src/exo/skills.py` (395 LOC — SkillRegistry, parsing, clone, sync)
> - `packages/exo-skills/src/exo_skills/watchers/github.py` (227 LOC)
> - `packages/exo-skills/src/exo_skills/watchers/local.py` (171 LOC)
> - `packages/exo-skills/src/exo_skills/__init__.py` (13 LOC)

---

## Counts

- **raise sites:** 6 (skills.py:127 [github.py], skills.py:183, skills.py:261, skills.py:277, skills.py:286; github.py:127)
- **error classes total / not inheriting ExoError:** 1 / 0 — `SkillError(ExoError)` at skills.py:48; no offenders
- **`except Exception` sites:** 7 ; **swallow-and-pass:** 0 ; **drop-cause:** 3 (skills.py:148 drops JSONDecodeError; skills.py:454, 465 log-and-continue without chaining; skills.py:525, 548, 564, 585 log-and-continue without chaining)
- **CancelledError handlers:** 2 — both correct: skills.py:463 (`pass` after `CancelledError` on cancelled task in `stop()`), skills.py:486–487 (`raise` in `_run_watcher`)
- **I/O call sites lacking timeout/retry:** 3 — `subprocess.run` git clone (skills.py:174–178), `proc.communicate()` git pull (github.py:184), `run_in_executor(_clone_github)` (github.py:146–148)

---

## Findings (prioritized)

### P0 — Silent skill drop / cause loss

**[P0] | #4 #5 | skills.py:194–209 | `_collect_skills` silently drops unreadable skill files**
`skill_file.read_text(encoding="utf-8")` is called with no try/except. A `PermissionError`, `UnicodeDecodeError`, or any other I/O error propagates raw and aborts the entire scan — silently skipping all subsequent files. The developer sees a raw OS traceback with no indication which file caused it or what to do.
Fix: wrap per-file processing in `try/except Exception as exc`, log a warning with the file path, and `continue` to allow loading the rest of the skills. Include the file path in the message so the developer knows exactly which file to fix.

**[P0] | #5 | skills.py:174–178, github.py:146–148 | Raw `CalledProcessError`/`FileNotFoundError` from git clone escapes package boundary**
`_clone_github` calls `subprocess.run(..., check=True)` with no try/except. If `git` is not on PATH or the repo/branch does not exist (404), the raw `subprocess.CalledProcessError` or `FileNotFoundError` propagates to the caller (`load_all`, `GitHubPollingWatcher.watch`) with no wrapping into `SkillError`, no hint, and no indication which URL caused the failure.
Fix: wrap `subprocess.run` in `try/except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc`, raise `SkillError(..., context={"url": url, "branch": branch}, hint="Ensure git is installed and the repository URL and branch exist.") from exc`.

**[P0] | #5 | skills.py:148–149 | `tool_list` JSON parse error silently replaced with `{}`**
`json.JSONDecodeError` and `TypeError` are caught and the field is silently replaced with `{}`. A malformed `tool_list` value in a skill file is dropped with zero logging or warning — the developer has no idea the data was discarded.
Fix: log a `logger.warning("Malformed tool_list in %s — expected JSON object, got %r: %s", skill_file, val, exc)` and optionally raise a `SkillError` for strict parsing mode. At minimum, emit the warning so the failure is visible.

---

### P1 — Unactionable / missing context

**[P1] | #2 #3 | github.py:127 | `SkillError("Invalid GitHub URL: ...")` has no hint**
`raise SkillError(f"Invalid GitHub URL: {source_url}")` names the bad value but doesn't tell the developer what a valid URL looks like.
Fix:
```python
raise SkillError(
    f"Invalid GitHub URL: {source_url!r}",
    hint="Expected format: https://github.com/<owner>/<repo>/tree/<branch>[/<subdir>]",
    context={"source_url": source_url},
)
```

**[P1] | #2 #3 | skills.py:183 | Subdirectory not found has no hint**
`raise SkillError(f"Subdirectory '{subdir}' not found in {owner}/{repo}@{branch}")` is contextual but has no actionable hint.
Fix: add `hint=f"Check that '{subdir}' exists in the {branch} branch of {owner}/{repo}."` and `context={"owner": owner, "repo": repo, "branch": branch, "subdir": subdir}`.

**[P1] | #2 #3 | skills.py:261 | Source directory not found has no hint**
`raise SkillError(f"Skill source directory not found: {root}")` — no hint about whether the user passed a path that needs to be created or if the URL is simply wrong.
Fix: `hint=f"Create the directory or pass a valid GitHub URL (https://github.com/...) instead of a local path."`, `context={"path": str(root)}`.

**[P1] | #2 #3 | skills.py:277 | Duplicate skill conflict error doesn't name the conflicting sources**
`raise SkillError(f"Duplicate skill '{name}' (conflict_strategy=raise)")` names the skill but not which two sources define it. The developer must guess.
Fix: thread source path into `_merge` and include `context={"skill": name, "existing_path": existing.path, "new_path": skill.path}`.

**[P1] | #2 #3 | skills.py:286 | `SkillRegistry.get` missing hint**
`raise SkillError(f"Skill '{name}' not found")` — no hint about which skills exist or how to discover them.
Fix: `hint=f"Call registry.list_names() to see available skills. Loaded: {list(self._skills)[:5]}"`, `context={"name": name}`.

**[P1] | #5 | skills.py:454, 465 | `stop()` except blocks drop cause**
`logger.warning("Error stopping watcher ...", exc_info=True)` and `logger.warning("Error awaiting watcher task", exc_info=True)` log the exception, but neither names the watcher type/source in the "awaiting task" message. Minor context gap.
Fix: pass `watcher` or `task` to the second log message to identify which watcher's task failed.

**[P1] | #5 | skills.py:525, 548, 564, 585 | Tool add/remove failures drop cause**
`logger.warning("Failed to add/remove tool '%s' to/from agent '%s'", ...)` with `exc_info=True` — the warning logs the exception, but these paths swallow failures without chaining. This is acceptable (resilience over strictness), but the skill name driving the tool operation is missing from the log line. Add `extra={"skill": event.skill_name}` or include it in the message.
Fix: add `, skill=%r` to each format string: `"Failed to add tool '%s' to agent '%s' (skill '%s')", t.name, agent.name, event.skill_name`.

---

### P2 — Polish

**[P2] | #9 | github.py:143 | `asyncio.get_event_loop()` is deprecated**
`loop = asyncio.get_event_loop()` should be `asyncio.get_running_loop()` in async context.
Fix: `loop = asyncio.get_running_loop()`.

**[P2] | #2 | skills.py:139–153 | Front-matter lines with no colon are silently skipped**
`if colon < 0: continue` in `extract_front_matter` silently drops malformed front-matter lines (e.g. a continuation line that looks like a field). No warning is emitted.
Fix: emit a `logger.debug("Skipping unparseable front-matter line in %s: %r", ...)` if a file path is available. (The function currently takes only text, not a path — consider adding an optional `source` param for the log.)

**[P2] | #8 | skills.py:199 | `_collect_skills` falls back silently to directory name when `name` is missing**
`name = meta.get("name") or skill_file.parent.name` — no warning when the `name` field is absent from front-matter. A skill author may not realise their `name:` line was malformed.
Fix: add `if not meta.get("name"): logger.warning("Skill file %s has no 'name' front-matter field; using directory name %r", skill_file, skill_file.parent.name)`.

---

## Resilience gaps

| Site | File:Line | Issue |
|---|---|---|
| `subprocess.run` git clone | skills.py:174–178 | No timeout. A stalled network clone hangs the entire thread-executor slot. Add `timeout=60` to `subprocess.run`. |
| `run_in_executor(_clone_github)` | github.py:146–148 | No `asyncio.wait_for` wrapping the executor call. The async side has no cancellation-friendly timeout for the blocking clone. Wrap with `asyncio.wait_for(..., timeout=120)`. |
| `proc.communicate()` git pull | github.py:184 | No timeout on `communicate()`. A hung git process blocks the watcher poll loop. Use `asyncio.wait_for(proc.communicate(), timeout=60)` and kill the process on `TimeoutError`. |
| `_collect_skills` file read | skills.py:197 | No per-file error isolation. An unreadable file aborts the entire scan. Wrap in `try/except` per file. |

---

## Effort estimate

**M** — The core fixes (wrapping `subprocess.run` in `SkillError`, adding timeout to `proc.communicate()`, isolating per-file errors in `_collect_skills`, and adding `hint=` / `context=` to the 5 existing `SkillError` raises) are mechanical and self-contained; the test suite is small; total estimated work is ~2–3 hours.
