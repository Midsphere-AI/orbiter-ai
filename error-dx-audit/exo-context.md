# exo-context — Error DX & Resilience Audit

## Counts
- raise sites: 20
- error classes total / not inheriting ExoError: 7 / 1 (offenders: `state.py:49` — `KeyError`)
- `except Exception` sites: 9 ; swallow-and-pass: 1 (`info.py:135`) ; drop-cause: 0
- CancelledError handlers: 0
- I/O call sites lacking timeout/retry: 5 (all in `workspace.py` — synchronous blocking disk I/O in an async context)

## Findings (prioritized)

| Priority | Checklist# | Location | What's wrong | Concrete fix |
|---|---|---|---|---|
| P0 | #4 | `info.py:135` | `except Exception: pass` silently swallows all errors from `token_tracker.get_trajectory()`, dropping the trajectory entirely with no log entry. Any bug in a custom `TokenTracker` vanishes. | Replace with `except Exception: _log.debug("trajectory unavailable for %r: %s", agent_name, exc, exc_info=True)` (log at DEBUG, not silent). |
| P0 | #8 | `workspace.py:311–317` | Synchronous `write_text` / `mkdir` / `shutil.rmtree` called from async `write()` and `delete()` methods. Blocks the event loop; under load this can stall all async tasks with no timeout or cleanup guard. | Offload to `asyncio.to_thread()` (or `aiofiles`), or document the methods as sync-only and call them from a thread executor. Add `OSError` handling around each I/O call. |
| P0 | #8 | `workspace.py:180–187` | `_notify()` catches `Exception`, logs, then re-raises. Fine in isolation, but the `raise` is bare — if the observer callback raises a `CancelledError`-subclass that is also an `Exception` (Python 3.11 `BaseException` hierarchy caveat), it is re-raised correctly. However, if a `CancelledError` propagates as `BaseException`, it bypasses the `except Exception` entirely and leaks uncaught from the public `write()` / `delete()` entrypoints with no cleanup. The `_persist()` call preceding `_notify()` has already mutated state; partial-commit on cancel. | Wrap the `_persist` + `_notify` pair in a try/finally; in the cancel path the artifact is already in `_artifacts` — ensure consistency or document the invariant. |
| P1 | #2 | `checkpoint.py:94` | `CheckpointError("task_id is required and must be non-empty")` — no `hint=` stating what to pass. Same pattern in `context.py:68` and `workspace.py:144`. | Add `hint="Pass a non-empty string as task_id."` (and equivalents). |
| P1 | #2 | `checkpoint.py:156` | `CheckpointError(f"Checkpoint version {version} not found (available: 1-{len(self._checkpoints)})")` — message is contextual but lacks a `hint=` pointing to `.list_versions()`. | Add `hint="Call .list_versions() to see which versions exist."` |
| P1 | #3 | `checkpoint.py:156` | No `context=` dict — `task_id` and `version` are known at the call site but not attached to the error. | Add `context={"task_id": self._task_id, "requested_version": version}`. |
| P1 | #2 | `context.py:154` | `ContextError(f"Context {child.task_id!r} is not a child of {self.task_id!r}")` — actionable text in the message body but no `hint=`. | Add `hint="Only merge a context returned by this context's .fork() method."` |
| P1 | #3 | `context.py:154` | `context=` not populated — parent and child task IDs are available. | Add `context={"parent": self._task_id, "child": child.task_id}`. |
| P1 | #2 | `context.py:232` | `CheckpointError(f"Expected Checkpoint, got {type(checkpoint).__name__}")` — no `hint=`. | Add `hint="Pass a Checkpoint object returned by .snapshot() or CheckpointStore.get()."` |
| P1 | #2 | `processor.py:59` | `ProcessorError("event must be a non-empty string")` — no `hint=`. | Add `hint="Pass a non-empty event string, e.g. 'pre_llm_call' or 'post_tool_call'."` |
| P1 | #2 | `prompt_builder.py:95–98` | `PromptBuilderError(f"PromptSection {section_name!r} not found in registry")` — good message but no `hint=` pointing to the registry or how to list available sections. | Add `hint=f"Available sections: {list(section_registry.keys())}. Register custom sections via section_registry.register()."` |
| P1 | #3 | `prompt_builder.py:95–98` | No `context=` dict. | Add `context={"section_name": section_name}`. |
| P1 | #2 | `_internal/knowledge.py:66,69` | `KnowledgeError("chunk_size must be positive")` / `KnowledgeError("chunk_overlap must be in [0, chunk_size)")` — no `hint=` with valid ranges. | Add `hint="chunk_size must be >= 1"` and `hint=f"chunk_overlap must be 0 <= chunk_overlap < chunk_size ({chunk_size})"`. |
| P1 | #3 | `_internal/knowledge.py:131` | `KnowledgeError("artifact name is required")` — no `context=` or `hint=`. | Add `hint="Pass a non-empty string as the artifact name."` |
| P1 | #1 | `state.py:49` | `raise KeyError(key)` — raw `KeyError`, not an `ExoError` subclass. This escapes the package boundary as a stdlib exception. | Raise `ContextError(f"Key {key!r} not found in context state", hint="Use .get(key) to avoid raising on missing keys, or check 'key in state' first.")`. |
| P1 | #2 | `variables.py:97,102,110` | `VariableResolveError(f"Variable path {path!r} not found at segment {part!r}")` — actionable path in the message, but no `hint=` on what to do (register a resolver, check state keys). | Add `hint="Register a resolver via DynamicVariableRegistry.register() or ensure the state contains the key path."` |
| P1 | #8 | `workspace.py:311,317` | `write_text()` calls have no `OSError` / `PermissionError` guard — disk-full or permission errors surface as raw `OSError` from inside `async write()`. | Wrap `_persist()` in `try/except OSError as e: raise WorkspaceError(..., hint="Check disk space and storage_path permissions.") from e`. |
| P1 | #8 | `workspace.py:334` | `shutil.rmtree(resolved)` has no error handling — a missing-path race or permission error surfaces as raw `OSError`. | Wrap in `try/except OSError`. |
| P2 | #9 | `info.py:108–109` | Bare `except ImportError: pass` with four dummy `type(None)` tuple entries — if the import fails, all four message counts silently return 0 with no warning. Acceptable for optional imports but should log at DEBUG. | Add `logger.debug("exo.types unavailable in build_context_window_info; message counts will be 0")`. |
| P2 | #2 | `tools.py:67` | `ToolError(f"Tool '{self.name}' requires a bound context...")` — not an `ExoError` subclass (inherits from `exo.tool.ToolError`); no `hint=` or `context=`. | Add `hint="Call tool.bind(ctx) before executing."` and `context={"tool": self.name}`. |

## Resilience gaps

| File:line | I/O system | Gap |
|---|---|---|
| `workspace.py:310–311` | Filesystem `mkdir` + `write_text` | Blocking sync I/O inside `async write()` — stalls event loop; no `OSError` guard for disk-full / permissions. |
| `workspace.py:317` | Filesystem `write_text` (meta.json) | Same: blocking, no error handling. A partial write (content written, meta fails) leaves the artifact directory in an inconsistent state. |
| `workspace.py:334` | Filesystem `shutil.rmtree` | Blocking sync I/O inside `async delete()`; no `OSError` guard for concurrent deletion or permission errors. |
| `workspace.py:300–305` | Path resolution | `Path.resolve()` can stat the filesystem (blocking) and raises `OSError` on broken symlinks — not caught. |
| `tools.py:331` | Filesystem `Path.read_text` | Already has specific `UnicodeDecodeError` / `OSError` guards — **this site is handled**; listed for completeness. |

The `_persist` and `_remove_persisted` methods should either be refactored to use `asyncio.to_thread()` or clearly documented as sync-only (and the async `write`/`delete` callers should be made sync too).

## Effort estimate
**S–M.** The taxonomy, hint, and context gaps are mechanical and plentiful (~18 sites), but all involve very simple edits. The blocking I/O in `workspace.py` is the one structural gap — refactoring `_persist`/`_remove_persisted` to use `asyncio.to_thread` is a few lines but needs test coverage. Total: roughly a half-day of focused work.
