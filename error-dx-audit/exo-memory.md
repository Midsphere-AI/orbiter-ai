# exo-memory — Error DX & Resilience Audit

## Counts
- raise sites: 17
- error classes total / not inheriting ExoError: 2 total (`ExoMemoryError`, `MigrationError`) / 0 (both inherit ExoError correctly); raw non-ExoError raises: 6 offenders:
  - `backends/sqlite.py:112` — `RuntimeError`
  - `backends/postgres.py:101` — `RuntimeError`
  - `long_term.py:401` — `KeyError`
  - `evolution/ace.py:161` — `ValueError`
  - `encrypted.py:62` — `ValueError`
  - `encrypted.py:123` — `NotImplementedError`
- `except Exception` sites: 8 ; swallow-and-pass: 1 (`encrypted.py:140` — bare `except Exception: continue`, silently drops bad-key items without any logging); drop-cause: 0 (all others chain properly)
- CancelledError handlers: 0 (no explicit `CancelledError` handling anywhere)
- I/O call sites lacking timeout/retry: 12 (all SQLite `aiosqlite` calls, all Postgres `asyncpg` pool calls, all `asyncio.to_thread` Chroma/embedding calls)

---

## Findings (prioritized)

**[P0] | #4 | `encrypted.py:140` | Silent swallow on decrypt failure**
`except Exception: continue` — if a memory item can't be decrypted, it is silently dropped from search results with no log, no hint. A wrong key or data corruption produces an empty result set with no indication of the root cause.
Fix: add `logger.warning("EncryptedMemoryStore: failed to decrypt item id=%s, skipping: %s", item.id, exc)` with `exc_info=False` before `continue`. The bare except already avoids CancelledError so swapping to `except Exception as exc` and logging is the right move.

**[P0] | #1,#2 | `backends/sqlite.py:112` and `backends/postgres.py:101` | `RuntimeError` escapes package boundary**
`_ensure_init()` raises a raw `RuntimeError("Store not initialized — call init() or use 'async with'")` on both backends. The message happens to be readable, but it's not an `ExoError` subclass, so it can't carry `hint=` or `context=` and doesn't satisfy the taxonomy rule.
Fix: `raise ExoMemoryError("Store not initialized.", hint="Call await store.init() or use 'async with SQLiteMemoryStore(path) as store:'")`.

**[P0] | #1 | `long_term.py:401` | `KeyError` escapes package boundary**
`MemoryOrchestrator.process()` raises a raw `KeyError` when a `task_id` is unknown. The raw `KeyError` reveals an internal dict lookup, not a meaningful API error.
Fix: `raise ExoMemoryError(f"No extraction task with id {task_id!r}.", context={"task_id": task_id}, hint="Check the task_id returned by submit().")`.

**[P0] | #1 | `evolution/ace.py:161` | `ValueError` escapes package boundary**
`ACEStrategy.record()` raises raw `ValueError` for an invalid label. This is a public API entry point.
Fix: `raise ExoMemoryError(f"Invalid feedback label {label!r}.", context={"label": label, "valid": sorted(_VALID_LABELS)}, hint=f"Pass one of: {sorted(_VALID_LABELS)}.")`.

**[P0] | #1 | `encrypted.py:62` | `ValueError` escapes package boundary**
`EncryptedMemoryStore.__init__` raises raw `ValueError` for bad key length.
Fix: `raise ExoMemoryError(f"AES key must be exactly 32 bytes (got {len(key)}).", hint="Use derive_key(password) to generate a valid 32-byte key.")`.

**[P1] | #2,#3 | `backends/sqlite.py:150` | Context-free ExoMemoryError on add**
`raise ExoMemoryError(f"add failed for item {item.id}: {exc}")` includes the item ID but not the db_path or backend name, making it hard to distinguish which store failed in a multi-store setup.
Fix: use `raise ExoMemoryError("SQLite add failed.", context={"item_id": item.id, "db_path": self.db_path}, hint="Check disk space and file permissions.") from exc`.

**[P1] | #2,#3 | `backends/sqlite.py:202` | Decontextualised search error**
`raise ExoMemoryError(f"search failed: {exc}")` — no db_path, no query hint.
Fix: `raise ExoMemoryError("SQLite search failed.", context={"db_path": self.db_path}, hint="Check the database file is readable and the schema is current.") from exc`.

**[P1] | #2 | `encrypted.py:101` | ExoMemoryError on get missing hint**
`raise ExoMemoryError(f"Failed to decrypt memory item {item_id}")` has no hint — developer doesn't know whether to check the key, data corruption, or encoding.
Fix: add `hint="Verify the AES key is the same one used when storing this item. Data may also be corrupted."`.

**[P1] | #2,#3 | All Postgres `add/get/search/clear` operations | No error wrapping at all**
`PostgresMemoryStore.add()`, `get()`, `search()`, and `clear()` have no `try/except`. Raw `asyncpg` exceptions (`asyncpg.PostgresError`, `asyncpg.TooManyConnectionsError`) leak directly to callers.
Fix: wrap each method body in `try/except Exception as exc: raise ExoMemoryError("Postgres <op> failed.", context={"dsn": self.dsn}, hint="Check the DSN and ensure the Postgres server is reachable.") from exc`.

**[P1] | #2 | `long_term.py:419-421` | Exception swallowed as task failure string**
`MemoryOrchestrator.process()` catches `Exception`, records `task.fail(str(exc))` and logs it but does NOT re-raise. The caller gets a completed `ExtractionTask` with `status=FAILED` — they have to inspect the task to discover the error. If the extractor raises a `CancelledError`, it's converted to a string in the task and re-raised? No — `CancelledError` is `BaseException`, so `except Exception` skips it, which is correct. But if extraction fails due to a transient error, there is no way for the caller's `await process(...)` call to raise. The error is fully swallowed at the method boundary.
Fix: at minimum, re-raise as `ExoMemoryError` after recording the failure, or at least document the swallow explicitly. Current behavior is a hidden contract break.

**[P1] | #2 | `migrations.py:116` | MigrationError missing hint for unsupported store type**
`raise MigrationError(f"Unsupported store type: {type(store).__name__}")` — no hint on what store types are accepted.
Fix: add `hint="Pass a SQLiteMemoryStore or PostgresMemoryStore instance. Got: {type(store).__name__}."`.

**[P1] | #2 | `short_term.py:42` | ExoMemoryError on bad scope has no hint**
`raise ExoMemoryError(f"Invalid scope {scope!r}, must be 'user', 'session', or 'task'")` — already names the valid options in the message but doesn't use `hint=`. Move the "must be ..." to `hint=` and keep the message terse.

**[P2] | #9 | `long_term.py` | Uses `logging` not `get_logger`**
All files in `exo-memory/` use `import logging` / `logging.getLogger(__name__)` which is correct per CLAUDE.md convention for non-`_internal` packages. No issue here. (Confirmed correct pattern.)

**[P2] | #5 | `migrations.py:59` | `MigrationError` for duplicate version drops cause**
`raise MigrationError(msg)` — no `from exc` needed (no prior exception), but the message could include hint about which versions are already registered.
Fix: `raise MigrationError(f"Duplicate migration version {migration.version}.", context={"version": migration.version}, hint=f"Already registered versions: {[m.version for m in self._migrations]}.")`.

**[P2] | #1 | `encrypted.py:123` | `NotImplementedError` escapes package boundary**
`EncryptedMemoryStore.search()` raises raw `NotImplementedError` when a query is given. The message is already descriptive. Minimal fix: wrap in `ExoMemoryError` with `hint=` so it's caught by the taxonomy rule.

---

## Resilience gaps

| Location | I/O System | Gap |
|---|---|---|
| `backends/sqlite.py:83` — `aiosqlite.connect()` | SQLite | No timeout; if DB file is on a slow/network mount, init hangs forever |
| `backends/sqlite.py:117–151` — `db.execute()` in `add()` | SQLite | No timeout per-operation; aiosqlite uses a background thread; `asyncio.wait_for` would provide a ceiling |
| `backends/sqlite.py:156–163` — `db.execute()` in `get()` | SQLite | Same as above |
| `backends/sqlite.py:198–203` — `db.execute()` in `search()` | SQLite | Same |
| `backends/postgres.py:73` — `asyncpg.create_pool()` | Postgres | No connection timeout; will hang on unreachable DSN |
| `backends/postgres.py:110–134` — `pool.acquire()` + `conn.execute()` in `add()` | Postgres | No query timeout, no retry on transient `asyncpg.TooManyConnectionsError` |
| `backends/postgres.py:137–147` — `get()` | Postgres | Same |
| `backends/postgres.py:183–186` — `search()` | Postgres | Same |
| `backends/vector.py:72` — `self._embeddings.embed(item.content)` in `VectorMemoryStore.add()` | Embedding API (network) | No timeout, no retry; a hanging embedding call blocks the whole add |
| `backends/vector.py:110` — `self._embeddings.embed(query)` in `search()` | Embedding API | Same |
| `backends/vector.py:312–320` — `asyncio.to_thread(collection.add, ...)` in `ChromaVectorMemoryStore.add()` | ChromaDB | No timeout on synchronous Chroma call; thread pool starvation possible |
| `backends/vector.py:357` — `asyncio.to_thread(collection.query, ...)` in `ChromaVectorMemoryStore.search()` | ChromaDB | Same; no retry on Chroma connection errors |

**Additional concern:** `ChromaVectorMemoryStore._ensure_collection()` is called synchronously from async methods. If ChromaDB's `PersistentClient(path=...)` blocks (e.g., acquires a file lock), it runs on the event-loop thread, stalling the entire event loop. This should be moved into `asyncio.to_thread`.

---

## Effort estimate

**M** — ~6 raw non-ExoError raises to fix, ~4 Postgres methods needing `try/except` wrapping, one critical silent swallow in `encrypted.py`, plus 12 I/O sites that each need a timeout comment/wrapper. The migration+orchestrator contract clarifications are minor. No architectural changes needed.
