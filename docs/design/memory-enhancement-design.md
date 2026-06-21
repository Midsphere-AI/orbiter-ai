# Memory Enhancement Design — Type Taxonomy, Encryption, Deduplication, Migration

**Status:** Proposed
**Epic:** 5 — Enhanced Memory System
**Date:** 2026-03-10

---

## 1. Motivation

Exo's `exo-memory` package provides a solid foundation:

- **MemoryStore protocol** — async `add`, `get`, `search`, `clear` with pluggable backends.
- **Typed item hierarchy** — `MemoryItem` base with `SystemMemory`, `HumanMemory`,
  `AIMemory`, `ToolMemory` subtypes (discriminated via `memory_type` field).
- **ShortTermMemory** — in-memory conversation context with scope filtering, round
  windowing, and tool call integrity.
- **LongTermMemory** — persistent knowledge store with content deduplication and
  namespace isolation.
- **MemoryOrchestrator** — async LLM-powered extraction (USER_PROFILE, AGENT_EXPERIENCE,
  FACTS) with task queuing.
- **Summary system** — trigger-based LLM summarization with configurable thresholds.
- **Event system** — `MemoryEventEmitter` wraps any store with EventBus integration.
- **Backends** — SQLite (aiosqlite), PostgreSQL (asyncpg), Vector (cosine similarity
  with pluggable embeddings).

However, it lacks several capabilities found in agent-core (`openjiuwen/core/memory/`):

1. **Memory type taxonomy** — agent-core distinguishes USER_PROFILE, SEMANTIC_MEMORY,
   EPISODIC_MEMORY, VARIABLE, and SUMMARY as first-class types. Exo's `memory_type`
   is an untyped string with only conversation-role subtypes (system/human/ai/tool).
2. **Encryption at rest** — agent-core provides AES-256 encryption (nonce + tag +
   ciphertext). Exo has Fernet encryption for API keys but nothing for memory items.
3. **Intelligent deduplication** — agent-core's `MemUpdateChecker` uses LLM-based
   semantic comparison (top-5 similarity @ 0.75 threshold) to decide ADD/DELETE/MERGE.
   Exo's LongTermMemory only checks exact content+type matches.
4. **Schema migration system** — agent-core has pluggable migration registries for SQL,
   vector store, and KV store backends with version tracking. Exo backends create
   tables on init but have no migration path for schema changes.

This document proposes adding four enhancement areas to `exo-memory`, all as
additive changes with zero modifications to existing APIs.

---

## 2. Key Decision: Memory Type Taxonomy

### Option A — Separate MemoryType StrEnum (rejected)

Create a new `MemoryType` enum distinct from the existing `memory_type: str` field.
This creates confusion about which field to use for filtering and breaks existing
code that checks `memory_type == "human"`.

### Option B — Extend memory_type as StrEnum with backward-compatible defaults (chosen)

Add a `MemoryCategory` StrEnum for the knowledge-level taxonomy (user_profile,
semantic, episodic, variable, summary) while keeping the existing `memory_type` string
field unchanged for conversation-role types (system, human, ai, tool).

**Why Option B:**

- Existing `memory_type` serves conversation-role discrimination (system/human/ai/tool)
  and is used everywhere for subtype dispatch in backends. Changing it would break
  serialization and database queries.
- Knowledge-level categories are orthogonal: a `MemoryItem` with `memory_type="ai"`
  might later be extracted as `category="episodic"` by the orchestrator.
- New `category` field with a default of `None` is fully backward-compatible — existing
  items without a category continue to work unchanged.
- Search filters can use both `memory_type` (conversation role) and `category`
  (knowledge classification) independently.

### Design

```python
class MemoryCategory(StrEnum):
    """Knowledge-level classification for memory items."""

    USER_PROFILE = "user_profile"
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    VARIABLE = "variable"
    SUMMARY = "summary"
```

Add to `MemoryItem`:

```python
class MemoryItem(BaseModel):
    # ... existing fields unchanged ...
    category: MemoryCategory | None = None  # New, optional, backward-compatible
```

Add to `MemoryStore.search()`:

```python
async def search(
    self,
    *,
    query: str = "",
    metadata: MemoryMetadata | None = None,
    memory_type: str | None = None,
    category: MemoryCategory | None = None,  # New filter
    status: MemoryStatus | None = None,
    limit: int = 10,
) -> list[MemoryItem]:
```

Default of `None` means existing callers that don't pass `category` see no change.
Backends add `category TEXT` column (nullable) to the schema — existing rows have
`NULL` and are unaffected.

---

## 3. Key Decision: Encryption

### Option A — Built into each backend (rejected)

Each backend (SQLite, Postgres, Vector) handles encryption internally. This
duplicates crypto logic across three implementations and makes it hard to add new
backends without repeating the pattern.

### Option B — Decorator/wrapper around MemoryStore (chosen)

An `EncryptedMemoryStore` wraps any `MemoryStore` implementation and transparently
encrypts `content` on write and decrypts on read. The underlying store sees only
ciphertext.

**Why Option B:**

- Single implementation — crypto logic in one place, tested once.
- Works with any backend (SQLite, Postgres, Vector, future backends).
- Opt-in — users who don't need encryption use the store directly.
- Follows the same decorator pattern as `MemoryEventEmitter`.

### Design

```python
class EncryptedMemoryStore:
    """Wraps a MemoryStore to encrypt content at rest.

    Uses Fernet (AES-128-CBC with HMAC-SHA256) for symmetric encryption.
    Only the `content` field is encrypted — metadata, timestamps, and IDs
    remain in plaintext for filtering and indexing.
    """

    def __init__(self, store: MemoryStore, secret: str) -> None:
        self._store = store
        self._fernet = Fernet(_derive_key(secret))

    async def add(self, item: MemoryItem) -> None:
        encrypted = item.model_copy(
            update={"content": self._fernet.encrypt(item.content.encode()).decode()}
        )
        await self._store.add(encrypted)

    async def get(self, item_id: str) -> MemoryItem | None:
        item = await self._store.get(item_id)
        if item is None:
            return None
        return item.model_copy(
            update={"content": self._fernet.decrypt(item.content.encode()).decode()}
        )

    async def search(self, **kwargs) -> list[MemoryItem]:
        items = await self._store.search(**kwargs)
        return [
            item.model_copy(
                update={"content": self._fernet.decrypt(item.content.encode()).decode()}
            )
            for item in items
        ]

    async def clear(self, **kwargs) -> int:
        return await self._store.clear(**kwargs)
```

**Trade-offs:**

- Keyword search on encrypted content doesn't work (ciphertext is opaque). This is
  acceptable — sensitive data that warrants encryption shouldn't be keyword-searchable.
  Vector search still works if embeddings are computed pre-encryption.
- Fernet uses AES-128-CBC. Agent-core uses AES-256. Fernet is simpler, well-tested,
  and sufficient for at-rest encryption.

---

## 4. Intelligent Deduplication (MemUpdateChecker)

### Overview

When new knowledge memories are added to long-term storage, an LLM-based checker
compares them against existing memories to decide whether to ADD, UPDATE (merge), or
SKIP (duplicate). This prevents memory bloat from repeated information.

### Design

```python
class UpdateAction(StrEnum):
    """Decision from MemUpdateChecker."""
    ADD = "add"
    UPDATE = "update"
    SKIP = "skip"


class UpdateDecision(BaseModel, frozen=True):
    """Result of deduplication check."""
    action: UpdateAction
    target_id: str | None = None  # ID of memory to update (for UPDATE)
    merged_content: str | None = None  # New content after merge (for UPDATE)
    reason: str = ""


class MemUpdateChecker:
    """LLM-based semantic deduplication for long-term memory."""

    def __init__(
        self,
        store: MemoryStore,
        checker: Callable[[str], Awaitable[str]],  # LLM call
        similarity_threshold: float = 0.75,
        top_k: int = 5,
    ) -> None: ...

    async def check(self, item: MemoryItem) -> UpdateDecision:
        """Check if item should be added, merged, or skipped.

        1. Search store for top_k similar items (by content).
        2. If no similar items found, return ADD.
        3. If similar items found, send to LLM with prompt asking
           whether to ADD (distinct info), UPDATE (merge with existing),
           or SKIP (duplicate).
        4. Return UpdateDecision with action and optional merge content.
        """
```

The `checker` callable is LLM-agnostic — the caller provides it (same pattern as
`DialogueCompressor.summarizer` and `MemoryOrchestrator.Extractor`).

### Integration with LongTermMemory

The checker is optional. `LongTermMemory` gains an optional constructor parameter:

```python
class LongTermMemory:
    def __init__(
        self,
        namespace: str = "default",
        update_checker: MemUpdateChecker | None = None,  # New, optional
    ) -> None: ...
```

When `update_checker` is set, `add()` calls `check()` before storing:

- `ADD` → store as normal.
- `UPDATE` → update the target item's content with merged version.
- `SKIP` → silently skip (no error, no storage).

When `update_checker` is `None`, behavior is unchanged (existing exact-match dedup).

---

## 5. Schema Migration System

### Overview

As the memory schema evolves (e.g., adding the `category` column), backends need a
migration path. A simple version-tracked migration system ensures schema changes
apply automatically and idempotently.

### Design

```python
class Migration(BaseModel, frozen=True):
    """A single schema migration step."""
    version: int
    description: str
    up_sql: str  # SQL to apply migration


class MigrationRunner:
    """Tracks and applies schema migrations for memory backends.

    Uses a `_memory_migrations` table to track which migrations have been applied.
    Migrations run automatically on backend initialization.
    """

    def __init__(self, migrations: list[Migration]) -> None: ...

    async def run(self, db) -> None:
        """Apply all pending migrations in order.

        1. Create _memory_migrations table if not exists.
        2. Read current version.
        3. Apply migrations with version > current, in order.
        4. Record each applied migration with timestamp.
        """
```

### Backend Integration

Each backend defines its own migration list:

```python
# In sqlite.py
_SQLITE_MIGRATIONS = [
    Migration(
        version=1,
        description="Initial schema (created on first init)",
        up_sql="",  # No-op — table already created in __init__
    ),
    Migration(
        version=2,
        description="Add category column",
        up_sql="ALTER TABLE memory_items ADD COLUMN category TEXT",
    ),
]
```

The `MigrationRunner` is called during backend `init()` after the initial table
creation. Version 1 is a no-op that records the baseline. Future schema changes
are additional `Migration` entries.

### Scope

This migration system covers SQL backends (SQLite, PostgreSQL) only. The vector
backend stores data in memory (or delegates to external vector DBs) and doesn't
need SQL migrations. If vector store schema changes are needed in the future, a
separate migration strategy can be added.

---

## 6. File Layout

All changes are within `packages/exo-memory/`:

| Addition | Location |
|----------|----------|
| `MemoryCategory` StrEnum | `base.py` (extend existing file) |
| `category` field on `MemoryItem` | `base.py` (extend existing model) |
| `EncryptedMemoryStore` class | `encryption.py` (new file, ~80 lines) |
| `UpdateAction`, `UpdateDecision`, `MemUpdateChecker` | `dedup.py` (new file, ~120 lines) |
| `Migration`, `MigrationRunner` | `migrations.py` (new file, ~80 lines) |
| Migration lists for SQLite | `backends/sqlite.py` (extend existing) |
| Migration lists for PostgreSQL | `backends/postgres.py` (extend existing) |
| `category` filter in `search()` | All backends + `MemoryStore` protocol |
| New exports | `__init__.py` |
| Tests | `tests/test_encryption.py`, `tests/test_dedup.py`, `tests/test_migrations.py` |

Estimated total new code: ~400 lines across 3 new files + backend extensions.

---

## 7. Backward Compatibility

### Existing APIs — no changes required

| Component | Impact |
|-----------|--------|
| `MemoryItem` | No changes — `category` defaults to `None` |
| `MemoryStore` protocol | `category` parameter added with default `None` |
| `ShortTermMemory` | No changes — `category` filter forwarded to `_filter()` |
| `LongTermMemory` | No changes — `update_checker` defaults to `None` |
| `MemoryOrchestrator` | No changes |
| `MemoryEventEmitter` | No changes — wraps store transparently |
| SQLite/Postgres backends | Migration adds nullable column — existing rows unaffected |
| Vector backend | `category` added to in-memory filter — existing items have `None` |
| Summary system | No changes |

### New additions (purely additive)

| Addition | Description |
|----------|-------------|
| `MemoryCategory` | New StrEnum for knowledge classification |
| `EncryptedMemoryStore` | New decorator class, opt-in |
| `MemUpdateChecker` | New dedup checker, opt-in on LongTermMemory |
| `MigrationRunner` | New migration helper, used internally by backends |
| `category` field | New optional field on MemoryItem (default `None`) |

### Existing code paths — unchanged

All existing `await store.add(item)`, `await store.search(memory_type="human")`,
and `await store.clear(metadata=meta)` calls work identically. Items without a
`category` are stored and retrieved without issue. The `EncryptedMemoryStore` and
`MemUpdateChecker` are entirely opt-in wrappers.

---

## 8. Interaction with Existing Components

### MemoryOrchestrator

The orchestrator's `ExtractionType` (USER_PROFILE, AGENT_EXPERIENCE, FACTS) maps
naturally to `MemoryCategory`:

- `ExtractionType.USER_PROFILE` → `MemoryCategory.USER_PROFILE`
- `ExtractionType.AGENT_EXPERIENCE` → `MemoryCategory.EPISODIC`
- `ExtractionType.FACTS` → `MemoryCategory.SEMANTIC`

When the orchestrator stores extracted knowledge in LongTermMemory, it can set the
`category` field accordingly. This is an enhancement to the orchestrator's `process()`
method, not a required change — without it, extracted items simply have `category=None`.

### MemoryEventEmitter

The event emitter wraps any `MemoryStore`. Since `EncryptedMemoryStore` also wraps
a `MemoryStore`, they compose:

```python
# Encryption first, then events
encrypted = EncryptedMemoryStore(sqlite_store, secret="...")
emitter = MemoryEventEmitter(encrypted, bus=bus)
# Events fire with encrypted store, content is encrypted at rest
```

### ShortTermMemory

Short-term memory stores conversation history. It does not need `MemoryCategory`
(conversation messages aren't knowledge items) or encryption (in-memory, ephemeral).
The `category` filter is supported but rarely used for short-term data.

---

## 9. Dependencies

- **cryptography** — `EncryptedMemoryStore` uses `cryptography.fernet.Fernet`. No new
  dependency added (already a transitive dependency).
- No other new dependencies.

---

## 10. Open Questions

1. **Should `EncryptedMemoryStore` encrypt `MemoryMetadata.extra` as well?**
   Recommendation: No — metadata must remain in plaintext for query filtering.
   If users store sensitive data in `extra`, they should encrypt those values
   at the application layer.

2. **Should `MemUpdateChecker` use vector similarity or keyword search for finding
   candidates?** Recommendation: Accept a `MemoryStore` and use its `search()`
   method — if backed by a vector store, similarity is automatic; if backed by
   SQLite, keyword search provides a reasonable fallback.

3. **Should migrations support rollback (`down_sql`)?** Recommendation: Not in v1.
   Forward-only migrations are simpler and sufficient. Rollback can be added later
   if needed.

---

## 11. Test Strategy

| Component | Key test cases |
|-----------|---------------|
| `MemoryCategory` | Enum values, assignment to MemoryItem, None default |
| `category` filter | Search with category, search without (backward compat), backends |
| `EncryptedMemoryStore` | Add+get roundtrip, search decrypts, clear works, invalid key |
| `MemUpdateChecker` | ADD decision, UPDATE with merge, SKIP for duplicate, no similar items |
| `MigrationRunner` | Fresh init, idempotent re-run, version ordering, migration table created |
| Integration | Category + encryption + dedup composed together |
