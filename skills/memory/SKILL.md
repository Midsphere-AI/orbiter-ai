---
name: exo:memory
description: "Use when configuring Exo agent memory — AgentMemory, ShortTermMemory, LongTermMemory, memory backends (SQLiteMemoryStore, VectorMemoryStore, ChromaVectorMemoryStore), MemoryPersistence, conversation_id scoping, embeddings (from exo.models), memory search, summarization. Triggers on: agent memory, ShortTermMemory, LongTermMemory, memory store, SQLite memory, ChromaDB, vector memory, conversation_id, memory persistence, memory search, embeddings."
---

> **Branch:** These skills are written for the `rename/orbiter-to-exo` branch. The Exo APIs referenced here may differ on other branches.

# Exo Memory — Memory System

## When To Use This Skill

Use this skill when the developer needs to:
- Configure short-term and long-term memory for agents
- Choose a memory backend (in-memory, SQLite, ChromaDB/vector)
- Scope memory by conversation, session, or user
- Understand auto-creation defaults and how to override them
- Configure embeddings for semantic search
- Understand the persistence hook system
- Use conversation_id for multi-conversation agents

## Decision Guide

1. **Want default memory (auto)?** → Just create an `Agent(name="bot")` — auto-creates `AgentMemory(ShortTermMemory(), SQLiteMemoryStore())` if exo-memory is installed
2. **Want no memory?** → `Agent(name="bot", memory=None)`
3. **Want keyword search instead of vector?** → `Agent(memory=AgentMemory(short_term=ShortTermMemory(), long_term=SQLiteMemoryStore()))`
4. **Want in-process semantic search?** → `VectorMemoryStore(OpenAIEmbeddings(api_key=..., dimension=1536))`
5. **Want persistent vector search?** → `ChromaVectorMemoryStore(OpenAIEmbeddingProvider(), path="/path/to/db")`
6. **Need multi-conversation support?** → Use `conversation_id` parameter on `agent.run()`
7. **Need custom scoping?** → Configure `ShortTermMemory(scope="session")` or `scope="user"`

## Reference

### AgentMemory

Composite memory bundling short-term and long-term stores:

```python
from exo.memory.base import AgentMemory
from exo.memory.short_term import ShortTermMemory
from exo.memory.backends.sqlite import SQLiteMemoryStore
from exo.memory.backends.vector import VectorMemoryStore, ChromaVectorMemoryStore, OpenAIEmbeddingProvider
from exo.models import OpenAIEmbeddings

# Default setup (SQLite keyword search)
memory = AgentMemory(
    short_term=ShortTermMemory(),
    long_term=SQLiteMemoryStore(db_path="~/.exo/memory.db"),
)

# Semantic search (in-process, no external DB)
memory = AgentMemory(
    short_term=ShortTermMemory(),
    long_term=VectorMemoryStore(OpenAIEmbeddings(api_key="...", dimension=1536)),
)

# Persistent semantic search via ChromaDB
memory = AgentMemory(
    short_term=ShortTermMemory(),
    long_term=ChromaVectorMemoryStore(OpenAIEmbeddingProvider()),  # requires chromadb
)

agent = Agent(name="bot", memory=memory)
```

### Auto-Creation Default

When `memory` is not passed to Agent (or is `_MEMORY_UNSET`):

1. Creates `AgentMemory(ShortTermMemory(), SQLiteMemoryStore())` — SQLite is the default long-term store
2. Returns `None` if `exo-memory` is not installed at all

```python
# These are equivalent:
agent = Agent(name="bot")  # auto-creates memory with SQLiteMemoryStore
agent = Agent(name="bot", memory=None)  # explicitly disables
```

### ShortTermMemory

In-memory conversation store with scope-based filtering:

```python
from exo.memory.short_term import ShortTermMemory

stm = ShortTermMemory(
    scope="task",        # "task" (default), "session", or "user"
    max_rounds=0,        # Max conversation rounds to keep (0 = unlimited)
)
```

**Scope filtering** (determines which metadata fields are used for filtering):
- `"task"` — Filters by `user_id` + `session_id` + `task_id` (most specific)
- `"session"` — Filters by `user_id` + `session_id`
- `"user"` — Filters by `user_id` only (broadest)

**Windowing:** When `max_rounds > 0`, keeps only the last N conversation rounds (human messages mark round boundaries). System messages before the cutoff are always preserved.

**Tool-call integrity:** Automatically removes dangling AI messages with unmatched tool_calls and orphaned tool results.

### MemoryStore Protocol

All memory backends implement this protocol:

```python
from exo.memory.base import MemoryStore, MemoryItem, MemoryMetadata

class MemoryStore(Protocol):
    async def add(self, item: MemoryItem) -> None: ...
    async def get(self, item_id: str) -> MemoryItem | None: ...
    async def search(
        self,
        *,
        query: str = "",
        metadata: MemoryMetadata | None = None,
        memory_type: str | None = None,
        category: MemoryCategory | None = None,
        status: MemoryStatus | None = None,
        limit: int = 10,
    ) -> list[MemoryItem]: ...
    async def clear(self, *, metadata: MemoryMetadata | None = None) -> int: ...
```

### Memory Item Types

```python
from exo.memory.base import MemoryItem, HumanMemory, AIMemory, ToolMemory, SystemMemory

# Base class fields (all items):
# id: str (auto UUID)
# content: str
# memory_type: str
# category: MemoryCategory | None
# status: MemoryStatus (DRAFT → ACCEPTED → DISCARD)
# metadata: MemoryMetadata (user_id, session_id, task_id, agent_id, extra)
# created_at: str (ISO timestamp)
# updated_at: str (ISO timestamp)

# Subtypes:
HumanMemory(content="User said this")           # memory_type = "human"
AIMemory(content="Assistant said this",          # memory_type = "ai"
         tool_calls=[{"id": "tc1", "name": "search", "arguments": "..."}])
ToolMemory(content="Tool result",                # memory_type = "tool"
           tool_call_id="tc1",
           tool_name="search",
           is_error=False)
SystemMemory(content="System message")           # memory_type = "system"

# Snapshot (context snapshot persistence):
from exo.memory.snapshot import SnapshotMemory
SnapshotMemory(                                  # memory_type = "snapshot"
    content="[serialized msg_list JSON]",
    snapshot_version=1,
    raw_item_count=42,
    latest_raw_id="item-99",
    latest_raw_created_at="2026-01-01T00:00:00",
    config_hash="abc123",
)
```

### MemoryMetadata

Scoping metadata attached to every memory item:

```python
from exo.memory.base import MemoryMetadata

meta = MemoryMetadata(
    user_id="user-123",
    session_id="sess-456",
    task_id="conv-789",       # Maps to conversation_id in Agent.run()
    agent_id="my-agent",      # Maps to agent.name
    extra={"custom": "data"},
)
```

### Backend: SQLiteMemoryStore

Persistent SQLite-backed store with keyword search:

```python
from exo.memory.backends.sqlite import SQLiteMemoryStore

store = SQLiteMemoryStore(db_path="/path/to/memory.db")
# db_path defaults to EXO_MEMORY_PATH env var, then ~/.exo/memory.db

await store.init()  # Opens DB, creates tables if needed

# Use as long-term store:
memory = AgentMemory(short_term=ShortTermMemory(), long_term=store)

# Cleanup:
await store.close()
```

**Supports:** `async with` context manager for auto-init/close.

### Backend: ChromaVectorMemoryStore

Persistent ChromaDB-backed store with semantic search:

```python
from exo.memory.backends.vector import ChromaVectorMemoryStore, OpenAIEmbeddingProvider

store = ChromaVectorMemoryStore(
    embedding_provider=OpenAIEmbeddingProvider(
        model="text-embedding-3-small",    # Default
        api_key=None,                       # Uses OPENAI_API_KEY env var
    ),
    collection_name="exo_memory",           # ChromaDB collection name
    path="/path/to/chroma/db",              # Persistent path (None = ephemeral)
)

# Semantic search:
results = await store.search(query="What did we discuss about AI safety?", limit=5)
```

**When `path` is None:** Uses `chromadb.EphemeralClient()` (in-memory, lost on restart).
**When `path` is set:** Uses `chromadb.PersistentClient(path=...)` (survives restarts).

### Backend: VectorMemoryStore (In-Memory)

Lightweight in-memory vector store (no persistence) backed by the canonical
:class:`exo.models.Embeddings` interface.  Uses ``httpx`` — no SDK required:

```python
from exo.memory.backends.vector import VectorMemoryStore
from exo.models import OpenAIEmbeddings, VertexEmbeddings

# OpenAI
embeddings = OpenAIEmbeddings(
    api_key="sk-...",              # required
    model="text-embedding-3-small",
    dimension=1536,
)
store = VectorMemoryStore(embeddings)

# Vertex AI
embeddings = VertexEmbeddings(
    api_key="<bearer-token>",      # required
    project="my-gcp-project",
    model="text-embedding-005",
    dimension=768,
    location="us-central1",
)
store = VectorMemoryStore(embeddings)
```

### Embedding Providers

Canonical embedding types come from ``exo.models``.  They are also
re-exported from ``exo.memory.backends.vector`` for convenience.

| Provider | Import | Default Model | Dimension |
|----------|--------|---------------|-----------|
| OpenAI (canonical) | `from exo.models import OpenAIEmbeddings` | `text-embedding-3-small` | 1536 |
| Vertex AI (canonical) | `from exo.models import VertexEmbeddings` | `text-embedding-005` | 768 |
| HTTP (generic) | `from exo.models import HTTPEmbeddings` | — | configurable |
| OpenAI legacy protocol | `from exo.memory.backends.vector import OpenAIEmbeddingProvider` | `text-embedding-3-small` | — |

```python
# Canonical (for VectorMemoryStore) — uses httpx, no SDK dep
from exo.models import OpenAIEmbeddings, VertexEmbeddings
embeddings = OpenAIEmbeddings(api_key="sk-...")

# Legacy protocol (for ChromaVectorMemoryStore) — uses openai SDK
from exo.memory.backends.vector import OpenAIEmbeddingProvider
provider = OpenAIEmbeddingProvider(model="text-embedding-3-small")
```

### Conversation ID Scoping

Each `agent.run()` call can be scoped to a conversation:

```python
# Conversation 1
r1 = await run(agent, "Hello!", provider=provider, conversation_id="conv-1")
r2 = await run(agent, "Follow up", provider=provider, conversation_id="conv-1")
# Both share history in conv-1

# Conversation 2 (separate history)
r3 = await run(agent, "Different topic", provider=provider, conversation_id="conv-2")
# Separate history from conv-1
```

**How it works internally:**
1. `conversation_id` maps to `MemoryMetadata.task_id`
2. If not provided, uses `agent.conversation_id` (auto-generated UUID on first run)
3. History loaded via `persistence.load_history(agent_name, conversation_id, rounds)`
4. All saved items (HumanMemory, AIMemory, ToolMemory) get `metadata.task_id = conversation_id`

### Memory Persistence (Auto-Attached Hooks)

When `memory` is provided, the agent auto-attaches `MemoryPersistence` hooks:

```python
# Internally (you don't need to do this manually):
persistence = MemoryPersistence(memory.short_term)
persistence.attach(agent)

# Hooks registered:
# POST_LLM_CALL → saves AIMemory (with tool_calls)
# POST_TOOL_CALL → saves ToolMemory (with tool_name, result, is_error)
```

**What gets saved automatically:**
- Every LLM response → `AIMemory` with content and tool_calls
- Every tool result → `ToolMemory` with tool_name, result, error status
- User input → `HumanMemory` (saved at run start)

### Context Snapshot Persistence

`MemoryPersistence` also handles context snapshot save/load when `ContextConfig.enable_snapshots=True`:

```python
# Save snapshot (called automatically at end of agent.run/run.stream)
await persistence.save_snapshot(
    agent_name="bot",
    conversation_id="conv-1",
    msg_list=processed_messages,
    context_config=ctx_config,
)

# Load snapshot
snap = await persistence.load_snapshot("bot", "conv-1")
if snap is not None:
    # Check freshness (stale if raw items are newer or config changed)
    if await persistence.is_snapshot_fresh(snap, "bot", "conv-1", context_config=cfg):
        messages = deserialize_msg_list(snap.content)

# Clear snapshot (force rebuild from raw)
await agent.clear_snapshot()  # or clear_snapshot(conversation_id="conv-1")
```

**Serialization helpers** (in `exo.memory.snapshot`):

```python
from exo.memory.snapshot import serialize_msg_list, deserialize_msg_list, has_message_content

# Serialize msg_list to JSON (excludes instruction SystemMessages)
json_str = serialize_msg_list(msg_list)

# Deserialize back to Message objects
messages = deserialize_msg_list(json_str)

# Check if a marker exists (for idempotent hook injection)
if not has_message_content(messages, "[MY_MARKER]"):
    messages.insert(1, UserMessage(content="[MY_MARKER] injected context"))
```

### Long-Term Memory Search (Knowledge Injection)

Long-term memory is automatically searched before each LLM call:

```python
# Internally (automatic when agent.memory is set):
items = await agent.memory.long_term.search(query=user_input, limit=5)
# Results injected into system message as:
# <knowledge>
#   [long_term_memory]: relevant fact 1
#   [long_term_memory]: relevant fact 2
# </knowledge>
```

### Summarization

For context management (works with memory):

```python
from exo.memory.summary import SummaryConfig, check_trigger, generate_summary

config = SummaryConfig(
    message_threshold=20,        # Trigger when messages exceed this
    token_threshold=4000,        # Trigger when tokens exceed this
    keep_recent=4,               # Items to preserve after compression
)

# Check if summarization should trigger
result = check_trigger(memory_items, config)
if result.triggered:
    summary = await generate_summary(memory_items, config, summarizer)
    # summary.summaries: dict[template_name, summary_text]
    # summary.compressed_items: list of kept items
```

## Patterns

### Stateless Agent (No Memory)

```python
agent = Agent(
    name="calculator",
    memory=None,
    context=None,
    tools=[calculate],
    max_steps=1,
)
```

### Multi-Session Chat Agent

```python
from exo.memory.base import AgentMemory
from exo.memory.short_term import ShortTermMemory
from exo.memory.backends.sqlite import SQLiteMemoryStore
from exo.memory.backends.vector import VectorMemoryStore, ChromaVectorMemoryStore, OpenAIEmbeddingProvider
from exo.models import OpenAIEmbeddings

# Option A: SQLite (keyword search, persistent, no external deps)
memory = AgentMemory(
    short_term=ShortTermMemory(scope="session", max_rounds=50),
    long_term=SQLiteMemoryStore(db_path="./data/memory.db"),
)

# Option B: In-process semantic search (no persistence)
memory = AgentMemory(
    short_term=ShortTermMemory(scope="session", max_rounds=50),
    long_term=VectorMemoryStore(OpenAIEmbeddings(api_key="sk-...")),
)

# Option C: ChromaDB persistent semantic search (requires chromadb)
memory = AgentMemory(
    short_term=ShortTermMemory(scope="session", max_rounds=50),
    long_term=ChromaVectorMemoryStore(
        OpenAIEmbeddingProvider(),
        path="./data/memory",  # Persistent
    ),
)

agent = Agent(name="assistant", memory=memory)

# Session 1
await run(agent, "My name is Alice", provider=p, conversation_id="sess-1")
await run(agent, "What's my name?", provider=p, conversation_id="sess-1")
# Agent knows: "Alice" (from short-term)

# Session 2 — fresh short-term, but long-term has knowledge
await run(agent, "What do you remember?", provider=p, conversation_id="sess-2")
# Agent can search long-term for facts from session 1
```

### Custom Memory Backend

```python
from exo.memory.base import MemoryStore, MemoryItem, MemoryMetadata

class RedisMemoryStore:
    """Custom Redis-backed memory store."""

    async def add(self, item: MemoryItem) -> None:
        await self.redis.set(f"mem:{item.id}", item.model_dump_json())

    async def get(self, item_id: str) -> MemoryItem | None:
        data = await self.redis.get(f"mem:{item_id}")
        return MemoryItem.model_validate_json(data) if data else None

    async def search(self, *, query: str = "", limit: int = 10, **_) -> list[MemoryItem]:
        # Implement search logic
        ...

    async def clear(self, *, metadata: MemoryMetadata | None = None) -> int:
        # Implement clear logic
        ...

agent = Agent(name="bot", memory=AgentMemory(
    short_term=ShortTermMemory(),
    long_term=RedisMemoryStore(),
))
```

## Gotchas

- **Auto-creation requires exo-memory installed** — if not installed, `memory` defaults to `None` silently
- **ChromaDB requires chromadb package** — falls back to SQLiteMemoryStore with a warning if not installed
- **`conversation_id` maps to `metadata.task_id`** — this is an internal mapping, not obvious from the API
- **Auto-generated conversation_id** — if `conversation_id` is not passed and `agent.conversation_id` is None, a UUID4 is auto-generated and stored on the agent
- **Persistence hooks use `short_term`** — `MemoryPersistence` is attached to `memory.short_term`, not `memory.long_term`
- **Long-term search is automatic** — happens before each LLM call when `agent.memory.long_term` exists
- **`memory=None` propagates to spawn_self children** — if parent has no memory, children don't either
- **ShortTermMemory is in-memory only** — data is lost when the process exits. Use a persistent backend for long-term.
- **OpenAIEmbeddingProvider uses OPENAI_API_KEY env var** — ensure it's set for vector search
- **Search results injected as `<knowledge>` blocks** — the LLM sees them in the system message, not as separate messages
- **Context snapshots use the same MemoryStore** — `SnapshotMemory` items are stored alongside regular items with `memory_type="snapshot"`. They use deterministic IDs (`snapshot_{agent}_{conversation}`) so upsert replaces the previous.
- **`load_history()` excludes snapshots** — raw history loading filters by `memory_type` so snapshots never leak into the regular history path
- **`inject_ephemeral()` messages never reach memory** — ephemeral messages are removed from msg_list immediately after the LLM call, before MemoryPersistence hooks (POST_LLM_CALL) fire on subsequent calls and before snapshots are saved. They are invisible to the memory system. Use `inject_message()` for content that should persist.
