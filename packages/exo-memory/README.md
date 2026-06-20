# exo-memory

> Pluggable memory backends for short-term conversation context and long-term knowledge extraction.

`exo-memory` gives Exo agents a structured place to store, retrieve, and evolve what they know. It covers the full range from ephemeral in-process conversation buffers to durable Postgres stores with vector search, and it ships the extraction pipeline that turns raw LLM exchanges into searchable long-term memories.

## Installation

```bash
pip install exo-memory
# or
uv add exo-memory

# Optional storage backends
pip install exo-memory[sqlite]     # SQLite persistence
pip install exo-memory[postgres]   # PostgreSQL + pgvector
pip install exo-memory[vector]     # ChromaDB vector search
```

## Quick start

```python
import asyncio
from exo.memory import (
    ShortTermMemory,
    HumanMemory,
    AIMemory,
    MemoryMetadata,
)

async def main() -> None:
    stm = ShortTermMemory(scope="session", max_rounds=20)

    meta = MemoryMetadata(session_id="s-1", user_id="u-1")

    await stm.add(HumanMemory(content="What is Python?", metadata=meta))
    await stm.add(AIMemory(content="A high-level programming language.", metadata=meta))

    results = await stm.search(query="Python", metadata=meta, limit=5)
    for item in results:
        print(item.memory_type, item.content)

asyncio.run(main())
```

## What's inside

- **`ShortTermMemory`** — in-process conversation store with scope-based filtering (`"user"`, `"session"`, `"task"`), configurable round windowing, and tool-call integrity enforcement
- **`LongTermMemory`** / **`MemoryOrchestrator`** — persistent memory layer with LLM-based extraction; `OrchestratorConfig` controls extraction intervals and strategies
- **`MemoryStore`** — `Protocol` that all backends implement; swap backends without changing agent code (`SQLiteMemoryStore`, `PostgresMemoryStore`, `VectorMemoryStore`)
- **`MemoryItem`** — typed item hierarchy: `HumanMemory`, `AIMemory`, `ToolMemory`, `SystemMemory`, `SnapshotMemory`; each carries `MemoryMetadata` for scoping
- **`EncryptedMemoryStore`** — transparent AES-GCM encryption wrapper for any `MemoryStore`
- **`MemoryEvolutionStrategy`** — base class for evolution algorithms; built-in strategies include `ACEStrategy`, `ReMeStrategy`, and `ReasoningBankStrategy`

## Part of [Exo](https://github.com/midsphere-ai/exo)

`exo-memory` plugs into the agent runtime via hooks and `MemoryPersistence`. Get the full framework with `pip install exo-ai`.

---

MIT © Midsphere AI
