"""Exo Memory: Pluggable memory backends."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

__version__: str = "0.1.0"

# SQLiteMemoryStore: eager import — aiosqlite is a hard dependency.
from exo.memory.backends.sqlite import (  # pyright: ignore[reportMissingImports]
    SQLiteMemoryStore,
)
from exo.memory.backends.vector import (  # pyright: ignore[reportMissingImports]
    Embeddings,
    OpenAIEmbeddings,
    VectorMemoryStore,
    VertexEmbeddings,
)
from exo.memory.base import (  # pyright: ignore[reportMissingImports]
    AgentMemory,
    AIMemory,
    ExoMemoryError,
    HumanMemory,
    MemoryCategory,
    MemoryItem,
    MemoryMetadata,
    MemoryStatus,
    MemoryStore,
    SystemMemory,
    ToolMemory,
)
from exo.memory.dedup import (  # pyright: ignore[reportMissingImports]
    MemUpdateChecker,
    MergeResult,
    UpdateDecision,
)
from exo.memory.encrypted import (  # pyright: ignore[reportMissingImports]
    EncryptedMemoryStore,
)
from exo.memory.events import (  # pyright: ignore[reportMissingImports]
    MEMORY_ADDED,
    MEMORY_CLEARED,
    MEMORY_SEARCHED,
    MemoryEventEmitter,
)
from exo.memory.evolution import (  # pyright: ignore[reportMissingImports]
    MemoryEvolutionStrategy,
)
from exo.memory.evolution.ace import (  # pyright: ignore[reportMissingImports]
    ACEStrategy,
)
from exo.memory.evolution.reasoning_bank import (  # pyright: ignore[reportMissingImports]
    ReasoningBankStrategy,
)
from exo.memory.evolution.reme import (  # pyright: ignore[reportMissingImports]
    ReMeStrategy,
)
from exo.memory.long_term import (  # pyright: ignore[reportMissingImports]
    ExtractionTask,
    ExtractionType,
    Extractor,
    LongTermMemory,
    MemoryOrchestrator,
    MemoryTaskStatus,
    OrchestratorConfig,
)
from exo.memory.migrations import (  # pyright: ignore[reportMissingImports]
    Migration,
    MigrationRegistry,
)
from exo.memory.persistence import (  # pyright: ignore[reportMissingImports]
    MemoryPersistence,
)
from exo.memory.search import (  # pyright: ignore[reportMissingImports]
    SearchManager,
)
from exo.memory.short_term import (  # pyright: ignore[reportMissingImports]
    ShortTermMemory,
)
from exo.memory.snapshot import (  # pyright: ignore[reportMissingImports]
    SnapshotMemory,
    deserialize_msg_list,
    has_message_content,
    serialize_msg_list,
)
from exo.memory.summary import (  # pyright: ignore[reportMissingImports]
    Summarizer,
    SummaryConfig,
    SummaryResult,
    SummaryTemplate,
    check_trigger,
    generate_summary,
)

# ChromaVectorMemoryStore (needs chromadb) and PostgresMemoryStore (needs asyncpg)
# are NOT imported eagerly to avoid breaking environments where those packages
# are absent.  Access them via this lazy __getattr__, or import directly:
#   from exo.memory.backends.vector import ChromaVectorMemoryStore
#   from exo.memory.backends.postgres import PostgresMemoryStore
_LAZY_BACKENDS: dict[str, tuple[str, str]] = {
    "ChromaVectorMemoryStore": ("exo.memory.backends.vector", "ChromaVectorMemoryStore"),
    "PostgresMemoryStore": ("exo.memory.backends.postgres", "PostgresMemoryStore"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_BACKENDS:
        module_path, attr = _LAZY_BACKENDS[name]
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__: list[str] = [
    "MEMORY_ADDED",
    "MEMORY_CLEARED",
    "MEMORY_SEARCHED",
    "ACEStrategy",
    "AIMemory",
    "AgentMemory",
    "ChromaVectorMemoryStore",
    "Embeddings",
    "EncryptedMemoryStore",
    "ExoMemoryError",
    "ExtractionTask",
    "ExtractionType",
    "Extractor",
    "HumanMemory",
    "LongTermMemory",
    "MemUpdateChecker",
    "MemoryCategory",
    "MemoryEventEmitter",
    "MemoryEvolutionStrategy",
    "MemoryItem",
    "MemoryMetadata",
    "MemoryOrchestrator",
    "MemoryPersistence",
    "MemoryStatus",
    "MemoryStore",
    "MemoryTaskStatus",
    "MergeResult",
    "Migration",
    "MigrationRegistry",
    "OpenAIEmbeddings",
    "OrchestratorConfig",
    "PostgresMemoryStore",
    "ReMeStrategy",
    "ReasoningBankStrategy",
    "SQLiteMemoryStore",
    "SearchManager",
    "ShortTermMemory",
    "SnapshotMemory",
    "Summarizer",
    "SummaryConfig",
    "SummaryResult",
    "SummaryTemplate",
    "SystemMemory",
    "ToolMemory",
    "UpdateDecision",
    "VectorMemoryStore",
    "VertexEmbeddings",
    "check_trigger",
    "deserialize_msg_list",
    "generate_summary",
    "has_message_content",
    "serialize_msg_list",
]
