"""Exo Memory: Pluggable memory backends."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

__version__: str = "0.1.0"

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

__all__: list[str] = [
    "MEMORY_ADDED",
    "MEMORY_CLEARED",
    "MEMORY_SEARCHED",
    "ACEStrategy",
    "AIMemory",
    "AgentMemory",
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
    "ReMeStrategy",
    "ReasoningBankStrategy",
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
