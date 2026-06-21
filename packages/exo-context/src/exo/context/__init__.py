"""Exo Context: hierarchical state, prompt building, processors, workspace."""

__version__: str = "0.1.0"

from exo.context.checkpoint import (  # pyright: ignore[reportMissingImports]
    Checkpoint,
    CheckpointStore,
)
from exo.context.config import (  # pyright: ignore[reportMissingImports]
    ContextConfig,
    OverflowStrategy,
)
from exo.context.context import (  # pyright: ignore[reportMissingImports]
    Context,
    ContextError,
)
from exo.context.hook import (  # pyright: ignore[reportMissingImports]
    ContextWindowHook,
)
from exo.context.info import (  # pyright: ignore[reportMissingImports]
    ContextWindowInfo,
    build_context_window_info,
)
from exo.context.neuron import (  # pyright: ignore[reportMissingImports]
    HistoryNeuron,
    HistorySection,
    Neuron,
    PromptSection,
    SystemNeuron,
    SystemSection,
    TaskNeuron,
    TaskSection,
    neuron_registry,
    section_registry,
)
from exo.context.processor import (  # pyright: ignore[reportMissingImports]
    ContextProcessor,
    DialogueCompressor,
    ProcessorPipeline,
    ToolResultOffloader,
)
from exo.context.prompt_builder import (  # pyright: ignore[reportMissingImports]
    PromptBuilder,
)
from exo.context.state import ContextState  # pyright: ignore[reportMissingImports]
from exo.context.token_tracker import TokenTracker  # pyright: ignore[reportMissingImports]
from exo.context.tools import (  # pyright: ignore[reportMissingImports]
    get_context_tools,
    get_file_tools,
    get_knowledge_tools,
    get_planning_tools,
)
from exo.context.workspace import (  # pyright: ignore[reportMissingImports]
    ArtifactType,
    Workspace,
)

__all__: list[str] = [
    "ArtifactType",
    "Checkpoint",
    "CheckpointStore",
    "Context",
    "ContextConfig",
    "ContextError",
    "ContextProcessor",
    "ContextState",
    "ContextWindowHook",
    "ContextWindowInfo",
    "DialogueCompressor",
    "HistoryNeuron",  # Deprecated alias for HistorySection
    "HistorySection",
    "Neuron",  # Deprecated alias for PromptSection
    "OverflowStrategy",
    "ProcessorPipeline",
    "PromptBuilder",
    "PromptSection",
    "SystemNeuron",  # Deprecated alias for SystemSection
    "SystemSection",
    "TaskNeuron",  # Deprecated alias for TaskSection
    "TaskSection",
    "TokenTracker",
    "ToolResultOffloader",
    "Workspace",
    "build_context_window_info",
    "get_context_tools",
    "get_file_tools",
    "get_knowledge_tools",
    "get_planning_tools",
    "neuron_registry",  # Deprecated alias for section_registry
    "section_registry",
]
