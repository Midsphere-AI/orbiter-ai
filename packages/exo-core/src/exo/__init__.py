"""Exo Core: Agent, Tool, Runner, Config, Events, Hooks, Swarm."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
__version__ = "0.1.0"

from exo._internal.agent_group import ParallelGroup
from exo._internal.nested import RalphNode, RefinementNode, SwarmNode
from exo._internal.workflow_checkpoint import WorkflowCheckpoint, WorkflowCheckpointStore
from exo.agent import Agent, AgentError
from exo.hooks import HookManager, HookPoint
from exo.human import ConsoleHandler, HumanInputHandler
from exo.observability.logging import (  # pyright: ignore[reportMissingImports]
    configure_logging as configure,
)
from exo.observability.logging import (  # pyright: ignore[reportMissingImports]
    get_logger,
)
from exo.parallel import (
    SubAgentError,
    SubAgentResult,
    SubAgentStatus,
    SubAgentTask,
    run_parallel,
    stream_parallel,
)
from exo.runner import run
from exo.swarm import Swarm
from exo.token_counter import TokenCounter, count_tokens

# FunctionTool is intentionally not re-exported at the top level: use the
# @tool decorator for functions, or the Tool ABC for custom tools. It remains
# importable from exo.tool for advanced/back-compat use.
from exo.tool import Tool, tool
from exo.tool_context import ToolContext
from exo.tool_result import tool_error, tool_ok
from exo.types import (
    EVENT_TYPES,
    AgentOutput,
    AssistantMessage,
    AudioBlock,
    ContentBlock,
    ContextEvent,
    DocumentBlock,
    ErrorEvent,
    EventType,
    ExoError,
    HITLApprovalEvent,
    ImageDataBlock,
    ImageURLBlock,
    MCPProgressEvent,
    Message,
    MessageContent,
    MessageInjectedEvent,
    ReasoningEvent,
    RunResult,
    StatusEvent,
    StepEvent,
    StreamEvent,
    SystemMessage,
    TextBlock,
    TextEvent,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEvent,
    ToolResult,
    ToolResultEvent,
    Usage,
    UsageEvent,
    UserMessage,
    VideoBlock,
)

__all__: list[str] = [
    "EVENT_TYPES",
    "Agent",
    "AgentError",
    "AgentOutput",
    "AssistantMessage",
    "AudioBlock",
    "ConsoleHandler",
    "ContentBlock",
    "ContextEvent",
    "DocumentBlock",
    "ErrorEvent",
    "EventType",
    "ExoError",
    "HITLApprovalEvent",
    "HookManager",
    "HookPoint",
    "HumanInputHandler",
    "ImageDataBlock",
    "ImageURLBlock",
    "MCPProgressEvent",
    "Message",
    "MessageContent",
    "MessageInjectedEvent",
    "ParallelGroup",
    "RalphNode",
    "ReasoningEvent",
    "RefinementNode",
    "RunResult",
    "StatusEvent",
    "StepEvent",
    "StreamEvent",
    "SubAgentError",
    "SubAgentResult",
    "SubAgentStatus",
    "SubAgentTask",
    "Swarm",
    "SwarmNode",
    "SystemMessage",
    "TextBlock",
    "TextEvent",
    "TokenCounter",
    "Tool",
    "ToolCall",
    "ToolCallDeltaEvent",
    "ToolCallEvent",
    "ToolContext",
    "ToolResult",
    "ToolResultEvent",
    "Usage",
    "UsageEvent",
    "UserMessage",
    "VideoBlock",
    "WorkflowCheckpoint",
    "WorkflowCheckpointStore",
    "configure",
    "count_tokens",
    "get_logger",
    "run",
    "run_parallel",
    "stream_parallel",
    "tool",
    "tool_error",
    "tool_ok",
]
