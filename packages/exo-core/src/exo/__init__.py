"""Exo Core: Agent, Tool, Runner, Config, Events, Hooks, Swarm."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
__version__ = "0.1.0"

from exo._internal.agent_group import ParallelGroup
from exo._internal.nested import RalphNode, SwarmNode
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
from exo.tool import FunctionTool, Tool, tool
from exo.tool_context import ToolContext
from exo.tool_result import tool_error, tool_ok
from exo.types import (
    AgentOutput,
    AssistantMessage,
    AudioBlock,
    ContentBlock,
    ContextEvent,
    DocumentBlock,
    ErrorEvent,
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
    "ExoError",
    "FunctionTool",
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
