"""A2A protocol types — agent cards, configs, and task events."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Transport & capabilities
# ---------------------------------------------------------------------------


class TransportMode(StrEnum):
    """Supported A2A transport protocols.

    Only ``JSONRPC`` is implemented; the enum exists to allow future transports
    to be added without a breaking API change.
    """

    JSONRPC = "jsonrpc"


class TaskState(StrEnum):
    """Lifecycle states for a remote A2A task."""

    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


# ---------------------------------------------------------------------------
# Agent card & skills
# ---------------------------------------------------------------------------


class AgentSkill(BaseModel):
    """A single capability advertised by an agent."""

    model_config = {"frozen": True}

    id: str = Field(description="Unique skill identifier")
    name: str = Field(description="Human-readable name")
    description: str = Field(default="", description="What the skill does")
    tags: tuple[str, ...] = Field(default=(), description="Classification tags")

    @model_validator(mode="before")
    @classmethod
    def _coerce_tags(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("tags"), list):
            data = {**data, "tags": tuple(data["tags"])}
        return data


class AgentCapabilities(BaseModel):
    """Runtime capabilities of an A2A agent."""

    model_config = {"frozen": True}

    streaming: bool = Field(default=False, description="Supports streaming responses")
    state_transition_history: bool = Field(default=False, description="Tracks state transitions")


class AgentCard(BaseModel):
    """Complete metadata descriptor for a remote A2A agent.

    Published at ``/.well-known/agent-card`` for discovery.
    """

    model_config = {"frozen": True}

    name: str = Field(description="Agent identifier")
    description: str = Field(default="", description="Agent purpose")
    version: str = Field(default="0.0.1", description="Agent version")
    url: str = Field(default="", description="Agent endpoint URL")
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    skills: tuple[AgentSkill, ...] = Field(default=(), description="Advertised skills")
    default_input_modes: tuple[str, ...] = Field(
        default=("text",), description="Accepted input formats"
    )
    default_output_modes: tuple[str, ...] = Field(
        default=("text",), description="Produced output formats"
    )
    supported_transports: tuple[TransportMode, ...] = Field(
        default=(TransportMode.JSONRPC,), description="Transport protocols"
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_sequences(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for key in (
                "skills",
                "default_input_modes",
                "default_output_modes",
                "supported_transports",
            ):
                val = data.get(key)
                if isinstance(val, list):
                    data = {**data, key: tuple(val)}
        return data


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


class ServingConfig(BaseModel):
    """Server-side configuration for publishing an agent via A2A.

    .. warning:: Ephemeral ports (``port=0``)

        When ``port=0`` the OS chooses a port at bind time, but the agent card
        URL is built *before* binding using ``http://{host}:{port}/``, which
        yields ``http://localhost:0/``.  Pass an explicit, non-zero ``port``
        (or override ``AgentCard.url`` after binding) so that remote callers
        receive a reachable advertised URL.
    """

    model_config = {"frozen": True}

    host: str = Field(default="localhost", description="Bind host")
    port: int = Field(default=0, description="Bind port (0 = auto; see class docstring)")
    endpoint: str = Field(default="/", description="Base URL path")
    streaming: bool = Field(default=False, description="Enable streaming")
    version: str = Field(default="0.0.1", description="Advertised version")
    skills: tuple[AgentSkill, ...] = Field(default=(), description="Skills to advertise")
    input_modes: tuple[str, ...] = Field(default=("text",), description="Accepted input formats")
    output_modes: tuple[str, ...] = Field(default=("text",), description="Produced output formats")
    transports: tuple[TransportMode, ...] = Field(
        default=(TransportMode.JSONRPC,), description="Enabled transports"
    )
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Extension point for custom config"
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_sequences(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for key in ("skills", "input_modes", "output_modes", "transports"):
                val = data.get(key)
                if isinstance(val, list):
                    data = {**data, key: tuple(val)}
        return data


class ClientConfig(BaseModel):
    """Client-side configuration for connecting to a remote A2A agent."""

    model_config = {"frozen": True}

    streaming: bool = Field(default=False, description="Request streaming")
    timeout: float = Field(default=600.0, gt=0, description="Request timeout (sec)")
    transports: tuple[TransportMode, ...] = Field(
        default=(TransportMode.JSONRPC,), description="Preferred transports"
    )
    accepted_output_modes: tuple[str, ...] = Field(
        default=(), description="Accepted output formats (empty = any)"
    )
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Extension point for custom config"
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_sequences(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for key in ("transports", "accepted_output_modes"):
                val = data.get(key)
                if isinstance(val, list):
                    data = {**data, key: tuple(val)}
        return data


# ---------------------------------------------------------------------------
# Task events
# ---------------------------------------------------------------------------


class A2ATaskStatus(BaseModel):
    """Current status of a remote A2A task."""

    model_config = {"frozen": True}

    state: TaskState = Field(description="Task lifecycle state")
    reason: str = Field(default="", description="Reason / error message")


class TaskStatusUpdateEvent(BaseModel):
    """Emitted when a remote task changes state."""

    model_config = {"frozen": True}

    task_id: str = Field(description="Task being updated")
    status: A2ATaskStatus = Field(description="New status")


class TaskArtifactUpdateEvent(BaseModel):
    """Emitted when a remote task produces output."""

    model_config = {"frozen": True}

    task_id: str = Field(description="Task being updated")
    text: str = Field(default="", description="Artifact text content")
    last_chunk: bool = Field(default=False, description="Whether this is the final chunk")
