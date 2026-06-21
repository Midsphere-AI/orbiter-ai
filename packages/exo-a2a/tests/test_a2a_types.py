"""Tests for exo.a2a.types — A2A protocol types."""

from __future__ import annotations

from exo.a2a.types import (  # pyright: ignore[reportMissingImports]
    A2ATaskStatus,
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    ClientConfig,
    ServingConfig,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
    TransportMode,
)

# ---------------------------------------------------------------------------
# TransportMode
# ---------------------------------------------------------------------------


class TestTransportMode:
    def test_jsonrpc_value(self) -> None:
        assert TransportMode.JSONRPC == "jsonrpc"

    def test_is_str_enum(self) -> None:
        assert isinstance(TransportMode.JSONRPC, str)

    def test_only_jsonrpc_implemented(self) -> None:
        """Only JSONRPC is implemented; GRPC and WEBSOCKET are not present."""
        assert not hasattr(TransportMode, "GRPC")
        assert not hasattr(TransportMode, "WEBSOCKET")


# ---------------------------------------------------------------------------
# TaskState
# ---------------------------------------------------------------------------


class TestTaskState:
    def test_values(self) -> None:
        assert TaskState.SUBMITTED == "submitted"
        assert TaskState.WORKING == "working"
        assert TaskState.COMPLETED == "completed"
        assert TaskState.FAILED == "failed"
        assert TaskState.CANCELED == "canceled"

    def test_is_str_enum(self) -> None:
        assert isinstance(TaskState.WORKING, str)


# ---------------------------------------------------------------------------
# AgentSkill
# ---------------------------------------------------------------------------


class TestAgentSkill:
    def test_creation(self) -> None:
        skill = AgentSkill(id="s1", name="Search")
        assert skill.id == "s1"
        assert skill.name == "Search"
        assert skill.description == ""
        assert skill.tags == ()

    def test_full_creation(self) -> None:
        skill = AgentSkill(id="s2", name="Code", description="Code generation", tags=["dev", "ai"])
        assert skill.description == "Code generation"
        assert skill.tags == ("dev", "ai")

    def test_frozen(self) -> None:
        skill = AgentSkill(id="s1", name="Search")
        try:
            skill.name = "Other"  # type: ignore[misc]
            raise AssertionError("should be frozen")
        except Exception:
            pass

    def test_tags_coercion_from_list(self) -> None:
        skill = AgentSkill(id="s1", name="S", tags=["a", "b"])
        assert skill.tags == ("a", "b")

    def test_serialization(self) -> None:
        skill = AgentSkill(id="s1", name="S", tags=["x"])
        data = skill.model_dump()
        assert data["id"] == "s1"
        assert data["tags"] == ("x",)
        restored = AgentSkill.model_validate(data)
        assert restored == skill


# ---------------------------------------------------------------------------
# AgentCapabilities
# ---------------------------------------------------------------------------


class TestAgentCapabilities:
    def test_defaults(self) -> None:
        caps = AgentCapabilities()
        assert caps.streaming is False
        assert caps.state_transition_history is False

    def test_no_push_notifications(self) -> None:
        """push_notifications was removed — no code path backs it."""
        assert not hasattr(AgentCapabilities(), "push_notifications")

    def test_custom(self) -> None:
        caps = AgentCapabilities(streaming=True)
        assert caps.streaming is True

    def test_frozen(self) -> None:
        caps = AgentCapabilities()
        try:
            caps.streaming = True  # type: ignore[misc]
            raise AssertionError("should be frozen")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# AgentCard
# ---------------------------------------------------------------------------


class TestAgentCard:
    def test_minimal(self) -> None:
        card = AgentCard(name="agent-1")
        assert card.name == "agent-1"
        assert card.description == ""
        assert card.version == "0.0.1"
        assert card.url == ""
        assert card.capabilities == AgentCapabilities()
        assert card.skills == ()
        assert card.default_input_modes == ("text",)
        assert card.default_output_modes == ("text",)
        assert card.supported_transports == (TransportMode.JSONRPC,)

    def test_full_creation(self) -> None:
        skill = AgentSkill(id="s1", name="Search")
        card = AgentCard(
            name="agent-2",
            description="A search agent",
            version="1.0.0",
            url="http://localhost:8080",
            capabilities=AgentCapabilities(streaming=True),
            skills=[skill],
            default_input_modes=["text", "json"],
            default_output_modes=["text"],
            supported_transports=["jsonrpc"],
        )
        assert card.skills == (skill,)
        assert card.default_input_modes == ("text", "json")
        assert card.supported_transports == (TransportMode.JSONRPC,)

    def test_frozen(self) -> None:
        card = AgentCard(name="a")
        try:
            card.name = "b"  # type: ignore[misc]
            raise AssertionError("should be frozen")
        except Exception:
            pass

    def test_json_roundtrip(self) -> None:
        card = AgentCard(
            name="rt-agent",
            url="http://example.com",
            skills=[AgentSkill(id="s1", name="S")],
            supported_transports=["jsonrpc"],
        )
        json_str = card.model_dump_json()
        restored = AgentCard.model_validate_json(json_str)
        assert restored.name == card.name
        assert restored.skills == card.skills
        assert restored.supported_transports == card.supported_transports

    def test_model_dump(self) -> None:
        card = AgentCard(name="x")
        data = card.model_dump()
        assert data["name"] == "x"
        assert data["version"] == "0.0.1"


# ---------------------------------------------------------------------------
# ServingConfig
# ---------------------------------------------------------------------------


class TestServingConfig:
    def test_defaults(self) -> None:
        cfg = ServingConfig()
        assert cfg.host == "localhost"
        assert cfg.port == 0
        assert cfg.endpoint == "/"
        assert cfg.streaming is False
        assert cfg.version == "0.0.1"
        assert cfg.skills == ()
        assert cfg.input_modes == ("text",)
        assert cfg.output_modes == ("text",)
        assert cfg.transports == (TransportMode.JSONRPC,)
        assert cfg.extra == {}

    def test_custom(self) -> None:
        skill = AgentSkill(id="s1", name="S")
        cfg = ServingConfig(
            host="0.0.0.0",
            port=8080,
            streaming=True,
            skills=[skill],
            transports=["jsonrpc"],
        )
        assert cfg.port == 8080
        assert cfg.streaming is True
        assert cfg.skills == (skill,)
        assert cfg.transports == (TransportMode.JSONRPC,)

    def test_frozen(self) -> None:
        cfg = ServingConfig()
        try:
            cfg.port = 9090  # type: ignore[misc]
            raise AssertionError("should be frozen")
        except Exception:
            pass

    def test_serialization(self) -> None:
        cfg = ServingConfig(port=3000, streaming=True)
        data = cfg.model_dump()
        assert data["port"] == 3000
        restored = ServingConfig.model_validate(data)
        assert restored == cfg


# ---------------------------------------------------------------------------
# ClientConfig
# ---------------------------------------------------------------------------


class TestClientConfig:
    def test_defaults(self) -> None:
        cfg = ClientConfig()
        assert cfg.streaming is False
        assert cfg.timeout == 600.0
        assert cfg.transports == (TransportMode.JSONRPC,)
        assert cfg.accepted_output_modes == ()
        assert cfg.extra == {}

    def test_custom(self) -> None:
        cfg = ClientConfig(
            streaming=True,
            timeout=30.0,
            transports=["jsonrpc"],
            accepted_output_modes=["text", "json"],
        )
        assert cfg.streaming is True
        assert cfg.timeout == 30.0
        assert cfg.transports == (TransportMode.JSONRPC,)
        assert cfg.accepted_output_modes == ("text", "json")

    def test_frozen(self) -> None:
        cfg = ClientConfig()
        try:
            cfg.timeout = 10.0  # type: ignore[misc]
            raise AssertionError("should be frozen")
        except Exception:
            pass

    def test_timeout_validation(self) -> None:
        try:
            ClientConfig(timeout=0)
            raise AssertionError("should reject timeout=0")
        except Exception:
            pass

    def test_serialization(self) -> None:
        cfg = ClientConfig(streaming=True, timeout=120.0)
        json_str = cfg.model_dump_json()
        restored = ClientConfig.model_validate_json(json_str)
        assert restored == cfg


# ---------------------------------------------------------------------------
# A2ATaskStatus
# ---------------------------------------------------------------------------


class TestA2ATaskStatus:
    def test_creation(self) -> None:
        status = A2ATaskStatus(state=TaskState.WORKING)
        assert status.state == TaskState.WORKING
        assert status.reason == ""

    def test_with_reason(self) -> None:
        status = A2ATaskStatus(state=TaskState.FAILED, reason="timeout")
        assert status.reason == "timeout"

    def test_frozen(self) -> None:
        status = A2ATaskStatus(state=TaskState.COMPLETED)
        try:
            status.state = TaskState.FAILED  # type: ignore[misc]
            raise AssertionError("should be frozen")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# TaskStatusUpdateEvent
# ---------------------------------------------------------------------------


class TestTaskStatusUpdateEvent:
    def test_creation(self) -> None:
        evt = TaskStatusUpdateEvent(
            task_id="t1",
            status=A2ATaskStatus(state=TaskState.WORKING),
        )
        assert evt.task_id == "t1"
        assert evt.status.state == TaskState.WORKING

    def test_failed_event(self) -> None:
        evt = TaskStatusUpdateEvent(
            task_id="t2",
            status=A2ATaskStatus(state=TaskState.FAILED, reason="error"),
        )
        assert evt.status.state == TaskState.FAILED
        assert evt.status.reason == "error"

    def test_serialization(self) -> None:
        evt = TaskStatusUpdateEvent(
            task_id="t3",
            status=A2ATaskStatus(state=TaskState.COMPLETED),
        )
        data = evt.model_dump()
        assert data["task_id"] == "t3"
        assert data["status"]["state"] == "completed"
        restored = TaskStatusUpdateEvent.model_validate(data)
        assert restored == evt


# ---------------------------------------------------------------------------
# TaskArtifactUpdateEvent
# ---------------------------------------------------------------------------


class TestTaskArtifactUpdateEvent:
    def test_creation(self) -> None:
        evt = TaskArtifactUpdateEvent(task_id="t1", text="hello")
        assert evt.task_id == "t1"
        assert evt.text == "hello"
        assert evt.last_chunk is False

    def test_last_chunk(self) -> None:
        evt = TaskArtifactUpdateEvent(task_id="t1", text="done", last_chunk=True)
        assert evt.last_chunk is True

    def test_defaults(self) -> None:
        evt = TaskArtifactUpdateEvent(task_id="t1")
        assert evt.text == ""
        assert evt.last_chunk is False

    def test_serialization(self) -> None:
        evt = TaskArtifactUpdateEvent(task_id="t1", text="result", last_chunk=True)
        json_str = evt.model_dump_json()
        restored = TaskArtifactUpdateEvent.model_validate_json(json_str)
        assert restored == evt
