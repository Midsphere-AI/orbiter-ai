"""Tests for exo.config — configuration types."""

import pytest
from pydantic import ValidationError

from exo.config import (
    AgentConfig,
    ModelConfig,
    RunConfig,
    TaskConfig,
    parse_model_string,
)

# --- parse_model_string ---


class TestParseModelString:
    def test_provider_and_model(self) -> None:
        assert parse_model_string("openai:gpt-4o") == ("openai", "gpt-4o")

    def test_anthropic_provider(self) -> None:
        assert parse_model_string("anthropic:claude-sonnet-4-20250514") == (
            "anthropic",
            "claude-sonnet-4-20250514",
        )

    def test_no_prefix_defaults_to_openai(self) -> None:
        assert parse_model_string("gpt-4o") == ("openai", "gpt-4o")

    def test_empty_string(self) -> None:
        assert parse_model_string("") == ("openai", "")

    def test_multiple_colons(self) -> None:
        assert parse_model_string("custom:my:model") == ("custom", "my:model")

    def test_colon_only(self) -> None:
        assert parse_model_string(":model") == ("", "model")


# --- ModelConfig ---


class TestModelConfig:
    def test_defaults(self) -> None:
        mc = ModelConfig()
        assert mc.provider == "openai"
        assert mc.model_name == "gpt-4o"
        assert mc.api_key is None
        assert mc.base_url is None
        assert mc.max_retries == 3
        assert mc.timeout == 30.0
        assert mc.context_window_tokens is None

    def test_create(self) -> None:
        mc = ModelConfig(
            provider="anthropic",
            model_name="claude-sonnet-4-20250514",
            api_key="sk-test",
            base_url="https://api.example.com",
            max_retries=5,
            timeout=60.0,
        )
        assert mc.provider == "anthropic"
        assert mc.api_key == "sk-test"

    def test_frozen(self) -> None:
        mc = ModelConfig()
        with pytest.raises(ValidationError):
            mc.provider = "changed"  # type: ignore[misc]

    def test_max_retries_ge_zero(self) -> None:
        ModelConfig(max_retries=0)  # should not raise
        with pytest.raises(ValidationError):
            ModelConfig(max_retries=-1)

    def test_timeout_gt_zero(self) -> None:
        ModelConfig(timeout=0.1)  # should not raise
        with pytest.raises(ValidationError):
            ModelConfig(timeout=0)
        with pytest.raises(ValidationError):
            ModelConfig(timeout=-1.0)

    def test_context_window_tokens_field(self) -> None:
        mc = ModelConfig(context_window_tokens=128000)
        assert mc.context_window_tokens == 128000

    def test_context_window_tokens_none(self) -> None:
        mc = ModelConfig(context_window_tokens=None)
        assert mc.context_window_tokens is None

    def test_roundtrip(self) -> None:
        mc = ModelConfig(provider="anthropic", model_name="claude", max_retries=1)
        data = mc.model_dump()
        restored = ModelConfig.model_validate(data)
        assert restored == mc

    def test_roundtrip_with_context_window(self) -> None:
        mc = ModelConfig(provider="openai", model_name="gpt-4o", context_window_tokens=128000)
        data = mc.model_dump()
        restored = ModelConfig.model_validate(data)
        assert restored == mc
        assert restored.context_window_tokens == 128000


# --- AgentConfig ---


class TestAgentConfig:
    def test_defaults(self) -> None:
        ac = AgentConfig(name="test")
        assert ac.name == "test"
        assert ac.model == "openai:gpt-4o-mini"
        assert ac.instructions == ""
        assert ac.temperature == 1.0
        assert ac.max_tokens is None
        assert ac.max_steps == 10
        assert ac.planning_enabled is False
        assert ac.planning_model is None
        assert ac.planning_instructions == ""
        assert ac.budget_awareness is None
        assert ac.hitl_tools == []
        assert ac.emit_mcp_progress is True
        assert ac.injected_tool_args == {}
        assert ac.allow_parallel_subagents is False
        assert ac.max_parallel_subagents == 3

    def test_create(self) -> None:
        ac = AgentConfig(
            name="researcher",
            model="anthropic:claude-sonnet-4-20250514",
            instructions="You research things.",
            temperature=0.7,
            max_tokens=4096,
            max_steps=20,
            planning_enabled=True,
            planning_model="openai:gpt-4o-mini",
            planning_instructions="Plan first.",
            budget_awareness="limit:70",
            hitl_tools=["deploy_service"],
            emit_mcp_progress=False,
            injected_tool_args={"ui_request_id": "Opaque request id"},
            allow_parallel_subagents=True,
            max_parallel_subagents=4,
        )
        assert ac.name == "researcher"
        assert ac.model == "anthropic:claude-sonnet-4-20250514"
        assert ac.max_tokens == 4096
        assert ac.planning_enabled is True
        assert ac.planning_model == "openai:gpt-4o-mini"
        assert ac.budget_awareness == "limit:70"
        assert ac.hitl_tools == ["deploy_service"]
        assert ac.emit_mcp_progress is False
        assert ac.injected_tool_args == {"ui_request_id": "Opaque request id"}
        assert ac.allow_parallel_subagents is True
        assert ac.max_parallel_subagents == 4

    def test_missing_name(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig()  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        ac = AgentConfig(name="test")
        with pytest.raises(ValidationError):
            ac.name = "changed"  # type: ignore[misc]

    def test_temperature_bounds(self) -> None:
        AgentConfig(name="t", temperature=0.0)  # lower bound OK
        AgentConfig(name="t", temperature=2.0)  # upper bound OK
        with pytest.raises(ValidationError):
            AgentConfig(name="t", temperature=-0.1)
        with pytest.raises(ValidationError):
            AgentConfig(name="t", temperature=2.1)

    def test_max_steps_ge_one(self) -> None:
        AgentConfig(name="t", max_steps=1)  # should not raise
        with pytest.raises(ValidationError):
            AgentConfig(name="t", max_steps=0)
        with pytest.raises(ValidationError):
            AgentConfig(name="t", max_steps=-1)

    def test_roundtrip(self) -> None:
        ac = AgentConfig(
            name="bot",
            temperature=0.5,
            max_steps=5,
            planning_enabled=True,
            planning_model="openai:gpt-4o-mini",
            budget_awareness="per-message",
            hitl_tools=["search"],
            injected_tool_args={"run_origin": "Surface label"},
            allow_parallel_subagents=True,
            max_parallel_subagents=2,
        )
        data = ac.model_dump()
        restored = AgentConfig.model_validate(data)
        assert restored == ac

    def test_invalid_planning_model_raises(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig(name="planner", planning_model="openai:")

    def test_invalid_budget_awareness_raises(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig(name="budget", budget_awareness="limit:abc")
        with pytest.raises(ValidationError):
            AgentConfig(name="budget", budget_awareness="limit:101")

    def test_invalid_max_parallel_subagents_raises(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig(name="parallel", max_parallel_subagents=8)


# --- TaskConfig ---


class TestTaskConfig:
    def test_defaults(self) -> None:
        tc = TaskConfig(name="my_task")
        assert tc.name == "my_task"
        assert tc.description == ""

    def test_create(self) -> None:
        tc = TaskConfig(name="analyze", description="Analyze the data.")
        assert tc.description == "Analyze the data."

    def test_missing_name(self) -> None:
        with pytest.raises(ValidationError):
            TaskConfig()  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        tc = TaskConfig(name="t")
        with pytest.raises(ValidationError):
            tc.name = "changed"  # type: ignore[misc]

    def test_roundtrip(self) -> None:
        tc = TaskConfig(name="t", description="desc")
        data = tc.model_dump()
        restored = TaskConfig.model_validate(data)
        assert restored == tc


# --- RunConfig ---


class TestRunConfig:
    def test_defaults(self) -> None:
        rc = RunConfig()
        assert rc.max_steps == 10
        assert rc.timeout is None
        assert rc.stream is False
        assert rc.verbose is False

    def test_create(self) -> None:
        rc = RunConfig(max_steps=20, timeout=120.0, stream=True, verbose=True)
        assert rc.max_steps == 20
        assert rc.timeout == 120.0
        assert rc.stream is True
        assert rc.verbose is True

    def test_frozen(self) -> None:
        rc = RunConfig()
        with pytest.raises(ValidationError):
            rc.stream = True  # type: ignore[misc]

    def test_max_steps_ge_one(self) -> None:
        RunConfig(max_steps=1)  # should not raise
        with pytest.raises(ValidationError):
            RunConfig(max_steps=0)

    def test_roundtrip(self) -> None:
        rc = RunConfig(max_steps=5, stream=True)
        data = rc.model_dump()
        restored = RunConfig.model_validate(data)
        assert restored == rc
