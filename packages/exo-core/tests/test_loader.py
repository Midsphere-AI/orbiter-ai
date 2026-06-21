"""Tests for exo.loader — YAML agent & swarm loader."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest

from exo.agent import Agent
from exo.loader import (
    LoaderError,
    _build_agent,
    _substitute,
    load_agents,
    load_swarm,
    load_yaml,
    register_agent_class,
)
from exo.swarm import Swarm

# ---------------------------------------------------------------------------
# Variable substitution
# ---------------------------------------------------------------------------


class TestSubstitute:
    """Unit tests for _substitute()."""

    def test_plain_string(self) -> None:
        assert _substitute("hello", {}, {}) == "hello"

    def test_env_var_full(self) -> None:
        assert _substitute("${FOO}", {"FOO": "bar"}, {}) == "bar"

    def test_env_var_partial(self) -> None:
        assert _substitute("prefix-${FOO}-suffix", {"FOO": "bar"}, {}) == "prefix-bar-suffix"

    def test_vars_full(self) -> None:
        assert _substitute("${vars.KEY}", {}, {"KEY": 42}) == 42

    def test_vars_full_preserves_type(self) -> None:
        assert _substitute("${vars.TEMP}", {}, {"TEMP": 0.7}) == 0.7

    def test_vars_partial_converts_to_str(self) -> None:
        result = _substitute("temp=${vars.TEMP}", {}, {"TEMP": 0.7})
        assert result == "temp=0.7"
        assert isinstance(result, str)

    def test_missing_env(self) -> None:
        assert _substitute("${MISSING}", {}, {}) == "${MISSING}"

    def test_missing_vars(self) -> None:
        assert _substitute("${vars.MISSING}", {}, {}) == "${vars.MISSING}"

    def test_dict_recursive(self) -> None:
        data = {"a": "${FOO}", "b": {"c": "${vars.X}"}}
        result = _substitute(data, {"FOO": "1"}, {"X": 2})
        assert result == {"a": "1", "b": {"c": 2}}

    def test_list_recursive(self) -> None:
        data = ["${FOO}", "${vars.X}"]
        result = _substitute(data, {"FOO": "a"}, {"X": "b"})
        assert result == ["a", "b"]

    def test_non_string_passthrough(self) -> None:
        assert _substitute(42, {}, {}) == 42
        assert _substitute(True, {}, {}) is True


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


class TestBuildAgent:
    """Unit tests for _build_agent()."""

    def test_builtin_minimal(self) -> None:
        agent = _build_agent("a", {})
        assert isinstance(agent, Agent)
        assert agent.name == "a"

    def test_builtin_with_model(self) -> None:
        agent = _build_agent("a", {"model": "anthropic:claude-3"})
        assert agent.model == "anthropic:claude-3"

    def test_builtin_with_instructions(self) -> None:
        agent = _build_agent("a", {"instructions": "Be helpful"})
        assert agent.instructions == "Be helpful"

    def test_builtin_with_system_prompt(self) -> None:
        """system_prompt is an alias for instructions."""
        agent = _build_agent("a", {"system_prompt": "Be concise"})
        assert agent.instructions == "Be concise"

    def test_builtin_with_params(self) -> None:
        agent = _build_agent("a", {"temperature": "0.5", "max_steps": "5"})
        assert agent.temperature == 0.5
        assert agent.max_steps == 5

    def test_builtin_with_runtime_controls(self) -> None:
        agent = _build_agent(
            "planner",
            {
                "planning_enabled": True,
                "planning_model": "openai:gpt-4o-mini",
                "planning_instructions": "Return a short plan.",
                "budget_awareness": "per-message",
                "emit_mcp_progress": False,
                "injected_tool_args": {"ui_request_id": "Opaque request id"},
            },
        )
        assert agent.planning_enabled is True
        assert agent.planning_model == "openai:gpt-4o-mini"
        assert agent.planning_instructions == "Return a short plan."
        assert agent.budget_awareness == "per-message"
        assert agent.emit_mcp_progress is False
        assert agent.injected_tool_args == {"ui_request_id": "Opaque request id"}

    def test_custom_class(self) -> None:
        class MyAgent:
            def __init__(self, *, name: str, **kw: Any) -> None:
                self.name = name
                self.extra = kw

        register_agent_class("custom", MyAgent)
        try:
            agent = _build_agent("a", {"type": "custom", "foo": "bar"})
            assert isinstance(agent, MyAgent)
            assert agent.name == "a"
            assert agent.extra == {"foo": "bar"}
        finally:
            from exo.loader import _AGENT_FACTORIES

            _AGENT_FACTORIES.pop("custom", None)


# ---------------------------------------------------------------------------
# load_yaml()
# ---------------------------------------------------------------------------


class TestLoadYaml:
    """Tests for load_yaml()."""

    def test_missing_file(self) -> None:
        with pytest.raises(LoaderError, match="not found"):
            load_yaml("/nonexistent.yaml")

    def test_non_dict(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text("- item1\n- item2\n")
        with pytest.raises(LoaderError, match="Expected YAML dict"):
            load_yaml(f)

    def test_basic_load(self, tmp_path: Path) -> None:
        f = tmp_path / "test.yaml"
        f.write_text("agents:\n  a:\n    model: gpt-4o\n")
        data = load_yaml(f)
        assert data["agents"]["a"]["model"] == "gpt-4o"

    def test_env_substitution(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_MODEL", "gpt-5")
        f = tmp_path / "test.yaml"
        f.write_text("agents:\n  a:\n    model: ${TEST_MODEL}\n")
        data = load_yaml(f)
        assert data["agents"]["a"]["model"] == "gpt-5"

    def test_vars_substitution(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            vars:
              MY_MODEL: gpt-4o-mini
            agents:
              a:
                model: ${vars.MY_MODEL}
        """)
        f = tmp_path / "test.yaml"
        f.write_text(content)
        data = load_yaml(f)
        assert data["agents"]["a"]["model"] == "gpt-4o-mini"

    def test_vars_type_preservation(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            vars:
              TEMP: 0.7
            agents:
              a:
                temperature: ${vars.TEMP}
        """)
        f = tmp_path / "test.yaml"
        f.write_text(content)
        data = load_yaml(f)
        assert data["agents"]["a"]["temperature"] == 0.7


# ---------------------------------------------------------------------------
# load_agents()
# ---------------------------------------------------------------------------


class TestLoadAgents:
    """Tests for load_agents()."""

    def test_single_agent(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            agents:
              researcher:
                model: openai:gpt-4o
                instructions: You research topics.
        """)
        f = tmp_path / "agents.yaml"
        f.write_text(content)
        agents = load_agents(f)
        assert "researcher" in agents
        assert isinstance(agents["researcher"], Agent)
        assert agents["researcher"].instructions == "You research topics."

    def test_multiple_agents(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            agents:
              a:
                model: gpt-4o
              b:
                model: gpt-4o-mini
        """)
        f = tmp_path / "agents.yaml"
        f.write_text(content)
        agents = load_agents(f)
        assert len(agents) == 2
        assert agents["a"].name == "a"
        assert agents["b"].name == "b"

    def test_no_agents_section(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.yaml"
        f.write_text("other: data\n")
        with pytest.raises(LoaderError, match="No 'agents'"):
            load_agents(f)

    def test_with_vars(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            vars:
              DEFAULT_MODEL: anthropic:claude-3
              TEMP: 0.5
            agents:
              a:
                model: ${vars.DEFAULT_MODEL}
                temperature: ${vars.TEMP}
        """)
        f = tmp_path / "agents.yaml"
        f.write_text(content)
        agents = load_agents(f)
        assert agents["a"].model == "anthropic:claude-3"
        assert agents["a"].temperature == 0.5

    def test_with_env_vars(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_INSTRUCTIONS", "Be smart")
        content = textwrap.dedent("""\
            agents:
              a:
                instructions: ${MY_INSTRUCTIONS}
        """)
        f = tmp_path / "agents.yaml"
        f.write_text(content)
        agents = load_agents(f)
        assert agents["a"].instructions == "Be smart"

    def test_with_runtime_controls(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            agents:
              planner:
                planning_enabled: true
                planning_model: openai:gpt-4o-mini
                planning_instructions: Return a short plan.
                budget_awareness: limit:70
                emit_mcp_progress: false
                injected_tool_args:
                  ui_request_id: Opaque request id
        """)
        f = tmp_path / "agents.yaml"
        f.write_text(content)
        agents = load_agents(f)
        agent = agents["planner"]
        assert agent.planning_enabled is True
        assert agent.planning_model == "openai:gpt-4o-mini"
        assert agent.budget_awareness == "limit:70"
        assert agent.emit_mcp_progress is False
        assert agent.injected_tool_args == {"ui_request_id": "Opaque request id"}


# ---------------------------------------------------------------------------
# load_swarm() — workflow mode
# ---------------------------------------------------------------------------


class TestLoadSwarmWorkflow:
    """Tests for load_swarm() in workflow mode."""

    def test_default_workflow(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            agents:
              a:
                model: gpt-4o
              b:
                model: gpt-4o
        """)
        f = tmp_path / "swarm.yaml"
        f.write_text(content)
        swarm = load_swarm(f)
        assert isinstance(swarm, Swarm)
        assert swarm.mode == "workflow"
        assert swarm.flow_order == ["a", "b"]

    def test_explicit_order(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            agents:
              a:
                model: gpt-4o
              b:
                model: gpt-4o
            swarm:
              type: workflow
              order: [b, a]
        """)
        f = tmp_path / "swarm.yaml"
        f.write_text(content)
        swarm = load_swarm(f)
        assert swarm.flow_order == ["b", "a"]

    def test_with_flow_dsl(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            agents:
              a:
                model: gpt-4o
              b:
                model: gpt-4o
              c:
                model: gpt-4o
            swarm:
              type: workflow
              flow: "a >> b >> c"
        """)
        f = tmp_path / "swarm.yaml"
        f.write_text(content)
        swarm = load_swarm(f)
        assert swarm.flow == "a >> b >> c"

    def test_unknown_order_agent(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            agents:
              a:
                model: gpt-4o
            swarm:
              type: workflow
              order: [a, unknown]
        """)
        f = tmp_path / "swarm.yaml"
        f.write_text(content)
        with pytest.raises(LoaderError, match="unknown agent"):
            load_swarm(f)


# ---------------------------------------------------------------------------
# load_swarm() — handoff mode
# ---------------------------------------------------------------------------


class TestLoadSwarmHandoff:
    """Tests for load_swarm() in handoff mode."""

    def test_handoff_with_edges(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            agents:
              a:
                model: gpt-4o
              b:
                model: gpt-4o
            swarm:
              type: handoff
              edges:
                - [a, b]
                - [b, a]
        """)
        f = tmp_path / "swarm.yaml"
        f.write_text(content)
        swarm = load_swarm(f)
        assert swarm.mode == "handoff"
        # Verify handoffs were wired
        a_agent = swarm.agents["a"]
        assert "b" in a_agent.handoffs

    def test_handoff_bad_edge(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            agents:
              a:
                model: gpt-4o
            swarm:
              type: handoff
              edges:
                - [a, missing]
        """)
        f = tmp_path / "swarm.yaml"
        f.write_text(content)
        with pytest.raises(LoaderError, match="unknown agent"):
            load_swarm(f)


# ---------------------------------------------------------------------------
# load_swarm() — team mode
# ---------------------------------------------------------------------------


class TestLoadSwarmTeam:
    """Tests for load_swarm() in team mode."""

    def test_team_mode(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            agents:
              lead:
                model: gpt-4o
                instructions: You are the lead.
              worker1:
                model: gpt-4o
              worker2:
                model: gpt-4o
            swarm:
              type: team
              order: [lead, worker1, worker2]
        """)
        f = tmp_path / "swarm.yaml"
        f.write_text(content)
        swarm = load_swarm(f)
        assert swarm.mode == "team"
        assert swarm.flow_order[0] == "lead"

    def test_team_custom_max_handoffs(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            agents:
              lead:
                model: gpt-4o
              worker:
                model: gpt-4o
            swarm:
              type: team
              max_handoffs: 5
        """)
        f = tmp_path / "swarm.yaml"
        f.write_text(content)
        swarm = load_swarm(f)
        assert swarm.max_handoffs == 5


# ---------------------------------------------------------------------------
# register_agent_class integration
# ---------------------------------------------------------------------------


class TestCustomAgentFactory:
    """Integration test for custom agent classes via YAML."""

    def test_custom_class_in_yaml(self, tmp_path: Path) -> None:
        class SpecialAgent:
            def __init__(self, *, name: str, model: str = "gpt-4o", **kw: Any) -> None:
                self.name = name
                self.model = model
                self.handoffs: dict[str, Any] = {}

            def describe(self) -> dict[str, Any]:
                return {"name": self.name}

        register_agent_class("special", SpecialAgent)
        try:
            content = textwrap.dedent("""\
                agents:
                  a:
                    type: special
                    model: gpt-5
            """)
            f = tmp_path / "agents.yaml"
            f.write_text(content)
            agents = load_agents(f)
            assert isinstance(agents["a"], SpecialAgent)
            assert agents["a"].model == "gpt-5"
        finally:
            from exo.loader import _AGENT_FACTORIES

            _AGENT_FACTORIES.pop("special", None)
