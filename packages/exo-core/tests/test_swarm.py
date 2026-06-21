"""Tests for exo.swarm — Swarm multi-agent orchestration."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest

from exo.agent import Agent
from exo.runner import run
from exo.swarm import Swarm, SwarmError, _DelegateTool
from exo.types import (
    AgentOutput,
    RunResult,
    StatusEvent,
    StreamEvent,
    TextEvent,
    ToolCall,
    Usage,
)

# ---------------------------------------------------------------------------
# Fixtures: mock provider
# ---------------------------------------------------------------------------


def _make_provider(responses: list[AgentOutput]) -> Any:
    """Create a mock provider returning pre-defined AgentOutput values."""
    call_count = 0

    async def complete(messages: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        resp = responses[min(call_count, len(responses) - 1)]
        call_count += 1

        class FakeResponse:
            content = resp.text
            tool_calls = resp.tool_calls
            usage = resp.usage

        return FakeResponse()

    mock = AsyncMock()
    mock.complete = complete
    return mock


# ---------------------------------------------------------------------------
# Swarm construction
# ---------------------------------------------------------------------------


class TestSwarmConstruction:
    def test_minimal_swarm(self) -> None:
        """Swarm can be created with a single agent and no flow DSL."""
        a = Agent(name="a")
        swarm = Swarm(agents=[a])

        assert swarm.mode == "workflow"
        assert swarm.flow_order == ["a"]
        assert "a" in swarm.agents

    def test_swarm_with_flow_dsl(self) -> None:
        """Swarm parses flow DSL and determines topological order."""
        a = Agent(name="a")
        b = Agent(name="b")
        c = Agent(name="c")
        swarm = Swarm(agents=[a, b, c], flow="a >> b >> c")

        assert swarm.flow_order == ["a", "b", "c"]
        assert swarm.flow == "a >> b >> c"

    def test_swarm_flow_order_from_dsl(self) -> None:
        """Flow DSL determines order independent of agent list order."""
        a = Agent(name="a")
        b = Agent(name="b")
        c = Agent(name="c")
        # Agents given in reverse, but flow specifies a >> b >> c
        swarm = Swarm(agents=[c, b, a], flow="a >> b >> c")

        assert swarm.flow_order == ["a", "b", "c"]

    def test_swarm_default_order_is_agent_list_order(self) -> None:
        """Without flow DSL, order matches agent list."""
        a = Agent(name="a")
        b = Agent(name="b")
        c = Agent(name="c")
        swarm = Swarm(agents=[c, a, b])

        assert swarm.flow_order == ["c", "a", "b"]

    def test_swarm_empty_agents_raises(self) -> None:
        """Swarm with no agents raises SwarmError."""
        with pytest.raises(SwarmError, match="at least one agent"):
            Swarm(agents=[])

    def test_swarm_duplicate_agent_names(self) -> None:
        """Swarm with duplicate agent names raises SwarmError."""
        a1 = Agent(name="a")
        a2 = Agent(name="a")

        with pytest.raises(SwarmError, match="Duplicate agent name"):
            Swarm(agents=[a1, a2])

    def test_swarm_flow_references_unknown_agent(self) -> None:
        """Flow DSL referencing an agent not in the swarm raises SwarmError."""
        a = Agent(name="a")

        with pytest.raises(SwarmError, match="unknown agent 'z'"):
            Swarm(agents=[a], flow="a >> z")

    def test_swarm_invalid_flow_dsl(self) -> None:
        """Invalid flow DSL string raises SwarmError."""
        a = Agent(name="a")

        with pytest.raises(SwarmError, match="Invalid flow DSL"):
            Swarm(agents=[a], flow="")

    def test_swarm_describe(self) -> None:
        """Swarm.describe() returns mode, flow, and agent info."""
        a = Agent(name="a")
        b = Agent(name="b")
        swarm = Swarm(agents=[a, b], flow="a >> b")

        desc = swarm.describe()

        assert desc["mode"] == "workflow"
        assert desc["flow"] == "a >> b"
        assert desc["flow_order"] == ["a", "b"]
        assert "a" in desc["agents"]
        assert "b" in desc["agents"]

    def test_swarm_repr(self) -> None:
        """Swarm.__repr__() includes mode, agents, flow."""
        a = Agent(name="a")
        swarm = Swarm(agents=[a], flow="a")

        r = repr(swarm)

        assert "workflow" in r
        assert "a" in r


# ---------------------------------------------------------------------------
# Workflow execution
# ---------------------------------------------------------------------------


class TestSwarmWorkflow:
    async def test_single_agent_workflow(self) -> None:
        """Workflow with one agent returns its output."""
        a = Agent(name="a", instructions="Be agent A.")
        swarm = Swarm(agents=[a])
        provider = _make_provider([AgentOutput(text="Hello from A")])

        result = await swarm.run("Hi", provider=provider)

        assert isinstance(result, RunResult)
        assert result.output == "Hello from A"

    async def test_two_agent_pipeline(self) -> None:
        """Workflow chains output→input between two agents."""
        a = Agent(name="a")
        b = Agent(name="b")
        swarm = Swarm(agents=[a, b], flow="a >> b")

        # Agent a receives "Hi" and outputs "from_a"
        # Agent b receives "from_a" and outputs "from_b"
        provider = _make_provider(
            [
                AgentOutput(text="from_a"),
                AgentOutput(text="from_b"),
            ]
        )

        result = await swarm.run("Hi", provider=provider)

        assert result.output == "from_b"

    async def test_three_agent_pipeline(self) -> None:
        """Workflow chains through 3 agents sequentially."""
        a = Agent(name="a")
        b = Agent(name="b")
        c = Agent(name="c")
        swarm = Swarm(agents=[a, b, c], flow="a >> b >> c")

        provider = _make_provider(
            [
                AgentOutput(text="step1"),
                AgentOutput(text="step2"),
                AgentOutput(text="step3"),
            ]
        )

        result = await swarm.run("start", provider=provider)

        assert result.output == "step3"

    async def test_workflow_output_becomes_next_input(self) -> None:
        """Each agent receives previous agent's output as input."""
        received_inputs: list[str] = []

        # Track what each agent receives by intercepting provider
        call_count = 0

        async def tracked_complete(messages: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            # Extract the last user message content to see what the agent received
            for m in reversed(messages):
                content = getattr(m, "content", None)
                role = getattr(m, "role", None)
                if role == "user" and content:
                    received_inputs.append(content)
                    break

            responses = [
                AgentOutput(text="output_from_a"),
                AgentOutput(text="output_from_b"),
                AgentOutput(text="output_from_c"),
            ]
            resp = responses[min(call_count, len(responses) - 1)]
            call_count += 1

            class FakeResponse:
                content = resp.text
                tool_calls = resp.tool_calls
                usage = resp.usage

            return FakeResponse()

        provider = AsyncMock()
        provider.complete = tracked_complete

        a = Agent(name="a")
        b = Agent(name="b")
        c = Agent(name="c")
        swarm = Swarm(agents=[a, b, c], flow="a >> b >> c")

        await swarm.run("initial_input", provider=provider)

        assert received_inputs[0] == "initial_input"
        assert received_inputs[1] == "output_from_a"
        assert received_inputs[2] == "output_from_b"


# ---------------------------------------------------------------------------
# Swarm via run() public API
# ---------------------------------------------------------------------------


class TestSwarmViaRun:
    async def test_run_with_swarm(self) -> None:
        """run(swarm, ...) detects Swarm and delegates correctly."""
        a = Agent(name="a")
        b = Agent(name="b")
        swarm = Swarm(agents=[a, b], flow="a >> b")

        provider = _make_provider(
            [
                AgentOutput(text="from_a"),
                AgentOutput(text="from_b"),
            ]
        )

        result = await run(swarm, "test", provider=provider)

        assert isinstance(result, RunResult)
        assert result.output == "from_b"

    def test_run_sync_with_swarm(self) -> None:
        """run.sync(swarm, ...) works for synchronous execution."""
        a = Agent(name="a")
        swarm = Swarm(agents=[a])

        provider = _make_provider([AgentOutput(text="sync_ok")])

        result = run.sync(swarm, "test", provider=provider)

        assert result.output == "sync_ok"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestSwarmEdgeCases:
    async def test_workflow_with_usage_tracking(self) -> None:
        """Workflow returns usage from the final agent."""
        a = Agent(name="a")
        b = Agent(name="b")
        swarm = Swarm(agents=[a, b], flow="a >> b")

        provider = _make_provider(
            [
                AgentOutput(
                    text="from_a",
                    usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
                ),
                AgentOutput(
                    text="from_b",
                    usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
                ),
            ]
        )

        result = await swarm.run("test", provider=provider)

        assert result.output == "from_b"
        # Usage comes from the final call_runner result
        assert result.usage.input_tokens == 20
        assert result.usage.output_tokens == 10

    async def test_unsupported_mode_raises(self) -> None:
        """Unsupported swarm mode raises SwarmError at construction time."""
        a = Agent(name="a")
        with pytest.raises(SwarmError, match="Unknown swarm mode"):
            Swarm(agents=[a], mode="invalid")  # type: ignore[arg-type]

    async def test_workflow_messages_passed_through(self) -> None:
        """Prior messages are forwarded to agents in workflow."""
        from exo.types import UserMessage

        a = Agent(name="a")
        swarm = Swarm(agents=[a])
        provider = _make_provider([AgentOutput(text="continued")])

        prior = [UserMessage(content="Earlier context")]
        result = await swarm.run("Continue", messages=prior, provider=provider)

        assert result.output == "continued"

    async def test_swarm_name_attribute(self) -> None:
        """Swarm has a name attribute for compatibility."""
        a = Agent(name="leader")
        swarm = Swarm(agents=[a])

        assert "leader" in swarm.name


# ---------------------------------------------------------------------------
# Handoff mode — construction
# ---------------------------------------------------------------------------


class TestSwarmHandoffConstruction:
    def test_handoff_mode_creation(self) -> None:
        """Swarm can be created in handoff mode."""
        triage = Agent(name="triage")
        billing = Agent(name="billing")
        swarm = Swarm(agents=[triage, billing], mode="handoff")

        assert swarm.mode == "handoff"
        assert swarm.max_handoffs == 10

    def test_handoff_mode_custom_max_handoffs(self) -> None:
        """max_handoffs can be configured."""
        a = Agent(name="a")
        b = Agent(name="b")
        swarm = Swarm(agents=[a, b], mode="handoff", max_handoffs=3)

        assert swarm.max_handoffs == 3

    def test_handoff_mode_no_flow_required(self) -> None:
        """Handoff mode works without flow DSL."""
        a = Agent(name="a")
        b = Agent(name="b")
        swarm = Swarm(agents=[a, b], mode="handoff")

        assert swarm.flow is None
        assert swarm.flow_order == ["a", "b"]


# ---------------------------------------------------------------------------
# Handoff mode — execution
# ---------------------------------------------------------------------------


class TestSwarmHandoff:
    async def test_simple_handoff_a_to_b(self) -> None:
        """Agent A hands off to agent B, B produces final output."""
        b = Agent(name="b")
        a = Agent(name="a", handoffs=[b])

        swarm = Swarm(agents=[a, b], mode="handoff")

        # Agent a outputs "b" (matching handoff target name) -> triggers handoff
        # Agent b outputs "final answer" (no handoff) -> stops
        provider = _make_provider(
            [
                AgentOutput(text="b"),
                AgentOutput(text="final answer"),
            ]
        )

        result = await swarm.run("Hello", provider=provider)

        assert result.output == "final answer"

    async def test_no_handoff_returns_immediately(self) -> None:
        """Agent with no handoff targets returns its output directly."""
        a = Agent(name="a")
        b = Agent(name="b")
        swarm = Swarm(agents=[a, b], mode="handoff")

        provider = _make_provider([AgentOutput(text="no handoff here")])

        result = await swarm.run("Hello", provider=provider)

        assert result.output == "no handoff here"

    async def test_handoff_chain_a_to_b_to_c(self) -> None:
        """Handoff chain: A -> B -> C, C produces final output."""
        c = Agent(name="c")
        b = Agent(name="b", handoffs=[c])
        a = Agent(name="a", handoffs=[b])

        swarm = Swarm(agents=[a, b, c], mode="handoff")

        provider = _make_provider(
            [
                AgentOutput(text="b"),  # a -> handoff to b
                AgentOutput(text="c"),  # b -> handoff to c
                AgentOutput(text="done!"),  # c -> final
            ]
        )

        result = await swarm.run("start", provider=provider)

        assert result.output == "done!"

    async def test_handoff_output_not_matching_target(self) -> None:
        """Agent with handoffs but output doesn't match any target name."""
        b = Agent(name="b")
        a = Agent(name="a", handoffs=[b])

        swarm = Swarm(agents=[a, b], mode="handoff")

        # Agent a outputs something that is NOT "b" -> no handoff
        provider = _make_provider([AgentOutput(text="some regular answer")])

        result = await swarm.run("Hello", provider=provider)

        assert result.output == "some regular answer"

    async def test_handoff_with_whitespace_stripping(self) -> None:
        """Handoff detection strips whitespace from output."""
        b = Agent(name="b")
        a = Agent(name="a", handoffs=[b])

        swarm = Swarm(agents=[a, b], mode="handoff")

        # Output has leading/trailing whitespace around target name
        provider = _make_provider(
            [
                AgentOutput(text="  b  "),
                AgentOutput(text="handled by b"),
            ]
        )

        result = await swarm.run("Hello", provider=provider)

        assert result.output == "handled by b"

    async def test_handoff_target_not_in_swarm(self) -> None:
        """Handoff target must exist in the swarm's agents dict."""
        external = Agent(name="external")
        a = Agent(name="a", handoffs=[external])

        # external is a handoff target on agent a, but NOT in the swarm
        swarm = Swarm(agents=[a], mode="handoff")

        # Agent a outputs "external" which matches handoff target name
        # but "external" is not in swarm.agents, so no handoff
        provider = _make_provider([AgentOutput(text="external")])

        result = await swarm.run("Hello", provider=provider)

        assert result.output == "external"

    async def test_handoff_conversation_history_transferred(self) -> None:
        """Handoff transfers conversation history to the next agent."""
        received_messages: list[list[Any]] = []

        call_count = 0

        async def tracked_complete(messages: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            received_messages.append(list(messages))

            responses = [
                AgentOutput(text="b"),  # a -> handoff to b
                AgentOutput(text="final"),  # b -> done
            ]
            resp = responses[min(call_count, len(responses) - 1)]
            call_count += 1

            class FakeResponse:
                content = resp.text
                tool_calls = resp.tool_calls
                usage = resp.usage

            return FakeResponse()

        provider = AsyncMock()
        provider.complete = tracked_complete

        b = Agent(name="b")
        a = Agent(name="a", handoffs=[b])
        swarm = Swarm(agents=[a, b], mode="handoff")

        result = await swarm.run("Hello", provider=provider)

        # Handoff happened: 2 LLM calls (one per agent)
        assert len(received_messages) == 2
        # Agent b receives the handoff output ("b") as its input
        last_user_msg = None
        for m in received_messages[1]:
            if getattr(m, "role", None) == "user":
                last_user_msg = m
        assert last_user_msg is not None
        assert last_user_msg.content == "b"
        # Final result comes from agent b
        assert result.output == "final"


# ---------------------------------------------------------------------------
# Handoff mode — loop detection
# ---------------------------------------------------------------------------


class TestSwarmHandoffLoopDetection:
    async def test_loop_detection_triggers(self) -> None:
        """Endless handoff loop raises SwarmError."""
        b = Agent(name="b")
        a = Agent(name="a", handoffs=[b])
        # Make b hand off back to a
        b.handoffs = {"a": a}

        swarm = Swarm(agents=[a, b], mode="handoff", max_handoffs=3)

        # a -> b -> a -> b -> exceeds max_handoffs=3
        provider = _make_provider(
            [
                AgentOutput(text="b"),  # a -> b
                AgentOutput(text="a"),  # b -> a
                AgentOutput(text="b"),  # a -> b
                AgentOutput(text="a"),  # b -> a (exceeds limit)
            ]
        )

        with pytest.raises(SwarmError, match=r"Max handoffs.*3.*exceeded"):
            await swarm.run("Hello", provider=provider)

    async def test_loop_detection_max_handoffs_1(self) -> None:
        """max_handoffs=1 allows exactly one handoff."""
        b = Agent(name="b")
        a = Agent(name="a", handoffs=[b])

        swarm = Swarm(agents=[a, b], mode="handoff", max_handoffs=1)

        provider = _make_provider(
            [
                AgentOutput(text="b"),  # a -> b (1 handoff, allowed)
                AgentOutput(text="result"),  # b -> final
            ]
        )

        result = await swarm.run("Hello", provider=provider)
        assert result.output == "result"

    async def test_loop_detection_max_handoffs_exceeded_exactly(self) -> None:
        """max_handoffs=1 fails on second handoff attempt."""
        b = Agent(name="b")
        a = Agent(name="a", handoffs=[b])
        b.handoffs = {"a": a}

        swarm = Swarm(agents=[a, b], mode="handoff", max_handoffs=1)

        provider = _make_provider(
            [
                AgentOutput(text="b"),  # a -> b (1 handoff, OK)
                AgentOutput(text="a"),  # b -> a (2nd handoff, exceeds 1)
            ]
        )

        with pytest.raises(SwarmError, match=r"Max handoffs.*1.*exceeded"):
            await swarm.run("Hello", provider=provider)


# ---------------------------------------------------------------------------
# Handoff via run() public API
# ---------------------------------------------------------------------------


class TestSwarmHandoffViaRun:
    async def test_run_handoff_swarm(self) -> None:
        """run(handoff_swarm, ...) works correctly."""
        b = Agent(name="b")
        a = Agent(name="a", handoffs=[b])

        swarm = Swarm(agents=[a, b], mode="handoff")

        provider = _make_provider(
            [
                AgentOutput(text="b"),
                AgentOutput(text="via run()"),
            ]
        )

        result = await run(swarm, "test", provider=provider)

        assert isinstance(result, RunResult)
        assert result.output == "via run()"

    def test_run_sync_handoff_swarm(self) -> None:
        """run.sync(handoff_swarm, ...) works for synchronous execution."""
        b = Agent(name="b")
        a = Agent(name="a", handoffs=[b])

        swarm = Swarm(agents=[a, b], mode="handoff")

        provider = _make_provider(
            [
                AgentOutput(text="b"),
                AgentOutput(text="sync handoff"),
            ]
        )

        result = run.sync(swarm, "test", provider=provider)

        assert result.output == "sync handoff"


# ---------------------------------------------------------------------------
# Team mode — construction
# ---------------------------------------------------------------------------


class TestSwarmTeamConstruction:
    def test_team_mode_creation(self) -> None:
        """Swarm can be created in team mode."""
        lead = Agent(name="lead")
        worker = Agent(name="worker")
        swarm = Swarm(agents=[lead, worker], mode="team")

        assert swarm.mode == "team"
        assert swarm.flow_order == ["lead", "worker"]

    async def test_team_mode_single_agent_raises(self) -> None:
        """Team mode with only one agent raises SwarmError."""
        a = Agent(name="a")
        swarm = Swarm(agents=[a], mode="team")

        provider = _make_provider([AgentOutput(text="ok")])

        with pytest.raises(SwarmError, match="at least two agents"):
            await swarm.run("test", provider=provider)


# ---------------------------------------------------------------------------
# Team mode — execution
# ---------------------------------------------------------------------------


_DEFAULT_USAGE = Usage(input_tokens=10, output_tokens=5, total_tokens=15)


def _make_team_provider(responses: list[Any]) -> Any:
    """Create a mock provider for team mode tests.

    Responses can be AgentOutput (text-only) or dicts with 'tool_calls'.
    Each response is consumed in order across all agents.
    """
    call_count = 0

    async def complete(messages: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        resp = responses[min(call_count, len(responses) - 1)]
        call_count += 1

        if isinstance(resp, AgentOutput):

            class FakeResponse:
                content = resp.text
                tool_calls = resp.tool_calls
                usage = resp.usage

            return FakeResponse()

        # Dict form for tool call responses
        class FakeToolCallResponse:
            content = resp.get("content", "")
            tool_calls = resp.get("tool_calls", [])
            usage = resp.get("usage", _DEFAULT_USAGE)

        return FakeToolCallResponse()

    mock = AsyncMock()
    mock.complete = complete
    return mock


class TestSwarmTeam:
    async def test_lead_delegates_to_worker(self) -> None:
        """Lead agent delegates to worker, then synthesizes result."""
        lead = Agent(name="lead", instructions="You are the lead.")
        worker = Agent(name="worker", instructions="You are a worker.")
        swarm = Swarm(agents=[lead, worker], mode="team")

        # Response 1 (lead): call delegate_to_worker tool
        tc = ToolCall(
            id="tc-1",
            name="delegate_to_worker",
            arguments=json.dumps({"task": "research topic X"}),
        )
        # Response 2 (worker): worker's output
        # Response 3 (lead): lead synthesizes final answer
        provider = _make_team_provider(
            [
                {"content": "", "tool_calls": [tc]},
                AgentOutput(text="worker found: topic X details"),
                AgentOutput(text="Based on research: final synthesis"),
            ]
        )

        result = await swarm.run("Summarize topic X", provider=provider)

        assert result.output == "Based on research: final synthesis"

    async def test_lead_delegates_to_multiple_workers(self) -> None:
        """Lead can delegate to different workers."""
        lead = Agent(name="lead")
        researcher = Agent(name="researcher")
        writer = Agent(name="writer")
        swarm = Swarm(agents=[lead, researcher, writer], mode="team")

        # Lead calls delegate_to_researcher, then delegate_to_writer, then returns
        tc1 = ToolCall(
            id="tc-1",
            name="delegate_to_researcher",
            arguments=json.dumps({"task": "research"}),
        )
        tc2 = ToolCall(
            id="tc-2",
            name="delegate_to_writer",
            arguments=json.dumps({"task": "write report"}),
        )
        provider = _make_team_provider(
            [
                {"content": "", "tool_calls": [tc1]},  # lead calls researcher
                AgentOutput(text="research results"),  # researcher responds
                {"content": "", "tool_calls": [tc2]},  # lead calls writer
                AgentOutput(text="written report"),  # writer responds
                AgentOutput(text="All done: final output"),  # lead final
            ]
        )

        result = await swarm.run("Do research and write", provider=provider)

        assert result.output == "All done: final output"

    async def test_lead_no_delegation_returns_directly(self) -> None:
        """Lead can return without delegating to any worker."""
        lead = Agent(name="lead")
        worker = Agent(name="worker")
        swarm = Swarm(agents=[lead, worker], mode="team")

        # Lead responds directly without calling any delegate tool
        provider = _make_team_provider([AgentOutput(text="I can handle this myself")])

        result = await swarm.run("Simple question", provider=provider)

        assert result.output == "I can handle this myself"

    async def test_worker_result_visible_to_lead(self) -> None:
        """Worker output is returned to lead as tool result."""
        received_messages: list[list[Any]] = []
        call_count = 0

        async def tracked_complete(messages: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            received_messages.append(list(messages))

            tc = ToolCall(
                id="tc-1",
                name="delegate_to_worker",
                arguments=json.dumps({"task": "do work"}),
            )
            responses: list[Any] = [
                {"content": "", "tool_calls": [tc]},  # lead delegates
                AgentOutput(text="worker result 42"),  # worker responds
                AgentOutput(text="synthesized"),  # lead final
            ]
            resp = responses[min(call_count, len(responses) - 1)]
            call_count += 1

            if isinstance(resp, AgentOutput):

                class FR:
                    content = resp.text
                    tool_calls = resp.tool_calls
                    usage = resp.usage

                return FR()

            class FTR:
                content = resp.get("content", "")
                tool_calls = resp.get("tool_calls", [])
                usage = resp.get("usage", _DEFAULT_USAGE)

            return FTR()

        provider = AsyncMock()
        provider.complete = tracked_complete

        lead = Agent(name="lead")
        worker = Agent(name="worker")
        swarm = Swarm(agents=[lead, worker], mode="team")

        result = await swarm.run("test", provider=provider)

        assert result.output == "synthesized"
        # At least 3 calls: lead (delegate), worker, lead (synthesize)
        assert len(received_messages) >= 3
        # The third call (lead's second) should contain the tool result
        # with the worker's output
        lead_second_call_msgs = received_messages[2]
        tool_results = [m for m in lead_second_call_msgs if getattr(m, "role", None) == "tool"]
        assert len(tool_results) > 0
        assert "worker result 42" in tool_results[0].content

    async def test_team_tools_restored_after_run(self) -> None:
        """Lead's original tools are restored after team execution."""
        from exo.tool import FunctionTool

        def my_tool(x: str) -> str:
            """A custom tool."""
            return x

        original_tool = FunctionTool(my_tool, name="my_tool")
        lead = Agent(name="lead", tools=[original_tool])
        worker = Agent(name="worker")
        swarm = Swarm(agents=[lead, worker], mode="team")

        provider = _make_team_provider([AgentOutput(text="no delegation")])

        await swarm.run("test", provider=provider)

        # Lead should have only its original tool, not delegate tools
        assert "my_tool" in lead.tools
        assert "delegate_to_worker" not in lead.tools
        # my_tool + retrieve_artifact + spawn_self + 7 context tools (auto-loaded)
        assert len(lead.tools) == 10

    async def test_team_tools_restored_on_error(self) -> None:
        """Lead's tools are restored even if run() raises."""
        from exo._internal.call_runner import CallRunnerError

        lead = Agent(name="lead")
        worker = Agent(name="worker")
        swarm = Swarm(agents=[lead, worker], mode="team")

        # Provider that always raises
        async def failing_complete(messages: Any, **kwargs: Any) -> Any:
            raise RuntimeError("LLM down")

        provider = AsyncMock()
        provider.complete = failing_complete

        with pytest.raises(CallRunnerError):
            await swarm.run("test", provider=provider)

        # Delegate tools should NOT remain on the lead
        assert "delegate_to_worker" not in lead.tools


# ---------------------------------------------------------------------------
# Team mode — via run() public API
# ---------------------------------------------------------------------------


class TestSwarmTeamViaRun:
    async def test_run_team_swarm(self) -> None:
        """run(team_swarm, ...) works correctly."""
        lead = Agent(name="lead")
        worker = Agent(name="worker")
        swarm = Swarm(agents=[lead, worker], mode="team")

        tc = ToolCall(
            id="tc-1",
            name="delegate_to_worker",
            arguments=json.dumps({"task": "work"}),
        )
        provider = _make_team_provider(
            [
                {"content": "", "tool_calls": [tc]},
                AgentOutput(text="worker done"),
                AgentOutput(text="via run()"),
            ]
        )

        result = await run(swarm, "test", provider=provider)

        assert isinstance(result, RunResult)
        assert result.output == "via run()"

    def test_run_sync_team_swarm(self) -> None:
        """run.sync(team_swarm, ...) works for synchronous execution."""
        lead = Agent(name="lead")
        worker = Agent(name="worker")
        swarm = Swarm(agents=[lead, worker], mode="team")

        provider = _make_team_provider([AgentOutput(text="sync team")])

        result = run.sync(swarm, "test", provider=provider)

        assert result.output == "sync team"


# ---------------------------------------------------------------------------
# _DelegateTool unit tests
# ---------------------------------------------------------------------------


class TestDelegateTool:
    def test_delegate_tool_name(self) -> None:
        """DelegateTool has correct name and schema."""
        worker = Agent(name="researcher")
        dtool = _DelegateTool(worker=worker)

        assert dtool.name == "delegate_to_researcher"
        assert "task" in dtool.parameters["properties"]
        assert dtool.parameters["required"] == ["task"]

    def test_delegate_tool_schema(self) -> None:
        """DelegateTool generates valid OpenAI function-calling schema."""
        worker = Agent(name="writer")
        dtool = _DelegateTool(worker=worker)

        schema = dtool.to_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "delegate_to_writer"
        assert "writer" in schema["function"]["description"]

    async def test_delegate_tool_execute(self) -> None:
        """DelegateTool executes the worker agent and returns output."""
        worker = Agent(name="worker")
        provider = _make_provider([AgentOutput(text="worker output")])
        dtool = _DelegateTool(worker=worker, provider=provider)

        result = await dtool.execute(task="do something")

        assert result == "worker output"


# ---------------------------------------------------------------------------
# Streaming helpers (mirrors test_runner.py helpers)
# ---------------------------------------------------------------------------


class _FakeStreamChunk:
    """Lightweight stream chunk for testing."""

    def __init__(
        self,
        delta: str = "",
        tool_call_deltas: list[Any] | None = None,
        finish_reason: str | None = None,
        usage: Usage | None = None,
    ) -> None:
        self.delta = delta
        self.tool_call_deltas = tool_call_deltas or []
        self.finish_reason = finish_reason
        self.usage = usage or Usage()


class _FakeToolCallDelta:
    """Mirrors ToolCallDelta fields for testing."""

    def __init__(
        self,
        index: int = 0,
        id: str | None = None,
        name: str | None = None,
        arguments: str = "",
    ) -> None:
        self.index = index
        self.id = id
        self.name = name
        self.arguments = arguments


def _make_stream_provider(
    stream_rounds: list[list[_FakeStreamChunk]],
) -> Any:
    """Create a mock provider with stream() returning pre-defined chunks.

    Each call to stream() consumes the next list of chunks from stream_rounds.
    """
    call_count = 0

    async def stream(messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
        nonlocal call_count
        chunks = stream_rounds[min(call_count, len(stream_rounds) - 1)]
        call_count += 1
        for c in chunks:
            yield c

    mock = AsyncMock()
    mock.stream = stream
    mock.complete = AsyncMock()
    return mock


# ---------------------------------------------------------------------------
# Swarm streaming — workflow mode
# ---------------------------------------------------------------------------


class TestSwarmStreamWorkflow:
    async def test_workflow_stream_single_agent(self) -> None:
        """Workflow streaming with one agent yields text events."""
        a = Agent(name="agent_a", instructions="Be agent A.")
        swarm = Swarm(agents=[a])
        chunks = [
            _FakeStreamChunk(delta="Hello from A"),
        ]
        provider = _make_stream_provider([chunks])

        events: list[StreamEvent] = []
        async for ev in run.stream(swarm, "Hi", provider=provider):
            events.append(ev)

        text_events = [e for e in events if isinstance(e, TextEvent)]
        assert len(text_events) == 1
        assert text_events[0].text == "Hello from A"
        assert text_events[0].agent_name == "agent_a"

    async def test_workflow_stream_two_agents(self) -> None:
        """Workflow streaming chains through two agents."""
        a = Agent(name="agent_a")
        b = Agent(name="agent_b")
        swarm = Swarm(agents=[a, b], flow="agent_a >> agent_b")

        # Agent a: produces "from_a"
        # Agent b: produces "from_b"
        chunks_a = [_FakeStreamChunk(delta="from_a")]
        chunks_b = [_FakeStreamChunk(delta="from_b")]
        provider = _make_stream_provider([chunks_a, chunks_b])

        events: list[StreamEvent] = []
        async for ev in run.stream(swarm, "Hi", provider=provider):
            events.append(ev)

        text_events = [e for e in events if isinstance(e, TextEvent)]
        assert len(text_events) == 2
        assert text_events[0].text == "from_a"
        assert text_events[0].agent_name == "agent_a"
        assert text_events[1].text == "from_b"
        assert text_events[1].agent_name == "agent_b"

    async def test_workflow_stream_detailed_emits_status_per_agent(self) -> None:
        """detailed=True emits StatusEvent for each agent in workflow."""
        a = Agent(name="agent_a")
        b = Agent(name="agent_b")
        swarm = Swarm(agents=[a, b], flow="agent_a >> agent_b")

        chunks_a = [_FakeStreamChunk(delta="A")]
        chunks_b = [_FakeStreamChunk(delta="B")]
        provider = _make_stream_provider([chunks_a, chunks_b])

        events: list[StreamEvent] = []
        async for ev in run.stream(swarm, "Hi", provider=provider, detailed=True):
            events.append(ev)

        status_events = [e for e in events if isinstance(e, StatusEvent)]
        # At minimum: workflow running events per agent + starting/completed from _stream
        running_events = [e for e in status_events if e.status == "running"]
        assert len(running_events) >= 2
        agent_names = [e.agent_name for e in running_events]
        assert "agent_a" in agent_names
        assert "agent_b" in agent_names

    async def test_workflow_stream_correct_agent_names(self) -> None:
        """All events have the correct agent_name in workflow streaming."""
        a = Agent(name="first")
        b = Agent(name="second")
        swarm = Swarm(agents=[a, b], flow="first >> second")

        chunks_a = [_FakeStreamChunk(delta="A output")]
        chunks_b = [_FakeStreamChunk(delta="B output")]
        provider = _make_stream_provider([chunks_a, chunks_b])

        events: list[StreamEvent] = []
        async for ev in run.stream(swarm, "go", provider=provider):
            events.append(ev)

        text_events = [e for e in events if isinstance(e, TextEvent)]
        assert text_events[0].agent_name == "first"
        assert text_events[1].agent_name == "second"


# ---------------------------------------------------------------------------
# Swarm streaming — handoff mode
# ---------------------------------------------------------------------------


class TestSwarmStreamHandoff:
    async def test_handoff_stream_no_handoff(self) -> None:
        """Handoff streaming with no handoff yields events from first agent."""
        a = Agent(name="agent_a")
        b = Agent(name="agent_b")
        swarm = Swarm(agents=[a, b], mode="handoff")

        chunks = [_FakeStreamChunk(delta="no handoff")]
        provider = _make_stream_provider([chunks])

        events: list[StreamEvent] = []
        async for ev in run.stream(swarm, "Hi", provider=provider):
            events.append(ev)

        text_events = [e for e in events if isinstance(e, TextEvent)]
        assert len(text_events) == 1
        assert text_events[0].text == "no handoff"
        assert text_events[0].agent_name == "agent_a"

    async def test_handoff_stream_a_to_b(self) -> None:
        """Handoff streaming follows handoff chain."""
        b = Agent(name="b")
        a = Agent(name="a", handoffs=[b])
        swarm = Swarm(agents=[a, b], mode="handoff")

        # Agent a outputs "b" -> handoff to b
        # Agent b outputs "final answer"
        chunks_a = [_FakeStreamChunk(delta="b")]
        chunks_b = [_FakeStreamChunk(delta="final answer")]
        provider = _make_stream_provider([chunks_a, chunks_b])

        events: list[StreamEvent] = []
        async for ev in run.stream(swarm, "Hello", provider=provider):
            events.append(ev)

        text_events = [e for e in events if isinstance(e, TextEvent)]
        assert len(text_events) == 2
        assert text_events[0].text == "b"
        assert text_events[0].agent_name == "a"
        assert text_events[1].text == "final answer"
        assert text_events[1].agent_name == "b"

    async def test_handoff_stream_detailed_emits_handoff_status(self) -> None:
        """detailed=True emits StatusEvent for handoff transitions."""
        b = Agent(name="b")
        a = Agent(name="a", handoffs=[b])
        swarm = Swarm(agents=[a, b], mode="handoff")

        chunks_a = [_FakeStreamChunk(delta="b")]
        chunks_b = [_FakeStreamChunk(delta="done")]
        provider = _make_stream_provider([chunks_a, chunks_b])

        events: list[StreamEvent] = []
        async for ev in run.stream(swarm, "Hi", provider=provider, detailed=True):
            events.append(ev)

        status_events = [e for e in events if isinstance(e, StatusEvent)]
        # Should have handoff status events
        running_events = [e for e in status_events if e.status == "running"]
        agent_names = [e.agent_name for e in running_events]
        assert "a" in agent_names
        assert "b" in agent_names

        # Verify handoff message
        handoff_events = [e for e in running_events if "Handoff from" in e.message]
        assert len(handoff_events) >= 1
        assert "a" in handoff_events[0].message
        assert "b" in handoff_events[0].message

    async def test_handoff_stream_max_handoffs_exceeded(self) -> None:
        """Handoff streaming raises SwarmError on excessive handoffs."""
        b = Agent(name="b")
        a = Agent(name="a", handoffs=[b])
        b.handoffs = {"a": a}

        swarm = Swarm(agents=[a, b], mode="handoff", max_handoffs=1)

        # a -> b -> a (exceeds max_handoffs=1)
        chunks_a = [_FakeStreamChunk(delta="b")]
        chunks_b = [_FakeStreamChunk(delta="a")]
        provider = _make_stream_provider([chunks_a, chunks_b, chunks_a])

        with pytest.raises(SwarmError, match=r"Max handoffs.*1.*exceeded"):
            async for _ in run.stream(swarm, "Hello", provider=provider):
                pass


# ---------------------------------------------------------------------------
# Swarm streaming — team mode
# ---------------------------------------------------------------------------


class TestSwarmStreamTeam:
    async def test_team_stream_no_delegation(self) -> None:
        """Team streaming works when lead doesn't delegate."""
        lead = Agent(name="lead")
        worker = Agent(name="worker")
        swarm = Swarm(agents=[lead, worker], mode="team")

        chunks = [_FakeStreamChunk(delta="I can handle this")]
        provider = _make_stream_provider([chunks])

        events: list[StreamEvent] = []
        async for ev in run.stream(swarm, "simple", provider=provider):
            events.append(ev)

        text_events = [e for e in events if isinstance(e, TextEvent)]
        assert len(text_events) == 1
        assert text_events[0].text == "I can handle this"
        assert text_events[0].agent_name == "lead"

    async def test_team_stream_tools_restored(self) -> None:
        """Team streaming restores lead's original tools after execution."""
        lead = Agent(name="lead")
        worker = Agent(name="worker")
        swarm = Swarm(agents=[lead, worker], mode="team")

        chunks = [_FakeStreamChunk(delta="done")]
        provider = _make_stream_provider([chunks])

        async for _ in run.stream(swarm, "test", provider=provider):
            pass

        assert "delegate_to_worker" not in lead.tools

    async def test_team_stream_single_agent_raises(self) -> None:
        """Team streaming with one agent raises SwarmError."""
        a = Agent(name="a")
        swarm = Swarm(agents=[a], mode="team")

        chunks = [_FakeStreamChunk(delta="ok")]
        provider = _make_stream_provider([chunks])

        with pytest.raises(SwarmError, match="at least two agents"):
            async for _ in run.stream(swarm, "test", provider=provider):
                pass

    async def test_team_stream_detailed_emits_status(self) -> None:
        """detailed=True emits StatusEvent for team lead."""
        lead = Agent(name="lead")
        worker = Agent(name="worker")
        swarm = Swarm(agents=[lead, worker], mode="team")

        chunks = [_FakeStreamChunk(delta="done")]
        provider = _make_stream_provider([chunks])

        events: list[StreamEvent] = []
        async for ev in run.stream(swarm, "test", provider=provider, detailed=True):
            events.append(ev)

        status_events = [e for e in events if isinstance(e, StatusEvent)]
        running_events = [e for e in status_events if e.status == "running"]
        assert any(e.agent_name == "lead" for e in running_events)


# ---------------------------------------------------------------------------
# Swarm streaming — unsupported mode
# ---------------------------------------------------------------------------


class TestSwarmStreamUnsupportedMode:
    async def test_unsupported_mode_raises(self) -> None:
        """Constructing a swarm with an unsupported mode raises SwarmError."""
        a = Agent(name="a")
        with pytest.raises(SwarmError, match="Unknown swarm mode"):
            Swarm(agents=[a], mode="invalid")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Swarm streaming via run.stream() public API
# ---------------------------------------------------------------------------


class TestSwarmStreamViaRun:
    async def test_run_stream_detects_swarm(self) -> None:
        """run.stream() detects Swarm and delegates to Swarm.stream()."""
        a = Agent(name="agent_a")
        swarm = Swarm(agents=[a])

        chunks = [_FakeStreamChunk(delta="via run.stream")]
        provider = _make_stream_provider([chunks])

        events: list[StreamEvent] = []
        async for ev in run.stream(swarm, "test", provider=provider):
            events.append(ev)

        text_events = [e for e in events if isinstance(e, TextEvent)]
        assert len(text_events) == 1
        assert text_events[0].text == "via run.stream"


# ---------------------------------------------------------------------------
# Swarm memory defaults (US-014)
# ---------------------------------------------------------------------------


class TestSwarmMemoryDefaults:
    def test_lead_agent_has_auto_created_memory(self) -> None:
        """Swarm lead agent has auto-created AgentMemory when not explicitly set."""
        try:
            from exo.memory.base import AgentMemory  # pyright: ignore[reportMissingImports]
        except ImportError:
            pytest.skip("exo-memory not installed")

        lead = Agent(name="lead")
        worker = Agent(name="worker")
        swarm = Swarm(agents=[lead, worker], mode="team")
        assert isinstance(swarm.agents["lead"].memory, AgentMemory)

    def test_lead_agent_with_disabled_memory_propagates(self) -> None:
        """Swarm lead agent with memory=None keeps memory disabled."""
        lead = Agent(name="lead", memory=None)
        worker = Agent(name="worker", memory=None)
        swarm = Swarm(agents=[lead, worker], mode="team")
        assert swarm.agents["lead"].memory is None


# ---------------------------------------------------------------------------
# Swarm context_mode propagation (US-016)
# ---------------------------------------------------------------------------


class TestSwarmContextMode:
    def test_swarm_propagates_context_mode_to_agents(self) -> None:
        """Swarm(context_mode='pilot') propagates Context(limit=100) to all agents."""
        try:
            from exo.context.context import Context  # pyright: ignore[reportMissingImports]
        except ImportError:
            pytest.skip("exo-context not installed")

        lead = Agent(name="lead")
        worker = Agent(name="worker")
        swarm = Swarm(agents=[lead, worker], context_mode="pilot")

        assert isinstance(swarm.agents["lead"].context, Context)
        assert swarm.agents["lead"].context.config.limit == 100
        assert isinstance(swarm.agents["worker"].context, Context)
        assert swarm.agents["worker"].context.config.limit == 100

    def test_swarm_context_mode_none_disables_context(self) -> None:
        """Swarm(context_mode=None) disables context on all agents."""
        lead = Agent(name="lead")
        worker = Agent(name="worker")
        swarm = Swarm(agents=[lead, worker], context_mode=None)

        assert swarm.agents["lead"].context is None
        assert swarm.agents["worker"].context is None

    def test_swarm_without_context_mode_agents_keep_defaults(self) -> None:
        """Swarm without context_mode leaves agent contexts unchanged."""
        try:
            from exo.context.context import Context  # pyright: ignore[reportMissingImports]
        except ImportError:
            pytest.skip("exo-context not installed")

        lead = Agent(name="lead")
        worker = Agent(name="worker")
        # Both agents should retain their auto-created Context (wrapping copilot config)
        swarm = Swarm(agents=[lead, worker])
        assert isinstance(swarm.agents["lead"].context, Context)
        assert isinstance(swarm.agents["worker"].context, Context)
