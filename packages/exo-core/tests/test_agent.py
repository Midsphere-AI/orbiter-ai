"""Tests for exo.agent — Agent class init, configuration & run."""

from __future__ import annotations

from typing import Any, ClassVar
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from exo.agent import Agent, AgentError
from exo.hooks import HookPoint
from exo.models.types import ModelError, ModelResponse  # pyright: ignore[reportMissingImports]
from exo.tool import FunctionTool, Tool, tool
from exo.types import ToolCall, Usage

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@tool
def greet(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}!"


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class ReportOutput(BaseModel):
    title: str
    body: str


# ---------------------------------------------------------------------------
# Agent creation: minimal & full
# ---------------------------------------------------------------------------


class TestAgentCreation:
    def test_minimal(self) -> None:
        """Agent with only a name uses sensible defaults."""
        agent = Agent(name="bot")
        assert agent.name == "bot"
        assert agent.model == "openai:gpt-4o"
        assert agent.provider_name == "openai"
        assert agent.model_name == "gpt-4o"
        assert agent.instructions == ""
        # retrieve_artifact is always auto-registered (needed for threshold-based offloading)
        assert "retrieve_artifact" in agent.tools
        # spawn_self is auto-registered by default (allow_self_spawn=True)
        assert "spawn_self" in agent.tools
        # 7 context tools are auto-loaded when context is active (default)
        auto_tool_names = {
            "retrieve_artifact",
            "spawn_self",
            "add_todo",
            "complete_todo",
            "get_todo",
            "get_knowledge",
            "grep_knowledge",
            "search_knowledge",
            "read_file",
        }
        non_auto = [k for k in agent.tools if k not in auto_tool_names]
        assert len(non_auto) == 0
        assert agent.handoffs == {}
        assert agent.output_type is None
        assert agent.max_steps == 10
        assert agent.temperature == 1.0
        assert agent.max_tokens is None
        # memory is auto-created (AgentMemory) when exo-memory is installed, else None
        try:
            from exo.memory.base import AgentMemory  # pyright: ignore[reportMissingImports]

            assert isinstance(agent.memory, AgentMemory)
        except ImportError:
            assert agent.memory is None
        assert agent.conversation_id is None
        # context is auto-created (Context wrapping ContextConfig copilot)
        # when exo-context is installed
        try:
            from exo.context.config import ContextConfig  # pyright: ignore[reportMissingImports]
            from exo.context.context import Context  # pyright: ignore[reportMissingImports]

            assert isinstance(agent.context, Context)
            assert isinstance(agent.context.config, ContextConfig)
        except ImportError:
            assert agent.context is None

    def test_full_config(self) -> None:
        """Agent accepts all configuration parameters."""

        async def my_hook(**data: Any) -> None:
            pass

        agent = Agent(
            name="researcher",
            model="anthropic:claude-sonnet-4-20250514",
            instructions="Research things.",
            tools=[greet, add],
            hooks=[(HookPoint.PRE_LLM_CALL, my_hook)],
            output_type=ReportOutput,
            max_steps=20,
            temperature=0.7,
            max_tokens=4096,
        )
        assert agent.name == "researcher"
        assert agent.provider_name == "anthropic"
        assert agent.model_name == "claude-sonnet-4-20250514"
        assert agent.instructions == "Research things."
        # 2 user tools + retrieve_artifact + spawn_self + 7 context tools (auto-loaded)
        assert len(agent.tools) == 11
        assert agent.output_type is ReportOutput
        assert agent.max_steps == 20
        assert agent.temperature == 0.7
        assert agent.max_tokens == 4096

    def test_name_is_required(self) -> None:
        """Agent() without name raises TypeError."""
        with pytest.raises(TypeError):
            Agent()  # type: ignore[call-arg]

    def test_all_params_keyword_only(self) -> None:
        """Positional arguments are not accepted."""
        with pytest.raises(TypeError):
            Agent("bot")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Model string parsing
# ---------------------------------------------------------------------------


class TestModelParsing:
    def test_provider_colon_model(self) -> None:
        agent = Agent(name="a", model="anthropic:claude-3-opus")
        assert agent.provider_name == "anthropic"
        assert agent.model_name == "claude-3-opus"

    def test_no_colon_defaults_openai(self) -> None:
        agent = Agent(name="a", model="gpt-4o-mini")
        assert agent.provider_name == "openai"
        assert agent.model_name == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


class TestToolRegistration:
    def test_tools_stored_by_name(self) -> None:
        agent = Agent(name="bot", tools=[greet, add])
        assert "greet" in agent.tools
        assert "add" in agent.tools
        assert isinstance(agent.tools["greet"], FunctionTool)

    def test_duplicate_tool_raises(self) -> None:
        dup = FunctionTool(lambda x: x, name="greet")
        with pytest.raises(AgentError, match="Duplicate tool name 'greet'"):
            Agent(name="bot", tools=[greet, dup])

    def test_tool_abc_subclass(self) -> None:
        """Custom Tool subclasses are accepted."""

        class MyTool(Tool):
            name = "custom"
            description = "A custom tool."
            parameters: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}

            async def execute(self, **kwargs: Any) -> str:
                return "done"

        agent = Agent(name="bot", tools=[MyTool()])
        assert "custom" in agent.tools

    def test_get_tool_schemas(self) -> None:
        agent = Agent(name="bot", tools=[greet])
        schemas = agent.get_tool_schemas()
        # greet + retrieve_artifact + spawn_self + 7 context tools (auto-loaded)
        assert len(schemas) == 10
        names = {s["function"]["name"] for s in schemas}
        assert "greet" in names
        assert schemas[0]["type"] == "function"

    def test_tool_schema_caching_returns_same_object(self) -> None:
        agent = Agent(name="bot", tools=[greet])
        first = agent.get_tool_schemas()
        second = agent.get_tool_schemas()
        assert first is second

    async def test_tool_schema_cache_invalidated_on_add_tool(self) -> None:
        agent = Agent(name="bot", tools=[greet])
        before = agent.get_tool_schemas()

        @tool
        def farewell(name: str) -> str:
            """Say goodbye."""
            return f"bye {name}"

        await agent.add_tool(farewell)
        after = agent.get_tool_schemas()
        assert before is not after
        names = {s["function"]["name"] for s in after}
        assert "farewell" in names

    def test_tool_schema_cache_invalidated_on_remove_tool(self) -> None:
        agent = Agent(name="bot", tools=[greet])
        before = agent.get_tool_schemas()
        agent.remove_tool("greet")
        after = agent.get_tool_schemas()
        assert before is not after
        names = {s["function"]["name"] for s in after}
        assert "greet" not in names


# ---------------------------------------------------------------------------
# Injected tool args — schema augmentation & argument stripping
# ---------------------------------------------------------------------------


class TestInjectedToolArgs:
    def test_schemas_augmented_with_injected_args(self) -> None:
        """injected_tool_args adds optional string properties to every tool schema."""
        agent = Agent(
            name="bot",
            tools=[greet],
            injected_tool_args={"reason": "Why you are calling this tool"},
        )
        schemas = agent.get_tool_schemas()
        greet_schema = next(s for s in schemas if s["function"]["name"] == "greet")
        props = greet_schema["function"]["parameters"]["properties"]
        assert "reason" in props
        assert props["reason"]["type"] == "string"
        assert props["reason"]["description"] == "Why you are calling this tool"

    def test_injected_args_are_optional(self) -> None:
        """Injected fields must NOT appear in the required list."""
        agent = Agent(
            name="bot",
            tools=[greet],
            injected_tool_args={"trace_id": "Correlation identifier"},
        )
        schemas = agent.get_tool_schemas()
        greet_schema = next(s for s in schemas if s["function"]["name"] == "greet")
        required = greet_schema["function"]["parameters"].get("required", [])
        assert "trace_id" not in required

    def test_original_tool_parameters_not_mutated(self) -> None:
        """The underlying Tool.parameters dict must never be mutated."""
        agent = Agent(
            name="bot",
            tools=[greet],
            injected_tool_args={"origin": "UI surface"},
        )
        original_props = greet.parameters.get("properties", {})
        agent.get_tool_schemas()
        assert "origin" not in original_props

    def test_no_augmentation_when_empty(self) -> None:
        """Empty injected_tool_args produces unchanged schemas."""
        agent_plain = Agent(name="bot", tools=[greet])
        agent_empty = Agent(name="bot", tools=[greet], injected_tool_args={})
        schemas_plain = agent_plain.get_tool_schemas()
        schemas_empty = agent_empty.get_tool_schemas()
        # Both should have the same properties for greet
        greet_plain = next(s for s in schemas_plain if s["function"]["name"] == "greet")
        greet_empty = next(s for s in schemas_empty if s["function"]["name"] == "greet")
        assert (
            greet_plain["function"]["parameters"]["properties"]
            == greet_empty["function"]["parameters"]["properties"]
        )

    def test_injected_does_not_overwrite_existing_params(self) -> None:
        """If an injected arg name collides with a real param, the real param wins."""
        agent = Agent(
            name="bot",
            tools=[greet],
            injected_tool_args={"name": "Should NOT overwrite the real 'name' param"},
        )
        schemas = agent.get_tool_schemas()
        greet_schema = next(s for s in schemas if s["function"]["name"] == "greet")
        props = greet_schema["function"]["parameters"]["properties"]
        # "name" should still be the original parameter, not the injected one
        assert props["name"]["type"] == "string"
        assert "Should NOT overwrite" not in props["name"].get("description", "")

    async def test_injected_args_stripped_before_execution(self) -> None:
        """Injected args filled by the LLM are stripped before tool.execute()."""
        received_kwargs: dict[str, Any] = {}

        @tool
        def capture(**kwargs: Any) -> str:
            """Capture all kwargs."""
            received_kwargs.update(kwargs)
            return "ok"

        agent = Agent(
            name="bot",
            tools=[capture],
            injected_tool_args={"reasoning": "Why you called this"},
        )

        # Simulate what the LLM returns: tool args including the injected field
        call = ToolCall(
            id="call_1",
            name="capture",
            arguments='{"reasoning": "because I need data", "real_arg": "value"}',
        )
        response = ModelResponse(
            content="",
            tool_calls=[call],
        )

        call_count = 0

        async def mock_complete(messages: Any, **kwargs: Any) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return response
            return ModelResponse(content="done")

        provider = AsyncMock()
        provider.complete = mock_complete

        await agent.run("test", provider=provider)

        # "reasoning" should have been stripped; "real_arg" should pass through
        assert "reasoning" not in received_kwargs
        assert received_kwargs.get("real_arg") == "value"

    def test_augmentation_applies_to_all_tools(self) -> None:
        """Injected args are added to every tool's schema, not just the first."""
        agent = Agent(
            name="bot",
            tools=[greet, add],
            injected_tool_args={"trace": "Trace ID"},
        )
        schemas = agent.get_tool_schemas()
        for schema in schemas:
            props = schema["function"]["parameters"].get("properties", {})
            if schema["function"]["name"] in ("greet", "add"):
                assert "trace" in props, f"Missing 'trace' in {schema['function']['name']}"


# ---------------------------------------------------------------------------
# Handoff registration
# ---------------------------------------------------------------------------


class TestHandoffRegistration:
    def test_handoffs_stored_by_name(self) -> None:
        billing = Agent(name="billing")
        support = Agent(name="support")
        triage = Agent(name="triage", handoffs=[billing, support])
        assert "billing" in triage.handoffs
        assert "support" in triage.handoffs

    def test_duplicate_handoff_raises(self) -> None:
        a1 = Agent(name="helper")
        a2 = Agent(name="helper")
        with pytest.raises(AgentError, match="Duplicate handoff agent 'helper'"):
            Agent(name="main", handoffs=[a1, a2])


# ---------------------------------------------------------------------------
# Hook integration
# ---------------------------------------------------------------------------


class TestHookIntegration:
    def test_hook_manager_initialized(self) -> None:
        agent = Agent(name="bot")
        assert not agent.hook_manager.has_hooks(HookPoint.PRE_LLM_CALL)

    async def test_hooks_registered(self) -> None:
        calls: list[str] = []

        async def on_pre(**data: Any) -> None:
            calls.append("pre")

        agent = Agent(
            name="bot",
            hooks=[(HookPoint.PRE_LLM_CALL, on_pre)],
        )
        assert agent.hook_manager.has_hooks(HookPoint.PRE_LLM_CALL)
        await agent.hook_manager.run(HookPoint.PRE_LLM_CALL)
        assert calls == ["pre"]

    async def test_multiple_hooks(self) -> None:
        calls: list[str] = []

        async def hook_a(**data: Any) -> None:
            calls.append("a")

        async def hook_b(**data: Any) -> None:
            calls.append("b")

        agent = Agent(
            name="bot",
            hooks=[
                (HookPoint.START, hook_a),
                (HookPoint.FINISHED, hook_b),
            ],
        )
        await agent.hook_manager.run(HookPoint.START)
        await agent.hook_manager.run(HookPoint.FINISHED)
        assert calls == ["a", "b"]


# ---------------------------------------------------------------------------
# Instructions (str and callable)
# ---------------------------------------------------------------------------


class TestInstructions:
    def test_string_instructions(self) -> None:
        agent = Agent(name="bot", instructions="Be helpful.")
        assert agent.instructions == "Be helpful."

    def test_callable_instructions(self) -> None:
        def make_instructions(agent_name: str) -> str:
            return f"You are {agent_name}."

        agent = Agent(name="bot", instructions=make_instructions)
        assert callable(agent.instructions)
        assert agent.instructions("bot") == "You are bot."  # type: ignore[operator]


# ---------------------------------------------------------------------------
# describe() and __repr__
# ---------------------------------------------------------------------------


class TestDescribeAndRepr:
    def test_describe_minimal(self) -> None:
        agent = Agent(name="bot")
        desc = agent.describe()
        assert desc["name"] == "bot"
        assert desc["model"] == "openai:gpt-4o"
        # retrieve_artifact is always auto-registered
        assert "retrieve_artifact" in desc["tools"]
        assert desc["handoffs"] == []
        assert desc["output_type"] is None

    def test_describe_with_tools_and_handoffs(self) -> None:
        helper = Agent(name="helper")
        agent = Agent(
            name="main",
            tools=[greet],
            handoffs=[helper],
            output_type=ReportOutput,
        )
        desc = agent.describe()
        assert "greet" in desc["tools"]
        assert desc["handoffs"] == ["helper"]
        assert desc["output_type"] == "ReportOutput"

    def test_repr_minimal(self) -> None:
        agent = Agent(name="bot")
        r = repr(agent)
        assert "Agent(" in r
        assert "name='bot'" in r
        assert "model='openai:gpt-4o'" in r

    def test_repr_with_tools(self) -> None:
        agent = Agent(name="bot", tools=[greet])
        r = repr(agent)
        # retrieve_artifact is always auto-registered alongside user tools
        assert "greet" in r
        assert "retrieve_artifact" in r


# ---------------------------------------------------------------------------
# Mock provider helper
# ---------------------------------------------------------------------------


def _mock_provider(
    content: str = "Hello!",
    tool_calls: list[ToolCall] | None = None,
    usage: Usage | None = None,
) -> AsyncMock:
    """Create a mock provider that returns a fixed ModelResponse."""
    resp = ModelResponse(
        id="resp-1",
        model="test-model",
        content=content,
        tool_calls=tool_calls or [],
        usage=usage or Usage(input_tokens=10, output_tokens=5, total_tokens=15),
    )
    provider = AsyncMock()
    provider.complete = AsyncMock(return_value=resp)
    return provider


# ---------------------------------------------------------------------------
# Agent.run() — single-turn execution
# ---------------------------------------------------------------------------


class TestAgentRun:
    async def test_successful_run(self) -> None:
        """Successful single-turn run returns parsed output."""
        provider = _mock_provider(content="Hi there!")
        agent = Agent(name="bot", instructions="Be helpful.")

        output = await agent.run("Hello", provider=provider)

        assert output.text == "Hi there!"
        assert output.usage.input_tokens == 10
        assert output.usage.output_tokens == 5
        provider.complete.assert_awaited_once()

    async def test_run_passes_messages(self) -> None:
        """History messages are included in the LLM call."""
        from exo.types import AssistantMessage, UserMessage

        provider = _mock_provider()
        agent = Agent(name="bot")

        history = [
            UserMessage(content="first"),
            AssistantMessage(content="reply"),
        ]
        await agent.run("second", messages=history, provider=provider)

        call_args = provider.complete.call_args
        messages = call_args[0][0]
        # history (2) + new user message (1) = 3 (no system since instructions="")
        assert len(messages) == 3
        assert messages[-1].content == "second"

    async def test_run_with_instructions(self) -> None:
        """System instructions are prepended to message list."""
        provider = _mock_provider()
        agent = Agent(name="bot", instructions="Be concise.")

        await agent.run("Hello", provider=provider)

        call_args = provider.complete.call_args
        messages = call_args[0][0]
        assert messages[0].role == "system"
        assert messages[0].content == "Be concise."

    async def test_run_with_callable_instructions(self) -> None:
        """Callable instructions are resolved before building messages."""
        provider = _mock_provider()
        agent = Agent(
            name="bot",
            instructions=lambda name: f"You are {name}.",
        )

        await agent.run("Hi", provider=provider)

        call_args = provider.complete.call_args
        messages = call_args[0][0]
        assert messages[0].content == "You are bot."

    async def test_async_callable_instructions(self) -> None:
        """Async callable instructions are awaited and produce the correct system prompt."""
        provider = _mock_provider()

        async def make_prompt(name: str) -> str:
            return f"You are {name}"

        agent = Agent(name="bot", instructions=make_prompt)

        await agent.run("Hi", provider=provider)

        call_args = provider.complete.call_args
        messages = call_args[0][0]
        assert messages[0].role == "system"
        assert messages[0].content == "You are bot"

    async def test_run_without_provider_raises(self) -> None:
        """run() without provider raises AgentError."""
        agent = Agent(name="bot")
        with pytest.raises(AgentError, match="requires a provider"):
            await agent.run("Hello")

    async def test_run_with_tool_calls_in_response(self) -> None:
        """Tool calls from the LLM trigger tool execution and re-call."""
        tc = ToolCall(id="tc-1", name="greet", arguments='{"name": "World"}')
        # First call returns tool call, second returns text
        resp_tool = ModelResponse(
            content="",
            tool_calls=[tc],
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        resp_text = ModelResponse(
            content="Done!",
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        provider = AsyncMock()
        provider.complete = AsyncMock(side_effect=[resp_tool, resp_text])
        agent = Agent(name="bot", tools=[greet])

        output = await agent.run("Hello", provider=provider)

        # Final output should be the text response after tool execution
        assert output.text == "Done!"
        assert provider.complete.await_count == 2

    async def test_run_passes_tool_schemas(self) -> None:
        """Tool schemas are passed to provider.complete()."""
        provider = _mock_provider()
        agent = Agent(name="bot", tools=[greet])

        await agent.run("Hello", provider=provider)

        call_args = provider.complete.call_args
        assert call_args[1]["tools"] is not None
        # greet + retrieve_artifact + spawn_self + 7 context tools (auto-loaded)
        assert len(call_args[1]["tools"]) == 10


# ---------------------------------------------------------------------------
# Agent.run() — retry logic
# ---------------------------------------------------------------------------


class TestAgentRunRetry:
    async def test_retry_on_transient_error(self) -> None:
        """Transient errors are retried up to max_retries."""
        provider = AsyncMock()
        # Fail twice, succeed on third attempt
        resp = ModelResponse(
            content="ok", usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2)
        )
        provider.complete = AsyncMock(
            side_effect=[RuntimeError("timeout"), RuntimeError("server error"), resp]
        )

        agent = Agent(name="bot")
        output = await agent.run("Hello", provider=provider, max_retries=3)

        assert output.text == "ok"
        assert provider.complete.await_count == 3

    async def test_all_retries_exhausted(self) -> None:
        """AgentError raised when all retries fail."""
        provider = AsyncMock()
        provider.complete = AsyncMock(side_effect=RuntimeError("server error"))

        agent = Agent(name="bot")
        with pytest.raises(AgentError, match="failed after 2 retries"):
            await agent.run("Hello", provider=provider, max_retries=2)

        assert provider.complete.await_count == 2

    async def test_context_length_error_no_retry(self) -> None:
        """Context-length errors fail immediately without retry."""
        provider = AsyncMock()
        err = ModelError("context window exceeded", model="gpt-4o", code="context_length")
        provider.complete = AsyncMock(side_effect=err)

        agent = Agent(name="bot")
        with pytest.raises(AgentError, match="Context length exceeded"):
            await agent.run("Hello", provider=provider, max_retries=5)

        # Should have called complete exactly once (no retries)
        assert provider.complete.await_count == 1

    async def test_context_length_detected_from_message(self) -> None:
        """Context-length errors detected from error message text."""
        provider = AsyncMock()
        provider.complete = AsyncMock(
            side_effect=RuntimeError("context length exceeded for this model")
        )

        agent = Agent(name="bot")
        with pytest.raises(AgentError, match="Context length exceeded"):
            await agent.run("Hello", provider=provider, max_retries=3)

        assert provider.complete.await_count == 1


# ---------------------------------------------------------------------------
# Agent.run() — hook invocation order
# ---------------------------------------------------------------------------


class TestAgentRunHooks:
    async def test_hook_invocation_order(self) -> None:
        """PRE_LLM_CALL fires before LLM call, POST_LLM_CALL fires after."""
        events: list[str] = []

        async def pre_hook(**data: Any) -> None:
            events.append("pre_llm")

        async def post_hook(**data: Any) -> None:
            events.append("post_llm")

        provider = _mock_provider()
        # Track when complete() is called
        original_complete = provider.complete

        async def tracked_complete(*args: Any, **kwargs: Any) -> ModelResponse:
            events.append("llm_call")
            return await original_complete(*args, **kwargs)

        provider.complete = tracked_complete

        agent = Agent(
            name="bot",
            hooks=[
                (HookPoint.PRE_LLM_CALL, pre_hook),
                (HookPoint.POST_LLM_CALL, post_hook),
            ],
        )

        await agent.run("Hello", provider=provider)

        assert events == ["pre_llm", "llm_call", "post_llm"]

    async def test_pre_hook_receives_agent_and_messages(self) -> None:
        """PRE_LLM_CALL hook receives agent and messages."""
        hook_data: dict[str, Any] = {}

        async def pre_hook(**data: Any) -> None:
            hook_data.update(data)

        provider = _mock_provider()
        agent = Agent(
            name="bot",
            instructions="Hi",
            hooks=[(HookPoint.PRE_LLM_CALL, pre_hook)],
        )

        await agent.run("Hello", provider=provider)

        assert hook_data["agent"] is agent
        assert len(hook_data["messages"]) >= 1

    async def test_post_hook_receives_response(self) -> None:
        """POST_LLM_CALL hook receives the model response."""
        hook_data: dict[str, Any] = {}

        async def post_hook(**data: Any) -> None:
            hook_data.update(data)

        provider = _mock_provider(content="I'm here!")
        agent = Agent(
            name="bot",
            hooks=[(HookPoint.POST_LLM_CALL, post_hook)],
        )

        await agent.run("Hello", provider=provider)

        assert hook_data["response"].content == "I'm here!"

    async def test_hooks_not_called_on_immediate_error(self) -> None:
        """POST_LLM_CALL hook is not called if LLM call raises immediately."""
        events: list[str] = []

        async def post_hook(**data: Any) -> None:
            events.append("post_llm")

        provider = AsyncMock()
        err = ModelError("context length exceeded", code="context_length")
        provider.complete = AsyncMock(side_effect=err)

        agent = Agent(
            name="bot",
            hooks=[(HookPoint.POST_LLM_CALL, post_hook)],
        )

        with pytest.raises(AgentError):
            await agent.run("Hello", provider=provider)

        assert "post_llm" not in events


# ---------------------------------------------------------------------------
# Agent.run() — tool execution loop
# ---------------------------------------------------------------------------


def _multi_step_provider(
    *responses: ModelResponse,
) -> AsyncMock:
    """Create a provider that returns responses sequentially."""
    provider = AsyncMock()
    provider.complete = AsyncMock(side_effect=list(responses))
    return provider


class TestAgentToolLoop:
    async def test_single_tool_call(self) -> None:
        """Single tool call → execute → LLM returns text."""
        tc = ToolCall(id="tc-1", name="greet", arguments='{"name": "Alice"}')
        provider = _multi_step_provider(
            ModelResponse(
                content="",
                tool_calls=[tc],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
            ModelResponse(
                content="I greeted Alice for you!",
                usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
            ),
        )
        agent = Agent(name="bot", tools=[greet])

        output = await agent.run("Say hi to Alice", provider=provider)

        assert output.text == "I greeted Alice for you!"
        assert provider.complete.await_count == 2

        # Second call should include the tool result in messages
        second_call_msgs = provider.complete.call_args_list[1][0][0]
        tool_result_msgs = [m for m in second_call_msgs if m.role == "tool"]
        assert len(tool_result_msgs) == 1
        assert "Hello, Alice!" in tool_result_msgs[0].content

    async def test_multi_tool_parallel_call(self) -> None:
        """Multiple tool calls execute in parallel."""
        tc1 = ToolCall(id="tc-1", name="greet", arguments='{"name": "Alice"}')
        tc2 = ToolCall(id="tc-2", name="add", arguments='{"a": 2, "b": 3}')
        provider = _multi_step_provider(
            ModelResponse(
                content="",
                tool_calls=[tc1, tc2],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
            ModelResponse(
                content="Done with both tools.",
                usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
            ),
        )
        agent = Agent(name="bot", tools=[greet, add])

        output = await agent.run("Do both tasks", provider=provider)

        assert output.text == "Done with both tools."
        assert provider.complete.await_count == 2

        # Both tool results should be in the second call
        second_call_msgs = provider.complete.call_args_list[1][0][0]
        tool_result_msgs = [m for m in second_call_msgs if m.role == "tool"]
        assert len(tool_result_msgs) == 2

    async def test_tool_error_returned_not_propagated(self) -> None:
        """Tool errors are captured as ToolResult(error=...), not raised."""

        @tool
        def failing_tool(x: str) -> str:
            """A tool that always fails."""
            raise ValueError("boom!")

        tc = ToolCall(id="tc-1", name="failing_tool", arguments='{"x": "hi"}')
        provider = _multi_step_provider(
            ModelResponse(
                content="",
                tool_calls=[tc],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
            ModelResponse(
                content="The tool failed, sorry.",
                usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
            ),
        )
        agent = Agent(name="bot", tools=[failing_tool])

        output = await agent.run("Try it", provider=provider)

        assert output.text == "The tool failed, sorry."
        # The error should be in the tool result message
        second_call_msgs = provider.complete.call_args_list[1][0][0]
        tool_result_msgs = [m for m in second_call_msgs if m.role == "tool"]
        assert len(tool_result_msgs) == 1
        assert tool_result_msgs[0].error is not None
        assert "boom" in tool_result_msgs[0].error

    async def test_unknown_tool_returns_error(self) -> None:
        """Calling an unknown tool returns an error ToolResult."""
        tc = ToolCall(id="tc-1", name="nonexistent", arguments="{}")
        provider = _multi_step_provider(
            ModelResponse(
                content="",
                tool_calls=[tc],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
            ModelResponse(
                content="Tool not found.",
                usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
            ),
        )
        agent = Agent(name="bot", tools=[greet])

        output = await agent.run("Use nonexistent", provider=provider)

        assert output.text == "Tool not found."
        second_call_msgs = provider.complete.call_args_list[1][0][0]
        tool_result_msgs = [m for m in second_call_msgs if m.role == "tool"]
        assert tool_result_msgs[0].error is not None
        assert "unknown tool" in tool_result_msgs[0].error

    async def test_max_steps_enforcement(self) -> None:
        """Tool loop stops at max_steps even if LLM keeps returning tool calls."""
        tc = ToolCall(id="tc-1", name="greet", arguments='{"name": "Loop"}')
        # All responses have tool calls — never returns text
        responses = [
            ModelResponse(
                content="",
                tool_calls=[tc],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            )
            for _ in range(5)
        ]
        provider = _multi_step_provider(*responses)
        agent = Agent(name="bot", tools=[greet], max_steps=3)

        output = await agent.run("Loop forever", provider=provider)

        # Should stop after 3 steps, returning last tool-call output
        assert provider.complete.await_count == 3
        assert len(output.tool_calls) == 1

    async def test_tool_hooks_fired(self) -> None:
        """PRE_TOOL_CALL and POST_TOOL_CALL hooks fire for each tool."""
        events: list[str] = []

        async def pre_tool(**data: Any) -> None:
            events.append(f"pre:{data['tool_name']}")

        async def post_tool(**data: Any) -> None:
            events.append(f"post:{data['tool_name']}")

        tc = ToolCall(id="tc-1", name="greet", arguments='{"name": "Bob"}')
        provider = _multi_step_provider(
            ModelResponse(
                content="",
                tool_calls=[tc],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
            ModelResponse(
                content="Done.",
                usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
            ),
        )
        agent = Agent(
            name="bot",
            tools=[greet],
            hooks=[
                (HookPoint.PRE_TOOL_CALL, pre_tool),
                (HookPoint.POST_TOOL_CALL, post_tool),
            ],
        )

        await agent.run("Hello", provider=provider)

        assert events == ["pre:greet", "post:greet"]

    async def test_multi_step_tool_loop(self) -> None:
        """Agent executes multiple rounds of tool calls before final text."""
        tc1 = ToolCall(id="tc-1", name="greet", arguments='{"name": "A"}')
        tc2 = ToolCall(id="tc-2", name="add", arguments='{"a": 1, "b": 2}')
        provider = _multi_step_provider(
            ModelResponse(
                content="",
                tool_calls=[tc1],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
            ModelResponse(
                content="",
                tool_calls=[tc2],
                usage=Usage(input_tokens=15, output_tokens=5, total_tokens=20),
            ),
            ModelResponse(
                content="All done!",
                usage=Usage(input_tokens=20, output_tokens=5, total_tokens=25),
            ),
        )
        agent = Agent(name="bot", tools=[greet, add])

        output = await agent.run("Do everything", provider=provider)

        assert output.text == "All done!"
        assert provider.complete.await_count == 3

    async def test_no_tools_returns_immediately(self) -> None:
        """Agent with no tools returns LLM response without looping."""
        provider = _mock_provider(content="Direct answer.")
        agent = Agent(name="bot")

        output = await agent.run("Hello", provider=provider)

        assert output.text == "Direct answer."
        provider.complete.assert_awaited_once()


# ---------------------------------------------------------------------------
# Agent edge cases
# ---------------------------------------------------------------------------


class TestAgentEdgeCases:
    async def test_retry_during_tool_loop(self) -> None:
        """Retry logic works on the second LLM call (after tool execution)."""
        tc = ToolCall(id="tc-1", name="greet", arguments='{"name": "Eve"}')
        resp_tool = ModelResponse(
            content="",
            tool_calls=[tc],
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        resp_text = ModelResponse(
            content="Recovered!",
            usage=Usage(input_tokens=15, output_tokens=5, total_tokens=20),
        )
        provider = AsyncMock()
        # First call succeeds (returns tool call), second fails, third succeeds
        provider.complete = AsyncMock(side_effect=[resp_tool, RuntimeError("transient"), resp_text])
        agent = Agent(name="bot", tools=[greet])

        output = await agent.run("Hello", provider=provider, max_retries=3)

        assert output.text == "Recovered!"
        assert provider.complete.await_count == 3

    async def test_agent_with_handoffs_runs_normally(self) -> None:
        """Agent with handoffs declared still runs normally (no handoff triggered)."""
        helper = Agent(name="helper")
        provider = _mock_provider(content="I handled it myself.")
        agent = Agent(name="triage", handoffs=[helper])

        output = await agent.run("Help me", provider=provider)

        assert output.text == "I handled it myself."
        assert "helper" in agent.handoffs

    async def test_handoffs_do_not_appear_as_tools(self) -> None:
        """Handoff agents are not included in tool schemas."""
        helper = Agent(name="helper")
        agent = Agent(name="triage", tools=[greet], handoffs=[helper])

        schemas = agent.get_tool_schemas()

        # greet + retrieve_artifact + spawn_self + 7 context tools; handoff should not appear
        assert len(schemas) == 10
        names = [s["function"]["name"] for s in schemas]
        assert "greet" in names
        assert "helper" not in names

    async def test_sequential_tool_calls_accumulate_messages(self) -> None:
        """Each tool loop iteration appends assistant + tool result messages."""
        tc1 = ToolCall(id="tc-1", name="add", arguments='{"a": 1, "b": 2}')
        tc2 = ToolCall(id="tc-2", name="add", arguments='{"a": 3, "b": 4}')
        provider = _multi_step_provider(
            ModelResponse(
                content="",
                tool_calls=[tc1],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
            ModelResponse(
                content="",
                tool_calls=[tc2],
                usage=Usage(input_tokens=20, output_tokens=5, total_tokens=25),
            ),
            ModelResponse(
                content="Results: 3 and 7",
                usage=Usage(input_tokens=30, output_tokens=10, total_tokens=40),
            ),
        )
        agent = Agent(name="bot", tools=[add])

        output = await agent.run("Add things", provider=provider)

        assert output.text == "Results: 3 and 7"
        # Third call should have: user + assistant(tc1) + tool(result1) + assistant(tc2) + tool(result2)
        third_call_msgs = provider.complete.call_args_list[2][0][0]
        assistant_msgs = [m for m in third_call_msgs if m.role == "assistant"]
        tool_msgs = [m for m in third_call_msgs if m.role == "tool"]
        assert len(assistant_msgs) == 2
        assert len(tool_msgs) == 2

    async def test_max_steps_one(self) -> None:
        """Agent with max_steps=1 makes exactly one LLM call."""
        tc = ToolCall(id="tc-1", name="greet", arguments='{"name": "X"}')
        provider = _multi_step_provider(
            ModelResponse(
                content="",
                tool_calls=[tc],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
        )
        agent = Agent(name="bot", tools=[greet], max_steps=1)

        output = await agent.run("Go", provider=provider)

        assert provider.complete.await_count == 1
        assert len(output.tool_calls) == 1

    async def test_empty_input(self) -> None:
        """Agent handles empty string input gracefully."""
        provider = _mock_provider(content="You said nothing.")
        agent = Agent(name="bot")

        output = await agent.run("", provider=provider)

        assert output.text == "You said nothing."

    async def test_usage_from_final_response(self) -> None:
        """Output usage comes from the final LLM call, not accumulated."""
        tc = ToolCall(id="tc-1", name="greet", arguments='{"name": "Z"}')
        provider = _multi_step_provider(
            ModelResponse(
                content="",
                tool_calls=[tc],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
            ModelResponse(
                content="Done.",
                usage=Usage(input_tokens=50, output_tokens=20, total_tokens=70),
            ),
        )
        agent = Agent(name="bot", tools=[greet])

        output = await agent.run("Go", provider=provider)

        assert output.usage.input_tokens == 50
        assert output.usage.output_tokens == 20

    async def test_tool_with_no_arguments(self) -> None:
        """Tool called with empty arguments dict works."""

        @tool
        def ping() -> str:
            """Ping."""
            return "pong"

        tc = ToolCall(id="tc-1", name="ping", arguments="{}")
        provider = _multi_step_provider(
            ModelResponse(
                content="",
                tool_calls=[tc],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
            ModelResponse(
                content="Pong received.",
                usage=Usage(input_tokens=20, output_tokens=5, total_tokens=25),
            ),
        )
        agent = Agent(name="bot", tools=[ping])

        output = await agent.run("Ping", provider=provider)

        assert output.text == "Pong received."
        second_call_msgs = provider.complete.call_args_list[1][0][0]
        tool_msgs = [m for m in second_call_msgs if m.role == "tool"]
        assert "pong" in tool_msgs[0].content


# ---------------------------------------------------------------------------
# Agent memory defaults (US-014)
# ---------------------------------------------------------------------------


class TestAgentMemoryDefaults:
    def test_default_memory_auto_created(self) -> None:
        """Agent() auto-creates AgentMemory when exo-memory is installed."""
        try:
            from exo.memory.base import AgentMemory  # pyright: ignore[reportMissingImports]
        except ImportError:
            pytest.skip("exo-memory not installed")

        agent = Agent(name="bot")
        assert isinstance(agent.memory, AgentMemory)

    def test_default_memory_has_short_and_long_term(self) -> None:
        """Auto-created AgentMemory contains ShortTermMemory and a MemoryStore for long_term."""
        try:
            from exo.memory.base import AgentMemory  # pyright: ignore[reportMissingImports]
            from exo.memory.short_term import (
                ShortTermMemory,  # pyright: ignore[reportMissingImports]
            )
        except ImportError:
            pytest.skip("exo-memory not installed")

        agent = Agent(name="bot")
        assert isinstance(agent.memory, AgentMemory)
        assert isinstance(agent.memory.short_term, ShortTermMemory)
        # long_term is SQLiteMemoryStore (no chromadb) or ChromaVectorMemoryStore (chromadb installed)
        assert agent.memory.long_term is not None
        assert hasattr(agent.memory.long_term, "search")

    def test_explicit_none_disables_memory(self) -> None:
        """Agent(memory=None) fully disables memory — no persistence."""
        agent = Agent(name="bot", memory=None)
        assert agent.memory is None
        assert agent._memory_persistence is None

    def test_explicit_agent_memory_used(self) -> None:
        """Agent(memory=AgentMemory(...)) uses the provided backends."""
        try:
            from exo.memory.base import AgentMemory  # pyright: ignore[reportMissingImports]
            from exo.memory.short_term import (
                ShortTermMemory,  # pyright: ignore[reportMissingImports]
            )
        except ImportError:
            pytest.skip("exo-memory not installed")

        short = ShortTermMemory()
        long_store = ShortTermMemory()
        custom = AgentMemory(short_term=short, long_term=long_store)
        agent = Agent(name="bot", memory=custom)
        assert agent.memory is custom
        assert agent.memory.short_term is short
        assert agent.memory.long_term is long_store

    def test_conversation_id_initially_none(self) -> None:
        """Agent has conversation_id=None on creation."""
        agent = Agent(name="bot")
        assert agent.conversation_id is None

    def test_auto_created_memory_attaches_persistence(self) -> None:
        """Auto-created AgentMemory wires up MemoryPersistence hooks."""
        try:
            from exo.memory.base import AgentMemory  # pyright: ignore[reportMissingImports]
            from exo.memory.persistence import (
                MemoryPersistence,  # pyright: ignore[reportMissingImports]
            )
        except ImportError:
            pytest.skip("exo-memory not installed")

        agent = Agent(name="bot")
        assert isinstance(agent.memory, AgentMemory)
        assert isinstance(agent._memory_persistence, MemoryPersistence)

    def test_explicit_none_memory_conversation_id_still_none(self) -> None:
        """Agent(memory=None) still has conversation_id=None."""
        agent = Agent(name="bot", memory=None)
        assert agent.conversation_id is None


# ---------------------------------------------------------------------------
# Long-term knowledge injection (US-022)
# ---------------------------------------------------------------------------


class TestLongTermKnowledgeInjection:
    """Tests for _inject_long_term_knowledge() helper and its wiring into Agent.run()."""

    async def test_vector_store_results_injected_into_system_message(self) -> None:
        """When long_term returns results, they appear in the system message."""
        from unittest.mock import AsyncMock

        from exo.agent import _inject_long_term_knowledge  # pyright: ignore[reportMissingImports]

        try:
            from exo.memory.base import (  # pyright: ignore[reportMissingImports]
                AgentMemory,
                HumanMemory,
            )
            from exo.memory.short_term import (
                ShortTermMemory,  # pyright: ignore[reportMissingImports]
            )
        except ImportError:
            pytest.skip("exo-memory not installed")

        from exo.types import SystemMessage, UserMessage

        # Mock long_term store that returns a hit
        mock_lt = AsyncMock()
        mock_lt.search = AsyncMock(
            return_value=[HumanMemory(content="Paris is the capital of France")]
        )
        mem = AgentMemory(short_term=ShortTermMemory(), long_term=mock_lt)

        msg_list = [
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="hello"),
        ]
        result = await _inject_long_term_knowledge(mem, "What is the capital of France?", msg_list)

        # System message should now contain the knowledge block
        sys_msgs = [m for m in result if isinstance(m, SystemMessage)]
        assert sys_msgs, "No SystemMessage in result"
        assert "<knowledge>" in sys_msgs[0].content
        assert "Paris is the capital of France" in sys_msgs[0].content
        assert "[long_term_memory]" in sys_msgs[0].content

    async def test_empty_results_no_injection(self) -> None:
        """When long_term returns no results, msg_list is unchanged."""
        from unittest.mock import AsyncMock

        from exo.agent import _inject_long_term_knowledge  # pyright: ignore[reportMissingImports]

        try:
            from exo.memory.base import AgentMemory  # pyright: ignore[reportMissingImports]
            from exo.memory.short_term import (
                ShortTermMemory,  # pyright: ignore[reportMissingImports]
            )
        except ImportError:
            pytest.skip("exo-memory not installed")

        from exo.types import SystemMessage, UserMessage

        mock_lt = AsyncMock()
        mock_lt.search = AsyncMock(return_value=[])
        mem = AgentMemory(short_term=ShortTermMemory(), long_term=mock_lt)

        msg_list = [SystemMessage(content="You are a bot."), UserMessage(content="hello")]
        result = await _inject_long_term_knowledge(mem, "query", msg_list)
        assert result == msg_list

    async def test_search_exception_returns_unchanged_msg_list(self) -> None:
        """When long_term.search() raises, msg_list is returned unchanged."""
        from unittest.mock import AsyncMock

        from exo.agent import _inject_long_term_knowledge  # pyright: ignore[reportMissingImports]

        try:
            from exo.memory.base import AgentMemory  # pyright: ignore[reportMissingImports]
            from exo.memory.short_term import (
                ShortTermMemory,  # pyright: ignore[reportMissingImports]
            )
        except ImportError:
            pytest.skip("exo-memory not installed")

        from exo.types import UserMessage

        mock_lt = AsyncMock()
        mock_lt.search = AsyncMock(side_effect=RuntimeError("embedding failed"))
        mem = AgentMemory(short_term=ShortTermMemory(), long_term=mock_lt)

        msg_list = [UserMessage(content="hello")]
        result = await _inject_long_term_knowledge(mem, "query", msg_list)
        assert result == msg_list

    async def test_no_system_message_creates_new_one(self) -> None:
        """When no SystemMessage exists, a new one is inserted with the knowledge block."""
        from unittest.mock import AsyncMock

        from exo.agent import _inject_long_term_knowledge  # pyright: ignore[reportMissingImports]

        try:
            from exo.memory.base import (  # pyright: ignore[reportMissingImports]
                AgentMemory,
                HumanMemory,
            )
            from exo.memory.short_term import (
                ShortTermMemory,  # pyright: ignore[reportMissingImports]
            )
        except ImportError:
            pytest.skip("exo-memory not installed")

        from exo.types import SystemMessage, UserMessage

        mock_lt = AsyncMock()
        mock_lt.search = AsyncMock(return_value=[HumanMemory(content="Some fact")])
        mem = AgentMemory(short_term=ShortTermMemory(), long_term=mock_lt)

        msg_list = [UserMessage(content="hello")]
        result = await _inject_long_term_knowledge(mem, "query", msg_list)

        sys_msgs = [m for m in result if isinstance(m, SystemMessage)]
        assert sys_msgs, "No SystemMessage created"
        assert "<knowledge>" in sys_msgs[0].content
        assert "Some fact" in sys_msgs[0].content

    async def test_none_memory_returns_unchanged(self) -> None:
        """When agent.memory is None, msg_list is unchanged."""
        from exo.agent import _inject_long_term_knowledge  # pyright: ignore[reportMissingImports]
        from exo.types import UserMessage

        msg_list = [UserMessage(content="hello")]
        result = await _inject_long_term_knowledge(None, "query", msg_list)
        assert result == msg_list

    async def test_keyword_search_used_for_sqlite_store(self) -> None:
        """SQLiteMemoryStore (keyword search) works correctly as long_term backend."""
        from unittest.mock import AsyncMock

        from exo.agent import _inject_long_term_knowledge  # pyright: ignore[reportMissingImports]

        try:
            from exo.memory.base import (  # pyright: ignore[reportMissingImports]
                AgentMemory,
                AIMemory,
            )
            from exo.memory.short_term import (
                ShortTermMemory,  # pyright: ignore[reportMissingImports]
            )
        except ImportError:
            pytest.skip("exo-memory not installed")

        from exo.types import SystemMessage, UserMessage

        mock_sqlite = AsyncMock()
        mock_sqlite.search = AsyncMock(return_value=[AIMemory(content="The sky is blue")])
        mem = AgentMemory(short_term=ShortTermMemory(), long_term=mock_sqlite)

        msg_list = [SystemMessage(content="You are a bot."), UserMessage(content="hi")]
        result = await _inject_long_term_knowledge(mem, "sky color", msg_list)

        sys_msgs = [m for m in result if isinstance(m, SystemMessage)]
        assert "<knowledge>" in sys_msgs[0].content
        assert "The sky is blue" in sys_msgs[0].content
        # Verify search was called with the user input as query
        mock_sqlite.search.assert_called_once_with(query="sky color", limit=5)


class TestDefaultLongTermCreation:
    """Tests for _make_default_long_term() — always SQLite, no chromadb dependency."""

    def test_default_long_term_is_sqlite(self) -> None:
        """Default long-term store is SQLiteMemoryStore in a temp dir."""
        try:
            from exo.memory.backends.sqlite import (
                SQLiteMemoryStore,  # pyright: ignore[reportMissingImports]
            )
        except ImportError:
            pytest.skip("exo-memory not installed")

        from exo.agent import _make_default_long_term  # pyright: ignore[reportMissingImports]

        store = _make_default_long_term()
        assert isinstance(store, SQLiteMemoryStore)
        # Path should be inside a temp directory (lazy — file not created yet)
        import os

        assert store.db_path.endswith("memory.db")
        assert not os.path.exists(store.db_path), "SQLite file should not be created at construction"

    def test_default_long_term_with_explicit_path(self, tmp_path: Any) -> None:
        """Explicit db_path is respected."""
        try:
            from exo.memory.backends.sqlite import (
                SQLiteMemoryStore,  # pyright: ignore[reportMissingImports]
            )
        except ImportError:
            pytest.skip("exo-memory not installed")

        from exo.agent import _make_default_long_term  # pyright: ignore[reportMissingImports]

        db_file = str(tmp_path / "custom.db")
        store = _make_default_long_term(db_path=db_file)
        assert isinstance(store, SQLiteMemoryStore)
        assert store.db_path == db_file


# ---------------------------------------------------------------------------
# Agent history persistence (US-015)
# ---------------------------------------------------------------------------


class TestAgentHistoryPersistence:
    async def test_second_run_picks_up_history(self) -> None:
        """Second run() on same agent instance sees messages from the first run."""
        try:
            from exo.memory.base import AgentMemory  # pyright: ignore[reportMissingImports]
            from exo.memory.short_term import (
                ShortTermMemory,  # pyright: ignore[reportMissingImports]
            )
        except ImportError:
            pytest.skip("exo-memory not installed")

        # Use in-memory stores for test isolation
        memory = AgentMemory(short_term=ShortTermMemory(), long_term=ShortTermMemory())
        provider1 = _mock_provider(content="First reply")
        agent = Agent(name="history-bot", memory=memory)

        # First run
        await agent.run("First input", provider=provider1)

        # Second run — capture messages sent to the LLM
        provider2 = _mock_provider(content="Second reply")
        original_complete = provider2.complete
        captured_messages: list[Any] = []

        async def capturing_complete(msgs: Any, **kwargs: Any) -> Any:
            captured_messages.extend(msgs)
            return await original_complete(msgs, **kwargs)

        provider2.complete = capturing_complete
        await agent.run("Second input", provider=provider2)

        # History from the first run should be present in the second run's LLM call
        user_msgs = [m for m in captured_messages if m.role == "user"]
        assistant_msgs = [m for m in captured_messages if m.role == "assistant"]
        user_contents = [m.content for m in user_msgs]
        assert "First input" in user_contents, "First run's user message not found in history"
        assert "Second input" in user_contents, "Current input not found"
        assert len(assistant_msgs) >= 1, "First run's assistant reply not found in history"

    async def test_conversation_id_auto_assigned_on_first_run(self) -> None:
        """conversation_id is auto-assigned (UUID4) on first run when memory is set."""
        try:
            from exo.memory.base import AgentMemory  # pyright: ignore[reportMissingImports]
            from exo.memory.short_term import (
                ShortTermMemory,  # pyright: ignore[reportMissingImports]
            )
        except ImportError:
            pytest.skip("exo-memory not installed")

        memory = AgentMemory(short_term=ShortTermMemory(), long_term=ShortTermMemory())
        agent = Agent(name="bot", memory=memory)
        assert agent.conversation_id is None

        provider = _mock_provider(content="hello")
        await agent.run("hi", provider=provider)

        assert agent.conversation_id is not None
        assert len(agent.conversation_id) > 0

    async def test_conversation_id_param_overrides_per_call_only(self) -> None:
        """conversation_id kwarg overrides instance conversation_id for that call only."""
        try:
            from exo.memory.base import AgentMemory  # pyright: ignore[reportMissingImports]
            from exo.memory.short_term import (
                ShortTermMemory,  # pyright: ignore[reportMissingImports]
            )
        except ImportError:
            pytest.skip("exo-memory not installed")

        memory = AgentMemory(short_term=ShortTermMemory(), long_term=ShortTermMemory())
        agent = Agent(name="bot", memory=memory)

        provider = _mock_provider(content="reply")
        # Run with explicit conversation_id
        await agent.run("hi", provider=provider, conversation_id="my-conv-123")

        # Instance conversation_id should NOT be updated when override is passed
        assert agent.conversation_id != "my-conv-123"

    async def test_explicit_messages_merged_with_history(self) -> None:
        """Explicit messages= are merged after loaded history, not replaced."""
        try:
            from exo.memory.base import AgentMemory  # pyright: ignore[reportMissingImports]
            from exo.memory.short_term import (
                ShortTermMemory,  # pyright: ignore[reportMissingImports]
            )
        except ImportError:
            pytest.skip("exo-memory not installed")

        from exo.types import AssistantMessage, UserMessage

        memory = AgentMemory(short_term=ShortTermMemory(), long_term=ShortTermMemory())
        provider1 = _mock_provider(content="First reply")
        agent = Agent(name="bot", memory=memory)

        # First run to populate history
        await agent.run("First input", provider=provider1)

        # Second run with explicit messages
        provider2 = _mock_provider(content="Second reply")
        original_complete = provider2.complete
        captured_messages: list[Any] = []

        async def capture(msgs: Any, **kwargs: Any) -> Any:
            captured_messages.extend(msgs)
            return await original_complete(msgs, **kwargs)

        provider2.complete = capture
        explicit_msgs = [
            UserMessage(content="explicit-prior"),
            AssistantMessage(content="explicit-reply"),
        ]
        await agent.run("New input", messages=explicit_msgs, provider=provider2)

        user_contents = [m.content for m in captured_messages if m.role == "user"]
        # Both DB history and explicit messages must be present
        assert "First input" in user_contents
        assert "explicit-prior" in user_contents
        assert "New input" in user_contents

    async def test_no_memory_no_history_loading(self) -> None:
        """Agent with memory=None does not attempt history loading."""
        provider = _mock_provider(content="Direct answer")
        agent = Agent(name="bot", memory=None)

        captured_messages: list[Any] = []
        original_complete = provider.complete

        async def capture(msgs: Any, **kwargs: Any) -> Any:
            captured_messages.extend(msgs)
            return await original_complete(msgs, **kwargs)

        provider.complete = capture
        await agent.run("Hello", provider=provider)

        # conversation_id still None after run (memory disabled)
        assert agent.conversation_id is None
        # Only the current user message should be present (no history)
        user_msgs = [m for m in captured_messages if m.role == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0].content == "Hello"


# ---------------------------------------------------------------------------
# Context defaults (US-016)
# ---------------------------------------------------------------------------


class TestAgentContextDefaults:
    def test_auto_creates_copilot_context(self) -> None:
        """Agent with no context params auto-creates Context wrapping ContextConfig defaults."""
        try:
            from exo.context.config import (  # pyright: ignore[reportMissingImports]
                ContextConfig,
                OverflowStrategy,
            )
            from exo.context.context import Context  # pyright: ignore[reportMissingImports]
        except ImportError:
            pytest.skip("exo-context not installed")

        agent = Agent(name="bot")
        assert isinstance(agent.context, Context)
        assert isinstance(agent.context.config, ContextConfig)
        assert agent.context.config.overflow == OverflowStrategy.SUMMARIZE
        assert agent.context.config.limit == 20
        assert agent._context_is_auto is True

    def test_context_none_disables_context(self) -> None:
        """Agent(context=None) fully disables the context engine."""
        agent = Agent(name="bot", context=None)
        assert agent.context is None
        assert agent._context_is_auto is False

    def test_context_mode_none_disables_context(self) -> None:
        """Agent(context_mode=None) fully disables the context engine."""
        agent = Agent(name="bot", context_mode=None)
        assert agent.context is None
        assert agent._context_is_auto is False

    def test_context_mode_pilot(self) -> None:
        """Agent(context_mode='pilot') creates Context wrapping ContextConfig(limit=100)."""
        try:
            from exo.context.config import ContextConfig  # pyright: ignore[reportMissingImports]
            from exo.context.context import Context  # pyright: ignore[reportMissingImports]
        except ImportError:
            pytest.skip("exo-context not installed")

        agent = Agent(name="bot", context_mode="pilot")
        assert isinstance(agent.context, Context)
        assert isinstance(agent.context.config, ContextConfig)
        assert agent.context.config.limit == 100
        assert agent._context_is_auto is False

    def test_context_mode_navigator(self) -> None:
        """Agent(context_mode='navigator') creates Context wrapping ContextConfig(limit=10, cache=True)."""
        try:
            from exo.context.config import ContextConfig  # pyright: ignore[reportMissingImports]
            from exo.context.context import Context  # pyright: ignore[reportMissingImports]
        except ImportError:
            pytest.skip("exo-context not installed")

        agent = Agent(name="bot", context_mode="navigator")
        assert isinstance(agent.context, Context)
        assert isinstance(agent.context.config, ContextConfig)
        assert agent.context.config.limit == 10
        assert agent.context.config.cache is True

    def test_explicit_context_takes_precedence_over_context_mode(self) -> None:
        """When context= is given, it takes precedence over context_mode."""
        try:
            from exo.context.config import ContextConfig  # pyright: ignore[reportMissingImports]
        except ImportError:
            pytest.skip("exo-context not installed")

        custom_ctx = ContextConfig(limit=10, overflow="summarize", cache=True)
        agent = Agent(name="bot", context=custom_ctx, context_mode="pilot")
        assert isinstance(agent.context, ContextConfig)
        assert agent.context.limit == 10
        assert agent.context.cache is True

    def test_explicit_context_config_accepted(self) -> None:
        """Agent(context=ContextConfig(...)) uses the provided config directly."""
        try:
            from exo.context.config import ContextConfig  # pyright: ignore[reportMissingImports]
        except ImportError:
            pytest.skip("exo-context not installed")

        cfg = ContextConfig(limit=50)
        agent = Agent(name="bot", context=cfg)
        assert agent.context is cfg
        assert agent.context.limit == 50
        assert agent._context_is_auto is False

    def test_auto_context_not_raised_in_to_dict(self) -> None:
        """Auto-created context does not prevent serialization."""
        agent = Agent(name="bot")  # auto-creates context
        # Should not raise even though context is set (it's auto-created)
        d = agent.to_dict()
        assert d["name"] == "bot"

    def test_explicit_context_raises_in_to_dict(self) -> None:
        """Explicitly provided ContextConfig raises ValueError in to_dict()."""
        try:
            from exo.context.config import ContextConfig  # pyright: ignore[reportMissingImports]
        except ImportError:
            pytest.skip("exo-context not installed")

        agent = Agent(name="bot", context=ContextConfig())
        with pytest.raises(ValueError, match="context engine"):
            agent.to_dict()
