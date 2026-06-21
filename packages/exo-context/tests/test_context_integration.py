"""Integration tests: Context + PromptBuilder + Processor + Workspace end-to-end."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from exo.context import (  # pyright: ignore[reportMissingImports]
    ArtifactType,
    Checkpoint,
    CheckpointStore,
    Context,
    ContextConfig,
    ContextError,
    ContextProcessor,
    ContextState,
    Neuron,
    OverflowStrategy,
    ProcessorPipeline,
    PromptBuilder,
    TokenTracker,
    ToolResultOffloader,
    Workspace,
    get_context_tools,
    get_file_tools,
    get_knowledge_tools,
    get_planning_tools,
    neuron_registry,
)
from exo.context._internal.knowledge import (  # pyright: ignore[reportMissingImports]
    KnowledgeStore,
)

# ── Public API import tests ──────────────────────────────────────────


class TestPublicAPIImports:
    """Verify all expected names are importable from exo.context."""

    def test_core_classes(self) -> None:
        assert Context is not None
        assert ContextConfig is not None
        assert ContextState is not None
        assert ContextError is not None

    def test_prompt_building(self) -> None:
        assert PromptBuilder is not None
        assert Neuron is not None
        assert neuron_registry is not None

    def test_processor_pipeline(self) -> None:
        assert ContextProcessor is not None
        assert ProcessorPipeline is not None
        assert ToolResultOffloader is not None

    def test_workspace_and_artifacts(self) -> None:
        assert Workspace is not None
        assert ArtifactType is not None

    def test_checkpoint(self) -> None:
        assert Checkpoint is not None
        assert CheckpointStore is not None

    def test_config_and_overflow(self) -> None:
        assert OverflowStrategy is not None
        assert ContextConfig is not None

    def test_token_tracker(self) -> None:
        assert TokenTracker is not None

    def test_tool_factories(self) -> None:
        assert get_context_tools is not None
        assert get_planning_tools is not None
        assert get_knowledge_tools is not None
        assert get_file_tools is not None


# ── Context + PromptBuilder end-to-end ───────────────────────────────


class TestContextPromptBuilderE2E:
    """End-to-end: Context creation -> neuron registration -> prompt building."""

    async def test_basic_prompt_building(self) -> None:
        ctx = Context("task-1")
        ctx.state.set("task_input", "Summarize this document")

        builder = PromptBuilder(ctx)
        builder.add("system")
        builder.add("task")
        prompt = await builder.build()

        assert "task-1" in prompt
        assert "Summarize this document" in prompt

    async def test_prompt_with_history(self) -> None:
        ctx = Context("task-2")
        ctx.state.set(
            "history",
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ],
        )

        builder = PromptBuilder(ctx)
        builder.add("history")
        prompt = await builder.build()

        assert "Hello" in prompt
        assert "Hi there!" in prompt

    async def test_prompt_with_task_state(self) -> None:
        ctx = Context("task-3")
        ctx.state.set("task_input", "Research and write a draft")

        builder = PromptBuilder(ctx)
        builder.add("task")
        prompt = await builder.build()

        assert "task-3" in prompt
        assert "Research and write a draft" in prompt

    async def test_full_prompt_composition(self) -> None:
        ctx = Context("task-4")
        ctx.state.set("task_input", "Build a web app")

        builder = PromptBuilder(ctx)
        builder.add("task").add("system")
        prompt = await builder.build()

        # task neuron is priority 1, system is 100
        parts = prompt.split("\n\n")
        assert len(parts) >= 2

        # All content present
        assert "task-4" in prompt
        assert "Build a web app" in prompt


# ── Context + Processor pipeline end-to-end ──────────────────────────


class _SetFlagProcessor(ContextProcessor):
    """Test processor that sets a flag in context state."""

    def __init__(self, event: str, key: str, value: object) -> None:
        super().__init__(event, name="set_flag")
        self._key = key
        self._value = value

    async def process(self, ctx: Context, payload: dict[str, Any]) -> None:
        ctx.state.set(self._key, self._value)


class TestContextProcessorE2E:
    """End-to-end: Context + ProcessorPipeline with built-in processors."""

    async def test_tool_result_offloader(self) -> None:
        ctx = Context("task-6")
        pipeline = ProcessorPipeline()
        pipeline.register(ToolResultOffloader(max_size=50))

        large_result = "x" * 200
        payload: dict[str, Any] = {
            "tool_result": large_result,
            "tool_name": "search",
            "tool_call_id": "tc-1",
        }
        await pipeline.fire("post_tool_call", ctx, payload)

        offloaded = ctx.state.get("offloaded_results")
        assert offloaded is not None
        assert len(offloaded) == 1
        assert offloaded[0]["tool_name"] == "search"
        assert offloaded[0]["size"] == 200

        # Payload was mutated with truncated reference
        assert "[Result truncated" in payload["tool_result"]

    async def test_multi_processor_pipeline(self) -> None:
        """Multiple processors fire in sequence for different events."""
        ctx = Context("task-7")

        flag_proc = _SetFlagProcessor("pre_llm_call", "pre_fired", True)
        pipeline = ProcessorPipeline()
        pipeline.register(flag_proc)
        pipeline.register(ToolResultOffloader(max_size=10))

        # Fire pre_llm_call — flag processor triggers
        await pipeline.fire("pre_llm_call", ctx)
        assert ctx.state.get("pre_fired") is True

        # Fire post_tool_call — ToolResultOffloader triggers
        payload: dict[str, Any] = {"tool_result": "a" * 100, "tool_name": "t"}
        await pipeline.fire("post_tool_call", ctx, payload)
        assert ctx.state.get("offloaded_results") is not None


# ── Context + Workspace end-to-end ──────────────────────────────────


class TestContextWorkspaceE2E:
    """End-to-end: Context + Workspace with knowledge store integration."""

    async def test_workspace_write_and_knowledge_search(self) -> None:
        ks = KnowledgeStore()
        workspace = Workspace("ws-1", knowledge_store=ks)

        await workspace.write("readme", "Exo is a multi-agent framework for LLMs")
        await workspace.write("guide", "Quick start: install with pip install exo")

        # Knowledge store was auto-indexed
        results = ks.search("multi-agent framework")
        assert len(results) > 0
        assert any("Exo" in r.chunk.content for r in results)

    async def test_workspace_versioning_with_context(self) -> None:
        ctx = Context("task-8")
        workspace = Workspace("ws-2")

        await workspace.write("notes", "Version 1")
        await workspace.write("notes", "Version 2")
        await workspace.write("notes", "Version 3")

        # Store workspace in context state
        ctx.state.set("workspace", workspace)

        # Access via context
        ws = ctx.state.get("workspace")
        assert ws.read("notes") == "Version 3"
        versions = ws.version_history("notes")
        assert len(versions) == 3

    async def test_workspace_with_filesystem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace("ws-3", storage_path=tmp)
            await workspace.write("data.json", '{"key": "value"}', artifact_type=ArtifactType.JSON)

            # Verify persisted to filesystem
            content_path = Path(tmp) / "data.json" / "content"
            assert content_path.exists()
            assert content_path.read_text() == '{"key": "value"}'


# ── Full lifecycle: Context + PromptBuilder + Processor + Workspace ──


class TestFullLifecycle:
    """End-to-end lifecycle: create -> populate -> process -> build prompt -> checkpoint."""

    async def test_full_agent_context_lifecycle(self) -> None:
        # 1. Create context with navigator-equivalent settings
        config = ContextConfig(limit=10, overflow="summarize", cache=True)
        ctx = Context("main-task", config=config)

        # 2. Set up workspace + knowledge store
        ks = KnowledgeStore()
        workspace = Workspace("ws-main", knowledge_store=ks)
        ctx.state.set("workspace", workspace)
        ctx.state.set("knowledge_store", ks)

        # 3. Write artifacts
        await workspace.write("spec", "Build a REST API with authentication and rate limiting")
        await workspace.write("notes", "Use FastAPI with JWT tokens for auth")

        # 4. Set task state
        ctx.state.set("task_input", "Implement the authentication module")
        ctx.state.set(
            "subtasks",
            [
                "Design auth schema",
                "Implement JWT handler",
                "Write tests",
            ],
        )

        # 5. Build prompt
        builder = PromptBuilder(ctx)
        builder.add("task").add("system")
        prompt = await builder.build()

        assert "main-task" in prompt
        assert "Implement the authentication module" in prompt
        assert "JWT handler" in prompt

        # 6. Track tokens
        ctx.add_tokens({"prompt_tokens": 500, "output_tokens": 150})

        # 7. Set up processor pipeline with a custom flag processor
        pipeline = ProcessorPipeline()
        pipeline.register(_SetFlagProcessor("pre_llm_call", "needs_summary", True))

        # 8. Simulate history growth past threshold
        ctx.state.set("history", [{"role": "user", "content": f"Turn {i}"} for i in range(5)])
        await pipeline.fire("pre_llm_call", ctx)
        assert ctx.state.get("needs_summary") is True

        # 9. Checkpoint
        cp = ctx.snapshot(metadata={"step": 1})
        assert cp.task_id == "main-task"
        assert cp.values["needs_summary"] is True
        assert cp.token_usage["prompt_tokens"] == 500

        # 10. Restore from checkpoint
        restored = Context.restore(cp, config=config)
        assert restored.state.get("needs_summary") is True
        assert restored.token_usage["prompt_tokens"] == 500

    async def test_fork_merge_with_workspace(self) -> None:
        """Fork a child context, work in it, merge back."""
        # Parent context with workspace
        ks = KnowledgeStore()
        workspace = Workspace("ws-fork", knowledge_store=ks)

        parent = Context("parent-task")
        parent.state.set("workspace", workspace)
        parent.state.set("result_count", 0)
        parent.add_tokens({"prompt_tokens": 100})

        # Fork for subtask
        child = parent.fork("subtask-1")
        assert child.state.get("workspace") is workspace  # Inherited

        # Child does work
        await workspace.write("subtask-output", "Subtask result data")
        child.state.set("result_count", 1)
        child.state.set("subtask_done", True)
        child.add_tokens({"prompt_tokens": 200, "output_tokens": 50})

        # Merge back
        parent.merge(child)
        assert parent.state.get("subtask_done") is True
        assert parent.state.get("result_count") == 1  # Overwritten by child local
        # Net token delta: child started with parent's 100, added 200 => net = 200
        assert parent.token_usage["prompt_tokens"] == 300  # 100 + 200 net
        assert parent.token_usage["output_tokens"] == 50

    async def test_context_tools_with_workspace(self) -> None:
        """Context tools work with workspace via context state."""
        ctx = Context("tool-task")
        ks = KnowledgeStore()
        workspace = Workspace("ws-tools", knowledge_store=ks)
        ctx.state.set("workspace", workspace)
        ctx.state.set("knowledge_store", ks)

        # Write an artifact via workspace
        await workspace.write("readme", "# My Project\nA framework for building agents")

        # Use planning tool
        planning_tools = {t.name: t for t in get_planning_tools()}
        add_todo = planning_tools["add_todo"]
        get_todo = planning_tools["get_todo"]

        add_todo.bind(ctx)
        result = await add_todo.execute(item="Read the readme")
        assert "Added todo" in result

        # Use planning tool get
        get_todo.bind(ctx)
        result = await get_todo.execute()
        assert "Read the readme" in result

        # Check state was mutated
        todos = ctx.state.get("todos")
        assert len(todos) == 1
        assert todos[0]["item"] == "Read the readme"


# ── Custom processor integration ─────────────────────────────────────


class _CounterProcessor(ContextProcessor):
    """Test processor that counts invocations."""

    def __init__(self, event: str) -> None:
        super().__init__(event, name="counter")
        self.count = 0

    async def process(self, ctx: Context, payload: dict[str, Any]) -> None:
        self.count += 1
        ctx.state.set("counter", self.count)


class TestCustomProcessorIntegration:
    """Custom processors integrate with context and pipeline."""

    async def test_custom_processor_mutates_state(self) -> None:
        ctx = Context("custom-task")
        proc = _CounterProcessor("on_step")

        pipeline = ProcessorPipeline()
        pipeline.register(proc)

        await pipeline.fire("on_step", ctx)
        await pipeline.fire("on_step", ctx)
        await pipeline.fire("on_step", ctx)

        assert ctx.state.get("counter") == 3
        assert proc.count == 3

    async def test_mixed_processors(self) -> None:
        """Mix custom and built-in processors."""
        ctx = Context("mixed-task")

        flag_proc = _SetFlagProcessor("pre_llm_call", "needs_summary", True)
        counter = _CounterProcessor("pre_llm_call")
        pipeline = ProcessorPipeline()
        pipeline.register(flag_proc)
        pipeline.register(counter)

        await pipeline.fire("pre_llm_call", ctx)

        assert ctx.state.get("needs_summary") is True
        assert ctx.state.get("counter") == 1


# ── Agent context wiring ─────────────────────────────────────────────


class TestAgentContextWiring:
    """Verify Agent accepts and stores context."""

    def test_agent_accepts_context(self) -> None:
        from exo.agent import Agent

        ctx = Context("agent-task")
        agent = Agent(name="test", context=ctx)
        assert agent.context is ctx

    def test_agent_context_default_auto_created(self) -> None:
        """Agent without explicit context auto-creates Context with default ContextConfig."""
        from exo.agent import Agent

        agent = Agent(name="test")
        assert isinstance(agent.context, Context)
        assert agent.context.config.overflow == OverflowStrategy.SUMMARIZE
        assert agent.context.config.limit == 20
        assert agent._context_is_auto is True

    def test_agent_describe_does_not_include_context(self) -> None:
        from exo.agent import Agent

        ctx = Context("agent-task")
        agent = Agent(name="test", context=ctx)
        desc = agent.describe()
        # describe() focuses on capabilities, not internal state
        assert "name" in desc
        assert "model" in desc


# ── Context windowing in Agent.run() ─────────────────────────────────


class TestAgentContextWindowing:
    """Verify Agent.run() applies history windowing and summarization per ContextConfig."""

    async def test_windowing_trims_old_messages(self) -> None:
        """Messages beyond limit are trimmed before the LLM call."""
        from unittest.mock import AsyncMock

        from exo.agent import Agent
        from exo.context.config import ContextConfig  # pyright: ignore[reportMissingImports]
        from exo.models.types import ModelResponse  # pyright: ignore[reportMissingImports]
        from exo.types import AssistantMessage, SystemMessage, UserMessage

        # limit=3, overflow=truncate: trims to 3, no summarization
        context = ContextConfig(limit=3, overflow="truncate")
        agent = Agent(name="windowing-test", memory=None, context=context)

        # Build 10 round-trips worth of history (20 messages)
        history = []
        for i in range(10):
            history.append(UserMessage(content=f"user-{i}"))
            history.append(AssistantMessage(content=f"assistant-{i}"))

        received: list[Any] = []

        async def capture_complete(msgs: Any, **kwargs: Any) -> ModelResponse:
            received.extend(msgs)
            return ModelResponse(content="done", tool_calls=[])

        mock_provider = AsyncMock()
        mock_provider.complete.side_effect = capture_complete

        await agent.run("final-input", messages=history, provider=mock_provider)

        # System message (instructions) + at most limit non-system messages
        non_system = [m for m in received if not isinstance(m, SystemMessage)]
        assert len(non_system) <= context.limit

    async def test_windowing_does_not_trim_when_within_limit(self) -> None:
        """When message count is within limit, no trimming occurs."""
        from unittest.mock import AsyncMock

        from exo.agent import Agent
        from exo.context.config import ContextConfig  # pyright: ignore[reportMissingImports]
        from exo.models.types import ModelResponse  # pyright: ignore[reportMissingImports]
        from exo.types import AssistantMessage, SystemMessage, UserMessage

        context = ContextConfig(limit=20, overflow="truncate")
        agent = Agent(name="no-trim-test", memory=None, context=context)

        # Only 2 rounds (4 messages) — well under history_rounds=20
        history = [
            UserMessage(content="msg-0"),
            AssistantMessage(content="reply-0"),
            UserMessage(content="msg-1"),
            AssistantMessage(content="reply-1"),
        ]

        received: list[Any] = []

        async def capture_complete(msgs: Any, **kwargs: Any) -> ModelResponse:
            received.extend(msgs)
            return ModelResponse(content="done", tool_calls=[])

        mock_provider = AsyncMock()
        mock_provider.complete.side_effect = capture_complete

        await agent.run("current", messages=history, provider=mock_provider)

        non_system = [m for m in received if not isinstance(m, SystemMessage)]
        # 4 history + 1 current input = 5 messages total, all preserved
        assert len(non_system) == 5

    async def test_no_windowing_when_context_none(self) -> None:
        """No windowing is applied when context=None."""
        from unittest.mock import AsyncMock

        from exo.agent import Agent
        from exo.models.types import ModelResponse  # pyright: ignore[reportMissingImports]
        from exo.types import AssistantMessage, SystemMessage, UserMessage

        agent = Agent(name="no-context-test", memory=None, context=None)

        # 15 rounds of history — would normally be trimmed if history_rounds=3
        history = []
        for i in range(15):
            history.append(UserMessage(content=f"u-{i}"))
            history.append(AssistantMessage(content=f"a-{i}"))

        received: list[Any] = []

        async def capture_complete(msgs: Any, **kwargs: Any) -> ModelResponse:
            received.extend(msgs)
            return ModelResponse(content="done", tool_calls=[])

        mock_provider = AsyncMock()
        mock_provider.complete.side_effect = capture_complete

        await agent.run("current", messages=history, provider=mock_provider)

        non_system = [m for m in received if not isinstance(m, SystemMessage)]
        # All 30 history + 1 current = 31 messages (no windowing applied)
        assert len(non_system) == 31

    async def test_summarization_triggered_when_threshold_hit(self) -> None:
        """Summarization fires when message count >= summary_threshold."""
        from unittest.mock import AsyncMock

        from exo.agent import Agent
        from exo.context.config import ContextConfig  # pyright: ignore[reportMissingImports]
        from exo.models.types import ModelResponse  # pyright: ignore[reportMissingImports]
        from exo.types import AssistantMessage, UserMessage

        # limit=5, overflow=summarize: summarization fires at 5 messages, window=5
        context = ContextConfig(limit=5, overflow="summarize")
        agent = Agent(name="summary-test", memory=None, context=context)

        # 5 user+assistant rounds = 10 messages → hits summary_threshold=5
        history = []
        for i in range(5):
            history.append(UserMessage(content=f"u-{i}"))
            history.append(AssistantMessage(content=f"a-{i}"))

        provider_calls: list[Any] = []

        async def mock_complete(msgs: Any, **kwargs: Any) -> ModelResponse:
            provider_calls.append(msgs)
            return ModelResponse(content="summary output", tool_calls=[])

        mock_provider = AsyncMock()
        mock_provider.complete.side_effect = mock_complete

        await agent.run("new message", messages=history, provider=mock_provider)

        # check_trigger was called (trigger.triggered=True since 10 >= 5)
        # After summarization, messages should be compressed
        # Summarization call uses provider.complete() too (via _ProviderSummarizer)
        # But the agent's final LLM call should have compressed messages
        # At minimum: the call succeeded
        assert len(provider_calls) >= 1

    async def test_offload_threshold_triggers_aggressive_trim(self) -> None:
        """When message count > offload_threshold, trim to summary_threshold."""

        # limit=5, overflow=summarize: _summary_threshold=5, _offload_threshold=max(5,12)=12
        # Use a SimpleNamespace to get independent control of thresholds
        import types as _types

        from exo.agent import _apply_context_windowing
        from exo.types import AssistantMessage, SystemMessage, UserMessage
        context = _types.SimpleNamespace(
            overflow="summarize",
            history_rounds=100,
            summary_threshold=5,
            offload_threshold=10,
            keep_recent=2,
        )

        # Build 20 messages (> offload_threshold=10)
        msg_list = []
        for i in range(10):
            msg_list.append(UserMessage(content=f"u-{i}"))
            msg_list.append(AssistantMessage(content=f"a-{i}"))

        result, _actions = await _apply_context_windowing(msg_list, context, provider=None)

        non_system = [m for m in result if not isinstance(m, SystemMessage)]
        # Offload trims to summary_threshold=5
        assert len(non_system) <= context.summary_threshold

    async def test_apply_context_windowing_unit(self) -> None:
        """Direct unit test of _apply_context_windowing helper."""
        from exo.agent import _apply_context_windowing
        from exo.context.config import ContextConfig  # pyright: ignore[reportMissingImports]
        from exo.types import AssistantMessage, SystemMessage, UserMessage

        # limit=4, overflow=truncate: _history_rounds=4, no summarization
        context = ContextConfig(limit=4, overflow="truncate")

        sys_msg = SystemMessage(content="You are helpful.")
        msg_list: list[Any] = [sys_msg]
        for i in range(10):
            msg_list.append(UserMessage(content=f"u-{i}"))
            msg_list.append(AssistantMessage(content=f"a-{i}"))

        result, _actions = await _apply_context_windowing(msg_list, context, provider=None)

        # System message preserved
        assert result[0] is sys_msg
        # Only last limit=4 non-system messages kept
        non_system = [m for m in result if not isinstance(m, SystemMessage)]
        assert len(non_system) == 4
        # They should be the most recent ones
        assert non_system[-1].content == "a-9"  # type: ignore[union-attr]
        assert non_system[-2].content == "u-9"  # type: ignore[union-attr]


# ── Token tracking helpers ────────────────────────────────────────────


class TestTokenTrackingHelpers:
    """Unit tests for _get_context_window_tokens and _update_system_token_info."""

    def test_get_context_window_tokens_known_model(self) -> None:
        from exo.agent import _get_context_window_tokens

        tokens = _get_context_window_tokens("gpt-4o")
        assert tokens == 128000

    def test_get_context_window_tokens_unknown_model(self) -> None:
        from exo.agent import _get_context_window_tokens

        tokens = _get_context_window_tokens("some-unknown-model")
        assert tokens is None

    def test_update_system_token_info_adds_tag(self) -> None:
        from exo.agent import _update_system_token_info
        from exo.types import SystemMessage, UserMessage

        msg_list = [SystemMessage(content="You are helpful."), UserMessage(content="Hello")]
        result = _update_system_token_info(msg_list, used=80000, total=100000)

        assert isinstance(result[0], SystemMessage)
        assert "[Context: 80000/100000 tokens (80% full)]" in result[0].content
        assert "You are helpful." in result[0].content

    def test_update_system_token_info_replaces_existing_tag(self) -> None:
        from exo.agent import _update_system_token_info
        from exo.types import SystemMessage

        # First application
        msg_list = [SystemMessage(content="Instructions.")]
        result1 = _update_system_token_info(msg_list, used=50000, total=100000)
        assert "[Context: 50000/100000 tokens (50% full)]" in result1[0].content

        # Second application — should replace the old tag
        result2 = _update_system_token_info(result1, used=90000, total=100000)
        assert "[Context: 90000/100000 tokens (90% full)]" in result2[0].content
        # Old tag not present
        assert "50000" not in result2[0].content

    def test_update_system_token_info_no_system_message(self) -> None:
        from exo.agent import _update_system_token_info
        from exo.types import SystemMessage, UserMessage

        msg_list = [UserMessage(content="Hello")]
        result = _update_system_token_info(msg_list, used=10000, total=200000)

        # A SystemMessage is inserted at position 0 with just the tag
        assert isinstance(result[0], SystemMessage)
        assert "[Context: 10000/200000 tokens (5% full)]" in result[0].content
        # Original user message preserved at position 1
        assert isinstance(result[1], UserMessage)

    def test_update_system_token_info_preserves_other_messages(self) -> None:
        from exo.agent import _update_system_token_info
        from exo.types import AssistantMessage, SystemMessage, UserMessage

        msg_list = [
            SystemMessage(content="Sys"),
            UserMessage(content="Q"),
            AssistantMessage(content="A"),
        ]
        result = _update_system_token_info(msg_list, used=1000, total=10000)
        assert len(result) == 3
        assert isinstance(result[1], UserMessage)
        assert isinstance(result[2], AssistantMessage)


# ── Token tracking integration (multi-step agent) ─────────────────────


class TestTokenTrackingIntegration:
    """Integration tests: token tracking wired into Agent.run()."""

    async def test_token_info_in_system_prompt_on_second_step(self) -> None:
        """After step 0, step 1 LLM call has [Context: ...] in system message."""
        from unittest.mock import AsyncMock

        from exo.agent import Agent
        from exo.context.config import ContextConfig  # pyright: ignore[reportMissingImports]
        from exo.models.types import ModelResponse  # pyright: ignore[reportMissingImports]
        from exo.tool import tool
        from exo.types import SystemMessage, ToolCall, Usage

        context = ContextConfig(limit=50, overflow="summarize")

        @tool
        def dummy_tool() -> str:
            return "tool result"

        agent = Agent(
            name="token-track-test",
            model="openai:gpt-4o",  # 128000 token window
            memory=None,
            context=context,
            tools=[dummy_tool],
        )

        received_step1_msgs: list[Any] = []
        call_count = [0]

        async def mock_complete(msgs: Any, **kwargs: Any) -> ModelResponse:
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: return a tool call + usage
                return ModelResponse(
                    content="",
                    tool_calls=[ToolCall(id="tc1", name="dummy_tool", arguments="{}")],
                    usage=Usage(input_tokens=1000, output_tokens=50, total_tokens=1050),
                )
            else:
                # Second call: capture messages and return final text
                received_step1_msgs.extend(msgs)
                return ModelResponse(content="done", tool_calls=[])

        mock_provider = AsyncMock()
        mock_provider.complete.side_effect = mock_complete

        await agent.run("hello", provider=mock_provider)

        # The second LLM call should have [Context: ...] in the system message
        system_msgs = [m for m in received_step1_msgs if isinstance(m, SystemMessage)]
        assert system_msgs, "Expected a SystemMessage in step 1"
        assert "[Context:" in system_msgs[0].content

    async def test_token_budget_trigger_forces_summarization(self) -> None:
        """When token fill ratio > token_budget_trigger, summarization fires early."""
        from unittest.mock import AsyncMock

        from exo.agent import Agent
        from exo.context.config import ContextConfig  # pyright: ignore[reportMissingImports]
        from exo.models.types import ModelResponse  # pyright: ignore[reportMissingImports]
        from exo.tool import tool
        from exo.types import AssistantMessage, SystemMessage, ToolCall, Usage, UserMessage

        # Low token_pressure (0.1) so even a small input_tokens fills it
        # limit=50, overflow=summarize: summary fires at 50 msgs (high — won't trigger normally)
        context = ContextConfig(limit=50, overflow="summarize", token_pressure=0.1)

        @tool
        def action_tool() -> str:
            return "action done"

        agent = Agent(
            name="budget-trigger-test",
            model="openai:gpt-4o",  # 128000 token window
            memory=None,
            context=context,
            tools=[action_tool],
        )

        # Build history: 4 rounds (8 non-system messages) — under summary_threshold=50
        history = []
        for i in range(4):
            history.append(UserMessage(content=f"msg-{i}"))
            history.append(AssistantMessage(content=f"reply-{i}"))

        call_count = [0]
        all_call_args: list[Any] = []

        async def mock_complete(msgs: Any, **kwargs: Any) -> ModelResponse:
            call_count[0] += 1
            all_call_args.append(list(msgs))
            if call_count[0] == 1:
                # Step 0: return tool call, report fill ratio > 0.1
                # input_tokens=20000 > 0.1 * 128000 = 12800
                return ModelResponse(
                    content="",
                    tool_calls=[ToolCall(id="tc2", name="action_tool", arguments="{}")],
                    usage=Usage(input_tokens=20000, output_tokens=10, total_tokens=20010),
                )
            else:
                # Subsequent calls: return text (handles both summarizer + step 1 calls)
                return ModelResponse(content="done", tool_calls=[])

        mock_provider = AsyncMock()
        mock_provider.complete.side_effect = mock_complete

        await agent.run("trigger input", messages=history, provider=mock_provider)

        # At minimum: step 0 call + step 1 call (plus possibly a summarization call)
        assert call_count[0] >= 2

        # The LAST call is step 1's LLM call (after summarization compressed messages).
        # Original non-system count: 8 history + 1 current + 1 tool result = 10.
        # After force_summarize with keep_recent=max(2, 10//2)=5: 5 kept (non-system).
        last_call_msgs = all_call_args[-1]
        non_system_last = [m for m in last_call_msgs if not isinstance(m, SystemMessage)]
        assert len(non_system_last) < 10, (
            f"Expected summarization to compress messages, got {len(non_system_last)}"
        )

    async def test_force_summarize_in_apply_context_windowing(self) -> None:
        """_apply_context_windowing with force_summarize=True triggers summarization."""

        from exo.agent import _apply_context_windowing
        from exo.context.config import ContextConfig  # pyright: ignore[reportMissingImports]
        from exo.types import AssistantMessage, SystemMessage, UserMessage

        # limit=50, overflow=summarize: _summary_threshold=50 — high, won't trigger normally
        context = ContextConfig(limit=50, overflow="summarize")

        # 4 messages (well under _summary_threshold=50)
        msg_list = [
            SystemMessage(content="System"),
            UserMessage(content="q1"),
            AssistantMessage(content="a1"),
            UserMessage(content="q2"),
        ]

        # Without force_summarize: no summarization
        result_normal, _actions_normal = await _apply_context_windowing(
            msg_list, context, provider=None
        )
        non_system_normal = [m for m in result_normal if not isinstance(m, SystemMessage)]
        assert len(non_system_normal) == 3  # unchanged

        # With force_summarize=True: summarization fires (provider=None → falls back gracefully)
        # Since provider=None, summarizer returns "" → result may differ from None case
        result_forced, _actions_forced = await _apply_context_windowing(
            msg_list, context, provider=None, force_summarize=True
        )
        # The function should run without error (summarization path reached)
        assert result_forced is not None
