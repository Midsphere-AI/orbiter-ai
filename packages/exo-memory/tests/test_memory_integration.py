"""Integration tests for the exo-memory public API and event system."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

from exo.events import EventBus  # pyright: ignore[reportMissingImports]
from exo.memory import (  # pyright: ignore[reportMissingImports]
    MEMORY_ADDED,
    MEMORY_CLEARED,
    MEMORY_SEARCHED,
    AIMemory,
    ExoMemoryError,
    ExtractionTask,
    ExtractionType,
    Extractor,
    HumanMemory,
    LongTermMemory,
    MemoryEventEmitter,
    MemoryItem,
    MemoryMetadata,
    MemoryOrchestrator,
    MemoryStatus,
    MemoryStore,
    MemoryTaskStatus,
    OrchestratorConfig,
    ShortTermMemory,
    Summarizer,
    SummaryConfig,
    SummaryResult,
    SummaryTemplate,
    SystemMemory,
    ToolMemory,
    check_trigger,
    generate_summary,
)

# ---------------------------------------------------------------------------
# Public API import tests
# ---------------------------------------------------------------------------


class TestPublicAPIImports:
    """Verify all public exports are importable from exo.memory."""

    def test_base_types(self) -> None:
        assert MemoryItem is not None
        assert MemoryMetadata is not None
        assert MemoryStatus is not None
        assert MemoryStore is not None
        assert ExoMemoryError is not None

    def test_memory_subtypes(self) -> None:
        assert SystemMemory is not None
        assert HumanMemory is not None
        assert AIMemory is not None
        assert ToolMemory is not None

    def test_short_term(self) -> None:
        assert ShortTermMemory is not None

    def test_long_term(self) -> None:
        assert LongTermMemory is not None
        assert MemoryOrchestrator is not None
        assert OrchestratorConfig is not None
        assert ExtractionTask is not None
        assert ExtractionType is not None
        assert Extractor is not None
        assert MemoryTaskStatus is not None

    def test_summary(self) -> None:
        assert SummaryConfig is not None
        assert SummaryResult is not None
        assert SummaryTemplate is not None
        assert Summarizer is not None
        assert check_trigger is not None
        assert generate_summary is not None

    def test_events(self) -> None:
        assert MemoryEventEmitter is not None
        assert MEMORY_ADDED == "memory:added"
        assert MEMORY_SEARCHED == "memory:searched"
        assert MEMORY_CLEARED == "memory:cleared"


# ---------------------------------------------------------------------------
# MemoryEventEmitter tests
# ---------------------------------------------------------------------------


class TestMemoryEventEmitterInit:
    def test_create_with_defaults(self) -> None:
        store = ShortTermMemory()
        emitter = MemoryEventEmitter(store)
        assert emitter.store is store
        assert isinstance(emitter.bus, EventBus)

    def test_create_with_custom_bus(self) -> None:
        store = ShortTermMemory()
        bus = EventBus()
        emitter = MemoryEventEmitter(store, bus=bus)
        assert emitter.bus is bus

    def test_repr(self) -> None:
        store = ShortTermMemory()
        emitter = MemoryEventEmitter(store)
        assert "MemoryEventEmitter" in repr(emitter)


class TestMemoryEventEmitterAdd:
    async def test_add_emits_event(self) -> None:
        store = ShortTermMemory()
        bus = EventBus()
        emitter = MemoryEventEmitter(store, bus=bus)

        captured: list[dict[str, Any]] = []

        async def handler(**data: Any) -> None:
            captured.append(data)

        bus.on(MEMORY_ADDED, handler)

        item = HumanMemory(content="hello")
        await emitter.add(item)

        assert len(captured) == 1
        assert captured[0]["item"] is item

    async def test_add_persists_to_store(self) -> None:
        store = ShortTermMemory()
        emitter = MemoryEventEmitter(store)

        item = HumanMemory(content="test")
        await emitter.add(item)

        result = await store.get(item.id)
        assert result is not None
        assert result.content == "test"


class TestMemoryEventEmitterGet:
    async def test_get_delegates_to_store(self) -> None:
        store = ShortTermMemory()
        emitter = MemoryEventEmitter(store)

        item = HumanMemory(content="data")
        await store.add(item)

        result = await emitter.get(item.id)
        assert result is not None
        assert result.content == "data"

    async def test_get_missing(self) -> None:
        store = ShortTermMemory()
        emitter = MemoryEventEmitter(store)
        result = await emitter.get("nonexistent")
        assert result is None


class TestMemoryEventEmitterSearch:
    async def test_search_emits_event(self) -> None:
        store = ShortTermMemory()
        bus = EventBus()
        emitter = MemoryEventEmitter(store, bus=bus)

        captured: list[dict[str, Any]] = []

        async def handler(**data: Any) -> None:
            captured.append(data)

        bus.on(MEMORY_SEARCHED, handler)

        await emitter.add(HumanMemory(content="hello world"))
        results = await emitter.search(query="hello")

        assert len(results) == 1
        assert len(captured) == 1
        assert captured[0]["query"] == "hello"
        assert len(captured[0]["results"]) == 1

    async def test_search_with_filters(self) -> None:
        store = ShortTermMemory()
        emitter = MemoryEventEmitter(store)

        await emitter.add(HumanMemory(content="user msg"))
        await emitter.add(AIMemory(content="ai reply"))

        results = await emitter.search(memory_type="human")
        assert len(results) == 1
        assert results[0].memory_type == "human"


class TestMemoryEventEmitterClear:
    async def test_clear_emits_event(self) -> None:
        store = ShortTermMemory()
        bus = EventBus()
        emitter = MemoryEventEmitter(store, bus=bus)

        captured: list[dict[str, Any]] = []

        async def handler(**data: Any) -> None:
            captured.append(data)

        bus.on(MEMORY_CLEARED, handler)

        await emitter.add(HumanMemory(content="to clear"))
        count = await emitter.clear()

        assert count == 1
        assert len(captured) == 1
        assert captured[0]["count"] == 1

    async def test_clear_with_metadata_filter(self) -> None:
        store = ShortTermMemory()
        emitter = MemoryEventEmitter(store)

        meta = MemoryMetadata(user_id="u1")
        await emitter.add(HumanMemory(content="a", metadata=meta))
        await emitter.add(HumanMemory(content="b"))

        count = await emitter.clear(metadata=meta)
        assert count == 1


# ---------------------------------------------------------------------------
# Agent + memory wiring integration
# ---------------------------------------------------------------------------


class TestAgentMemoryWiring:
    def test_agent_accepts_memory_store(self) -> None:
        from exo.agent import Agent  # pyright: ignore[reportMissingImports]

        store = ShortTermMemory()
        agent = Agent(name="test", memory=store)
        assert agent.memory is store

    def test_agent_accepts_event_emitter(self) -> None:
        from exo.agent import Agent  # pyright: ignore[reportMissingImports]

        store = ShortTermMemory()
        emitter = MemoryEventEmitter(store)
        agent = Agent(name="test", memory=emitter)
        assert agent.memory is emitter

    def test_agent_default_memory_auto_created(self) -> None:
        from exo.agent import Agent  # pyright: ignore[reportMissingImports]
        from exo.memory.base import AgentMemory

        agent = Agent(name="test")
        assert isinstance(agent.memory, AgentMemory)


# ---------------------------------------------------------------------------
# End-to-end integration scenarios
# ---------------------------------------------------------------------------


class TestEndToEndScenarios:
    async def test_short_term_to_summary_pipeline(self) -> None:
        """Full flow: add messages → check trigger → summarize."""
        store = ShortTermMemory()
        emitter = MemoryEventEmitter(store)

        # Add enough messages to trigger summarization
        config = SummaryConfig(message_threshold=3)
        for i in range(5):
            await emitter.add(HumanMemory(content=f"Message {i}"))

        items = await emitter.search()
        trigger = check_trigger(items, config)
        assert trigger.triggered is True

        # Generate summary with mock
        mock_summarizer = AsyncMock()
        mock_summarizer.summarize.return_value = "Summary of conversation"

        result = await generate_summary(items, config, mock_summarizer)
        assert result.original_count == 5
        assert "conversation" in result.summaries

    async def test_short_term_to_long_term_extraction(self) -> None:
        """Full flow: add messages → extract to long-term memory."""
        short = ShortTermMemory()
        long = LongTermMemory()
        orchestrator = MemoryOrchestrator(
            long,
            config=OrchestratorConfig(
                extraction_types=(ExtractionType.FACTS,),
            ),
        )

        # Add conversation items
        await short.add(HumanMemory(content="The meeting is at 3pm"))
        await short.add(AIMemory(content="I'll schedule it for 3pm"))

        # Search short-term for items
        items = await short.search()
        assert len(items) == 2

        # Submit extraction
        tasks = orchestrator.submit(items)
        assert len(tasks) == 1
        assert tasks[0].status == MemoryTaskStatus.PENDING

        # Process with mock extractor
        mock_extractor = AsyncMock()
        mock_extractor.extract.return_value = "Meeting scheduled at 3pm"

        processed = await orchestrator.process_all(mock_extractor)
        assert len(processed) == 1
        assert processed[0].status == MemoryTaskStatus.COMPLETED

        # Verify stored in long-term memory
        facts = await long.search(memory_type="facts")
        assert len(facts) == 1
        assert "3pm" in facts[0].content

    async def test_event_driven_extraction_pipeline(self) -> None:
        """Event emitter triggers extraction on add."""
        short = ShortTermMemory()
        LongTermMemory()
        bus = EventBus()
        emitter = MemoryEventEmitter(short, bus=bus)

        extracted: list[str] = []

        async def on_memory_added(item: MemoryItem, **_: Any) -> None:
            """Simple handler that stores content for later extraction."""
            extracted.append(item.content)

        bus.on(MEMORY_ADDED, on_memory_added)

        await emitter.add(HumanMemory(content="Task: buy groceries"))
        await emitter.add(AIMemory(content="Added to your list"))

        assert len(extracted) == 2
        assert "buy groceries" in extracted[0]

    async def test_memory_protocol_conformance(self) -> None:
        """MemoryEventEmitter implements MemoryStore protocol methods."""
        store = ShortTermMemory()
        emitter = MemoryEventEmitter(store)

        # Verify all protocol methods exist and work
        item = HumanMemory(content="protocol test")
        await emitter.add(item)
        assert await emitter.get(item.id) is not None
        results = await emitter.search(query="protocol")
        assert len(results) == 1
        count = await emitter.clear()
        assert count == 1

    async def test_multi_event_handler_pipeline(self) -> None:
        """Multiple handlers fire on same event."""
        store = ShortTermMemory()
        bus = EventBus()
        emitter = MemoryEventEmitter(store, bus=bus)

        log: list[str] = []

        async def handler_a(**_: Any) -> None:
            log.append("a")

        async def handler_b(**_: Any) -> None:
            log.append("b")

        bus.on(MEMORY_ADDED, handler_a)
        bus.on(MEMORY_ADDED, handler_b)

        await emitter.add(HumanMemory(content="test"))
        assert log == ["a", "b"]
