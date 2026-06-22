"""Tests for long-term memory and memory orchestrator."""

from __future__ import annotations

from exo.memory.base import (  # pyright: ignore[reportMissingImports]
    AIMemory,
    ExoMemoryError,
    HumanMemory,
    MemoryItem,
    MemoryMetadata,
    MemoryStatus,
    SystemMemory,
)
from exo.memory.long_term import (  # pyright: ignore[reportMissingImports]
    ExtractionTask,
    ExtractionType,
    Extractor,
    LongTermMemory,
    MemoryOrchestrator,
    MemoryTaskStatus,
    OrchestratorConfig,
    _format_extraction_items,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockExtractor:
    """Mock extractor that returns canned responses."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses or ["extracted knowledge"])
        self._call_count = 0

    async def extract(self, prompt: str) -> str:
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[idx]


class FailingExtractor:
    """Mock extractor that raises an exception."""

    async def extract(self, prompt: str) -> str:
        msg = "LLM service unavailable"
        raise RuntimeError(msg)


def _make_items(n: int, *, user_id: str = "u1", session_id: str = "s1") -> list[MemoryItem]:
    """Create n memory items for testing."""
    items: list[MemoryItem] = []
    for i in range(n):
        if i % 2 == 0:
            items.append(
                HumanMemory(
                    content=f"User message {i}",
                    metadata=MemoryMetadata(user_id=user_id, session_id=session_id),
                )
            )
        else:
            items.append(
                AIMemory(
                    content=f"AI response {i}",
                    metadata=MemoryMetadata(user_id=user_id, session_id=session_id),
                )
            )
    return items


# ===========================================================================
# ExtractionType
# ===========================================================================


class TestExtractionType:
    def test_values(self) -> None:
        assert ExtractionType.USER_PROFILE == "user_profile"
        assert ExtractionType.AGENT_EXPERIENCE == "agent_experience"
        assert ExtractionType.FACTS == "facts"

    def test_all_have_default_prompts(self) -> None:
        from exo.memory.long_term import (  # pyright: ignore[reportMissingImports]
            _DEFAULT_EXTRACTION_PROMPTS,
        )

        for etype in ExtractionType:
            assert etype in _DEFAULT_EXTRACTION_PROMPTS


# ===========================================================================
# TaskStatus
# ===========================================================================


class TestTaskStatus:
    def test_values(self) -> None:
        assert MemoryTaskStatus.PENDING == "pending"
        assert MemoryTaskStatus.RUNNING == "running"
        assert MemoryTaskStatus.COMPLETED == "completed"
        assert MemoryTaskStatus.FAILED == "failed"


# ===========================================================================
# ExtractionTask
# ===========================================================================


class TestExtractionTask:
    def test_creation(self) -> None:
        items = _make_items(2)
        task = ExtractionTask(extraction_type=ExtractionType.FACTS, source_items=items)
        assert task.status == MemoryTaskStatus.PENDING
        assert task.result is None
        assert task.error is None
        assert task.completed_at is None
        assert len(task.task_id) > 0

    def test_start(self) -> None:
        task = ExtractionTask(extraction_type=ExtractionType.FACTS, source_items=[])
        task.start()
        assert task.status == MemoryTaskStatus.RUNNING

    def test_complete(self) -> None:
        task = ExtractionTask(extraction_type=ExtractionType.FACTS, source_items=[])
        task.start()
        task.complete("some facts")
        assert task.status == MemoryTaskStatus.COMPLETED
        assert task.result == "some facts"
        assert task.completed_at is not None

    def test_fail(self) -> None:
        task = ExtractionTask(extraction_type=ExtractionType.FACTS, source_items=[])
        task.start()
        task.fail("connection error")
        assert task.status == MemoryTaskStatus.FAILED
        assert task.error == "connection error"
        assert task.completed_at is not None


# ===========================================================================
# Extractor protocol
# ===========================================================================


class TestExtractorProtocol:
    def test_mock_implements_protocol(self) -> None:
        assert isinstance(MockExtractor(), Extractor)

    def test_failing_implements_protocol(self) -> None:
        assert isinstance(FailingExtractor(), Extractor)


# ===========================================================================
# LongTermMemory
# ===========================================================================


class TestLongTermMemoryInit:
    def test_defaults(self) -> None:
        store = LongTermMemory()
        assert store.namespace == "default"
        assert len(store) == 0

    def test_custom_namespace(self) -> None:
        store = LongTermMemory(namespace="test-ns")
        assert store.namespace == "test-ns"

    def test_repr(self) -> None:
        store = LongTermMemory(namespace="ns")
        assert "ns" in repr(store)
        assert "items=0" in repr(store)


class TestLongTermMemoryAdd:
    async def test_add_and_get(self) -> None:
        store = LongTermMemory()
        item = MemoryItem(content="fact 1", memory_type="facts")
        await store.add(item)
        assert len(store) == 1
        got = await store.get(item.id)
        assert got is not None
        assert got.content == "fact 1"

    async def test_get_missing(self) -> None:
        store = LongTermMemory()
        assert await store.get("nonexistent") is None

    async def test_deduplication(self) -> None:
        store = LongTermMemory()
        item1 = MemoryItem(content="duplicate", memory_type="facts")
        item2 = MemoryItem(content="duplicate", memory_type="facts")
        await store.add(item1)
        await store.add(item2)
        assert len(store) == 1

    async def test_different_types_not_deduplicated(self) -> None:
        store = LongTermMemory()
        item1 = MemoryItem(content="same content", memory_type="facts")
        item2 = MemoryItem(content="same content", memory_type="user_profile")
        await store.add(item1)
        await store.add(item2)
        assert len(store) == 2

    async def test_different_content_not_deduplicated(self) -> None:
        store = LongTermMemory()
        await store.add(MemoryItem(content="fact A", memory_type="facts"))
        await store.add(MemoryItem(content="fact B", memory_type="facts"))
        assert len(store) == 2


class TestLongTermMemorySearch:
    async def test_search_by_query(self) -> None:
        store = LongTermMemory()
        await store.add(MemoryItem(content="User likes Python", memory_type="user_profile"))
        await store.add(MemoryItem(content="Agent learned retry", memory_type="agent_experience"))
        results = await store.search(query="python")
        assert len(results) == 1
        assert "Python" in results[0].content

    async def test_search_by_type(self) -> None:
        store = LongTermMemory()
        await store.add(MemoryItem(content="fact 1", memory_type="facts"))
        await store.add(MemoryItem(content="profile 1", memory_type="user_profile"))
        results = await store.search(memory_type="facts")
        assert len(results) == 1

    async def test_search_by_metadata(self) -> None:
        store = LongTermMemory()
        meta_a = MemoryMetadata(user_id="alice")
        meta_b = MemoryMetadata(user_id="bob")
        await store.add(MemoryItem(content="a fact", memory_type="facts", metadata=meta_a))
        await store.add(MemoryItem(content="b fact", memory_type="facts", metadata=meta_b))
        results = await store.search(metadata=MemoryMetadata(user_id="alice"))
        assert len(results) == 1
        assert results[0].content == "a fact"

    async def test_search_by_status(self) -> None:
        store = LongTermMemory()
        item = MemoryItem(content="draft fact", memory_type="facts", status=MemoryStatus.DRAFT)
        await store.add(item)
        await store.add(MemoryItem(content="accepted", memory_type="facts"))
        results = await store.search(status=MemoryStatus.DRAFT)
        assert len(results) == 1

    async def test_search_limit(self) -> None:
        store = LongTermMemory()
        for i in range(10):
            await store.add(MemoryItem(content=f"fact {i}", memory_type="facts"))
        results = await store.search(limit=3)
        assert len(results) == 3

    async def test_search_empty(self) -> None:
        store = LongTermMemory()
        results = await store.search(query="anything")
        assert results == []


class TestLongTermMemoryClear:
    async def test_clear_all(self) -> None:
        store = LongTermMemory()
        for i in range(5):
            await store.add(MemoryItem(content=f"item {i}", memory_type="facts"))
        removed = await store.clear()
        assert removed == 5
        assert len(store) == 0

    async def test_clear_by_metadata(self) -> None:
        store = LongTermMemory()
        meta_a = MemoryMetadata(user_id="alice")
        meta_b = MemoryMetadata(user_id="bob")
        await store.add(MemoryItem(content="a", memory_type="facts", metadata=meta_a))
        await store.add(MemoryItem(content="b", memory_type="facts", metadata=meta_b))
        removed = await store.clear(metadata=MemoryMetadata(user_id="alice"))
        assert removed == 1
        assert len(store) == 1

    async def test_clear_empty(self) -> None:
        store = LongTermMemory()
        assert await store.clear() == 0


# ===========================================================================
# OrchestratorConfig
# ===========================================================================


class TestOrchestratorConfig:
    def test_defaults(self) -> None:
        config = OrchestratorConfig()
        assert len(config.extraction_types) == 3
        assert config.min_items == 3

    def test_custom(self) -> None:
        config = OrchestratorConfig(
            extraction_types=(ExtractionType.FACTS,),
            min_items=5,
        )
        assert len(config.extraction_types) == 1
        assert config.min_items == 5

    def test_get_prompt_default(self) -> None:
        config = OrchestratorConfig()
        prompt = config.get_prompt(ExtractionType.FACTS)
        assert "{content}" in prompt

    def test_get_prompt_custom(self) -> None:
        config = OrchestratorConfig(
            prompts={ExtractionType.FACTS: "Custom: {content}"},
        )
        prompt = config.get_prompt(ExtractionType.FACTS)
        assert prompt == "Custom: {content}"


# ===========================================================================
# MemoryOrchestrator — submit
# ===========================================================================


class TestOrchestratorSubmit:
    def test_submit_all_types(self) -> None:
        store = LongTermMemory()
        orch = MemoryOrchestrator(store)
        items = _make_items(4)
        tasks = orch.submit(items)
        assert len(tasks) == 3  # default: all 3 extraction types
        assert all(t.status == MemoryTaskStatus.PENDING for t in tasks)

    def test_submit_single_type(self) -> None:
        store = LongTermMemory()
        orch = MemoryOrchestrator(store)
        items = _make_items(4)
        tasks = orch.submit(items, extraction_type=ExtractionType.FACTS)
        assert len(tasks) == 1
        assert tasks[0].extraction_type == ExtractionType.FACTS

    def test_submit_with_custom_config(self) -> None:
        store = LongTermMemory()
        config = OrchestratorConfig(extraction_types=(ExtractionType.USER_PROFILE,))
        orch = MemoryOrchestrator(store, config=config)
        tasks = orch.submit(_make_items(4))
        assert len(tasks) == 1
        assert tasks[0].extraction_type == ExtractionType.USER_PROFILE


# ===========================================================================
# MemoryOrchestrator — process
# ===========================================================================


class TestOrchestratorProcess:
    async def test_process_single_task(self) -> None:
        store = LongTermMemory()
        orch = MemoryOrchestrator(store)
        items = _make_items(4)
        tasks = orch.submit(items, extraction_type=ExtractionType.FACTS)
        task_id = tasks[0].task_id

        extractor = MockExtractor(["fact: sky is blue"])
        result = await orch.process(task_id, extractor)
        assert result.status == MemoryTaskStatus.COMPLETED
        assert result.result == "fact: sky is blue"

        # Should be stored in long-term memory
        stored = await store.search(memory_type="facts")
        assert len(stored) == 1
        assert stored[0].content == "fact: sky is blue"

    async def test_process_with_metadata(self) -> None:
        store = LongTermMemory()
        orch = MemoryOrchestrator(store)
        tasks = orch.submit(_make_items(4), extraction_type=ExtractionType.USER_PROFILE)
        meta = MemoryMetadata(user_id="alice", session_id="s1")

        await orch.process(tasks[0].task_id, MockExtractor(), metadata=meta)
        stored = await store.search(metadata=MemoryMetadata(user_id="alice"))
        assert len(stored) == 1
        assert stored[0].metadata.user_id == "alice"

    async def test_process_failed_task(self) -> None:
        store = LongTermMemory()
        orch = MemoryOrchestrator(store)
        tasks = orch.submit(_make_items(4), extraction_type=ExtractionType.FACTS)

        result = await orch.process(tasks[0].task_id, FailingExtractor())
        assert result.status == MemoryTaskStatus.FAILED
        assert "LLM service unavailable" in (result.error or "")
        assert len(store) == 0  # Nothing stored on failure

    async def test_process_unknown_task(self) -> None:
        store = LongTermMemory()
        orch = MemoryOrchestrator(store)
        import pytest

        with pytest.raises(ExoMemoryError, match="No extraction task"):
            await orch.process("nonexistent", MockExtractor())


# ===========================================================================
# MemoryOrchestrator — process_all
# ===========================================================================


class TestOrchestratorProcessAll:
    async def test_process_all_pending(self) -> None:
        store = LongTermMemory()
        orch = MemoryOrchestrator(store)
        items = _make_items(4)
        orch.submit(items)  # Creates 3 tasks (all types)

        extractor = MockExtractor(["profile info", "experience info", "facts info"])
        results = await orch.process_all(extractor)
        assert len(results) == 3
        assert all(t.status == MemoryTaskStatus.COMPLETED for t in results)
        assert len(store) == 3

    async def test_process_all_skips_non_pending(self) -> None:
        store = LongTermMemory()
        orch = MemoryOrchestrator(store)
        tasks = orch.submit(_make_items(4), extraction_type=ExtractionType.FACTS)

        # Process the first task
        await orch.process(tasks[0].task_id, MockExtractor())
        assert tasks[0].status == MemoryTaskStatus.COMPLETED

        # Submit a second task
        new_tasks = orch.submit(_make_items(4), extraction_type=ExtractionType.USER_PROFILE)
        assert new_tasks[0].status == MemoryTaskStatus.PENDING

        # process_all should only process the new pending task
        results = await orch.process_all(MockExtractor(["new profile"]))
        assert len(results) == 1
        assert results[0].extraction_type == ExtractionType.USER_PROFILE


# ===========================================================================
# MemoryOrchestrator — task management
# ===========================================================================


class TestOrchestratorTaskManagement:
    def test_get_task(self) -> None:
        store = LongTermMemory()
        orch = MemoryOrchestrator(store)
        tasks = orch.submit(_make_items(4), extraction_type=ExtractionType.FACTS)
        task = orch.get_task(tasks[0].task_id)
        assert task is not None
        assert task.task_id == tasks[0].task_id

    def test_get_task_missing(self) -> None:
        store = LongTermMemory()
        orch = MemoryOrchestrator(store)
        assert orch.get_task("nonexistent") is None

    def test_list_tasks_all(self) -> None:
        store = LongTermMemory()
        orch = MemoryOrchestrator(store)
        orch.submit(_make_items(4))
        assert len(orch.list_tasks()) == 3

    def test_list_tasks_by_status(self) -> None:
        store = LongTermMemory()
        orch = MemoryOrchestrator(store)
        orch.submit(_make_items(4))
        pending = orch.list_tasks(status=MemoryTaskStatus.PENDING)
        assert len(pending) == 3
        completed = orch.list_tasks(status=MemoryTaskStatus.COMPLETED)
        assert len(completed) == 0

    def test_repr(self) -> None:
        store = LongTermMemory()
        orch = MemoryOrchestrator(store)
        orch.submit(_make_items(4))
        r = repr(orch)
        assert "tasks=3" in r


# ===========================================================================
# _format_extraction_items helper
# ===========================================================================


class TestFormatExtractionItems:
    def test_empty(self) -> None:
        assert _format_extraction_items([]) == ""

    def test_single(self) -> None:
        items = [HumanMemory(content="hello")]
        result = _format_extraction_items(items)
        assert "[HUMAN]: hello" in result

    def test_mixed(self) -> None:
        items = [
            SystemMemory(content="system prompt"),
            HumanMemory(content="user msg"),
            AIMemory(content="ai response"),
        ]
        result = _format_extraction_items(items)
        assert "[SYSTEM]: system prompt" in result
        assert "[HUMAN]: user msg" in result
        assert "[AI]: ai response" in result
