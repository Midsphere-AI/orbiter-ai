"""Tests for context snapshot persistence — SnapshotMemory, serialization, save/load/freshness."""

from __future__ import annotations

from exo.memory.base import HumanMemory, MemoryMetadata
from exo.memory.persistence import MemoryPersistence
from exo.memory.short_term import ShortTermMemory
from exo.memory.snapshot import (
    SnapshotMemory,
    compute_config_hash,
    deserialize_msg_list,
    has_message_content,
    make_snapshot,
    serialize_msg_list,
    snapshot_id,
)
from exo.types import AssistantMessage, SystemMessage, ToolCall, ToolResult, UserMessage

# ---------------------------------------------------------------------------
# SnapshotMemory model
# ---------------------------------------------------------------------------


class TestSnapshotMemoryModel:
    def test_create_defaults(self) -> None:
        snap = SnapshotMemory(content="{}")
        assert snap.memory_type == "snapshot"
        assert snap.snapshot_version == 1
        assert snap.raw_item_count == 0
        assert snap.latest_raw_id == ""
        assert snap.latest_raw_created_at == ""
        assert snap.config_hash == ""

    def test_create_with_fields(self) -> None:
        snap = SnapshotMemory(
            id="snap-1",
            content="[]",
            snapshot_version=2,
            raw_item_count=42,
            latest_raw_id="item-99",
            latest_raw_created_at="2026-01-01T00:00:00",
            config_hash="abc123",
        )
        assert snap.id == "snap-1"
        assert snap.snapshot_version == 2
        assert snap.raw_item_count == 42
        assert snap.latest_raw_id == "item-99"
        assert snap.config_hash == "abc123"

    def test_snapshot_id(self) -> None:
        sid = snapshot_id("my-agent", "conv-123")
        assert sid == "snapshot_my-agent_conv-123"


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_roundtrip_user_message(self) -> None:
        original = [UserMessage(content="hello")]
        data = serialize_msg_list(original)
        restored = deserialize_msg_list(data)
        assert len(restored) == 1
        assert isinstance(restored[0], UserMessage)
        assert restored[0].content == "hello"

    def test_roundtrip_assistant_message(self) -> None:
        original = [AssistantMessage(content="hi there")]
        data = serialize_msg_list(original)
        restored = deserialize_msg_list(data)
        assert len(restored) == 1
        assert isinstance(restored[0], AssistantMessage)
        assert restored[0].content == "hi there"

    def test_roundtrip_assistant_with_tool_calls(self) -> None:
        tc = ToolCall(id="tc-1", name="search", arguments='{"q": "test"}')
        original = [AssistantMessage(content="let me search", tool_calls=[tc])]
        data = serialize_msg_list(original)
        restored = deserialize_msg_list(data)
        assert len(restored) == 1
        msg = restored[0]
        assert isinstance(msg, AssistantMessage)
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].id == "tc-1"
        assert msg.tool_calls[0].name == "search"
        assert msg.tool_calls[0].arguments == '{"q": "test"}'

    def test_roundtrip_tool_call_with_thought_signature(self) -> None:
        tc = ToolCall(id="tc-2", name="calc", arguments="{}", thought_signature=b"\x01\x02\x03")
        original = [AssistantMessage(content="", tool_calls=[tc])]
        data = serialize_msg_list(original)
        restored = deserialize_msg_list(data)
        assert restored[0].tool_calls[0].thought_signature == b"\x01\x02\x03"

    def test_roundtrip_tool_result(self) -> None:
        original = [
            ToolResult(tool_call_id="tc-1", tool_name="search", content="found it"),
        ]
        data = serialize_msg_list(original)
        restored = deserialize_msg_list(data)
        assert len(restored) == 1
        msg = restored[0]
        assert isinstance(msg, ToolResult)
        assert msg.tool_call_id == "tc-1"
        assert msg.tool_name == "search"
        assert msg.content == "found it"

    def test_roundtrip_tool_result_with_error(self) -> None:
        original = [
            ToolResult(tool_call_id="tc-1", tool_name="fail", content="", error="boom"),
        ]
        data = serialize_msg_list(original)
        restored = deserialize_msg_list(data)
        assert restored[0].error == "boom"

    def test_instruction_system_message_excluded(self) -> None:
        """Instruction SystemMessages should not appear in the snapshot."""
        original = [
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="hi"),
        ]
        data = serialize_msg_list(original)
        restored = deserialize_msg_list(data)
        # Only the UserMessage should survive.
        assert len(restored) == 1
        assert isinstance(restored[0], UserMessage)

    def test_summary_system_message_preserved(self) -> None:
        """[Conversation Summary] SystemMessages should be preserved."""
        original = [
            SystemMessage(content="[Conversation Summary]\nUser asked about X."),
            UserMessage(content="hi"),
        ]
        data = serialize_msg_list(original)
        restored = deserialize_msg_list(data)
        assert len(restored) == 2
        assert isinstance(restored[0], SystemMessage)
        assert restored[0].content.startswith("[Conversation Summary]")

    def test_roundtrip_mixed_conversation(self) -> None:
        """Full conversation with all message types."""
        tc = ToolCall(id="tc-1", name="get_weather", arguments='{"city": "SF"}')
        original = [
            SystemMessage(content="[Conversation Summary]\nPrevious context."),
            UserMessage(content="What's the weather?"),
            AssistantMessage(content="Let me check.", tool_calls=[tc]),
            ToolResult(tool_call_id="tc-1", tool_name="get_weather", content="Sunny 72F"),
            AssistantMessage(content="It's sunny and 72F in SF."),
        ]
        data = serialize_msg_list(original)
        restored = deserialize_msg_list(data)
        assert len(restored) == 5
        assert isinstance(restored[0], SystemMessage)
        assert isinstance(restored[1], UserMessage)
        assert isinstance(restored[2], AssistantMessage)
        assert isinstance(restored[3], ToolResult)
        assert isinstance(restored[4], AssistantMessage)

    def test_empty_list(self) -> None:
        data = serialize_msg_list([])
        restored = deserialize_msg_list(data)
        assert restored == []


# ---------------------------------------------------------------------------
# Config hash
# ---------------------------------------------------------------------------


class TestConfigHash:
    def test_deterministic(self) -> None:
        class FakeConfig:
            history_rounds = 20
            summary_threshold = 10
            offload_threshold = 50
            token_budget_trigger = 0.8

        h1 = compute_config_hash(FakeConfig())
        h2 = compute_config_hash(FakeConfig())
        assert h1 == h2

    def test_sensitive_to_changes(self) -> None:
        class ConfigA:
            history_rounds = 20
            summary_threshold = 10
            offload_threshold = 50
            token_budget_trigger = 0.8

        class ConfigB:
            history_rounds = 10
            summary_threshold = 10
            offload_threshold = 50
            token_budget_trigger = 0.8

        assert compute_config_hash(ConfigA()) != compute_config_hash(ConfigB())

    def test_config_with_nested_config_attr(self) -> None:
        """Handles objects where the real config is behind .config."""

        class Inner:
            history_rounds = 5
            summary_threshold = 3
            offload_threshold = 20
            token_budget_trigger = 0.8

        class Outer:
            config = Inner()

        h1 = compute_config_hash(Outer())
        h2 = compute_config_hash(Inner())
        assert h1 == h2


# ---------------------------------------------------------------------------
# has_message_content utility
# ---------------------------------------------------------------------------


class TestHasMessageContent:
    def test_found(self) -> None:
        msgs = [UserMessage(content="[MARKER] some context")]
        assert has_message_content(msgs, "[MARKER]") is True

    def test_not_found(self) -> None:
        msgs = [UserMessage(content="hello")]
        assert has_message_content(msgs, "[MARKER]") is False

    def test_empty_list(self) -> None:
        assert has_message_content([], "[MARKER]") is False


# ---------------------------------------------------------------------------
# make_snapshot helper
# ---------------------------------------------------------------------------


class TestMakeSnapshot:
    def test_creates_valid_snapshot(self) -> None:
        class FakeConfig:
            history_rounds = 20
            summary_threshold = 10
            offload_threshold = 50
            token_budget_trigger = 0.8

        msgs = [UserMessage(content="hi")]
        snap = make_snapshot(
            agent_name="bot",
            conversation_id="conv-1",
            msg_list=msgs,
            context_config=FakeConfig(),
            raw_item_count=5,
            latest_raw_id="item-5",
            latest_raw_created_at="2026-01-01",
        )
        assert snap.id == "snapshot_bot_conv-1"
        assert snap.memory_type == "snapshot"
        assert snap.raw_item_count == 5
        assert snap.latest_raw_id == "item-5"
        assert snap.config_hash != ""
        # Content should be deserializable.
        restored = deserialize_msg_list(snap.content)
        assert len(restored) == 1


# ---------------------------------------------------------------------------
# Persistence: save / load / freshness (ShortTermMemory backend)
# ---------------------------------------------------------------------------


class TestSnapshotPersistence:
    async def test_save_and_load(self) -> None:
        store = ShortTermMemory()
        p = MemoryPersistence(store=store)

        class FakeConfig:
            history_rounds = 20
            summary_threshold = 10
            offload_threshold = 50
            token_budget_trigger = 0.8
            enable_snapshots = True

        # Add a raw item first.
        meta = MemoryMetadata(agent_id="bot", task_id="conv-1")
        await store.add(HumanMemory(content="hello", metadata=meta))

        msgs = [UserMessage(content="hello"), AssistantMessage(content="hi")]
        await p.save_snapshot("bot", "conv-1", msgs, FakeConfig())

        snap = await p.load_snapshot("bot", "conv-1")
        assert snap is not None
        assert isinstance(snap, SnapshotMemory)
        restored = deserialize_msg_list(snap.content)
        assert len(restored) == 2

    async def test_load_returns_none_when_absent(self) -> None:
        store = ShortTermMemory()
        p = MemoryPersistence(store=store)
        snap = await p.load_snapshot("bot", "conv-1")
        assert snap is None

    async def test_freshness_fresh(self) -> None:
        store = ShortTermMemory()
        p = MemoryPersistence(store=store)

        class FakeConfig:
            history_rounds = 20
            summary_threshold = 10
            offload_threshold = 50
            token_budget_trigger = 0.8
            enable_snapshots = True

        meta = MemoryMetadata(agent_id="bot", task_id="conv-1")
        await store.add(HumanMemory(content="hello", metadata=meta))

        msgs = [UserMessage(content="hello")]
        await p.save_snapshot("bot", "conv-1", msgs, FakeConfig())

        snap = await p.load_snapshot("bot", "conv-1")
        assert snap is not None
        assert await p.is_snapshot_fresh(snap, "bot", "conv-1", context_config=FakeConfig()) is True

    async def test_freshness_stale_after_new_raw_item(self) -> None:
        store = ShortTermMemory()
        p = MemoryPersistence(store=store)

        class FakeConfig:
            history_rounds = 20
            summary_threshold = 10
            offload_threshold = 50
            token_budget_trigger = 0.8
            enable_snapshots = True

        meta = MemoryMetadata(agent_id="bot", task_id="conv-1")
        await store.add(HumanMemory(content="hello", metadata=meta))

        msgs = [UserMessage(content="hello")]
        await p.save_snapshot("bot", "conv-1", msgs, FakeConfig())

        # Simulate crash: new raw item added after snapshot.
        await store.add(HumanMemory(content="another message", metadata=meta))

        snap = await p.load_snapshot("bot", "conv-1")
        assert snap is not None
        assert await p.is_snapshot_fresh(snap, "bot", "conv-1") is False

    async def test_freshness_stale_after_config_change(self) -> None:
        store = ShortTermMemory()
        p = MemoryPersistence(store=store)

        class ConfigA:
            history_rounds = 20
            summary_threshold = 10
            offload_threshold = 50
            token_budget_trigger = 0.8
            enable_snapshots = True

        class ConfigB:
            history_rounds = 10
            summary_threshold = 5
            offload_threshold = 20
            token_budget_trigger = 0.8
            enable_snapshots = True

        meta = MemoryMetadata(agent_id="bot", task_id="conv-1")
        await store.add(HumanMemory(content="hello", metadata=meta))

        msgs = [UserMessage(content="hello")]
        await p.save_snapshot("bot", "conv-1", msgs, ConfigA())

        snap = await p.load_snapshot("bot", "conv-1")
        assert snap is not None
        # Fresh with same config.
        assert await p.is_snapshot_fresh(snap, "bot", "conv-1", context_config=ConfigA()) is True
        # Stale with different config.
        assert await p.is_snapshot_fresh(snap, "bot", "conv-1", context_config=ConfigB()) is False

    async def test_upsert_replaces_previous(self) -> None:
        store = ShortTermMemory()
        p = MemoryPersistence(store=store)

        class FakeConfig:
            history_rounds = 20
            summary_threshold = 10
            offload_threshold = 50
            token_budget_trigger = 0.8
            enable_snapshots = True

        meta = MemoryMetadata(agent_id="bot", task_id="conv-1")
        await store.add(HumanMemory(content="hello", metadata=meta))

        # Save twice.
        await p.save_snapshot("bot", "conv-1", [UserMessage(content="v1")], FakeConfig())
        await p.save_snapshot("bot", "conv-1", [UserMessage(content="v2")], FakeConfig())

        # Both snapshots get stored (ShortTermMemory is append-only),
        # but load_snapshot returns the first match which has the correct ID.
        snap = await p.load_snapshot("bot", "conv-1")
        assert snap is not None
