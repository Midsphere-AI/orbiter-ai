"""Tests for checkpoint — snapshot, restore, version incrementing, state preservation."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from exo.context.checkpoint import (  # pyright: ignore[reportMissingImports]
    Checkpoint,
    CheckpointError,
    CheckpointStore,
)
from exo.context.config import ContextConfig  # pyright: ignore[reportMissingImports]
from exo.context.context import Context  # pyright: ignore[reportMissingImports]

# ── Checkpoint dataclass ─────────────────────────────────────────────


class TestCheckpoint:
    def test_creation(self) -> None:
        cp = Checkpoint(
            task_id="t1",
            version=1,
            values={"key": "val"},
            token_usage={"prompt": 100},
        )
        assert cp.task_id == "t1"
        assert cp.version == 1
        assert cp.values == {"key": "val"}
        assert cp.token_usage == {"prompt": 100}
        assert cp.metadata == {}
        assert isinstance(cp.created_at, datetime)

    def test_with_metadata(self) -> None:
        cp = Checkpoint(
            task_id="t1",
            version=2,
            values={},
            token_usage={},
            metadata={"step": 5, "description": "after tool call"},
        )
        assert cp.metadata == {"step": 5, "description": "after tool call"}

    def test_immutable(self) -> None:
        cp = Checkpoint(task_id="t1", version=1, values={}, token_usage={})
        with pytest.raises(AttributeError):
            cp.version = 2  # type: ignore[misc]

    def test_repr(self) -> None:
        cp = Checkpoint(task_id="t1", version=3, values={"a": 1, "b": 2}, token_usage={})
        r = repr(cp)
        assert "t1" in r
        assert "version=3" in r
        assert "keys=2" in r

    def test_to_dict(self) -> None:
        cp = Checkpoint(
            task_id="t1",
            version=1,
            values={"key": "val"},
            token_usage={"prompt": 50},
            metadata={"note": "test"},
        )
        d = cp.to_dict()
        assert d["task_id"] == "t1"
        assert d["version"] == 1
        assert d["values"] == {"key": "val"}
        assert d["token_usage"] == {"prompt": 50}
        assert d["metadata"] == {"note": "test"}
        assert isinstance(d["created_at"], str)

    def test_from_dict(self) -> None:
        now = datetime.now(UTC)
        d = {
            "task_id": "t1",
            "version": 2,
            "values": {"x": 42},
            "token_usage": {"output": 30},
            "metadata": {},
            "created_at": now.isoformat(),
        }
        cp = Checkpoint.from_dict(d)
        assert cp.task_id == "t1"
        assert cp.version == 2
        assert cp.values == {"x": 42}
        assert cp.token_usage == {"output": 30}

    def test_from_dict_roundtrip(self) -> None:
        cp1 = Checkpoint(
            task_id="t1",
            version=1,
            values={"a": 1},
            token_usage={"prompt": 10},
            metadata={"desc": "round trip"},
        )
        d = cp1.to_dict()
        cp2 = Checkpoint.from_dict(d)
        assert cp2.task_id == cp1.task_id
        assert cp2.version == cp1.version
        assert cp2.values == cp1.values
        assert cp2.token_usage == cp1.token_usage
        assert cp2.metadata == cp1.metadata


# ── CheckpointStore ──────────────────────────────────────────────────


class TestCheckpointStore:
    def test_creation(self) -> None:
        store = CheckpointStore("t1")
        assert store.task_id == "t1"
        assert store.version == 0
        assert len(store) == 0

    def test_empty_task_id(self) -> None:
        with pytest.raises(CheckpointError, match="task_id"):
            CheckpointStore("")

    def test_save(self) -> None:
        store = CheckpointStore("t1")
        cp = store.save(values={"a": 1}, token_usage={"prompt": 10})
        assert cp.version == 1
        assert cp.task_id == "t1"
        assert cp.values == {"a": 1}
        assert cp.token_usage == {"prompt": 10}
        assert store.version == 1
        assert len(store) == 1

    def test_save_increments_version(self) -> None:
        store = CheckpointStore("t1")
        cp1 = store.save(values={"a": 1}, token_usage={})
        cp2 = store.save(values={"a": 2}, token_usage={})
        cp3 = store.save(values={"a": 3}, token_usage={})
        assert cp1.version == 1
        assert cp2.version == 2
        assert cp3.version == 3
        assert store.version == 3

    def test_save_with_metadata(self) -> None:
        store = CheckpointStore("t1")
        cp = store.save(values={}, token_usage={}, metadata={"step": 5})
        assert cp.metadata == {"step": 5}

    def test_get(self) -> None:
        store = CheckpointStore("t1")
        store.save(values={"a": 1}, token_usage={})
        store.save(values={"a": 2}, token_usage={})
        cp = store.get(1)
        assert cp.values == {"a": 1}
        cp2 = store.get(2)
        assert cp2.values == {"a": 2}

    def test_get_invalid_version(self) -> None:
        store = CheckpointStore("t1")
        store.save(values={}, token_usage={})
        with pytest.raises(CheckpointError, match="version 0"):
            store.get(0)
        with pytest.raises(CheckpointError, match="version 5"):
            store.get(5)

    def test_latest(self) -> None:
        store = CheckpointStore("t1")
        assert store.latest is None
        store.save(values={"a": 1}, token_usage={})
        store.save(values={"a": 2}, token_usage={})
        assert store.latest is not None
        assert store.latest.version == 2
        assert store.latest.values == {"a": 2}

    def test_list_versions(self) -> None:
        store = CheckpointStore("t1")
        assert store.list_versions() == []
        store.save(values={}, token_usage={})
        store.save(values={}, token_usage={})
        store.save(values={}, token_usage={})
        assert store.list_versions() == [1, 2, 3]

    def test_repr(self) -> None:
        store = CheckpointStore("t1")
        store.save(values={}, token_usage={})
        r = repr(store)
        assert "t1" in r
        assert "checkpoints=1" in r


# ── Context.snapshot / Context.restore ───────────────────────────────


class TestContextSnapshot:
    def test_snapshot_basic(self) -> None:
        ctx = Context("main")
        ctx.state.set("key", "value")
        ctx.add_tokens({"prompt": 100, "output": 50})
        cp = ctx.snapshot()
        assert cp.task_id == "main"
        assert cp.version == 1
        assert cp.values == {"key": "value"}
        assert cp.token_usage == {"prompt": 100, "output": 50}

    def test_snapshot_empty_state(self) -> None:
        ctx = Context("main")
        cp = ctx.snapshot()
        assert cp.values == {}
        assert cp.token_usage == {}

    def test_snapshot_with_metadata(self) -> None:
        ctx = Context("main")
        cp = ctx.snapshot(metadata={"step": 3})
        assert cp.metadata == {"step": 3}

    def test_snapshot_increments_version(self) -> None:
        ctx = Context("main")
        cp1 = ctx.snapshot()
        cp2 = ctx.snapshot()
        cp3 = ctx.snapshot()
        assert cp1.version == 1
        assert cp2.version == 2
        assert cp3.version == 3

    def test_snapshot_captures_current_state(self) -> None:
        """Each snapshot captures the state at the time it was taken."""
        ctx = Context("main")
        ctx.state.set("x", 1)
        cp1 = ctx.snapshot()

        ctx.state.set("x", 2)
        ctx.state.set("y", 10)
        ctx.add_tokens({"prompt": 50})
        cp2 = ctx.snapshot()

        assert cp1.values == {"x": 1}
        assert cp1.token_usage == {}
        assert cp2.values == {"x": 2, "y": 10}
        assert cp2.token_usage == {"prompt": 50}


class TestContextRestore:
    def test_restore_basic(self) -> None:
        ctx = Context("main")
        ctx.state.set("key", "value")
        ctx.add_tokens({"prompt": 100})
        cp = ctx.snapshot()

        restored = Context.restore(cp)
        assert restored.task_id == "main"
        assert restored.state.get("key") == "value"
        assert restored.token_usage == {"prompt": 100}

    def test_restore_with_config(self) -> None:
        config = ContextConfig(limit=5, overflow="truncate")
        ctx = Context("main")
        cp = ctx.snapshot()

        restored = Context.restore(cp, config=config)
        assert restored.config.limit == 5

    def test_restore_is_fresh_context(self) -> None:
        """Restored context is a new independent instance."""
        ctx = Context("main")
        ctx.state.set("key", "value")
        cp = ctx.snapshot()

        restored = Context.restore(cp)
        # Modifications to restored don't affect original
        restored.state.set("key", "changed")
        assert ctx.state.get("key") == "value"
        assert restored.state.get("key") == "changed"

    def test_restore_no_parent(self) -> None:
        """Restored context has no parent (it's a fresh root)."""
        ctx = Context("main")
        child = ctx.fork("child")
        cp = child.snapshot()

        restored = Context.restore(cp)
        assert restored.parent is None
        assert restored.children == []

    def test_restore_no_checkpoints(self) -> None:
        """Restored context starts with empty checkpoint history."""
        ctx = Context("main")
        ctx.snapshot()
        ctx.snapshot()
        cp = ctx.checkpoints.get(1)

        restored = Context.restore(cp)
        assert len(restored.checkpoints) == 0

    def test_restore_invalid_type(self) -> None:
        with pytest.raises(CheckpointError, match="Expected Checkpoint"):
            Context.restore("not a checkpoint")  # type: ignore[arg-type]


class TestContextCheckpoints:
    def test_checkpoints_property(self) -> None:
        ctx = Context("main")
        store = ctx.checkpoints
        assert isinstance(store, CheckpointStore)
        assert store.task_id == "main"

    def test_checkpoint_store_per_context(self) -> None:
        """Each context has its own checkpoint store."""
        ctx1 = Context("t1")
        ctx2 = Context("t2")
        ctx1.snapshot()
        assert len(ctx1.checkpoints) == 1
        assert len(ctx2.checkpoints) == 0

    def test_forked_context_has_own_store(self) -> None:
        """Forked context gets its own checkpoint store."""
        parent = Context("parent")
        parent.snapshot()
        child = parent.fork("child")
        assert len(child.checkpoints) == 0
        child.snapshot()
        assert len(child.checkpoints) == 1
        assert len(parent.checkpoints) == 1


class TestSnapshotRestoreRoundTrip:
    def test_full_roundtrip(self) -> None:
        """Snapshot -> restore -> verify all state preserved."""
        ctx = Context("main")
        ctx.state.set("model", "gpt-4o")
        ctx.state.set("temperature", 0.7)
        ctx.state.set("todos", [{"item": "do stuff", "done": False}])
        ctx.add_tokens({"prompt": 500, "output": 200})

        cp = ctx.snapshot(metadata={"desc": "before tool call"})

        # Mutate original
        ctx.state.set("model", "gpt-3.5")
        ctx.add_tokens({"prompt": 100})

        # Restore from checkpoint
        restored = Context.restore(cp)
        assert restored.state.get("model") == "gpt-4o"
        assert restored.state.get("temperature") == 0.7
        assert restored.state.get("todos") == [{"item": "do stuff", "done": False}]
        assert restored.token_usage == {"prompt": 500, "output": 200}

    def test_restore_old_version(self) -> None:
        """Restore from an earlier checkpoint, not the latest."""
        ctx = Context("main")
        ctx.state.set("step", 1)
        ctx.snapshot()

        ctx.state.set("step", 2)
        ctx.snapshot()

        ctx.state.set("step", 3)
        ctx.snapshot()

        # Restore from version 1
        cp_v1 = ctx.checkpoints.get(1)
        restored = Context.restore(cp_v1)
        assert restored.state.get("step") == 1

    def test_snapshot_serialization_roundtrip(self) -> None:
        """Snapshot -> to_dict -> from_dict -> restore."""
        ctx = Context("main")
        ctx.state.set("data", {"nested": [1, 2, 3]})
        ctx.add_tokens({"prompt": 42})

        cp = ctx.snapshot()
        d = cp.to_dict()
        cp2 = Checkpoint.from_dict(d)
        restored = Context.restore(cp2)

        assert restored.state.get("data") == {"nested": [1, 2, 3]}
        assert restored.token_usage == {"prompt": 42}

    def test_fork_snapshot_restore(self) -> None:
        """Snapshot a forked child, restore it independently."""
        parent = Context("parent")
        parent.state.set("shared", "parent_val")

        child = parent.fork("child")
        child.state.set("local", "child_val")
        child.add_tokens({"prompt": 100})

        cp = child.snapshot()

        # Restore child independently (no parent relationship)
        restored = Context.restore(cp)
        assert restored.task_id == "child"
        assert restored.parent is None
        # to_dict() on child includes inherited values
        assert restored.state.get("shared") == "parent_val"
        assert restored.state.get("local") == "child_val"
        assert restored.token_usage == {"prompt": 100}
