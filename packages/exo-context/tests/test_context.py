"""Tests for Context — core lifecycle with fork/merge."""

import pytest

from exo.context.config import (  # pyright: ignore[reportMissingImports]
    ContextConfig,
    OverflowStrategy,
)
from exo.context.context import Context, ContextError  # pyright: ignore[reportMissingImports]
from exo.context.state import ContextState  # pyright: ignore[reportMissingImports]

# ── Creation ─────────────────────────────────────────────────────────


class TestContextCreation:
    def test_minimal(self) -> None:
        ctx = Context("task-1")
        assert ctx.task_id == "task-1"
        assert ctx.config.overflow == OverflowStrategy.SUMMARIZE
        assert ctx.parent is None
        assert ctx.children == []
        assert ctx.token_usage == {}

    def test_empty_task_id_raises(self) -> None:
        with pytest.raises(ContextError, match="task_id is required"):
            Context("")

    def test_custom_config(self) -> None:
        cfg = ContextConfig(limit=5, overflow="summarize")
        ctx = Context("task-2", config=cfg)
        assert ctx.config is cfg
        assert ctx.config.overflow == OverflowStrategy.SUMMARIZE
        assert ctx.config.limit == 5

    def test_custom_state(self) -> None:
        state = ContextState({"key": "value"})
        ctx = Context("task-3", state=state)
        assert ctx.state is state
        assert ctx.state["key"] == "value"

    def test_default_state_is_empty(self) -> None:
        ctx = Context("task-4")
        assert ctx.state.local_dict() == {}
        assert len(ctx.state) == 0


# ── Properties ───────────────────────────────────────────────────────


class TestContextProperties:
    def test_task_id_property(self) -> None:
        ctx = Context("my-task")
        assert ctx.task_id == "my-task"

    def test_config_property(self) -> None:
        ctx = Context("task")
        assert isinstance(ctx.config, ContextConfig)

    def test_parent_none_for_root(self) -> None:
        ctx = Context("root")
        assert ctx.parent is None

    def test_children_returns_copy(self) -> None:
        ctx = Context("root")
        children = ctx.children
        children.append(Context("fake"))  # type: ignore[arg-type]
        assert len(ctx.children) == 0  # original not modified


# ── Token tracking ───────────────────────────────────────────────────


class TestTokenTracking:
    def test_add_tokens(self) -> None:
        ctx = Context("task")
        ctx.add_tokens({"prompt": 100, "output": 50})
        assert ctx.token_usage == {"prompt": 100, "output": 50}

    def test_add_tokens_accumulates(self) -> None:
        ctx = Context("task")
        ctx.add_tokens({"prompt": 100})
        ctx.add_tokens({"prompt": 50, "output": 30})
        assert ctx.token_usage == {"prompt": 150, "output": 30}

    def test_token_usage_returns_copy(self) -> None:
        ctx = Context("task")
        ctx.add_tokens({"prompt": 100})
        usage = ctx.token_usage
        usage["prompt"] = 999
        assert ctx.token_usage["prompt"] == 100  # original unchanged


# ── Fork ─────────────────────────────────────────────────────────────


class TestFork:
    def test_fork_creates_child(self) -> None:
        parent = Context("parent-task")
        child = parent.fork("child-task")
        assert child.task_id == "child-task"
        assert child.parent is parent
        assert len(parent.children) == 1
        assert parent.children[0] is child

    def test_fork_inherits_config(self) -> None:
        cfg = ContextConfig(limit=10, overflow="summarize", cache=True)
        parent = Context("parent", config=cfg)
        child = parent.fork("child")
        assert child.config is cfg

    def test_fork_state_inherits_parent(self) -> None:
        parent = Context("parent")
        parent.state.set("color", "blue")
        child = parent.fork("child")
        assert child.state.get("color") == "blue"

    def test_fork_state_write_isolation(self) -> None:
        parent = Context("parent")
        parent.state.set("shared", "original")
        child = parent.fork("child")
        child.state.set("shared", "modified")
        assert child.state["shared"] == "modified"
        assert parent.state["shared"] == "original"

    def test_fork_child_local_state_independent(self) -> None:
        parent = Context("parent")
        child = parent.fork("child")
        child.state.set("child_key", "child_value")
        assert "child_key" not in parent.state

    def test_fork_token_snapshot(self) -> None:
        parent = Context("parent")
        parent.add_tokens({"prompt": 100})
        child = parent.fork("child")
        # Child starts with parent's usage
        assert child.token_usage == {"prompt": 100}

    def test_multiple_forks(self) -> None:
        parent = Context("parent")
        c1 = parent.fork("child-1")
        c2 = parent.fork("child-2")
        assert len(parent.children) == 2
        assert c1.task_id == "child-1"
        assert c2.task_id == "child-2"

    def test_nested_fork(self) -> None:
        root = Context("root")
        child = root.fork("child")
        grandchild = child.fork("grandchild")
        assert grandchild.parent is child
        assert child.parent is root
        assert len(child.children) == 1


# ── Merge ────────────────────────────────────────────────────────────


class TestMerge:
    def test_merge_state(self) -> None:
        parent = Context("parent")
        parent.state.set("parent_key", "parent_value")
        child = parent.fork("child")
        child.state.set("child_key", "child_value")
        parent.merge(child)
        assert parent.state["child_key"] == "child_value"
        assert parent.state["parent_key"] == "parent_value"

    def test_merge_state_override(self) -> None:
        parent = Context("parent")
        parent.state.set("shared", "original")
        child = parent.fork("child")
        child.state.set("shared", "updated")
        parent.merge(child)
        assert parent.state["shared"] == "updated"

    def test_merge_empty_child_state(self) -> None:
        parent = Context("parent")
        parent.state.set("key", "value")
        child = parent.fork("child")
        parent.merge(child)
        # Parent state unchanged
        assert parent.state["key"] == "value"

    def test_merge_net_token_calculation(self) -> None:
        parent = Context("parent")
        parent.add_tokens({"prompt": 100, "output": 50})
        child = parent.fork("child")
        # Child adds its own tokens
        child.add_tokens({"prompt": 200, "output": 100})
        parent.merge(child)
        # Parent should get net: prompt +200, output +100
        assert parent.token_usage == {"prompt": 300, "output": 150}

    def test_merge_token_no_double_counting(self) -> None:
        parent = Context("parent")
        parent.add_tokens({"prompt": 100})
        child = parent.fork("child")
        # Child doesn't add any new tokens
        parent.merge(child)
        # Parent should remain at 100 (net = 0)
        assert parent.token_usage == {"prompt": 100}

    def test_merge_token_new_key_in_child(self) -> None:
        parent = Context("parent")
        parent.add_tokens({"prompt": 100})
        child = parent.fork("child")
        child.add_tokens({"completion": 75})
        parent.merge(child)
        assert parent.token_usage == {"prompt": 100, "completion": 75}

    def test_merge_non_child_raises(self) -> None:
        ctx_a = Context("a")
        ctx_b = Context("b")
        with pytest.raises(ContextError, match="not a child"):
            ctx_a.merge(ctx_b)

    def test_merge_wrong_parent_raises(self) -> None:
        parent1 = Context("parent1")
        parent2 = Context("parent2")
        child = parent1.fork("child")
        with pytest.raises(ContextError, match="not a child"):
            parent2.merge(child)

    def test_merge_removes_child_from_children(self) -> None:
        parent = Context("parent")
        child = parent.fork("child")
        assert len(parent.children) == 1
        parent.merge(child)
        assert len(parent.children) == 0

    def test_double_merge_raises(self) -> None:
        """Merging the same child twice must raise ContextError."""
        parent = Context("parent")
        child = parent.fork("child")
        parent.merge(child)
        with pytest.raises(ContextError, match="already been merged"):
            parent.merge(child)


# ── Hierarchical state isolation ─────────────────────────────────────


class TestHierarchicalIsolation:
    def test_three_level_state_inheritance(self) -> None:
        root = Context("root")
        root.state.set("root_key", "root_val")
        child = root.fork("child")
        child.state.set("child_key", "child_val")
        grandchild = child.fork("grandchild")
        # Grandchild sees both root and child state
        assert grandchild.state["root_key"] == "root_val"
        assert grandchild.state["child_key"] == "child_val"

    def test_grandchild_write_isolation(self) -> None:
        root = Context("root")
        root.state.set("val", 1)
        child = root.fork("child")
        grandchild = child.fork("grandchild")
        grandchild.state.set("val", 99)
        assert grandchild.state["val"] == 99
        assert child.state["val"] == 1
        assert root.state["val"] == 1

    def test_cascade_merge(self) -> None:
        """Merge grandchild into child, then child into root."""
        root = Context("root")
        root.add_tokens({"prompt": 50})
        child = root.fork("child")
        child.add_tokens({"prompt": 30})
        grandchild = child.fork("grandchild")
        grandchild.add_tokens({"prompt": 20})
        grandchild.state.set("gc_key", "gc_val")

        # Merge grandchild → child
        child.merge(grandchild)
        assert child.state["gc_key"] == "gc_val"
        assert child.token_usage["prompt"] == 80 + 20  # 80 (50 inherited + 30 own) + 20 net

        # Merge child → root
        child.state.set("child_key", "child_val")
        root.merge(child)
        assert root.state["gc_key"] == "gc_val"
        assert root.state["child_key"] == "child_val"
        # Root: 50 + net from child (child had 100 after grandchild merge, snapshot was 50, net=50)
        assert root.token_usage["prompt"] == 100

    def test_sibling_isolation(self) -> None:
        parent = Context("parent")
        child_a = parent.fork("child-a")
        child_b = parent.fork("child-b")
        child_a.state.set("exclusive", "a")
        child_b.state.set("exclusive", "b")
        # Siblings don't see each other's state
        assert child_a.state["exclusive"] == "a"
        assert child_b.state["exclusive"] == "b"


# ── Repr ─────────────────────────────────────────────────────────────


class TestContextRepr:
    def test_root_repr(self) -> None:
        ctx = Context("task-1")
        r = repr(ctx)
        assert "task_id='task-1'" in r
        assert "parent" not in r
        assert "children" not in r

    def test_child_repr(self) -> None:
        parent = Context("parent")
        child = parent.fork("child")
        r = repr(child)
        assert "task_id='child'" in r
        assert "parent='parent'" in r

    def test_parent_with_children_repr(self) -> None:
        parent = Context("parent")
        parent.fork("c1")
        parent.fork("c2")
        r = repr(parent)
        assert "children=2" in r
