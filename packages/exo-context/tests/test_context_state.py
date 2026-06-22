"""Tests for ContextState — hierarchical key-value state with parent inheritance."""

import pytest

from exo.context.state import (  # pyright: ignore[reportMissingImports]
    ContextState,
    ContextStateKeyError,
)

# ── Basic read/write ─────────────────────────────────────────────────


class TestBasicReadWrite:
    def test_set_and_get(self) -> None:
        s = ContextState()
        s.set("key", "value")
        assert s.get("key") == "value"

    def test_bracket_set_get(self) -> None:
        s = ContextState()
        s["a"] = 1
        assert s["a"] == 1

    def test_get_default(self) -> None:
        s = ContextState()
        assert s.get("missing") is None
        assert s.get("missing", 42) == 42

    def test_getitem_missing_raises(self) -> None:
        s = ContextState()
        with pytest.raises(ContextStateKeyError, match="missing"):
            _ = s["missing"]

    def test_getitem_missing_has_actionable_hint(self) -> None:
        s = ContextState()
        with pytest.raises(ContextStateKeyError) as exc_info:
            _ = s["my_key"]
        assert exc_info.value.hint is not None
        assert "get" in exc_info.value.hint

    def test_contains(self) -> None:
        s = ContextState()
        s["x"] = 1
        assert "x" in s
        assert "y" not in s

    def test_initial_data(self) -> None:
        s = ContextState({"a": 1, "b": 2})
        assert s["a"] == 1
        assert s["b"] == 2
        assert len(s) == 2


# ── Delete / pop / clear ────────────────────────────────────────────


class TestDeletePopClear:
    def test_delete(self) -> None:
        s = ContextState({"a": 1})
        s.delete("a")
        assert "a" not in s

    def test_delete_missing_raises(self) -> None:
        s = ContextState()
        with pytest.raises(KeyError):
            s.delete("missing")

    def test_delitem(self) -> None:
        s = ContextState({"x": 10})
        del s["x"]
        assert "x" not in s

    def test_pop(self) -> None:
        s = ContextState({"k": 99})
        val = s.pop("k")
        assert val == 99
        assert "k" not in s

    def test_pop_default(self) -> None:
        s = ContextState()
        assert s.pop("missing", "default") == "default"

    def test_clear(self) -> None:
        s = ContextState({"a": 1, "b": 2})
        s.clear()
        assert len(s) == 0
        assert s.local_dict() == {}


# ── Update ───────────────────────────────────────────────────────────


class TestUpdate:
    def test_update_from_dict(self) -> None:
        s = ContextState()
        s.update({"a": 1, "b": 2})
        assert s["a"] == 1
        assert s["b"] == 2

    def test_update_from_kwargs(self) -> None:
        s = ContextState()
        s.update(x=10, y=20)
        assert s["x"] == 10
        assert s["y"] == 20

    def test_update_from_context_state(self) -> None:
        other = ContextState({"k": "v"})
        s = ContextState()
        s.update(other)
        assert s["k"] == "v"

    def test_update_mixed(self) -> None:
        s = ContextState()
        s.update({"a": 1}, b=2)
        assert s["a"] == 1
        assert s["b"] == 2


# ── Parent inheritance ───────────────────────────────────────────────


class TestParentInheritance:
    def test_child_reads_parent(self) -> None:
        parent = ContextState({"color": "blue"})
        child = ContextState(parent=parent)
        assert child.get("color") == "blue"
        assert child["color"] == "blue"
        assert "color" in child

    def test_child_overrides_parent(self) -> None:
        parent = ContextState({"color": "blue"})
        child = ContextState({"color": "red"}, parent=parent)
        assert child["color"] == "red"
        assert parent["color"] == "blue"

    def test_write_isolation(self) -> None:
        parent = ContextState({"shared": "original"})
        child = ContextState(parent=parent)
        child["shared"] = "modified"
        assert child["shared"] == "modified"
        assert parent["shared"] == "original"

    def test_delete_does_not_affect_parent(self) -> None:
        parent = ContextState({"key": "parent_val"})
        child = ContextState({"key": "child_val"}, parent=parent)
        child.delete("key")
        # After deleting local, child should see parent value
        assert child["key"] == "parent_val"

    def test_three_level_chain(self) -> None:
        grandparent = ContextState({"a": 1})
        parent = ContextState({"b": 2}, parent=grandparent)
        child = ContextState({"c": 3}, parent=parent)
        assert child["a"] == 1
        assert child["b"] == 2
        assert child["c"] == 3

    def test_deep_override(self) -> None:
        grandparent = ContextState({"x": "gp"})
        parent = ContextState({"x": "p"}, parent=grandparent)
        child = ContextState({"x": "c"}, parent=parent)
        assert child["x"] == "c"
        assert parent["x"] == "p"
        assert grandparent["x"] == "gp"

    def test_parent_property(self) -> None:
        parent = ContextState()
        child = ContextState(parent=parent)
        assert child.parent is parent
        assert parent.parent is None


# ── Introspection ────────────────────────────────────────────────────


class TestIntrospection:
    def test_local_dict(self) -> None:
        parent = ContextState({"a": 1})
        child = ContextState({"b": 2}, parent=parent)
        assert child.local_dict() == {"b": 2}

    def test_to_dict_merges(self) -> None:
        parent = ContextState({"a": 1, "b": 2})
        child = ContextState({"b": 3, "c": 4}, parent=parent)
        merged = child.to_dict()
        assert merged == {"a": 1, "b": 3, "c": 4}

    def test_to_dict_no_parent(self) -> None:
        s = ContextState({"x": 1})
        assert s.to_dict() == {"x": 1}

    def test_keys_includes_parent(self) -> None:
        parent = ContextState({"a": 1})
        child = ContextState({"b": 2}, parent=parent)
        assert child.keys() == {"a", "b"}

    def test_len_includes_parent(self) -> None:
        parent = ContextState({"a": 1, "b": 2})
        child = ContextState({"c": 3}, parent=parent)
        assert len(child) == 3

    def test_len_deduplicates(self) -> None:
        parent = ContextState({"a": 1})
        child = ContextState({"a": 2}, parent=parent)
        assert len(child) == 1

    def test_iter(self) -> None:
        s = ContextState({"a": 1, "b": 2})
        assert set(s) == {"a", "b"}


# ── Bool ─────────────────────────────────────────────────────────────


class TestBool:
    def test_empty_is_falsy(self) -> None:
        assert not ContextState()

    def test_nonempty_is_truthy(self) -> None:
        assert ContextState({"a": 1})

    def test_empty_with_nonempty_parent_is_truthy(self) -> None:
        parent = ContextState({"a": 1})
        child = ContextState(parent=parent)
        assert child  # truthy because parent has data


# ── Repr ─────────────────────────────────────────────────────────────


class TestRepr:
    def test_repr_no_parent(self) -> None:
        s = ContextState({"a": 1, "b": 2})
        assert "local=2" in repr(s)
        assert "inherited" not in repr(s)

    def test_repr_with_parent(self) -> None:
        parent = ContextState({"x": 1})
        child = ContextState({"y": 2}, parent=parent)
        r = repr(child)
        assert "local=1" in r
        assert "inherited=1" in r
