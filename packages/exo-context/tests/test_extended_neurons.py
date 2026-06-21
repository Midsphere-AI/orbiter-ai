"""Tests for dynamic variable registry."""

from __future__ import annotations

from typing import Any

import pytest

from exo.context.neuron import (  # pyright: ignore[reportMissingImports]
    HistoryNeuron,
    SystemNeuron,
    TaskNeuron,
    neuron_registry,
)
from exo.context.state import ContextState  # pyright: ignore[reportMissingImports]
from exo.context.variables import (  # pyright: ignore[reportMissingImports]
    DynamicVariableRegistry,
    VariableResolveError,
)

# ── Registry ──────────────────────────────────────────────────────────


class TestCoreNeuronRegistry:
    def test_core_neurons_registered(self) -> None:
        for name in ("system", "task", "history"):
            assert name in neuron_registry

    def test_speculative_neurons_removed(self) -> None:
        """Speculative neurons (todo/knowledge/workspace/skill/fact/entity) are gone."""
        for name in ("todo", "knowledge", "workspace", "skill", "fact", "entity"):
            assert name not in neuron_registry

    def test_get_system(self) -> None:
        assert isinstance(neuron_registry.get("system"), SystemNeuron)

    def test_get_task(self) -> None:
        assert isinstance(neuron_registry.get("task"), TaskNeuron)

    def test_get_history(self) -> None:
        assert isinstance(neuron_registry.get("history"), HistoryNeuron)


class TestCorePriorityOrdering:
    def test_core_priority_order(self) -> None:
        """Core neurons are ordered by priority."""
        all_names = neuron_registry.list_all()
        neurons = [neuron_registry.get(n) for n in all_names]
        sorted_neurons = sorted(neurons, key=lambda n: n.priority)
        priorities = [(n.name, n.priority) for n in sorted_neurons]
        expected = [
            ("task", 1),
            ("history", 10),
            ("system", 100),
        ]
        assert priorities == expected


# ── DynamicVariableRegistry ─────────────────────────────────────────


class TestDynamicVariableRegistry:
    def test_register_and_resolve_callable(self) -> None:
        reg = DynamicVariableRegistry()
        reg.register("user.name", lambda state: state.get("user_name", "anon"))
        state = ContextState({"user_name": "Alice"})
        assert reg.resolve("user.name", state) == "Alice"

    def test_register_and_resolve_static(self) -> None:
        reg = DynamicVariableRegistry()
        reg.register("app.version", "1.0.0")
        state = ContextState()
        assert reg.resolve("app.version", state) == "1.0.0"

    def test_register_decorator_form(self) -> None:
        reg = DynamicVariableRegistry()

        @reg.register("env.mode")
        def _resolve_mode(state: Any) -> str:
            return state.get("mode", "dev")

        state = ContextState({"mode": "production"})
        assert reg.resolve("env.mode", state) == "production"

    def test_nested_path_resolution(self) -> None:
        reg = DynamicVariableRegistry()
        state = ContextState({"user": {"name": "Bob", "age": 30}})
        assert reg.resolve("user.name", state) == "Bob"
        assert reg.resolve("user.age", state) == 30

    def test_nested_path_with_dict(self) -> None:
        reg = DynamicVariableRegistry()
        state = {"config": {"db": {"host": "localhost"}}}
        assert reg.resolve("config.db.host", state) == "localhost"

    def test_resolver_takes_priority(self) -> None:
        """Registered resolvers take priority over nested path lookup."""
        reg = DynamicVariableRegistry()
        reg.register("user.name", lambda state: "from-resolver")
        state = ContextState({"user": {"name": "from-state"}})
        assert reg.resolve("user.name", state) == "from-resolver"

    def test_missing_path_raises(self) -> None:
        reg = DynamicVariableRegistry()
        state = ContextState()
        with pytest.raises(VariableResolveError, match="not found"):
            reg.resolve("nonexistent.path", state)

    def test_has(self) -> None:
        reg = DynamicVariableRegistry()
        reg.register("x.y", "val")
        assert reg.has("x.y")
        assert not reg.has("a.b")

    def test_list_all(self) -> None:
        reg = DynamicVariableRegistry()
        reg.register("a", "1")
        reg.register("b", "2")
        assert reg.list_all() == ["a", "b"]

    def test_resolve_template(self) -> None:
        reg = DynamicVariableRegistry()
        reg.register("name", "World")
        reg.register("version", "2.0")
        state = ContextState()
        result = reg.resolve_template("Hello ${name}! v${version}", state)
        assert result == "Hello World! v2.0"

    def test_resolve_template_missing_var(self) -> None:
        """Unresolvable variables are left as-is."""
        reg = DynamicVariableRegistry()
        state = ContextState()
        result = reg.resolve_template("Hello ${missing}", state)
        assert result == "Hello ${missing}"

    def test_resolve_template_mixed(self) -> None:
        """Template with some resolvable, some not."""
        reg = DynamicVariableRegistry()
        reg.register("found", "yes")
        state = ContextState()
        result = reg.resolve_template("${found} and ${not_found}", state)
        assert result == "yes and ${not_found}"

    def test_repr(self) -> None:
        reg = DynamicVariableRegistry()
        reg.register("a", "1")
        reg.register("b", "2")
        assert "2" in repr(reg)

    def test_nested_path_partial_raises(self) -> None:
        """Nested path where an intermediate segment is missing raises."""
        reg = DynamicVariableRegistry()
        state = ContextState({"user": {"name": "Alice"}})
        with pytest.raises(VariableResolveError, match="not found"):
            reg.resolve("user.email", state)
