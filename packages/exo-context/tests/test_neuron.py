"""Tests for exo.context.neuron — Neuron ABC and built-in neurons."""

from __future__ import annotations

import platform
from datetime import UTC
from typing import Any
from unittest.mock import patch

from exo.context.config import ContextConfig  # pyright: ignore[reportMissingImports]
from exo.context.context import Context  # pyright: ignore[reportMissingImports]
from exo.context.neuron import (  # pyright: ignore[reportMissingImports]
    HistoryNeuron,
    Neuron,
    SystemNeuron,
    TaskNeuron,
    neuron_registry,
)

# ── Helpers ──────────────────────────────────────────────────────────


def _make_ctx(
    task_id: str = "test-task",
    state: dict[str, Any] | None = None,
    config: ContextConfig | None = None,
) -> Context:
    ctx = Context(task_id, config=config)
    if state:
        for k, v in state.items():
            ctx.state.set(k, v)
    return ctx


class ConcreteNeuron(Neuron):
    """Minimal concrete implementation for ABC tests."""

    async def format(self, ctx: Context, **kwargs: Any) -> str:
        return f"concrete:{ctx.task_id}"


# ── Neuron ABC ───────────────────────────────────────────────────────


class TestNeuronABC:
    def test_name_and_priority(self) -> None:
        n = ConcreteNeuron("my-neuron", priority=42)
        assert n.name == "my-neuron"
        assert n.priority == 42

    def test_default_priority(self) -> None:
        n = ConcreteNeuron("default")
        assert n.priority == 50

    async def test_format(self) -> None:
        n = ConcreteNeuron("test")
        ctx = _make_ctx()
        result = await n.format(ctx)
        assert result == "concrete:test-task"

    def test_repr(self) -> None:
        n = ConcreteNeuron("my-neuron", priority=42)
        r = repr(n)
        assert "ConcreteNeuron" in r
        assert "my-neuron" in r
        assert "42" in r

    def test_cannot_instantiate_abc(self) -> None:
        """Neuron ABC cannot be instantiated directly."""
        import pytest

        with pytest.raises(TypeError, match="abstract"):
            Neuron("fail")  # type: ignore[abstract]


# ── SystemNeuron ─────────────────────────────────────────────────────


class TestSystemNeuron:
    async def test_format_contains_date_time_platform(self) -> None:
        neuron = SystemNeuron()
        ctx = _make_ctx()
        result = await neuron.format(ctx)

        assert "<system_info>" in result
        assert "</system_info>" in result
        assert "Current date:" in result
        assert "Current time:" in result
        assert "Platform:" in result
        assert platform.system() in result

    async def test_default_name_and_priority(self) -> None:
        neuron = SystemNeuron()
        assert neuron.name == "system"
        assert neuron.priority == 100

    async def test_custom_priority(self) -> None:
        neuron = SystemNeuron(priority=200)
        assert neuron.priority == 200

    async def test_format_utc(self) -> None:
        """Output should include UTC time."""
        neuron = SystemNeuron()
        ctx = _make_ctx()
        result = await neuron.format(ctx)
        assert "UTC" in result

    async def test_format_date_format(self) -> None:
        """Date format should be YYYY-MM-DD."""
        from datetime import datetime

        fixed_dt = datetime(2025, 6, 15, 12, 30, 45, tzinfo=UTC)
        with patch("exo.context.neuron.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_dt
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            neuron = SystemNeuron()
            ctx = _make_ctx()
            result = await neuron.format(ctx)
            assert "2025-06-15" in result
            assert "12:30:45 UTC" in result


# ── TaskNeuron ───────────────────────────────────────────────────────


class TestTaskNeuron:
    async def test_basic_task_info(self) -> None:
        neuron = TaskNeuron()
        ctx = _make_ctx("my-task-123")
        result = await neuron.format(ctx)

        assert "<task_info>" in result
        assert "</task_info>" in result
        assert "Task ID: my-task-123" in result

    async def test_task_with_input(self) -> None:
        neuron = TaskNeuron()
        ctx = _make_ctx(state={"task_input": "Find all bugs"})
        result = await neuron.format(ctx)

        assert "Input: Find all bugs" in result

    async def test_task_with_output(self) -> None:
        neuron = TaskNeuron()
        ctx = _make_ctx(state={"task_output": "Found 3 bugs"})
        result = await neuron.format(ctx)

        assert "Output: Found 3 bugs" in result

    async def test_task_with_subtasks(self) -> None:
        neuron = TaskNeuron()
        ctx = _make_ctx(state={"subtasks": ["Analyze code", "Write tests", "Deploy"]})
        result = await neuron.format(ctx)

        assert "Plan:" in result
        assert "<step1>Analyze code</step1>" in result
        assert "<step2>Write tests</step2>" in result
        assert "<step3>Deploy</step3>" in result

    async def test_task_empty_state(self) -> None:
        """Task neuron with no state still produces task_id."""
        neuron = TaskNeuron()
        ctx = _make_ctx("empty-task")
        result = await neuron.format(ctx)

        assert "Task ID: empty-task" in result
        assert "Input:" not in result
        assert "Output:" not in result
        assert "Plan:" not in result

    async def test_default_name_and_priority(self) -> None:
        neuron = TaskNeuron()
        assert neuron.name == "task"
        assert neuron.priority == 1

    async def test_full_task(self) -> None:
        """Task neuron with all state fields populated."""
        neuron = TaskNeuron()
        ctx = _make_ctx(
            "full-task",
            state={
                "task_input": "Solve problem X",
                "task_output": "Partial result Y",
                "subtasks": ["Step A"],
            },
        )
        result = await neuron.format(ctx)

        assert "Task ID: full-task" in result
        assert "Input: Solve problem X" in result
        assert "Output: Partial result Y" in result
        assert "<step1>Step A</step1>" in result


# ── HistoryNeuron ────────────────────────────────────────────────────


class TestHistoryNeuron:
    async def test_empty_history(self) -> None:
        neuron = HistoryNeuron()
        ctx = _make_ctx()
        result = await neuron.format(ctx)
        assert result == ""

    async def test_empty_list_history(self) -> None:
        neuron = HistoryNeuron()
        ctx = _make_ctx(state={"history": []})
        result = await neuron.format(ctx)
        assert result == ""

    async def test_basic_history(self) -> None:
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        neuron = HistoryNeuron()
        ctx = _make_ctx(state={"history": history})
        result = await neuron.format(ctx)

        assert "<conversation_history>" in result
        assert "</conversation_history>" in result
        assert "[user]: Hello" in result
        assert "[assistant]: Hi there!" in result

    async def test_windowing(self) -> None:
        """Only last N rounds are included based on config.limit."""
        config = ContextConfig(limit=1, overflow="summarize")
        history = [
            {"role": "user", "content": "Old message 1"},
            {"role": "assistant", "content": "Old response 1"},
            {"role": "user", "content": "Recent message"},
            {"role": "assistant", "content": "Recent response"},
        ]
        neuron = HistoryNeuron()
        ctx = _make_ctx(state={"history": history}, config=config)
        result = await neuron.format(ctx)

        # Only last 2 messages (1 round = 2 messages)
        assert "Old message 1" not in result
        assert "Old response 1" not in result
        assert "Recent message" in result
        assert "Recent response" in result

    async def test_large_window(self) -> None:
        """When window is larger than history, all messages shown."""
        config = ContextConfig(limit=100, overflow="summarize")
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        neuron = HistoryNeuron()
        ctx = _make_ctx(state={"history": history}, config=config)
        result = await neuron.format(ctx)

        assert "Hello" in result
        assert "Hi" in result

    async def test_default_name_and_priority(self) -> None:
        neuron = HistoryNeuron()
        assert neuron.name == "history"
        assert neuron.priority == 10

    async def test_history_with_tool_messages(self) -> None:
        """History neuron handles tool messages."""
        history = [
            {"role": "user", "content": "Search for X"},
            {"role": "assistant", "content": "Searching..."},
            {"role": "tool", "content": "Found result Y"},
            {"role": "assistant", "content": "I found Y"},
        ]
        neuron = HistoryNeuron()
        ctx = _make_ctx(state={"history": history})
        result = await neuron.format(ctx)

        assert "[tool]: Found result Y" in result

    async def test_windowing_boundary(self) -> None:
        """limit=2 keeps exactly 4 messages (2 rounds * 2 messages per round)."""
        config = ContextConfig(limit=2, overflow="summarize")
        history = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
            {"role": "user", "content": "msg3"},
            {"role": "assistant", "content": "msg4"},
            {"role": "user", "content": "msg5"},
            {"role": "assistant", "content": "msg6"},
        ]
        neuron = HistoryNeuron()
        ctx = _make_ctx(state={"history": history}, config=config)
        result = await neuron.format(ctx)

        assert "msg1" not in result
        assert "msg2" not in result
        assert "msg3" in result
        assert "msg4" in result
        assert "msg5" in result
        assert "msg6" in result


# ── Registry ─────────────────────────────────────────────────────────


class TestNeuronRegistry:
    def test_built_ins_registered(self) -> None:
        """All built-in neurons are registered."""
        assert "system" in neuron_registry
        assert "task" in neuron_registry
        assert "history" in neuron_registry

    def test_get_system(self) -> None:
        neuron = neuron_registry.get("system")
        assert isinstance(neuron, SystemNeuron)

    def test_get_task(self) -> None:
        neuron = neuron_registry.get("task")
        assert isinstance(neuron, TaskNeuron)

    def test_get_history(self) -> None:
        neuron = neuron_registry.get("history")
        assert isinstance(neuron, HistoryNeuron)

    def test_list_all(self) -> None:
        names = neuron_registry.list_all()
        assert "system" in names
        assert "task" in names
        assert "history" in names


# ── Priority ordering ────────────────────────────────────────────────


class TestPriorityOrdering:
    def test_task_before_history_before_system(self) -> None:
        """Priority ordering: task(1) < history(10) < system(100)."""
        task = neuron_registry.get("task")
        history = neuron_registry.get("history")
        system = neuron_registry.get("system")

        assert task.priority < history.priority < system.priority

    def test_sorted_by_priority(self) -> None:
        """Sorting neurons by priority gives correct order."""
        neurons = [
            neuron_registry.get("system"),
            neuron_registry.get("history"),
            neuron_registry.get("task"),
        ]
        sorted_neurons = sorted(neurons, key=lambda n: n.priority)
        assert sorted_neurons[0].name == "task"
        assert sorted_neurons[1].name == "history"
        assert sorted_neurons[2].name == "system"
