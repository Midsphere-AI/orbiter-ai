"""Tests for LLMCallOperator, ToolCallOperator, and MemoryCallOperator."""

from __future__ import annotations

from typing import Any

import pytest

from exo.train.operator.base import (  # pyright: ignore[reportMissingImports]
    Operator,
    TunableKind,
    TunableSpec,
)
from exo.train.operator.llm_call import (  # pyright: ignore[reportMissingImports]
    LLMCallOperator,
    LLMCallTrace,
)
from exo.train.operator.memory_call import (  # pyright: ignore[reportMissingImports]
    MemoryCallOperator,
    MemoryCallTrace,
)
from exo.train.operator.tool_call import (  # pyright: ignore[reportMissingImports]
    ToolCallOperator,
    ToolCallTrace,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _fake_llm(*, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
    return f"Response to: {user_prompt}"


async def _failing_llm(*, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
    raise ValueError("LLM unavailable")


async def _fake_tool(**kwargs: Any) -> dict[str, Any]:
    return {"result": "ok", **kwargs}


async def _failing_tool(**kwargs: Any) -> dict[str, Any]:
    raise RuntimeError("tool broke")


async def _fake_memory(**kwargs: Any) -> list[str]:
    return ["memory_1", "memory_2"]


def _make_flaky_memory(fail_times: int = 1) -> Any:
    """Return a memory fn that fails *fail_times* times then succeeds."""
    call_count = 0

    async def _flaky_memory(**kwargs: Any) -> list[str]:
        nonlocal call_count
        call_count += 1
        if call_count <= fail_times:
            raise ConnectionError("transient error")
        return ["recovered"]

    return _flaky_memory


async def _always_failing_memory(**kwargs: Any) -> list[str]:
    raise ConnectionError("permanent error")


# ---------------------------------------------------------------------------
# LLMCallOperator
# ---------------------------------------------------------------------------


class TestLLMCallOperator:
    def test_is_operator(self) -> None:
        op = LLMCallOperator("llm-1", _fake_llm, system_prompt="Be helpful.")
        assert isinstance(op, Operator)

    def test_name(self) -> None:
        op = LLMCallOperator("my-llm", _fake_llm)
        assert op.name == "my-llm"

    def test_get_tunables(self) -> None:
        op = LLMCallOperator(
            "llm-1",
            _fake_llm,
            system_prompt="Be concise.",
            user_prompt="Summarise: {text}",
        )
        tunables = op.get_tunables()
        assert len(tunables) == 2
        assert tunables[0] == TunableSpec(
            name="system_prompt",
            kind=TunableKind.PROMPT,
            current_value="Be concise.",
        )
        assert tunables[1] == TunableSpec(
            name="user_prompt",
            kind=TunableKind.PROMPT,
            current_value="Summarise: {text}",
        )

    async def test_execute(self) -> None:
        op = LLMCallOperator(
            "llm-1",
            _fake_llm,
            system_prompt="sys",
            user_prompt="hello",
        )
        result = await op.execute()
        assert result == "Response to: hello"

    async def test_execute_with_extra_kwargs(self) -> None:
        async def _llm_with_extras(
            *, system_prompt: str, user_prompt: str, temperature: float = 0.7
        ) -> str:
            return f"temp={temperature}"

        op = LLMCallOperator("llm-1", _llm_with_extras, user_prompt="hi")
        result = await op.execute(temperature=0.3)
        assert result == "temp=0.3"

    async def test_trace_recorded_on_success(self) -> None:
        op = LLMCallOperator("llm-1", _fake_llm, user_prompt="test")
        await op.execute()
        traces = op.traces
        assert len(traces) == 1
        t = traces[0]
        assert isinstance(t, LLMCallTrace)
        assert t.operator_name == "llm-1"
        assert t.user_prompt == "test"
        assert t.result == "Response to: test"
        assert t.error is None
        assert t.duration_ms >= 0
        assert t.timestamp > 0

    async def test_trace_recorded_on_failure(self) -> None:
        op = LLMCallOperator("llm-fail", _failing_llm)
        with pytest.raises(ValueError, match="LLM unavailable"):
            await op.execute()
        traces = op.traces
        assert len(traces) == 1
        assert traces[0].error == "LLM unavailable"
        assert traces[0].result is None

    async def test_multiple_traces(self) -> None:
        op = LLMCallOperator("llm-1", _fake_llm, user_prompt="a")
        await op.execute()
        await op.execute()
        assert len(op.traces) == 2

    def test_trace_to_dict(self) -> None:
        t = LLMCallTrace(
            operator_name="x",
            system_prompt="sys",
            user_prompt="usr",
            result="ok",
        )
        d = t.to_dict()
        assert d["operator_name"] == "x"
        assert d["system_prompt"] == "sys"

    def test_get_state(self) -> None:
        op = LLMCallOperator("llm-1", _fake_llm, system_prompt="a", user_prompt="b")
        assert op.get_state() == {"system_prompt": "a", "user_prompt": "b"}

    def test_load_state(self) -> None:
        op = LLMCallOperator("llm-1", _fake_llm)
        op.load_state({"system_prompt": "new-sys", "user_prompt": "new-usr"})
        assert op.get_state() == {
            "system_prompt": "new-sys",
            "user_prompt": "new-usr",
        }

    def test_state_roundtrip(self) -> None:
        op = LLMCallOperator("llm-1", _fake_llm, system_prompt="orig-sys", user_prompt="orig-usr")
        original = op.get_state()
        op.load_state({"system_prompt": "changed", "user_prompt": "changed"})
        assert op.get_state() != original
        op.load_state(original)
        assert op.get_state() == original

    def test_tunables_reflect_state(self) -> None:
        op = LLMCallOperator("llm-1", _fake_llm)
        op.load_state({"system_prompt": "updated", "user_prompt": "also updated"})
        tunables = op.get_tunables()
        assert tunables[0].current_value == "updated"
        assert tunables[1].current_value == "also updated"


# ---------------------------------------------------------------------------
# ToolCallOperator
# ---------------------------------------------------------------------------


class TestToolCallOperator:
    def test_is_operator(self) -> None:
        op = ToolCallOperator("tool-1", _fake_tool, tool_description="A search tool")
        assert isinstance(op, Operator)

    def test_name(self) -> None:
        op = ToolCallOperator("search", _fake_tool)
        assert op.name == "search"

    def test_get_tunables(self) -> None:
        op = ToolCallOperator("tool-1", _fake_tool, tool_description="Searches the web")
        tunables = op.get_tunables()
        assert len(tunables) == 1
        assert tunables[0] == TunableSpec(
            name="tool_description",
            kind=TunableKind.TEXT,
            current_value="Searches the web",
        )

    async def test_execute(self) -> None:
        op = ToolCallOperator("tool-1", _fake_tool)
        result = await op.execute(query="test")
        assert result == {"result": "ok", "query": "test"}

    async def test_trace_recorded_on_success(self) -> None:
        op = ToolCallOperator("tool-1", _fake_tool, tool_description="desc")
        await op.execute(q="hello")
        traces = op.traces
        assert len(traces) == 1
        t = traces[0]
        assert isinstance(t, ToolCallTrace)
        assert t.operator_name == "tool-1"
        assert t.tool_description == "desc"
        assert t.kwargs == {"q": "hello"}
        assert t.error is None
        assert t.duration_ms >= 0

    async def test_trace_recorded_on_failure(self) -> None:
        op = ToolCallOperator("tool-fail", _failing_tool)
        with pytest.raises(RuntimeError, match="tool broke"):
            await op.execute()
        traces = op.traces
        assert len(traces) == 1
        assert traces[0].error == "tool broke"

    def test_trace_to_dict(self) -> None:
        t = ToolCallTrace(operator_name="t", tool_description="d", kwargs={"a": 1})
        d = t.to_dict()
        assert d["operator_name"] == "t"
        assert d["kwargs"] == {"a": 1}

    def test_get_state(self) -> None:
        op = ToolCallOperator("t", _fake_tool, tool_description="desc")
        assert op.get_state() == {"tool_description": "desc"}

    def test_load_state(self) -> None:
        op = ToolCallOperator("t", _fake_tool)
        op.load_state({"tool_description": "new desc"})
        assert op.get_state() == {"tool_description": "new desc"}

    def test_state_roundtrip(self) -> None:
        op = ToolCallOperator("t", _fake_tool, tool_description="original")
        original = op.get_state()
        op.load_state({"tool_description": "changed"})
        assert op.get_state() != original
        op.load_state(original)
        assert op.get_state() == original

    def test_tunables_reflect_state(self) -> None:
        op = ToolCallOperator("t", _fake_tool)
        op.load_state({"tool_description": "updated"})
        assert op.get_tunables()[0].current_value == "updated"


# ---------------------------------------------------------------------------
# MemoryCallOperator
# ---------------------------------------------------------------------------


class TestMemoryCallOperator:
    def test_is_operator(self) -> None:
        op = MemoryCallOperator("mem-1", _fake_memory)
        assert isinstance(op, Operator)

    def test_name(self) -> None:
        op = MemoryCallOperator("mem-retriever", _fake_memory)
        assert op.name == "mem-retriever"

    def test_get_tunables(self) -> None:
        op = MemoryCallOperator("mem-1", _fake_memory, enabled=True, max_retries=3)
        tunables = op.get_tunables()
        assert len(tunables) == 2
        assert tunables[0] == TunableSpec(
            name="enabled",
            kind=TunableKind.DISCRETE,
            current_value=True,
            constraints={"choices": [True, False]},
        )
        assert tunables[1] == TunableSpec(
            name="max_retries",
            kind=TunableKind.DISCRETE,
            current_value=3,
            constraints={"min": 1, "max": 10},
        )

    async def test_execute(self) -> None:
        op = MemoryCallOperator("mem-1", _fake_memory)
        result = await op.execute(query="search")
        assert result == ["memory_1", "memory_2"]

    async def test_execute_disabled(self) -> None:
        op = MemoryCallOperator("mem-1", _fake_memory, enabled=False)
        result = await op.execute(query="search")
        assert result is None

    async def test_disabled_records_trace(self) -> None:
        op = MemoryCallOperator("mem-1", _fake_memory, enabled=False)
        await op.execute()
        traces = op.traces
        assert len(traces) == 1
        assert traces[0].enabled is False
        assert traces[0].result is None

    async def test_retry_on_transient_failure(self) -> None:
        op = MemoryCallOperator("mem-1", _make_flaky_memory(fail_times=1), max_retries=3)
        result = await op.execute()
        assert result == ["recovered"]
        traces = op.traces
        assert len(traces) == 1
        assert traces[0].error is None

    async def test_exhausted_retries_raises(self) -> None:
        op = MemoryCallOperator("mem-1", _always_failing_memory, max_retries=2)
        with pytest.raises(ConnectionError, match="permanent error"):
            await op.execute()
        traces = op.traces
        assert len(traces) == 1
        assert traces[0].error == "permanent error"

    async def test_trace_recorded_on_success(self) -> None:
        op = MemoryCallOperator("mem-1", _fake_memory, max_retries=2)
        await op.execute(query="test")
        traces = op.traces
        assert len(traces) == 1
        t = traces[0]
        assert isinstance(t, MemoryCallTrace)
        assert t.operator_name == "mem-1"
        assert t.enabled is True
        assert t.max_retries == 2
        assert t.kwargs == {"query": "test"}
        assert t.error is None
        assert t.duration_ms >= 0

    def test_trace_to_dict(self) -> None:
        t = MemoryCallTrace(operator_name="m", enabled=True, max_retries=1, kwargs={"k": "v"})
        d = t.to_dict()
        assert d["operator_name"] == "m"
        assert d["enabled"] is True

    def test_get_state(self) -> None:
        op = MemoryCallOperator("m", _fake_memory, enabled=True, max_retries=5)
        assert op.get_state() == {"enabled": True, "max_retries": 5}

    def test_load_state(self) -> None:
        op = MemoryCallOperator("m", _fake_memory)
        op.load_state({"enabled": False, "max_retries": 2})
        assert op.get_state() == {"enabled": False, "max_retries": 2}

    def test_load_state_invalid_max_retries(self) -> None:
        op = MemoryCallOperator("m", _fake_memory)
        with pytest.raises(ValueError, match="max_retries must be >= 1"):
            op.load_state({"enabled": False, "max_retries": 0})

    def test_state_roundtrip(self) -> None:
        op = MemoryCallOperator("m", _fake_memory, enabled=True, max_retries=3)
        original = op.get_state()
        op.load_state({"enabled": False, "max_retries": 2})
        assert op.get_state() != original
        op.load_state(original)
        assert op.get_state() == original

    def test_tunables_reflect_state(self) -> None:
        op = MemoryCallOperator("m", _fake_memory)
        op.load_state({"enabled": False, "max_retries": 7})
        tunables = op.get_tunables()
        assert tunables[0].current_value is False
        assert tunables[1].current_value == 7
