"""Unit tests for exo._internal.run_helpers.

Tests the shared helpers used by both the non-streaming (agent._run_inner)
and the streaming (runner._stream) run loops.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from exo._internal.run_helpers import (
    check_token_budget_pressure,
    drain_ephemeral_messages,
    drain_injected_messages,
    resolve_instructions,
)
from exo.types import UserMessage

# ---------------------------------------------------------------------------
# resolve_instructions
# ---------------------------------------------------------------------------


class TestResolveInstructions:
    """Tests for resolve_instructions()."""

    async def test_string_instructions(self) -> None:
        agent = SimpleNamespace(instructions="You are a helpful assistant.", name="test")
        result = await resolve_instructions(agent)
        assert result == "You are a helpful assistant."

    async def test_empty_string_instructions(self) -> None:
        agent = SimpleNamespace(instructions="", name="test")
        result = await resolve_instructions(agent)
        assert result == ""

    async def test_none_instructions(self) -> None:
        agent = SimpleNamespace(instructions=None, name="test")
        result = await resolve_instructions(agent)
        assert result == ""

    async def test_sync_callable_instructions(self) -> None:
        def instr(name: str) -> str:
            return f"Hello from {name}"

        agent = SimpleNamespace(instructions=instr, name="my_agent")
        result = await resolve_instructions(agent)
        assert result == "Hello from my_agent"

    async def test_async_callable_instructions(self) -> None:
        async def instr(name: str) -> str:
            await asyncio.sleep(0)
            return f"Async hello from {name}"

        agent = SimpleNamespace(instructions=instr, name="async_agent")
        result = await resolve_instructions(agent)
        assert result == "Async hello from async_agent"

    async def test_callable_returning_non_string(self) -> None:
        def instr(name: str) -> int:
            return 42

        agent = SimpleNamespace(instructions=instr, name="test")
        result = await resolve_instructions(agent)
        assert result == "42"

    async def test_string_coercion(self) -> None:
        # Non-string truthy value passed as instructions
        agent = SimpleNamespace(instructions=123, name="test")
        result = await resolve_instructions(agent)
        assert result == "123"


# ---------------------------------------------------------------------------
# drain_ephemeral_messages
# ---------------------------------------------------------------------------


class TestDrainEphemeralMessages:
    """Tests for drain_ephemeral_messages()."""

    def _make_agent(self, *messages: str) -> SimpleNamespace:
        q: asyncio.Queue = asyncio.Queue()
        for msg in messages:
            q.put_nowait(UserMessage(content=msg))
        return SimpleNamespace(_ephemeral_messages=q)

    async def test_empty_queue(self) -> None:
        agent = self._make_agent()
        result = drain_ephemeral_messages(agent)
        assert result == []

    async def test_single_message(self) -> None:
        agent = self._make_agent("ephemeral content")
        result = drain_ephemeral_messages(agent)
        assert len(result) == 1
        assert isinstance(result[0], UserMessage)
        assert result[0].content == "ephemeral content"

    async def test_multiple_messages(self) -> None:
        agent = self._make_agent("first", "second", "third")
        result = drain_ephemeral_messages(agent)
        assert len(result) == 3
        contents = [m.content for m in result]
        assert contents == ["first", "second", "third"]

    async def test_queue_drained(self) -> None:
        agent = self._make_agent("msg1", "msg2")
        drain_ephemeral_messages(agent)
        # Queue should be empty after drain
        assert agent._ephemeral_messages.empty()

    async def test_drain_returns_list_not_generator(self) -> None:
        agent = self._make_agent("x")
        result = drain_ephemeral_messages(agent)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# drain_injected_messages
# ---------------------------------------------------------------------------


class TestDrainInjectedMessages:
    """Tests for drain_injected_messages()."""

    def _make_agent(self, *messages: str) -> SimpleNamespace:
        q: asyncio.Queue = asyncio.Queue()
        for msg in messages:
            q.put_nowait(msg)
        return SimpleNamespace(_injected_messages=q)

    async def test_empty_queue(self) -> None:
        agent = self._make_agent()
        msg_list: list = []
        count = drain_injected_messages(agent, msg_list)
        assert count == 0
        assert msg_list == []

    async def test_single_injection(self) -> None:
        agent = self._make_agent("injected content")
        msg_list: list = []
        count = drain_injected_messages(agent, msg_list)
        assert count == 1
        assert len(msg_list) == 1
        assert isinstance(msg_list[0], UserMessage)
        assert msg_list[0].content == "injected content"

    async def test_multiple_injections(self) -> None:
        agent = self._make_agent("one", "two", "three")
        msg_list: list = []
        count = drain_injected_messages(agent, msg_list)
        assert count == 3
        assert [m.content for m in msg_list] == ["one", "two", "three"]

    async def test_appends_to_existing_list(self) -> None:
        agent = self._make_agent("new")
        existing = [UserMessage(content="old")]
        drain_injected_messages(agent, existing)
        assert len(existing) == 2
        assert existing[0].content == "old"
        assert existing[1].content == "new"

    async def test_queue_drained_after_call(self) -> None:
        agent = self._make_agent("a", "b")
        drain_injected_messages(agent, [])
        assert agent._injected_messages.empty()


# ---------------------------------------------------------------------------
# check_token_budget_pressure
# ---------------------------------------------------------------------------


class TestCheckTokenBudgetPressure:
    """Tests for check_token_budget_pressure()."""

    def _ctx(self, pressure: float = 0.8) -> SimpleNamespace:
        """Create a minimal context-like object."""
        cfg = SimpleNamespace(token_pressure=pressure)
        return SimpleNamespace(config=cfg)

    def test_no_context_window_returns_no_pressure(self) -> None:
        force, ratio, _trigger = check_token_budget_pressure(5000, None, self._ctx())
        assert force is False
        assert ratio == 0.0

    def test_zero_input_tokens_returns_no_pressure(self) -> None:
        force, _ratio, _trigger = check_token_budget_pressure(0, 100_000, self._ctx())
        assert force is False

    def test_below_threshold(self) -> None:
        # 60k / 100k = 0.6, threshold = 0.8 → no pressure
        force, ratio, trigger = check_token_budget_pressure(60_000, 100_000, self._ctx(0.8))
        assert force is False
        assert abs(ratio - 0.6) < 1e-9
        assert trigger == 0.8

    def test_above_threshold(self) -> None:
        # 90k / 100k = 0.9, threshold = 0.8 → pressure
        force, ratio, trigger = check_token_budget_pressure(90_000, 100_000, self._ctx(0.8))
        assert force is True
        assert abs(ratio - 0.9) < 1e-9
        assert trigger == 0.8

    def test_exactly_at_threshold_no_pressure(self) -> None:
        # fill_ratio > trigger (not >=), so exactly 0.8 is no pressure
        force, _ratio, _trigger = check_token_budget_pressure(80_000, 100_000, self._ctx(0.8))
        assert force is False

    def test_custom_threshold(self) -> None:
        # 55k / 100k = 0.55, threshold = 0.5 → pressure
        force, _ratio, trigger = check_token_budget_pressure(55_000, 100_000, self._ctx(0.5))
        assert force is True
        assert trigger == 0.5

    def test_token_budget_trigger_alias(self) -> None:
        """Legacy config with token_budget_trigger (no token_pressure)."""
        cfg = SimpleNamespace(token_budget_trigger=0.7)
        ctx = SimpleNamespace(config=cfg)
        force, _ratio, trigger = check_token_budget_pressure(80_000, 100_000, ctx)
        assert force is True
        assert trigger == 0.7

    def test_default_trigger_when_no_attribute(self) -> None:
        """Config with neither attribute — should default to 0.8."""
        ctx = SimpleNamespace(config=SimpleNamespace())
        force, _ratio, trigger = check_token_budget_pressure(85_000, 100_000, ctx)
        assert force is True
        assert trigger == 0.8

    def test_bare_context_without_config_attribute(self) -> None:
        """Context without .config — the context itself is used as config."""
        ctx = SimpleNamespace(token_pressure=0.75)
        force, _ratio, trigger = check_token_budget_pressure(80_000, 100_000, ctx)
        assert force is True
        assert trigger == 0.75
