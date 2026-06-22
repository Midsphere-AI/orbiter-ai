"""Tests for ContextWindowInfo dataclass and build_context_window_info builder."""

from __future__ import annotations

import dataclasses
import types
from typing import Any

from exo.context.info import ContextWindowInfo, build_context_window_info
from exo.context.token_tracker import TokenTracker
from exo.types import AssistantMessage, SystemMessage, ToolResult, UserMessage

# ---------------------------------------------------------------------------
# 1. ContextWindowInfo is frozen (immutable)
# ---------------------------------------------------------------------------


def test_context_window_info_is_frozen():
    info = ContextWindowInfo(
        step=0,
        max_steps=10,
        is_initial=False,
        total_messages=5,
        system_count=1,
        user_count=2,
        assistant_count=1,
        tool_result_count=1,
        context_window=128_000,
        input_tokens=500,
        output_tokens=100,
        fill_ratio=0.004,
        token_pressure_threshold=0.8,
        force=False,
        cumulative_input_tokens=500,
        cumulative_output_tokens=100,
    )
    try:
        info.step = 99  # type: ignore[misc]
        raise AssertionError("Should have raised FrozenInstanceError")
    except dataclasses.FrozenInstanceError:
        pass


# ---------------------------------------------------------------------------
# 2. build_context_window_info counts message types correctly
# ---------------------------------------------------------------------------


def test_build_basic_msg_list_counts():
    msg_list = [
        SystemMessage(content="system prompt"),
        UserMessage(content="hello"),
        AssistantMessage(content="hi"),
        UserMessage(content="more"),
        ToolResult(tool_call_id="t1", tool_name="search", content="result"),
    ]
    cfg = types.SimpleNamespace(overflow="summarize", limit=20, keep_recent=5, history_rounds=20)
    info = build_context_window_info(msg_list, cfg)

    assert info.total_messages == 5
    assert info.system_count == 1
    assert info.user_count == 2
    assert info.assistant_count == 1
    assert info.tool_result_count == 1


# ---------------------------------------------------------------------------
# 3. build_context_window_info extracts tokens from last_usage
# ---------------------------------------------------------------------------


def test_build_extracts_last_usage_tokens():
    msg_list = [UserMessage(content="hi")]
    cfg = types.SimpleNamespace(overflow="summarize", limit=20, keep_recent=5, history_rounds=20)
    usage = types.SimpleNamespace(input_tokens=500, output_tokens=100)

    info = build_context_window_info(msg_list, cfg, last_usage=usage)

    assert info.input_tokens == 500
    assert info.output_tokens == 100


# ---------------------------------------------------------------------------
# 4. build_context_window_info computes fill_ratio correctly
# ---------------------------------------------------------------------------


def test_build_fill_ratio():
    msg_list = [UserMessage(content="hi")]
    cfg = types.SimpleNamespace(overflow="summarize", limit=20, keep_recent=5, history_rounds=20)
    usage = types.SimpleNamespace(input_tokens=500, output_tokens=100)

    info = build_context_window_info(msg_list, cfg, context_window_tokens=1000, last_usage=usage)

    assert info.fill_ratio == 0.5  # 500 / 1000


# ---------------------------------------------------------------------------
# 5. fill_ratio=0.0 when no context_window_tokens
# ---------------------------------------------------------------------------


def test_build_fill_ratio_zero_without_context_window():
    msg_list = [UserMessage(content="hi")]
    cfg = types.SimpleNamespace(overflow="summarize", limit=20, keep_recent=5, history_rounds=20)
    usage = types.SimpleNamespace(input_tokens=500, output_tokens=100)

    info = build_context_window_info(msg_list, cfg, last_usage=usage)

    assert info.context_window is None
    assert info.fill_ratio == 0.0


# ---------------------------------------------------------------------------
# 6. build_context_window_info extracts trajectory from token tracker
# ---------------------------------------------------------------------------


def test_build_extracts_trajectory_from_tracker():
    tracker = TokenTracker()
    tracker.add_step("agent-a", prompt_tokens=100, output_tokens=50)
    tracker.add_step("agent-a", prompt_tokens=200, output_tokens=80)
    tracker.add_step("agent-b", prompt_tokens=300, output_tokens=90)

    msg_list = [UserMessage(content="hi")]
    cfg = types.SimpleNamespace(overflow="summarize", limit=20, keep_recent=5, history_rounds=20)

    info = build_context_window_info(msg_list, cfg, agent_name="agent-a", token_tracker=tracker)

    assert len(info.trajectory) == 2
    assert info.cumulative_input_tokens == 300  # 100 + 200
    assert info.cumulative_output_tokens == 130  # 50 + 80


# ---------------------------------------------------------------------------
# 7. build_context_window_info reads config fields via duck-typing
# ---------------------------------------------------------------------------


def test_build_reads_config_duck_typed():
    msg_list = [UserMessage(content="hi")]
    cfg = types.SimpleNamespace(
        overflow="truncate",
        limit=15,
        keep_recent=3,
        token_pressure=0.7,
        history_rounds=15,
    )

    info = build_context_window_info(msg_list, cfg)

    assert info.overflow == "truncate"
    assert info.limit == 15
    assert info.keep_recent == 3
    assert info.token_pressure_threshold == 0.7


# ---------------------------------------------------------------------------
# 8. step=-1 sets is_initial=True
# ---------------------------------------------------------------------------


def test_build_step_minus_one_is_initial():
    msg_list = [UserMessage(content="hi")]
    cfg = types.SimpleNamespace(overflow="summarize", limit=20, keep_recent=5, history_rounds=20)

    info = build_context_window_info(msg_list, cfg, step=-1)

    assert info.step == -1
    assert info.is_initial is True


# ---------------------------------------------------------------------------
# 9. step=3 sets is_initial=False
# ---------------------------------------------------------------------------


def test_build_step_positive_not_initial():
    msg_list = [UserMessage(content="hi")]
    cfg = types.SimpleNamespace(overflow="summarize", limit=20, keep_recent=5, history_rounds=20)

    info = build_context_window_info(msg_list, cfg, step=3)

    assert info.step == 3
    assert info.is_initial is False


# ---------------------------------------------------------------------------
# 10. No tracker/usage gives zeros
# ---------------------------------------------------------------------------


def test_build_no_tracker_no_usage_gives_zeros():
    msg_list = [UserMessage(content="hi")]
    cfg = types.SimpleNamespace(overflow="summarize", limit=20, keep_recent=5, history_rounds=20)

    info = build_context_window_info(msg_list, cfg)

    assert info.input_tokens == 0
    assert info.output_tokens == 0
    assert info.fill_ratio == 0.0
    assert info.cumulative_input_tokens == 0
    assert info.cumulative_output_tokens == 0
    assert info.trajectory == ()


# ---------------------------------------------------------------------------
# 11. Token-tracker failure is logged at DEBUG, not silently dropped
# ---------------------------------------------------------------------------


def test_token_tracker_failure_is_logged_not_swallowed(caplog: Any) -> None:
    """A broken token tracker must log at DEBUG, not silently discard the error."""

    class BrokenTracker:
        def get_trajectory(self, agent_name: str) -> list[object]:
            raise RuntimeError("tracker exploded")

    msg_list = [UserMessage(content="hi")]
    cfg = types.SimpleNamespace(overflow="summarize", limit=20, keep_recent=5, history_rounds=20)

    import logging

    with caplog.at_level(logging.DEBUG, logger="exo.context.info"):
        info = build_context_window_info(
            msg_list,
            cfg,
            agent_name="test-agent",
            token_tracker=BrokenTracker(),
        )

    # The result must be usable (trajectory empty, no exception raised)
    assert info.trajectory == ()
    assert info.cumulative_input_tokens == 0

    # A DEBUG log record must have been emitted mentioning the agent name
    debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert debug_records, "Expected at least one DEBUG log entry for the tracker failure"
    combined = " ".join(r.getMessage() for r in debug_records)
    assert "test-agent" in combined or "trajectory" in combined
