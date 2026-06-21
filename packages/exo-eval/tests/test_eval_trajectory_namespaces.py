"""Tests for TrajectoryItem SAR namespace grouping (Tier-3 DX).

Covers:
- Grouped construction (state=, action=, reward=) populates flat fields
- Flat construction still works unchanged
- Conflict between grouped + flat raises ValueError
- to_dict / from_dict round-trip unchanged (wire format unchanged)
- Read accessor views return correct TrajectoryState / TrajectoryAction / TrajectoryReward
- Both styles compare equal when they represent the same data
"""

from __future__ import annotations

import pytest

from exo.eval import (
    TrajectoryAction,
    TrajectoryItem,
    TrajectoryReward,
    TrajectoryState,
)

# ---------------------------------------------------------------------------
# Namespace dataclasses smoke tests
# ---------------------------------------------------------------------------


def test_trajectory_state_defaults() -> None:
    s = TrajectoryState()
    assert s.input == ""
    assert s.messages == ()
    assert s.context == {}


def test_trajectory_action_defaults() -> None:
    a = TrajectoryAction()
    assert a.output == ""
    assert a.tool_calls == ()


def test_trajectory_reward_defaults() -> None:
    r = TrajectoryReward()
    assert r.score is None
    assert r.status == "success"
    assert r.metadata == {}


def test_trajectory_state_fields() -> None:
    msgs: tuple[dict, ...] = ({"role": "user", "content": "hi"},)
    s = TrajectoryState(input="task", messages=msgs, context={"k": "v"})
    assert s.input == "task"
    assert s.messages == msgs
    assert s.context == {"k": "v"}


# ---------------------------------------------------------------------------
# Grouped construction populates flat fields
# ---------------------------------------------------------------------------


def test_grouped_state_populates_flat() -> None:
    msgs: tuple[dict, ...] = ({"role": "user", "content": "hello"},)
    item = TrajectoryItem(state=TrajectoryState(input="hello", messages=msgs, context={"x": 1}))
    assert item.input == "hello"
    assert item.messages == msgs
    assert item.context == {"x": 1}


def test_grouped_action_populates_flat() -> None:
    tcs: tuple[dict, ...] = ({"name": "search", "args": {}},)
    item = TrajectoryItem(action=TrajectoryAction(output="done", tool_calls=tcs))
    assert item.output == "done"
    assert item.tool_calls == tcs


def test_grouped_reward_populates_flat() -> None:
    item = TrajectoryItem(
        reward=TrajectoryReward(score=0.9, status="failure", metadata={"reason": "x"})
    )
    assert item.score == 0.9
    assert item.status == "failure"
    assert item.metadata == {"reason": "x"}


def test_all_grouped_together() -> None:
    item = TrajectoryItem(
        id="abc",
        task_id="t1",
        agent_id="a1",
        step=3,
        state=TrajectoryState(input="q", context={"env": "prod"}),
        action=TrajectoryAction(output="ans"),
        reward=TrajectoryReward(score=1.0, status="success"),
    )
    assert item.input == "q"
    assert item.context == {"env": "prod"}
    assert item.output == "ans"
    assert item.score == 1.0
    assert item.status == "success"
    # identity fields unaffected
    assert item.id == "abc"
    assert item.task_id == "t1"
    assert item.step == 3


# ---------------------------------------------------------------------------
# Flat construction still works (backward compat)
# ---------------------------------------------------------------------------


def test_flat_construction_unchanged() -> None:
    item = TrajectoryItem(
        id="flat1",
        task_id="task",
        agent_id="agent",
        step=2,
        input="flat input",
        messages=({"role": "user", "content": "flat input"},),
        context={"mode": "test"},
        output="flat output",
        tool_calls=({"name": "tool", "args": {}},),
        score=0.5,
        status="success",
        metadata={"tag": "flat"},
    )
    assert item.id == "flat1"
    assert item.input == "flat input"
    assert item.output == "flat output"
    assert item.score == 0.5
    assert item.metadata == {"tag": "flat"}


def test_flat_defaults_unchanged() -> None:
    item = TrajectoryItem()
    assert item.input == ""
    assert item.output == ""
    assert item.messages == ()
    assert item.tool_calls == ()
    assert item.score is None
    assert item.status == "success"
    assert item.metadata == {}
    assert item.context == {}


# ---------------------------------------------------------------------------
# Conflict detection raises ValueError
# ---------------------------------------------------------------------------


def test_conflict_state_input_raises() -> None:
    with pytest.raises(ValueError, match="Cannot specify both"):
        TrajectoryItem(
            input="flat",
            state=TrajectoryState(input="grouped"),
        )


def test_conflict_state_messages_raises() -> None:
    flat_msgs: tuple[dict, ...] = ({"role": "user", "content": "a"},)
    grouped_msgs: tuple[dict, ...] = ({"role": "user", "content": "b"},)
    with pytest.raises(ValueError, match="Cannot specify both"):
        TrajectoryItem(
            messages=flat_msgs,
            state=TrajectoryState(messages=grouped_msgs),
        )


def test_conflict_action_output_raises() -> None:
    with pytest.raises(ValueError, match="Cannot specify both"):
        TrajectoryItem(
            output="flat",
            action=TrajectoryAction(output="grouped"),
        )


def test_conflict_action_tool_calls_raises() -> None:
    flat_tc: tuple[dict, ...] = ({"name": "a"},)
    grouped_tc: tuple[dict, ...] = ({"name": "b"},)
    with pytest.raises(ValueError, match="Cannot specify both"):
        TrajectoryItem(
            tool_calls=flat_tc,
            action=TrajectoryAction(tool_calls=grouped_tc),
        )


def test_conflict_reward_score_raises() -> None:
    with pytest.raises(ValueError, match="Cannot specify both"):
        TrajectoryItem(
            score=0.5,
            reward=TrajectoryReward(score=0.9),
        )


def test_conflict_reward_status_raises() -> None:
    with pytest.raises(ValueError, match="Cannot specify both"):
        TrajectoryItem(
            status="failure",
            reward=TrajectoryReward(status="success"),
        )


def test_conflict_reward_metadata_raises() -> None:
    with pytest.raises(ValueError, match="Cannot specify both"):
        TrajectoryItem(
            metadata={"a": 1},
            reward=TrajectoryReward(metadata={"b": 2}),
        )


# ---------------------------------------------------------------------------
# No conflict when grouped + flat agree on same value
# ---------------------------------------------------------------------------


def test_no_conflict_when_values_agree() -> None:
    """Grouped + flat with the same value is allowed (idempotent)."""
    item = TrajectoryItem(
        input="same",
        state=TrajectoryState(input="same"),
    )
    assert item.input == "same"


# ---------------------------------------------------------------------------
# to_dict / from_dict wire format unchanged
# ---------------------------------------------------------------------------


def test_to_dict_flat_unchanged() -> None:
    item = TrajectoryItem(
        id="w1",
        task_id="t",
        agent_id="a",
        step=0,
        input="in",
        output="out",
        score=1.0,
        status="success",
    )
    d = item.to_dict()
    assert set(d.keys()) == {
        "id",
        "task_id",
        "agent_id",
        "step",
        "timestamp",
        "input",
        "messages",
        "context",
        "output",
        "tool_calls",
        "score",
        "status",
        "metadata",
    }
    assert d["input"] == "in"
    assert d["output"] == "out"
    assert d["score"] == 1.0
    # grouped namespace keys must NOT appear in wire format
    assert "state" not in d
    assert "action" not in d
    assert "reward" not in d


def test_to_dict_grouped_wire_format_same() -> None:
    """Grouped construction produces the same to_dict as flat."""
    flat = TrajectoryItem(id="x", input="q", output="a", score=0.8)
    grouped = TrajectoryItem(
        id="x",
        state=TrajectoryState(input="q"),
        action=TrajectoryAction(output="a"),
        reward=TrajectoryReward(score=0.8),
    )
    flat_d = flat.to_dict()
    grouped_d = grouped.to_dict()
    # timestamp will differ; override for comparison
    flat_d["timestamp"] = grouped_d["timestamp"] = 0.0
    assert flat_d == grouped_d


def test_from_dict_round_trip() -> None:
    original = TrajectoryItem(
        id="rt1",
        task_id="task",
        input="rt input",
        output="rt output",
        messages=({"role": "user", "content": "rt input"},),
        score=0.75,
        status="success",
        metadata={"k": "v"},
    )
    restored = TrajectoryItem.from_dict(original.to_dict())
    assert restored.id == original.id
    assert restored.input == original.input
    assert restored.output == original.output
    assert restored.messages == original.messages
    assert restored.score == original.score
    assert restored.status == original.status
    assert restored.metadata == original.metadata


def test_from_dict_extra_keys_ignored() -> None:
    """from_dict must not choke if 'state'/'action'/'reward' keys happen to be absent."""
    d = TrajectoryItem(id="a", input="x").to_dict()
    restored = TrajectoryItem.from_dict(d)
    assert restored.id == "a"
    assert restored.input == "x"


# ---------------------------------------------------------------------------
# Read accessor views
# ---------------------------------------------------------------------------


def test_state_view_from_flat() -> None:
    msgs: tuple[dict, ...] = ({"role": "user", "content": "v"},)
    item = TrajectoryItem(input="v", messages=msgs, context={"c": 1})
    sv = item.state_view
    assert isinstance(sv, TrajectoryState)
    assert sv.input == "v"
    assert sv.messages == msgs
    assert sv.context == {"c": 1}


def test_action_view_from_flat() -> None:
    tcs: tuple[dict, ...] = ({"name": "t"},)
    item = TrajectoryItem(output="result", tool_calls=tcs)
    av = item.action_view
    assert isinstance(av, TrajectoryAction)
    assert av.output == "result"
    assert av.tool_calls == tcs


def test_reward_view_from_flat() -> None:
    item = TrajectoryItem(score=0.4, status="failure", metadata={"info": "z"})
    rv = item.reward_view
    assert isinstance(rv, TrajectoryReward)
    assert rv.score == 0.4
    assert rv.status == "failure"
    assert rv.metadata == {"info": "z"}


def test_state_view_from_grouped() -> None:
    item = TrajectoryItem(state=TrajectoryState(input="grouped"))
    assert item.state_view.input == "grouped"


def test_action_view_from_grouped() -> None:
    item = TrajectoryItem(action=TrajectoryAction(output="g-out"))
    assert item.action_view.output == "g-out"


def test_reward_view_from_grouped() -> None:
    item = TrajectoryItem(reward=TrajectoryReward(score=0.3))
    assert item.reward_view.score == 0.3


# ---------------------------------------------------------------------------
# Equality — dataclass equality is on flat fields only
# ---------------------------------------------------------------------------


def test_grouped_and_flat_are_equal_instances() -> None:
    """Two items with same flat-field values are equal regardless of how built."""
    flat = TrajectoryItem(id="e1", input="hi", output="there", step=0, timestamp=0.0)
    grouped = TrajectoryItem(
        id="e1",
        step=0,
        timestamp=0.0,
        state=TrajectoryState(input="hi"),
        action=TrajectoryAction(output="there"),
    )
    assert flat == grouped


# ---------------------------------------------------------------------------
# Grouped params excluded from repr (transparency)
# ---------------------------------------------------------------------------


def test_grouped_params_not_in_repr() -> None:
    item = TrajectoryItem(state=TrajectoryState(input="x"))
    r = repr(item)
    assert "state=" not in r or "state=None" in r
