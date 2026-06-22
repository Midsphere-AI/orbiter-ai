"""Tests for OperatorTrainer and FileCheckpointStore."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from exo.train.operator.base import (  # pyright: ignore[reportMissingImports]
    Operator,
    TunableKind,
    TunableSpec,
)
from exo.train.operator_trainer import (  # pyright: ignore[reportMissingImports]
    FileCheckpointStore,
    OperatorTrainConfig,
    OperatorTrainer,
)
from exo.train.trainer import (  # pyright: ignore[reportMissingImports]
    TrainConfig,
    Trainer,
    TrainerError,
    TrainerState,
    TrainMetrics,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubOperator(Operator):
    """Minimal operator for testing."""

    def __init__(self, op_name: str, prompt: str = "default") -> None:
        self._name = op_name
        self._prompt = prompt

    @property
    def name(self) -> str:
        return self._name

    async def execute(self, **kwargs: Any) -> Any:
        return self._prompt

    def get_tunables(self) -> list[TunableSpec]:
        return [TunableSpec(name="prompt", kind=TunableKind.PROMPT, current_value=self._prompt)]

    def get_state(self) -> dict[str, Any]:
        return {"prompt": self._prompt}

    def load_state(self, state: dict[str, Any]) -> None:
        self._prompt = state.get("prompt", self._prompt)


class _StubAgent:
    """Agent with operators."""

    def __init__(self, operators: list[Operator] | None = None) -> None:
        self.operators = operators or []


class _StubOptimizer:
    """Mock optimizer with backward/step."""

    def __init__(self) -> None:
        self.backward_calls: list[list[dict[str, Any]]] = []
        self.step_calls = 0

    async def backward(self, evaluated_cases: Sequence[dict[str, Any]]) -> list[str]:
        self.backward_calls.append(list(evaluated_cases))
        return ["gradient"]

    async def step(self) -> dict[str, str]:
        self.step_calls += 1
        return {"op": "improved"}


async def _good_eval_fn(agent: Any, data: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Eval function that returns perfect scores."""
    return [{"input": d["input"], "output": "ok", "score": 1.0} for d in data]


async def _mediocre_eval_fn(agent: Any, data: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Eval function that returns 50% scores."""
    return [{"input": d["input"], "output": "meh", "score": 0.5} for d in data]


def _make_improving_eval_fn() -> Any:
    """Return an eval function that improves its score over successive calls."""
    call_count = 0

    async def _improving_eval_fn(
        agent: Any, data: Sequence[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        nonlocal call_count
        call_count += 1
        score = min(1.0, 0.3 * call_count)
        return [{"input": d["input"], "output": "better", "score": score} for d in data]

    return _improving_eval_fn


async def _failing_eval_fn(agent: Any, data: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    raise RuntimeError("eval failed")


# ---------------------------------------------------------------------------
# FileCheckpointStore
# ---------------------------------------------------------------------------


class TestFileCheckpointStore:
    def test_save_and_load(self, tmp_path: Path) -> None:
        store = FileCheckpointStore(tmp_path / "ckpts")
        data = {"epoch": 0, "accuracy": 0.9, "operator_states": {"op1": {"p": "v"}}}
        store.save(0, data)
        loaded = store.load(0)
        assert loaded == data

    def test_load_nonexistent(self, tmp_path: Path) -> None:
        store = FileCheckpointStore(tmp_path / "empty")
        assert store.load(0) is None
        assert store.load() is None

    def test_latest_epoch(self, tmp_path: Path) -> None:
        store = FileCheckpointStore(tmp_path / "ckpts")
        store.save(0, {"epoch": 0})
        store.save(1, {"epoch": 1})
        store.save(3, {"epoch": 3})
        assert store.latest_epoch() == 3

    def test_latest_epoch_empty(self, tmp_path: Path) -> None:
        store = FileCheckpointStore(tmp_path / "nope")
        assert store.latest_epoch() is None

    def test_load_latest(self, tmp_path: Path) -> None:
        store = FileCheckpointStore(tmp_path / "ckpts")
        store.save(0, {"epoch": 0})
        store.save(2, {"epoch": 2})
        loaded = store.load()  # should load epoch 2
        assert loaded is not None
        assert loaded["epoch"] == 2

    def test_directory_created_on_save(self, tmp_path: Path) -> None:
        store = FileCheckpointStore(tmp_path / "nested" / "dir")
        store.save(0, {"test": True})
        assert (tmp_path / "nested" / "dir").exists()

    def test_directory_property(self, tmp_path: Path) -> None:
        store = FileCheckpointStore(tmp_path / "mydir")
        assert store.directory == tmp_path / "mydir"


# ---------------------------------------------------------------------------
# OperatorTrainer — is a Trainer
# ---------------------------------------------------------------------------


class TestOperatorTrainerIsTrainer:
    def test_extends_trainer(self) -> None:
        t = OperatorTrainer()
        assert isinstance(t, Trainer)

    def test_default_state(self) -> None:
        t = OperatorTrainer()
        assert t.state == TrainerState.CREATED

    def test_custom_config(self) -> None:
        cfg = OperatorTrainConfig(epochs=5, checkpoint_dir="/tmp/ckpt")
        t = OperatorTrainer(cfg)
        assert t.config.epochs == 5


# ---------------------------------------------------------------------------
# OperatorTrainer — check_agent
# ---------------------------------------------------------------------------


class TestOperatorTrainerCheckAgent:
    def test_valid_agent(self) -> None:
        t = OperatorTrainer()
        agent = _StubAgent([_StubOperator("op1")])
        t.check_agent(agent)  # should not raise

    def test_none_agent(self) -> None:
        t = OperatorTrainer()
        with pytest.raises(TrainerError, match="agent is required"):
            t.check_agent(None)

    def test_no_operators(self) -> None:
        t = OperatorTrainer()
        with pytest.raises(TrainerError, match="non-empty 'operators'"):
            t.check_agent(_StubAgent([]))

    def test_no_operators_attr(self) -> None:
        t = OperatorTrainer()
        with pytest.raises(TrainerError, match="non-empty 'operators'"):
            t.check_agent(object())

    def test_invalid_operator_type(self) -> None:
        t = OperatorTrainer()
        agent = _StubAgent()
        agent.operators = ["not-an-operator"]  # type: ignore[assignment]
        with pytest.raises(TrainerError, match="Operator instances"):
            t.check_agent(agent)


# ---------------------------------------------------------------------------
# OperatorTrainer — check_dataset
# ---------------------------------------------------------------------------


class TestOperatorTrainerCheckDataset:
    def test_valid_data(self) -> None:
        t = OperatorTrainer()
        t.check_dataset([{"input": "hello"}])

    def test_empty_data(self) -> None:
        t = OperatorTrainer()
        with pytest.raises(TrainerError, match="non-empty"):
            t.check_dataset([])

    def test_not_list(self) -> None:
        t = OperatorTrainer()
        with pytest.raises(TrainerError, match="list or tuple"):
            t.check_dataset("not a list")

    def test_missing_input_key(self) -> None:
        t = OperatorTrainer()
        with pytest.raises(TrainerError, match="'input' key"):
            t.check_dataset([{"output": "no input"}])

    def test_with_test_data(self) -> None:
        t = OperatorTrainer()
        t.check_dataset(
            [{"input": "train"}],
            test_data=[{"input": "test"}],
        )

    def test_invalid_test_data(self) -> None:
        t = OperatorTrainer()
        with pytest.raises(TrainerError, match="test_data must be"):
            t.check_dataset([{"input": "ok"}], test_data="bad")


# ---------------------------------------------------------------------------
# OperatorTrainer — check_reward / check_config
# ---------------------------------------------------------------------------


class TestOperatorTrainerCheckRewardConfig:
    def test_check_reward_noop(self) -> None:
        t = OperatorTrainer()
        t.check_reward()  # should not raise

    def test_check_config_valid(self) -> None:
        t = OperatorTrainer(TrainConfig(epochs=3))
        t.check_config()  # should not raise

    def test_check_config_invalid_epochs(self) -> None:
        t = OperatorTrainer(TrainConfig(epochs=0))
        with pytest.raises(TrainerError, match="epochs must be"):
            t.check_config()


# ---------------------------------------------------------------------------
# OperatorTrainer — train()
# ---------------------------------------------------------------------------


class TestOperatorTrainerTrain:
    async def test_train_full_loop(self) -> None:
        optimizer = _StubOptimizer()
        t = OperatorTrainer(
            OperatorTrainConfig(epochs=3),
            optimizer=optimizer,
            eval_fn=_good_eval_fn,
        )
        agent = _StubAgent([_StubOperator("op1")])
        t.check_agent(agent)
        t.check_dataset([{"input": "a"}, {"input": "b"}])
        t.check_reward()
        t.check_config()
        t.mark_validated()

        metrics = await t.train()

        assert t.state == TrainerState.COMPLETED
        assert isinstance(metrics, TrainMetrics)
        assert metrics.accuracy == 1.0
        assert metrics.steps == 3
        assert optimizer.step_calls == 3
        assert len(optimizer.backward_calls) == 3

    async def test_train_without_validation_fails(self) -> None:
        t = OperatorTrainer(optimizer=_StubOptimizer(), eval_fn=_good_eval_fn)
        with pytest.raises(TrainerError, match="must be validated"):
            await t.train()

    async def test_train_without_optimizer_fails(self) -> None:
        t = OperatorTrainer(eval_fn=_good_eval_fn)
        agent = _StubAgent([_StubOperator("op1")])
        t.check_agent(agent)
        t.check_dataset([{"input": "x"}])
        t.check_reward()
        t.check_config()
        t.mark_validated()
        with pytest.raises(TrainerError, match="optimizer is required"):
            await t.train()

    async def test_train_without_eval_fn_fails(self) -> None:
        t = OperatorTrainer(optimizer=_StubOptimizer())
        agent = _StubAgent([_StubOperator("op1")])
        t.check_agent(agent)
        t.check_dataset([{"input": "x"}])
        t.check_reward()
        t.check_config()
        t.mark_validated()
        with pytest.raises(TrainerError, match="eval_fn is required"):
            await t.train()

    async def test_train_failure_sets_failed_state(self) -> None:
        t = OperatorTrainer(
            OperatorTrainConfig(epochs=2),
            optimizer=_StubOptimizer(),
            eval_fn=_failing_eval_fn,
        )
        agent = _StubAgent([_StubOperator("op1")])
        t.check_agent(agent)
        t.check_dataset([{"input": "a"}])
        t.check_reward()
        t.check_config()
        t.mark_validated()
        with pytest.raises(RuntimeError, match="eval failed"):
            await t.train()
        assert t.state == TrainerState.FAILED

    async def test_train_with_test_data(self) -> None:
        t = OperatorTrainer(
            OperatorTrainConfig(epochs=1),
            optimizer=_StubOptimizer(),
            eval_fn=_mediocre_eval_fn,
        )
        agent = _StubAgent([_StubOperator("op1")])
        t.check_agent(agent)
        t.check_dataset(
            [{"input": "train"}],
            test_data=[{"input": "test"}],
        )
        t.check_reward()
        t.check_config()
        t.mark_validated()

        metrics = await t.train()
        assert metrics.accuracy == 0.5
        assert metrics.loss == 0.5


# ---------------------------------------------------------------------------
# OperatorTrainer — checkpoint/resume
# ---------------------------------------------------------------------------


class TestOperatorTrainerCheckpoint:
    async def test_saves_checkpoints(self, tmp_path: Path) -> None:
        store = FileCheckpointStore(tmp_path / "ckpts")
        t = OperatorTrainer(
            OperatorTrainConfig(epochs=2),
            optimizer=_StubOptimizer(),
            eval_fn=_good_eval_fn,
            checkpoint_store=store,
        )
        agent = _StubAgent([_StubOperator("op1", prompt="hello")])
        t.check_agent(agent)
        t.check_dataset([{"input": "x"}])
        t.check_reward()
        t.check_config()
        t.mark_validated()

        await t.train()

        # Two checkpoints should exist (epochs 0 and 1).
        assert store.latest_epoch() == 1
        ckpt = store.load(0)
        assert ckpt is not None
        assert ckpt["epoch"] == 0
        assert "op1" in ckpt["operator_states"]

    async def test_resume_from_checkpoint(self, tmp_path: Path) -> None:
        store = FileCheckpointStore(tmp_path / "ckpts")

        # Simulate a previous run that saved epoch 0.
        store.save(
            0,
            {
                "epoch": 0,
                "best_accuracy": 0.7,
                "operator_states": {"op1": {"prompt": "resumed-prompt"}},
            },
        )

        optimizer = _StubOptimizer()
        t = OperatorTrainer(
            OperatorTrainConfig(epochs=3),
            optimizer=optimizer,
            eval_fn=_good_eval_fn,
            checkpoint_store=store,
            resume_from=0,
        )
        op = _StubOperator("op1", prompt="original")
        agent = _StubAgent([op])
        t.check_agent(agent)
        t.check_dataset([{"input": "x"}])
        t.check_reward()
        t.check_config()
        t.mark_validated()

        metrics = await t.train()

        # Should have resumed from epoch 0, so only epochs 1 and 2 ran.
        assert optimizer.step_calls == 2
        assert metrics.steps == 2
        # Operator state should have been restored before training.
        assert op.get_state()["prompt"] == "resumed-prompt"

    async def test_resume_nonexistent_starts_fresh(self, tmp_path: Path) -> None:
        store = FileCheckpointStore(tmp_path / "empty")
        optimizer = _StubOptimizer()
        t = OperatorTrainer(
            OperatorTrainConfig(epochs=2),
            optimizer=optimizer,
            eval_fn=_good_eval_fn,
            checkpoint_store=store,
            resume_from=5,  # doesn't exist
        )
        agent = _StubAgent([_StubOperator("op1")])
        t.check_agent(agent)
        t.check_dataset([{"input": "x"}])
        t.check_reward()
        t.check_config()
        t.mark_validated()

        metrics = await t.train()
        # Should run all epochs since checkpoint doesn't exist.
        assert optimizer.step_calls == 2
        assert metrics.steps == 2

    async def test_auto_checkpoint_store_from_config(self, tmp_path: Path) -> None:
        cfg = OperatorTrainConfig(epochs=1, checkpoint_dir=str(tmp_path / "auto_ckpt"))
        t = OperatorTrainer(
            cfg,
            optimizer=_StubOptimizer(),
            eval_fn=_good_eval_fn,
        )
        agent = _StubAgent([_StubOperator("op1")])
        t.check_agent(agent)
        t.check_dataset([{"input": "x"}])
        t.check_reward()
        t.check_config()
        t.mark_validated()

        await t.train()

        # Checkpoint file should exist.
        assert (tmp_path / "auto_ckpt" / "checkpoint_0.json").exists()


# ---------------------------------------------------------------------------
# OperatorTrainer — evaluate()
# ---------------------------------------------------------------------------


class TestOperatorTrainerEvaluate:
    async def test_evaluate(self) -> None:
        t = OperatorTrainer(eval_fn=_good_eval_fn)
        agent = _StubAgent([_StubOperator("op1")])
        t.check_agent(agent)
        t.check_dataset([{"input": "a"}])
        metrics = await t.evaluate()
        assert metrics.accuracy == 1.0
        assert metrics.loss == 0.0

    async def test_evaluate_with_explicit_data(self) -> None:
        t = OperatorTrainer(eval_fn=_mediocre_eval_fn)
        agent = _StubAgent([_StubOperator("op1")])
        t.check_agent(agent)
        t.check_dataset([{"input": "train"}])
        metrics = await t.evaluate(test_data=[{"input": "test1"}, {"input": "test2"}])
        assert metrics.accuracy == 0.5
        assert metrics.steps == 2

    async def test_evaluate_without_eval_fn_fails(self) -> None:
        t = OperatorTrainer()
        with pytest.raises(TrainerError, match="eval_fn is required"):
            await t.evaluate()


# ---------------------------------------------------------------------------
# Full lifecycle integration
# ---------------------------------------------------------------------------


class TestOperatorTrainerLifecycle:
    async def test_full_lifecycle(self, tmp_path: Path) -> None:
        cfg = OperatorTrainConfig(
            epochs=2,
            checkpoint_dir=str(tmp_path / "ckpts"),
        )
        optimizer = _StubOptimizer()
        t = OperatorTrainer(cfg, optimizer=optimizer, eval_fn=_mediocre_eval_fn)

        # Phase 1: Validation
        agent = _StubAgent([_StubOperator("llm1"), _StubOperator("llm2")])
        t.check_agent(agent)
        t.check_dataset(
            [{"input": "q1"}, {"input": "q2"}],
            test_data=[{"input": "t1"}],
        )
        t.check_reward()
        t.check_config()
        t.mark_validated()
        assert t.state == TrainerState.VALIDATED

        # Phase 2: Training
        metrics = await t.train()
        assert t.state == TrainerState.COMPLETED
        assert metrics.steps == 2
        assert metrics.accuracy == 0.5

        # Phase 3: Evaluation
        eval_metrics = await t.evaluate()
        assert eval_metrics.accuracy == 0.5

        # Checkpoints saved
        store = FileCheckpointStore(tmp_path / "ckpts")
        assert store.latest_epoch() == 1
