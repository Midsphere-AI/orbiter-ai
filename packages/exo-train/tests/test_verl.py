"""Tests for VeRL integration."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from exo.train.trainer import (  # pyright: ignore[reportMissingImports]
    TrainConfig,
    TrainerError,
    TrainerState,
    TrainMetrics,
)
from exo.train.verl import (  # pyright: ignore[reportMissingImports]
    RewardSpec,
    VeRLAlgorithm,
    VeRLConfig,
    VeRLTrainer,
    _build_verl_params,
    _check_verl_available,
)

# ---------------------------------------------------------------------------
# VeRLAlgorithm
# ---------------------------------------------------------------------------


class TestVeRLAlgorithm:
    def test_values(self) -> None:
        assert VeRLAlgorithm.PPO == "ppo"
        assert VeRLAlgorithm.GRPO == "grpo"

    def test_is_str(self) -> None:
        assert isinstance(VeRLAlgorithm.PPO, str)


# ---------------------------------------------------------------------------
# RewardSpec
# ---------------------------------------------------------------------------


class TestRewardSpec:
    def test_with_callable(self) -> None:
        def my_reward(x: str) -> float:
            return 1.0

        spec = RewardSpec(callable=my_reward)
        assert spec.resolve() is my_reward

    def test_with_module_path(self) -> None:
        # Use a known importable function
        spec = RewardSpec(module_path="math", func_name="sqrt")
        fn = spec.resolve()
        assert fn(4.0) == 2.0

    def test_missing_both_raises(self) -> None:
        with pytest.raises(TrainerError, match="requires either"):
            RewardSpec()

    def test_missing_func_name_raises(self) -> None:
        with pytest.raises(TrainerError, match="requires either"):
            RewardSpec(module_path="math")

    def test_resolve_missing_func(self) -> None:
        spec = RewardSpec(module_path="math", func_name="nonexistent_fn_xyz")
        with pytest.raises(TrainerError, match="Cannot find"):
            spec.resolve()

    def test_resolve_non_callable(self) -> None:
        spec = RewardSpec(module_path="math", func_name="pi")
        with pytest.raises(TrainerError, match="not callable"):
            spec.resolve()


# ---------------------------------------------------------------------------
# VeRLConfig
# ---------------------------------------------------------------------------


class TestVeRLConfig:
    def test_defaults(self) -> None:
        cfg = VeRLConfig()
        assert cfg.algorithm == VeRLAlgorithm.GRPO
        assert cfg.rollout_batch_size == 4
        assert cfg.ppo_epochs == 4
        assert cfg.kl_coeff == 0.1
        assert cfg.clip_range == 0.2
        assert cfg.gamma == 1.0
        assert cfg.lam == 0.95
        assert cfg.model_name == ""
        assert cfg.tokenizer_name == ""
        assert cfg.max_prompt_length == 1024
        assert cfg.max_response_length == 512
        # Inherited from TrainConfig
        assert cfg.epochs == 1
        assert cfg.batch_size == 8
        assert cfg.learning_rate == 1e-5

    def test_custom(self) -> None:
        cfg = VeRLConfig(
            algorithm=VeRLAlgorithm.PPO,
            epochs=3,
            rollout_batch_size=8,
            model_name="my-model",
        )
        assert cfg.algorithm == VeRLAlgorithm.PPO
        assert cfg.epochs == 3
        assert cfg.rollout_batch_size == 8
        assert cfg.model_name == "my-model"

    def test_invalid_rollout_batch(self) -> None:
        with pytest.raises(ValueError, match="rollout_batch_size"):
            VeRLConfig(rollout_batch_size=0)

    def test_invalid_ppo_epochs(self) -> None:
        with pytest.raises(ValueError, match="ppo_epochs"):
            VeRLConfig(ppo_epochs=0)

    def test_invalid_clip_range_low(self) -> None:
        with pytest.raises(ValueError, match="clip_range"):
            VeRLConfig(clip_range=-0.1)

    def test_invalid_clip_range_high(self) -> None:
        with pytest.raises(ValueError, match="clip_range"):
            VeRLConfig(clip_range=1.5)

    def test_clip_range_boundaries(self) -> None:
        cfg_zero = VeRLConfig(clip_range=0.0)
        assert cfg_zero.clip_range == 0.0
        cfg_one = VeRLConfig(clip_range=1.0)
        assert cfg_one.clip_range == 1.0

    def test_is_train_config(self) -> None:
        cfg = VeRLConfig()
        assert isinstance(cfg, TrainConfig)


# ---------------------------------------------------------------------------
# VeRLTrainer — init
# ---------------------------------------------------------------------------


class TestVeRLTrainerInit:
    def test_default(self) -> None:
        t = VeRLTrainer()
        assert t.state == TrainerState.CREATED
        assert isinstance(t.verl_config, VeRLConfig)
        assert t.verl_config.algorithm == VeRLAlgorithm.GRPO

    def test_custom_config(self) -> None:
        cfg = VeRLConfig(algorithm=VeRLAlgorithm.PPO, epochs=5)
        t = VeRLTrainer(cfg)
        assert t.verl_config.algorithm == VeRLAlgorithm.PPO
        assert t.verl_config.epochs == 5

    def test_repr(self) -> None:
        t = VeRLTrainer()
        r = repr(t)
        assert "VeRLTrainer" in r
        assert "grpo" in r
        assert "created" in r


# ---------------------------------------------------------------------------
# VeRLTrainer — check_agent
# ---------------------------------------------------------------------------


class TestVeRLTrainerCheckAgent:
    def test_valid_agent(self) -> None:
        t = VeRLTrainer()
        t.check_agent("my-agent")
        # No error

    def test_none_agent_raises(self) -> None:
        t = VeRLTrainer()
        with pytest.raises(TrainerError, match="non-None agent"):
            t.check_agent(None)


# ---------------------------------------------------------------------------
# VeRLTrainer — check_dataset
# ---------------------------------------------------------------------------


class TestVeRLTrainerCheckDataset:
    def test_valid_data(self) -> None:
        t = VeRLTrainer()
        t.check_dataset([{"input": "hello"}])

    def test_with_test_data(self) -> None:
        t = VeRLTrainer()
        t.check_dataset(
            [{"input": "train"}],
            test_data=[{"input": "test"}],
        )

    def test_empty_raises(self) -> None:
        t = VeRLTrainer()
        with pytest.raises(TrainerError, match="non-empty"):
            t.check_dataset([])

    def test_wrong_type_raises(self) -> None:
        t = VeRLTrainer()
        with pytest.raises(TrainerError, match="list or tuple"):
            t.check_dataset("not a list")

    def test_non_dict_item_raises(self) -> None:
        t = VeRLTrainer()
        with pytest.raises(TrainerError, match="must be a dict"):
            t.check_dataset(["not a dict"])

    def test_missing_input_key_raises(self) -> None:
        t = VeRLTrainer()
        with pytest.raises(TrainerError, match="missing required 'input'"):
            t.check_dataset([{"output": "no input key"}])


# ---------------------------------------------------------------------------
# VeRLTrainer — check_reward
# ---------------------------------------------------------------------------


class TestVeRLTrainerCheckReward:
    def test_none_reward(self) -> None:
        t = VeRLTrainer()
        t.check_reward(None)  # OK — uses VeRL default

    def test_callable_reward(self) -> None:
        t = VeRLTrainer()
        t.check_reward(lambda x: 1.0)

    def test_reward_spec(self) -> None:
        t = VeRLTrainer()
        spec = RewardSpec(module_path="math", func_name="sqrt")
        t.check_reward(spec)

    def test_invalid_type_raises(self) -> None:
        t = VeRLTrainer()
        with pytest.raises(TrainerError, match="callable or RewardSpec"):
            t.check_reward("not callable")


# ---------------------------------------------------------------------------
# VeRLTrainer — check_config
# ---------------------------------------------------------------------------


class TestVeRLTrainerCheckConfig:
    def test_none_config(self) -> None:
        t = VeRLTrainer()
        t.check_config(None)

    def test_dict_config(self) -> None:
        t = VeRLTrainer()
        t.check_config({"custom_key": "value"})
        assert t.verl_config.extra.get("custom_key") == "value"

    def test_verl_config_override(self) -> None:
        t = VeRLTrainer()
        new_cfg = VeRLConfig(algorithm=VeRLAlgorithm.PPO, epochs=10)
        t.check_config(new_cfg)
        assert t.verl_config.algorithm == VeRLAlgorithm.PPO
        assert t.verl_config.epochs == 10

    def test_base_train_config(self) -> None:
        t = VeRLTrainer()
        t.check_config(TrainConfig(epochs=3))
        # Accepts without error — keeps VeRL defaults


# ---------------------------------------------------------------------------
# VeRLTrainer — train
# ---------------------------------------------------------------------------


class TestVeRLTrainerTrain:
    async def test_train_without_validation_fails(self) -> None:
        t = VeRLTrainer()
        with pytest.raises(TrainerError, match="must be validated"):
            await t.train()

    async def test_train_verl_not_installed(self) -> None:
        t = VeRLTrainer()
        t.check_agent("agent")
        t.check_dataset([{"input": "q1"}])
        t.check_reward(None)
        t.check_config(None)
        t.mark_validated()

        # VeRL is not installed in test env, so _check_verl_available raises
        with pytest.raises(TrainerError, match="VeRL is required"):
            await t.train()
        assert t.state == TrainerState.FAILED

    async def test_train_with_mocked_verl(self) -> None:
        t = VeRLTrainer(VeRLConfig(epochs=2, batch_size=4))
        t.check_agent("agent")
        t.check_dataset([{"input": "q1"}, {"input": "q2"}, {"input": "q3"}])
        t.check_reward(lambda data, sol, gt, extra=None: 1.0)
        t.check_config(None)
        t.mark_validated()

        # Mock the VeRL availability check
        with patch("exo.train.verl._check_verl_available"):
            metrics = await t.train()

        assert t.state == TrainerState.COMPLETED
        assert isinstance(metrics, TrainMetrics)
        assert metrics.steps > 0
        assert metrics.extra["algorithm"] == "grpo"
        assert metrics.extra["train_items"] == 3


# ---------------------------------------------------------------------------
# VeRLTrainer — evaluate
# ---------------------------------------------------------------------------


class TestVeRLTrainerEvaluate:
    async def test_evaluate_verl_not_installed(self) -> None:
        t = VeRLTrainer()
        with pytest.raises(TrainerError, match="VeRL is required"):
            await t.evaluate(test_data=[{"input": "t1"}])

    async def test_evaluate_with_data(self) -> None:
        t = VeRLTrainer()
        with patch("exo.train.verl._check_verl_available", return_value=object()):
            metrics = await t.evaluate(test_data=[{"input": "t1"}, {"input": "t2"}])
        assert metrics.steps == 2
        assert metrics.extra["eval_items"] == 2

    async def test_evaluate_no_data(self) -> None:
        t = VeRLTrainer()
        with patch("exo.train.verl._check_verl_available", return_value=object()):
            metrics = await t.evaluate()
        assert metrics.steps == 0
        assert metrics.extra["eval_items"] == 0

    async def test_evaluate_uses_check_dataset_data(self) -> None:
        t = VeRLTrainer()
        t.check_dataset(
            [{"input": "train"}],
            test_data=[{"input": "t1"}, {"input": "t2"}, {"input": "t3"}],
        )
        with patch("exo.train.verl._check_verl_available", return_value=object()):
            metrics = await t.evaluate()
        assert metrics.steps == 3


# ---------------------------------------------------------------------------
# Full lifecycle integration
# ---------------------------------------------------------------------------


class TestVeRLTrainerLifecycle:
    async def test_full_lifecycle(self) -> None:
        cfg = VeRLConfig(
            algorithm=VeRLAlgorithm.PPO,
            epochs=2,
            batch_size=4,
            model_name="test-model",
        )
        t = VeRLTrainer(cfg)

        # Phase 1: Validation
        assert t.state == TrainerState.CREATED
        t.check_agent("my-agent")
        t.check_dataset(
            [{"input": "q1"}, {"input": "q2"}],
            test_data=[{"input": "t1"}],
        )
        t.check_reward(lambda d, s, g, e=None: 1.0)
        t.check_config({"extra_setting": True})
        t.mark_validated()
        assert t.state == TrainerState.VALIDATED

        # Phase 2: Training (mocked VeRL)
        with patch("exo.train.verl._check_verl_available"):
            train_metrics = await t.train()
        assert t.state == TrainerState.COMPLETED
        assert train_metrics.extra["algorithm"] == "ppo"

        # Phase 3: Evaluation (also requires VeRL)
        with patch("exo.train.verl._check_verl_available", return_value=object()):
            eval_metrics = await t.evaluate()
        assert eval_metrics.steps == 1  # Uses test_data from check_dataset

    async def test_cannot_train_twice(self) -> None:
        t = VeRLTrainer()
        t.check_agent("agent")
        t.check_dataset([{"input": "q"}])
        t.mark_validated()

        with patch("exo.train.verl._check_verl_available"):
            await t.train()
        assert t.state == TrainerState.COMPLETED

        with pytest.raises(TrainerError, match="must be validated"):
            await t.train()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestCheckVeRLAvailable:
    def test_raises_when_not_installed(self) -> None:
        # VeRL is not installed in test env
        with pytest.raises(ImportError, match="VeRL is required"):
            _check_verl_available()

    def test_passes_when_installed(self) -> None:
        with patch.dict("sys.modules", {"verl": object()}):
            _check_verl_available()  # No error


class TestBuildVeRLParams:
    def test_default_config(self) -> None:
        cfg = VeRLConfig()
        params = _build_verl_params(cfg, None)
        assert params["algorithm"] == "grpo"
        assert params["epochs"] == 1
        assert params["batch_size"] == 8
        assert params["rollout_batch_size"] == 4
        assert params["ppo_epochs"] == 4
        assert params["kl_coeff"] == 0.1
        assert params["clip_range"] == 0.2
        assert "reward_fn" not in params
        assert "model_name" not in params

    def test_with_model_name(self) -> None:
        cfg = VeRLConfig(model_name="my-model", tokenizer_name="my-tok")
        params = _build_verl_params(cfg, None)
        assert params["model_name"] == "my-model"
        assert params["tokenizer_name"] == "my-tok"

    def test_with_callable_reward(self) -> None:
        def my_reward() -> float:
            return 1.0

        spec = RewardSpec(callable=my_reward)
        params = _build_verl_params(VeRLConfig(), spec)
        assert params["reward_fn"] == "my_reward"

    def test_with_module_reward(self) -> None:
        spec = RewardSpec(module_path="my.module", func_name="compute_reward")
        params = _build_verl_params(VeRLConfig(), spec)
        assert params["reward_module"] == "my.module"
        assert params["reward_func"] == "compute_reward"

    def test_with_extra(self) -> None:
        cfg = VeRLConfig(extra={"custom": "value"})
        params = _build_verl_params(cfg, None)
        assert params["extra"]["custom"] == "value"

    def test_with_output_dir(self) -> None:
        cfg = VeRLConfig(output_dir="/tmp/out")
        params = _build_verl_params(cfg, None)
        assert params["output_dir"] == "/tmp/out"
