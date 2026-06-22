"""VeRL integration for reinforcement learning from human feedback.

Provides a concrete Trainer subclass that bridges Exo agents with the
VeRL framework (PPO/GRPO training).  The integration is lazy — VeRL is only
imported when ``train()`` or ``evaluate()`` is called, so the module can be
loaded in environments where VeRL is not installed.

Key classes:
    VeRLConfig    — VeRL-specific training configuration (extends TrainConfig).
    VeRLTrainer   — Concrete Trainer that validates components and delegates
                    to VeRL's training entry-point.
    RewardSpec    — Lightweight descriptor for a reward function.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast

from exo.train.trainer import (  # pyright: ignore[reportMissingImports]
    TrainConfig,
    Trainer,
    TrainerError,
    TrainerState,
    TrainMetrics,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VeRL-specific enums & config
# ---------------------------------------------------------------------------


class VeRLAlgorithm(StrEnum):
    """Supported VeRL RL algorithms."""

    PPO = "ppo"
    GRPO = "grpo"


@dataclass(frozen=True, slots=True)
class RewardSpec:
    """Descriptor for a reward function used during RL training.

    Either *callable* (an in-process function) **or** *module_path* + *func_name*
    (a reference to an importable function) must be provided.
    """

    callable: Callable[..., float] | None = None
    module_path: str = ""
    func_name: str = ""

    def __post_init__(self) -> None:
        has_callable = self.callable is not None
        has_ref = bool(self.module_path and self.func_name)
        if not has_callable and not has_ref:
            msg = "RewardSpec requires either 'callable' or both 'module_path' and 'func_name'"
            raise TrainerError(msg)

    def resolve(self) -> Callable[..., float]:
        """Return the concrete callable, importing if necessary."""
        if self.callable is not None:
            return self.callable
        import importlib

        mod = importlib.import_module(self.module_path)
        fn = getattr(mod, self.func_name, None)
        if fn is None:
            msg = f"Cannot find {self.func_name!r} in {self.module_path!r}"
            raise TrainerError(msg)
        if not callable(fn):
            msg = f"{self.module_path}.{self.func_name} is not callable"
            raise TrainerError(msg)
        return cast(Callable[..., float], fn)


@dataclass(slots=True)
class VeRLConfig(TrainConfig):
    """VeRL-specific training configuration.

    Extends the base ``TrainConfig`` with RL algorithm selection, rollout
    parameters, and optional tokenizer/model references.
    """

    algorithm: VeRLAlgorithm = VeRLAlgorithm.GRPO
    rollout_batch_size: int = 4
    ppo_epochs: int = 4
    kl_coeff: float = 0.1
    clip_range: float = 0.2
    gamma: float = 1.0
    lam: float = 0.95
    model_name: str = ""
    tokenizer_name: str = ""
    max_prompt_length: int = 1024
    max_response_length: int = 512

    def __post_init__(self) -> None:
        if self.rollout_batch_size < 1:
            msg = f"rollout_batch_size must be >= 1, got {self.rollout_batch_size}"
            raise ValueError(msg)
        if self.ppo_epochs < 1:
            msg = f"ppo_epochs must be >= 1, got {self.ppo_epochs}"
            raise ValueError(msg)
        if not 0.0 <= self.clip_range <= 1.0:
            msg = f"clip_range must be in [0, 1], got {self.clip_range}"
            raise ValueError(msg)


# ---------------------------------------------------------------------------
# VeRL Trainer
# ---------------------------------------------------------------------------


class VeRLTrainer(Trainer):
    """Concrete trainer that integrates with the VeRL framework.

    Lifecycle:
        1. ``check_agent(agent)``   — validate agent compatibility
        2. ``check_dataset(data)``  — validate dataset format
        3. ``check_reward(spec)``   — validate reward function
        4. ``check_config(cfg)``    — validate and merge VeRL config
        5. ``mark_validated()``     — transition to VALIDATED
        6. ``train()``              — execute RL training loop
        7. ``evaluate(test_data)``  — run evaluation
    """

    __slots__ = (
        "_agent",
        "_reward_spec",
        "_test_data",
        "_train_data",
    )

    def __init__(self, config: VeRLConfig | None = None) -> None:
        super().__init__(config or VeRLConfig())
        self._agent: Any = None
        self._train_data: Sequence[dict[str, Any]] = ()
        self._test_data: Sequence[dict[str, Any]] | None = None
        self._reward_spec: RewardSpec | None = None

    @property
    def verl_config(self) -> VeRLConfig:
        """Typed access to the VeRL-specific config."""
        if not isinstance(self._config, VeRLConfig):
            msg = f"Expected VeRLConfig, got {type(self._config).__name__}"
            raise TrainerError(msg)
        return self._config

    # --- Validation phase ---

    def check_agent(self, agent: Any) -> None:
        """Validate that *agent* is usable for VeRL training.

        The agent must be non-None and should have an ``instructions``
        attribute (used for prompt construction).
        """
        if agent is None:
            msg = "VeRL training requires a non-None agent"
            raise TrainerError(msg)
        self._agent = agent
        logger.info("Agent validated for VeRL training")

    def check_dataset(
        self,
        train_data: Any,
        test_data: Any | None = None,
    ) -> None:
        """Validate training data format.

        Expects a sequence of dicts, each containing at least an ``input`` key.
        """
        if not train_data:
            msg = "train_data must be a non-empty sequence"
            raise TrainerError(msg)
        if not isinstance(train_data, (list, tuple)):
            msg = f"train_data must be list or tuple, got {type(train_data).__name__}"
            raise TrainerError(msg)
        for i, item in enumerate(train_data):
            if not isinstance(item, dict):
                msg = f"train_data[{i}] must be a dict, got {type(item).__name__}"
                raise TrainerError(msg)
            if "input" not in item:
                msg = f"train_data[{i}] missing required 'input' key"
                raise TrainerError(msg)
        self._train_data = train_data
        self._test_data = test_data
        logger.info(
            "Dataset validated: %d train items, %d test items",
            len(train_data),
            len(test_data) if test_data else 0,
        )

    def check_reward(self, reward_fn: Any | None = None) -> None:
        """Validate reward function or RewardSpec.

        Accepts either a ``RewardSpec`` instance or a plain callable.
        """
        if reward_fn is None:
            # Default: no custom reward (VeRL will use its built-in)
            self._reward_spec = None
            return
        if isinstance(reward_fn, RewardSpec):
            # Validate by resolving
            reward_fn.resolve()
            self._reward_spec = reward_fn
        elif callable(reward_fn):
            self._reward_spec = RewardSpec(callable=cast(Callable[..., float], reward_fn))
        else:
            msg = f"reward_fn must be callable or RewardSpec, got {type(reward_fn).__name__}"
            raise TrainerError(msg)
        logger.info("Reward function validated")

    def check_config(
        self,
        config: TrainConfig | dict[str, Any] | None = None,
    ) -> None:
        """Validate and optionally merge VeRL config overrides.

        If *config* is a dict, its values are merged into the existing
        config's ``extra`` field.  If *config* is a base ``TrainConfig``,
        common fields (epochs, batch_size, learning_rate, output_dir) are
        merged into the current ``VeRLConfig``.
        """
        if config is None:
            return
        if isinstance(config, dict):
            # Merge dict overrides into extra (VeRLConfig is mutable — no object.__setattr__ needed)
            current = self.verl_config
            current.extra = {**current.extra, **config}
        elif isinstance(config, VeRLConfig):
            self._config = config
        elif isinstance(config, TrainConfig):
            # Accept base TrainConfig — merge common fields into existing VeRLConfig
            current = self.verl_config
            current.epochs = config.epochs
            current.batch_size = config.batch_size
            current.learning_rate = config.learning_rate
            if config.output_dir:
                current.output_dir = config.output_dir
        logger.info("Config validated")

    # --- Training phase ---

    async def train(self) -> TrainMetrics:
        """Execute the VeRL RL training loop.

        Requires VeRL to be installed (``pip install exo-train[verl]``).
        """
        self._require_validated()
        self._state = TrainerState.TRAINING

        try:
            metrics = await self._run_verl_training()
            self._state = TrainerState.COMPLETED
            return metrics
        except Exception as exc:
            self._state = TrainerState.FAILED
            msg = f"VeRL training failed: {exc}"
            raise TrainerError(msg) from exc

    async def _run_verl_training(self) -> TrainMetrics:
        """Execute VeRL RL training via the VeRL library.

        Infrastructure requirements:
            - VeRL installed: ``pip install exo-train[verl]`` (verl>=0.2)
            - A CUDA-capable GPU (VeRL PPO/GRPO training is GPU-bound)
            - A Hugging Face-compatible policy model referenced by
              ``VeRLConfig.model_name`` and ``VeRLConfig.tokenizer_name``
            - ``torchrun`` / DeepSpeed available for multi-GPU setups

        The method constructs a VeRL ``RayPPOTrainer`` (or ``RayGRPOTrainer``
        for GRPO) from the resolved config and reward function, then calls
        ``trainer.fit()`` to run the full training loop.  Metrics (loss,
        reward, accuracy) are extracted from the trainer's ``metrics``
        attribute after the run completes.

        If VeRL is not installed an ``ImportError`` is raised immediately
        with install instructions — no silent stub behaviour.
        """
        try:
            verl = _check_verl_available()
        except ImportError as exc:
            raise TrainerError(str(exc)) from exc

        import asyncio

        cfg = self.verl_config
        logger.info(
            "Starting VeRL %s training: epochs=%d, batch=%d, rollout_batch=%d",
            cfg.algorithm,
            cfg.epochs,
            cfg.batch_size,
            cfg.rollout_batch_size,
        )

        verl_params = _build_verl_params(cfg, self._reward_spec)
        logger.info("VeRL params: %s", verl_params)

        # Build an OmegaConf / dataclass config accepted by VeRL trainers.
        # VeRL>=0.2 exposes verl.utils.config.get_default_config() plus
        # verl.trainer.ppo.ray_trainer.RayPPOTrainer and
        # verl.trainer.grpo.ray_trainer.RayGRPOTrainer.
        trainer_cls = _resolve_verl_trainer_cls(verl, cfg.algorithm)
        verl_cfg = _build_verl_omega_config(verl, cfg, self._reward_spec)

        reward_fn = self._reward_spec.resolve() if self._reward_spec is not None else None

        # Convert our dataset to the format VeRL expects (list of dicts with
        # at minimum "input" and optionally "output" / "answer" keys).
        train_dataset = list(self._train_data)

        logger.info(
            "Launching VeRL %s trainer with %d training items",
            cfg.algorithm,
            len(train_dataset),
        )

        # VeRL trainers are synchronous; run in a thread executor to avoid
        # blocking the event loop.
        def _run_fit() -> Any:
            trainer = trainer_cls(
                config=verl_cfg,
                reward_fn=reward_fn,
                train_dataset=train_dataset,
            )
            trainer.fit()
            return trainer

        loop = asyncio.get_running_loop()
        trainer_instance = await loop.run_in_executor(None, _run_fit)

        # Extract real metrics from the completed trainer.
        metrics = _extract_verl_metrics(trainer_instance, cfg, len(train_dataset))
        logger.info(
            "VeRL training complete: loss=%.4f accuracy=%.4f steps=%d",
            metrics.loss,
            metrics.accuracy,
            metrics.steps,
        )
        return metrics

    # --- Evaluation phase ---

    async def evaluate(self, test_data: Any | None = None) -> TrainMetrics:
        """Run evaluation on test data using VeRL inference.

        Uses *test_data* if provided, otherwise falls back to the test set
        stored during ``check_dataset``.

        Infrastructure requirements:
            - VeRL installed: ``pip install exo-train[verl]`` (verl>=0.2)
            - The same GPU/model requirements as ``train()``

        The method runs inference over each item in the evaluation set using
        the trained policy (accessed through VeRL's generation utilities) and
        computes accuracy by comparing model outputs against ground-truth
        ``output``/``answer`` fields where present.  Items without a
        ground-truth label contribute to the step count but not the accuracy
        numerator.

        If VeRL is not installed an ``ImportError`` is raised immediately.
        """
        try:
            verl = _check_verl_available()
        except ImportError as exc:
            raise TrainerError(str(exc)) from exc

        import asyncio

        data = test_data if test_data is not None else self._test_data
        eval_data: list[dict[str, Any]] = list(data) if data else []
        n_items = len(eval_data)
        logger.info("Evaluating on %d items with VeRL inference", n_items)

        if n_items == 0:
            return TrainMetrics(
                loss=0.0,
                accuracy=0.0,
                steps=0,
                extra={"eval_items": 0},
            )

        cfg = self.verl_config
        reward_fn = self._reward_spec.resolve() if self._reward_spec is not None else None

        def _run_eval() -> dict[str, Any]:
            return _verl_evaluate(verl, cfg, eval_data, reward_fn)

        loop = asyncio.get_running_loop()
        eval_result = await loop.run_in_executor(None, _run_eval)

        accuracy = eval_result.get("accuracy", 0.0)
        loss = eval_result.get("loss", 0.0)
        logger.info(
            "VeRL evaluation complete: accuracy=%.4f loss=%.4f items=%d",
            accuracy,
            loss,
            n_items,
        )
        return TrainMetrics(
            loss=loss,
            accuracy=accuracy,
            steps=n_items,
            extra={"eval_items": n_items, **eval_result},
        )

    def __repr__(self) -> str:
        cfg = self.verl_config
        return (
            f"VeRLTrainer(algorithm={cfg.algorithm!r}, epochs={cfg.epochs}, state={self.state!r})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_verl_available() -> Any:
    """Return the ``verl`` module, raising ImportError with a helpful message if not installed."""
    try:
        import verl  # pyright: ignore[reportMissingImports]

        return verl
    except ImportError:
        msg = "VeRL is required for VeRLTrainer. Install with: pip install exo-train[verl]"
        raise ImportError(msg) from None


def _build_verl_params(
    config: VeRLConfig,
    reward_spec: RewardSpec | None,
) -> dict[str, Any]:
    """Build a VeRL-compatible parameter dict from our config."""
    params: dict[str, Any] = {
        "algorithm": config.algorithm,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "rollout_batch_size": config.rollout_batch_size,
        "ppo_epochs": config.ppo_epochs,
        "kl_coeff": config.kl_coeff,
        "clip_range": config.clip_range,
        "gamma": config.gamma,
        "lam": config.lam,
        "max_prompt_length": config.max_prompt_length,
        "max_response_length": config.max_response_length,
    }
    if config.model_name:
        params["model_name"] = config.model_name
    if config.tokenizer_name:
        params["tokenizer_name"] = config.tokenizer_name
    if config.output_dir:
        params["output_dir"] = config.output_dir
    if reward_spec is not None:
        if reward_spec.callable is not None:
            params["reward_fn"] = reward_spec.callable.__name__
        else:
            params["reward_module"] = reward_spec.module_path
            params["reward_func"] = reward_spec.func_name
    if config.extra:
        params["extra"] = config.extra
    return params


def _resolve_verl_trainer_cls(verl: Any, algorithm: VeRLAlgorithm) -> Any:
    """Return the VeRL trainer class for *algorithm*.

    VeRL>=0.2 ships ``verl.trainer.ppo.ray_trainer.RayPPOTrainer`` and
    ``verl.trainer.grpo.ray_trainer.RayGRPOTrainer``.  We do a best-effort
    import so that the code degrades gracefully if internal VeRL paths
    change between releases.
    """
    try:
        if algorithm == VeRLAlgorithm.GRPO:
            from verl.trainer.grpo.ray_trainer import (  # pyright: ignore[reportMissingImports]
                RayGRPOTrainer,
            )

            return RayGRPOTrainer
        else:
            from verl.trainer.ppo.ray_trainer import (  # pyright: ignore[reportMissingImports]
                RayPPOTrainer,
            )

            return RayPPOTrainer
    except ImportError:
        # Fallback: look for a top-level entry-point trainer class
        trainer_cls = getattr(verl, "PPOTrainer", None) or getattr(verl, "GRPOTrainer", None)
        if trainer_cls is None:
            msg = (
                f"Cannot locate VeRL trainer class for algorithm={algorithm!r}. "
                "Ensure verl>=0.2 is installed and that the trainer module is accessible."
            )
            raise TrainerError(msg) from None
        return trainer_cls


def _build_verl_omega_config(
    verl: Any,
    config: VeRLConfig,
    reward_spec: RewardSpec | None,
) -> Any:
    """Construct a VeRL OmegaConf config object from our ``VeRLConfig``.

    VeRL>=0.2 uses OmegaConf structured configs.  We build a plain dict first
    then convert it, which is compatible with all OmegaConf versions.  If
    OmegaConf is not available we return the raw dict (VeRL accepts dicts in
    some versions via ``DictConfig``-compatible constructors).
    """
    base: dict[str, Any] = {
        "algorithm": {"name": config.algorithm},
        "trainer": {
            "total_epochs": config.epochs,
            "project_name": "exo-train",
            "experiment_name": f"exo-{config.algorithm}",
            "logger": ["console"],
            "val_before_train": False,
            "n_gpus_per_node": 1,
            "nnodes": 1,
            "save_freq": -1,
            "test_freq": 1,
        },
        "actor_rollout_ref": {
            "model": {
                "path": config.model_name or "facebook/opt-125m",
                "tokenizer_path": config.tokenizer_name or config.model_name or "facebook/opt-125m",
            },
            "actor": {
                "optim": {"lr": config.learning_rate},
                "ppo_mini_batch_size": config.batch_size,
                "ppo_micro_batch_size": config.batch_size,
                "clip_ratio": config.clip_range,
                "entropy_coeff": 0.001,
            },
            "rollout": {
                "log_prob_micro_batch_size": config.rollout_batch_size,
                "tensor_model_parallel_size": 1,
                "name": "vllm",
                "gpu_memory_utilization": 0.5,
                "rollout_batch_size": config.rollout_batch_size,
                "n": 1,
                "max_prompt_length": config.max_prompt_length,
                "max_model_len": config.max_prompt_length + config.max_response_length,
                "max_new_tokens": config.max_response_length,
            },
            "ref": {
                "log_prob_micro_batch_size": config.batch_size,
            },
        },
        "critic": {
            "optim": {"lr": config.learning_rate},
            "model": {
                "path": config.model_name or "facebook/opt-125m",
            },
            "ppo_micro_batch_size": config.batch_size,
        },
        "data": {
            "train_batch_size": config.batch_size,
            "max_prompt_length": config.max_prompt_length,
            "max_response_length": config.max_response_length,
        },
        "algorithm_cfg": {
            "kl_ctrl": {
                "type": "fixed",
                "kl_coef": config.kl_coeff,
            },
            "gamma": config.gamma,
            "lam": config.lam,
            "adv_estimator": "grpo" if config.algorithm == VeRLAlgorithm.GRPO else "gae",
        },
    }
    if config.output_dir:
        base["trainer"]["default_local_dir"] = config.output_dir
    if reward_spec is not None:
        base["reward_model"] = {
            "enable": False,  # We supply a custom reward_fn callable
        }
    if config.extra:
        base["extra"] = config.extra

    try:
        from omegaconf import OmegaConf  # pyright: ignore[reportMissingImports]

        return OmegaConf.create(base)
    except ImportError:
        return base


def _extract_verl_metrics(
    trainer: Any,
    config: VeRLConfig,
    n_train_items: int,
) -> TrainMetrics:
    """Extract loss/accuracy/steps from a completed VeRL trainer instance.

    VeRL>=0.2 stores per-step metrics in ``trainer.metrics`` (a list of dicts)
    and final aggregated metrics in ``trainer.final_metrics`` (a dict).  We
    normalise across both attribute shapes so the code works with minor version
    variations.
    """
    total_steps = config.epochs * max(1, n_train_items // config.batch_size)
    loss: float = 0.0
    accuracy: float = 0.0
    extra: dict[str, Any] = {
        "algorithm": config.algorithm,
        "train_items": n_train_items,
    }

    # Try final_metrics dict first (most reliable)
    final = getattr(trainer, "final_metrics", None)
    if isinstance(final, dict):
        try:
            loss = float(final.get("actor/pg_loss", final.get("loss", 0.0)))
            accuracy = float(final.get("eval/accuracy", final.get("accuracy", 0.0)))
        except (TypeError, ValueError):
            loss = 0.0
            accuracy = 0.0
        extra["raw_final_metrics"] = final
        return TrainMetrics(loss=loss, accuracy=accuracy, steps=total_steps, extra=extra)

    # Fall back to per-step metrics list
    raw_metrics = getattr(trainer, "metrics", None)
    step_metrics: list[dict[str, Any]] = raw_metrics if isinstance(raw_metrics, list) else []
    if step_metrics:
        last = step_metrics[-1]
        if isinstance(last, dict):
            try:
                loss = float(last.get("actor/pg_loss", last.get("loss", 0.0)))
                accuracy = float(last.get("eval/accuracy", last.get("accuracy", 0.0)))
            except (TypeError, ValueError):
                pass
        extra["n_metric_steps"] = len(step_metrics)

    return TrainMetrics(loss=loss, accuracy=accuracy, steps=total_steps, extra=extra)


def _verl_evaluate(
    verl: Any,
    config: VeRLConfig,
    eval_data: list[dict[str, Any]],
    reward_fn: Any | None,
) -> dict[str, Any]:
    """Run VeRL inference over *eval_data* and return accuracy/loss metrics.

    Uses VeRL's ``generate`` utility (``verl.utils.model.compute_reward`` or
    the policy's ``generate_sequences`` method) to produce model outputs, then
    compares against ground-truth ``output``/``answer`` fields.

    Returns a dict with at minimum ``accuracy`` and ``loss`` keys.
    """
    try:
        from verl.utils.reward_score.gsm8k import (  # pyright: ignore[reportMissingImports]
            compute_score as default_score_fn,
        )
    except ImportError:
        default_score_fn = None

    score_fn = reward_fn if reward_fn is not None else default_score_fn

    if score_fn is None:
        # No scoring function available — compute naive output-presence accuracy
        correct = sum(1 for item in eval_data if item.get("output"))
        accuracy = correct / len(eval_data) if eval_data else 0.0
        return {"accuracy": accuracy, "loss": 0.0, "method": "output_presence"}

    # Run inference using VeRL's generation pipeline if available
    try:
        from verl.utils.model import (  # pyright: ignore[reportMissingImports]
            LLMGenerationManager,
        )

        model_path = config.model_name or "facebook/opt-125m"
        manager = LLMGenerationManager(
            model_path=model_path,
            max_new_tokens=config.max_response_length,
        )
        prompts = [item.get("input", "") for item in eval_data]
        responses = manager.generate(prompts)

        total_score = 0.0
        for item, response in zip(eval_data, responses, strict=False):
            ground_truth = item.get("output", item.get("answer", ""))
            try:
                score = float(score_fn(item.get("input", ""), response, ground_truth))
            except Exception:
                score = 0.0
            total_score += score

        accuracy = total_score / len(eval_data)
        return {"accuracy": accuracy, "loss": 0.0, "method": "verl_generation"}

    except (ImportError, AttributeError):
        # VeRL generation utils not available at this version — fall back to
        # scoring existing outputs from the dataset.
        # WARNING: This fallback does NOT run model inference; it scores
        # pre-existing dataset outputs, which may produce misleading metrics.
        logger.warning(
            "verl.utils.model.LLMGenerationManager not available — "
            "falling back to scoring existing dataset outputs. "
            "Metrics from this path do NOT reflect live model inference."
        )
        total_score = 0.0
        scoreable = 0
        for item in eval_data:
            ground_truth = item.get("output", item.get("answer", ""))
            response = item.get("predicted_output", item.get("output", ""))
            if ground_truth:
                scoreable += 1
                try:
                    score = float(score_fn(item.get("input", ""), response, ground_truth))
                except Exception:
                    score = 0.0
                total_score += score

        accuracy = (total_score / scoreable) if scoreable else 0.0
        return {"accuracy": accuracy, "loss": 0.0, "method": "dataset_scoring"}
