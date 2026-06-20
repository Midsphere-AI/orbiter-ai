# exo-train

> Training framework for Exo agents: data synthesis, evolutionary optimization, and VeRL integration.

`exo-train` closes the agent improvement loop. It captures execution trajectories, synthesises labelled training data from them, runs multi-epoch evolution pipelines, and bridges to VeRL for PPO/GRPO reinforcement learning — all built on top of `exo-core` agents. Install the optional `[verl]` extra to unlock RL training.

## Installation

```bash
pip install exo-train
# or, with VeRL reinforcement-learning support
pip install exo-train[verl]
# or
uv add exo-train
```

## Quick start

```python
from exo.train import (
    TrajectoryExtractor,
    SynthesisPipeline,
    SynthesisConfig,
    EvolutionPipeline,
    EvolutionConfig,
)

# 1. Extract trajectories from agent runs
extractor = TrajectoryExtractor()
dataset = await extractor.extract(agent, prompts=["Solve x^2 = 4"])

# 2. Synthesise training pairs
config = SynthesisConfig(num_samples=200, train_ratio=0.9)
result = await SynthesisPipeline(config).run(dataset)

# 3. Evolve over multiple epochs
evo = EvolutionPipeline(EvolutionConfig(max_epochs=3))
await evo.run(agent, result.train_items)
```

## What's inside

- **`TrajectoryExtractor`** — captures agent execution traces (`TrajectoryItem`, `TrajectoryDataset`) for downstream training
- **`SynthesisPipeline` / `DataSynthesiser`** — generates and augments training samples; supports LLM, template, and augment strategies (`augment_add_noise`, `augment_swap_io`, `deduplicate`, `filter_by_score`)
- **`EvolutionPipeline`** — multi-epoch synthesis → training → evaluation loop with pluggable `EvolutionStrategy`
- **`Trainer` / `OperatorTrainer`** — abstract training lifecycle (validate → train → evaluate) with `FileCheckpointStore` for checkpoint management
- **`InstructionOptimizer` / `ToolOptimizer`** — optimise agent instructions and tool definitions from feedback
- **`VeRLTrainer`** — PPO/GRPO training via VeRL; requires `pip install exo-train[verl]`

## Part of [Exo](https://github.com/midsphere-ai/exo)

Get the full framework with `pip install exo-ai`.

---

MIT © Midsphere AI
