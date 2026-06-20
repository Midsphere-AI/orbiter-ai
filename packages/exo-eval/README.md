# exo-eval

> Evaluation and scoring framework for agent outputs

exo-eval provides a composable system for measuring the quality of Exo agent outputs. It pairs an `Evaluator` runner — which runs a dataset through any `EvalTarget` in parallel — with a library of scorers that range from lightweight rule checks to full LLM-as-judge assessments. A separate reflection layer lets agents review their own past outputs and iterate toward improvement.

## Installation

```bash
pip install exo-eval
# or
uv add exo-eval
```

## Quick start

```python
import asyncio
from exo.eval import (
    Evaluator,
    EvalTarget,
    EvalCriteria,
    OutputRelevanceScorer,
    LLMAsJudgeScorer,
)

class MyAgent(EvalTarget):
    async def predict(self, case_id: str, input):
        return f"Answer to: {input}"  # replace with real agent call

async def main():
    evaluator = Evaluator(
        scorers=[OutputRelevanceScorer(), LLMAsJudgeScorer()],
        criteria=[EvalCriteria(metric_name="relevance", threshold=0.7)],
        parallel=8,
    )
    dataset = [
        {"id": "q1", "input": "What is the capital of France?"},
        {"id": "q2", "input": "Explain gradient descent."},
    ]
    result = await evaluator.evaluate(MyAgent(), dataset)
    print(result.summary)        # {"relevance": 0.85, ...}
    print(result.pass_at_k)     # {1: 0.9, ...} when repeat_times > 1

asyncio.run(main())
```

## What's inside

- **`Evaluator`** — runs an `EvalTarget` over a dataset with concurrency control, optional `repeat_times`, and pass@k metrics
- **`LLMAsJudgeScorer`** — delegates scoring to an LLM; subclass `build_prompt` / `parse_response` for custom rubrics
- **`OutputRelevanceScorer` / `OutputCorrectnessScorer` / `OutputCompletenessScorer`** — lightweight rule-based scorers for common quality dimensions
- **`FormatValidationScorer` / `SchemaValidationScorer`** — structural checks for JSON, XML, YAML, Markdown, and Pydantic schemas
- **`TrajectoryValidator`** — evaluates the step-by-step execution path of an agent run
- **`GeneralReflector`** — wraps an agent to perform iterative self-reflection and improvement using `ReflectionHistory`

## Part of [Exo](https://github.com/midsphere-ai/exo)

Get the full framework with `pip install exo-ai`.

---

MIT © Midsphere AI
