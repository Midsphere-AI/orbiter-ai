"""Tests for InstructionOptimizer — textual gradient prompt optimization."""

from __future__ import annotations

from typing import Any

from exo.train.evolution import (  # pyright: ignore[reportMissingImports]
    EvolutionConfig,
    EvolutionPipeline,
    EvolutionStrategy,
)
from exo.train.operator.llm_call import (  # pyright: ignore[reportMissingImports]
    LLMCallOperator,
)
from exo.train.optimizer import (  # pyright: ignore[reportMissingImports]
    InstructionOptimizer,
)

# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------


async def _mock_llm(*, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
    """Simple mock LLM used by operators."""
    return f"Response to: {user_prompt}"


async def _mock_provider(*, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
    """Mock meta-LLM provider for the optimizer.

    Returns predictable responses based on the system prompt content.
    """
    if "analyse" in system_prompt.lower() or "failures" in system_prompt.lower():
        return "The prompt should be more specific about output format and examples."
    if "rewrite" in system_prompt.lower():
        # Simulate a rewrite that incorporates improvements.
        # Extract template variables from user_prompt to preserve them.
        if "{{query}}" in user_prompt:
            return "Be concise. Format output as JSON. Answer: {{query}}"
        return "Be concise. Format output as JSON."
    return "mock response"


async def _mock_provider_drops_template(
    *, system_prompt: str, user_prompt: str, **kwargs: Any
) -> str:
    """Mock provider that intentionally drops template variables."""
    if "rewrite" in system_prompt.lower():
        return "Be concise and specific."
    return "gradient: improve formatting"


def _make_operator(
    name: str = "summarizer",
    system_prompt: str = "Be helpful.",
    user_prompt: str = "",
) -> LLMCallOperator:
    return LLMCallOperator(
        name,
        _mock_llm,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


def _make_cases(
    *,
    n_pass: int = 2,
    n_fail: int = 2,
) -> list[dict[str, Any]]:
    """Build a mix of passing and failing evaluation cases."""
    cases: list[dict[str, Any]] = []
    for i in range(n_pass):
        cases.append(
            {
                "input": f"good-input-{i}",
                "output": f"good-output-{i}",
                "expected": f"good-output-{i}",
                "score": 1.0,
                "status": "success",
            }
        )
    for i in range(n_fail):
        cases.append(
            {
                "input": f"bad-input-{i}",
                "output": f"wrong-output-{i}",
                "expected": f"correct-output-{i}",
                "score": 0.3,
                "status": "failed",
            }
        )
    return cases


# ---------------------------------------------------------------------------
# Init & properties
# ---------------------------------------------------------------------------


class TestInstructionOptimizerInit:
    def test_basic_init(self) -> None:
        op = _make_operator()
        opt = InstructionOptimizer("gpt-4", [op])
        assert opt.model == "gpt-4"
        assert len(opt.operators) == 1
        assert opt.gradients == []

    def test_multiple_operators(self) -> None:
        ops = [_make_operator("op1"), _make_operator("op2")]
        opt = InstructionOptimizer("gpt-4", ops)
        assert len(opt.operators) == 2

    def test_is_evolution_strategy(self) -> None:
        opt = InstructionOptimizer("gpt-4", [_make_operator()])
        assert isinstance(opt, EvolutionStrategy)

    def test_with_llm_fn(self) -> None:
        opt = InstructionOptimizer("gpt-4", [_make_operator()], llm_fn=_mock_provider)
        assert opt.model == "gpt-4"

    def test_operators_are_copies(self) -> None:
        ops = [_make_operator()]
        opt = InstructionOptimizer("gpt-4", ops)
        assert opt.operators is not ops


# ---------------------------------------------------------------------------
# backward()
# ---------------------------------------------------------------------------


class TestBackward:
    async def test_empty_cases(self) -> None:
        opt = InstructionOptimizer("gpt-4", [_make_operator()])
        gradients = await opt.backward([])
        assert gradients == []
        assert opt.gradients == []

    async def test_no_operators(self) -> None:
        opt = InstructionOptimizer("gpt-4", [])
        gradients = await opt.backward(_make_cases())
        assert gradients == []

    async def test_without_llm_fn(self) -> None:
        op = _make_operator("my-op")
        opt = InstructionOptimizer("gpt-4", [op])
        gradients = await opt.backward(_make_cases(n_fail=3))
        assert len(gradients) == 1
        assert "my-op" in gradients[0]
        assert "3 weak case(s)" in gradients[0]
        # Gradients stored.
        assert opt.gradients == gradients

    async def test_with_mock_provider(self) -> None:
        op = _make_operator()
        opt = InstructionOptimizer("gpt-4", [op], llm_fn=_mock_provider)
        gradients = await opt.backward(_make_cases())
        assert len(gradients) == 1
        assert "specific" in gradients[0].lower() or "format" in gradients[0].lower()

    async def test_multiple_operators(self) -> None:
        ops = [_make_operator("op-a"), _make_operator("op-b")]
        opt = InstructionOptimizer("gpt-4", ops, llm_fn=_mock_provider)
        gradients = await opt.backward(_make_cases())
        assert len(gradients) == 2

    async def test_all_passing_cases_still_analysed(self) -> None:
        """When no cases are clearly failing, all cases are used."""
        op = _make_operator()
        opt = InstructionOptimizer("gpt-4", [op])
        cases = [{"input": "a", "output": "b", "score": 1.0, "status": "success"}]
        gradients = await opt.backward(cases)
        assert len(gradients) == 1
        assert "1 weak case(s)" in gradients[0]

    async def test_model_passed_to_llm_fn(self) -> None:
        """Verify the model string is forwarded to llm_fn."""
        received_model = []

        async def _capture(*, system_prompt: str, user_prompt: str, **kw: Any) -> str:
            received_model.append(kw.get("model"))
            return "gradient"

        opt = InstructionOptimizer("claude-3", [_make_operator()], llm_fn=_capture)
        await opt.backward(_make_cases())
        assert received_model == ["claude-3"]


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------


class TestStep:
    async def test_empty_gradients(self) -> None:
        opt = InstructionOptimizer("gpt-4", [_make_operator()])
        result = await opt.step()
        assert result == {}

    async def test_without_llm_fn(self) -> None:
        op = _make_operator(system_prompt="Original prompt.")
        opt = InstructionOptimizer("gpt-4", [op])
        # Generate gradients first.
        await opt.backward(_make_cases())
        result = await opt.step()
        # Without llm_fn, prompt is unchanged.
        assert result["summarizer"] == "Original prompt."
        assert op.get_state()["system_prompt"] == "Original prompt."
        # Gradients cleared after step.
        assert opt.gradients == []

    async def test_with_mock_provider(self) -> None:
        op = _make_operator(system_prompt="Be helpful.")
        opt = InstructionOptimizer("gpt-4", [op], llm_fn=_mock_provider)
        await opt.backward(_make_cases())
        result = await opt.step()
        assert "summarizer" in result
        # Provider rewrites the prompt.
        new_prompt = result["summarizer"]
        assert new_prompt != "Be helpful."
        # Operator state updated.
        assert op.get_state()["system_prompt"] == new_prompt

    async def test_step_clears_gradients(self) -> None:
        opt = InstructionOptimizer("gpt-4", [_make_operator()], llm_fn=_mock_provider)
        await opt.backward(_make_cases())
        assert len(opt.gradients) == 1
        await opt.step()
        assert opt.gradients == []

    async def test_multiple_operators_updated(self) -> None:
        ops = [
            _make_operator("op-a", system_prompt="Prompt A"),
            _make_operator("op-b", system_prompt="Prompt B"),
        ]
        opt = InstructionOptimizer("gpt-4", ops, llm_fn=_mock_provider)
        await opt.backward(_make_cases())
        result = await opt.step()
        assert "op-a" in result
        assert "op-b" in result

    async def test_user_prompt_preserved(self) -> None:
        """step() only changes system_prompt, not user_prompt."""
        op = _make_operator(
            system_prompt="Be helpful.",
            user_prompt="Summarise: {{text}}",
        )
        opt = InstructionOptimizer("gpt-4", [op], llm_fn=_mock_provider)
        await opt.backward(_make_cases())
        await opt.step()
        assert op.get_state()["user_prompt"] == "Summarise: {{text}}"


# ---------------------------------------------------------------------------
# Template preservation
# ---------------------------------------------------------------------------


class TestTemplatePreservation:
    async def test_templates_preserved_by_provider(self) -> None:
        """When the provider includes the template, it's kept."""
        op = _make_operator(system_prompt="Answer: {{query}}")
        opt = InstructionOptimizer("gpt-4", [op], llm_fn=_mock_provider)
        await opt.backward(_make_cases())
        result = await opt.step()
        assert "{{query}}" in result["summarizer"]

    async def test_templates_restored_when_dropped(self) -> None:
        """When the provider drops a template, _preserve_templates adds it back."""
        op = _make_operator(system_prompt="Process {{query}} and {{context}}")
        opt = InstructionOptimizer("gpt-4", [op], llm_fn=_mock_provider_drops_template)
        await opt.backward(_make_cases())
        result = await opt.step()
        assert "{{query}}" in result["summarizer"]
        assert "{{context}}" in result["summarizer"]

    def test_preserve_templates_static(self) -> None:
        original = "Hello {{name}}, your {{item}} is ready."
        rewritten = "Hi there, your order is ready."
        result = InstructionOptimizer._preserve_templates(original, rewritten)
        assert "{{name}}" in result
        assert "{{item}}" in result

    def test_preserve_templates_noop_when_present(self) -> None:
        original = "Hello {{name}}"
        rewritten = "Greetings {{name}}"
        result = InstructionOptimizer._preserve_templates(original, rewritten)
        assert result == rewritten  # No change needed.

    def test_preserve_templates_no_templates(self) -> None:
        original = "Be helpful."
        rewritten = "Be concise."
        result = InstructionOptimizer._preserve_templates(original, rewritten)
        assert result == rewritten


# ---------------------------------------------------------------------------
# EvolutionStrategy integration
# ---------------------------------------------------------------------------


class TestEvolutionStrategyIntegration:
    async def test_synthesise_passthrough(self) -> None:
        opt = InstructionOptimizer("gpt-4", [_make_operator()])
        data = [{"input": "a", "output": "b"}]
        result = await opt.synthesise(None, data, 0)
        assert result == data

    async def test_train_runs_backward_and_step(self) -> None:
        op = _make_operator(system_prompt="Original.")
        opt = InstructionOptimizer("gpt-4", [op], llm_fn=_mock_provider)
        cases = _make_cases()
        loss = await opt.train(None, cases, 0)
        assert loss == 0.0
        # Prompt should have been updated.
        assert op.get_state()["system_prompt"] != "Original."

    async def test_evaluate_returns_zero(self) -> None:
        opt = InstructionOptimizer("gpt-4", [_make_operator()])
        result = await opt.evaluate(None, [], 0)
        assert result == 0.0

    async def test_works_in_evolution_pipeline(self) -> None:
        """Verify InstructionOptimizer plugs into EvolutionPipeline."""
        op = _make_operator(system_prompt="Be helpful.")
        opt = InstructionOptimizer("gpt-4", [op], llm_fn=_mock_provider)
        config = EvolutionConfig(max_epochs=1)
        pipeline = EvolutionPipeline(opt, config)

        cases = _make_cases()
        result = await pipeline.run(None, cases)
        assert result.total_epochs == 1
        # Prompt was optimized during training phase.
        assert op.get_state()["system_prompt"] != "Be helpful."


# ---------------------------------------------------------------------------
# backward + step cycle
# ---------------------------------------------------------------------------


class TestBackwardStepCycle:
    async def test_full_cycle(self) -> None:
        op = _make_operator(system_prompt="Be helpful. Answer: {{query}}")
        opt = InstructionOptimizer("gpt-4", [op], llm_fn=_mock_provider)

        # backward
        gradients = await opt.backward(_make_cases())
        assert len(gradients) == 1
        assert len(opt.gradients) == 1

        # step
        result = await opt.step()
        assert "summarizer" in result
        assert "{{query}}" in result["summarizer"]
        assert opt.gradients == []

    async def test_multiple_cycles(self) -> None:
        op = _make_operator(system_prompt="V1")
        opt = InstructionOptimizer("gpt-4", [op], llm_fn=_mock_provider)

        # Cycle 1
        await opt.backward(_make_cases())
        r1 = await opt.step()
        prompt_after_1 = r1["summarizer"]

        # Cycle 2
        await opt.backward(_make_cases())
        r2 = await opt.step()

        # Both cycles produced results.
        assert prompt_after_1 != "V1"
        assert "summarizer" in r2


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestBuildPrompts:
    def test_backward_prompt_structure(self) -> None:
        state = {"system_prompt": "Be helpful.", "user_prompt": "Q: {{q}}"}
        cases = [
            {"input": "x", "output": "wrong", "expected": "right", "score": 0.2},
        ]
        text = InstructionOptimizer._build_backward_prompt(state, cases)
        assert "Be helpful." in text
        assert "Q: {{q}}" in text
        assert "x" in text
        assert "wrong" in text
        assert "right" in text

    def test_backward_prompt_caps_at_10(self) -> None:
        state = {"system_prompt": "", "user_prompt": ""}
        cases = [{"input": f"i{i}", "output": f"o{i}"} for i in range(20)]
        text = InstructionOptimizer._build_backward_prompt(state, cases)
        # Should only include first 10.
        assert "i9" in text
        assert "i10" not in text

    def test_step_prompt_structure(self) -> None:
        text = InstructionOptimizer._build_step_prompt("Be helpful.", "Add more examples.")
        assert "Be helpful." in text
        assert "Add more examples." in text
        assert "template variables" in text.lower()
