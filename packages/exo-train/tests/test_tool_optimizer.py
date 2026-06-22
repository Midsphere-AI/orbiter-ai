"""Tests for ToolOptimizer — description beam search optimization."""

from __future__ import annotations

from typing import Any

from exo.train.operator.tool_call import (  # pyright: ignore[reportMissingImports]
    ToolCallOperator,
)
from exo.train.optimizer import (  # pyright: ignore[reportMissingImports]
    ToolOptimizer,
)

# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------


async def _mock_tool_fn(**kwargs: Any) -> str:
    """Simple mock tool function."""
    return f"result: {kwargs}"


async def _mock_provider(*, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
    """Mock meta-LLM provider for the ToolOptimizer.

    Returns predictable responses based on the system prompt content.
    """
    if "generate" in system_prompt.lower() or "improved tool description" in system_prompt.lower():
        # Stage 1: Generate candidate descriptions.
        return "Improved description for tool (candidate)"
    if "rate" in system_prompt.lower() or "evaluating" in system_prompt.lower():
        # Stage 2: Evaluate — return a numeric score.
        return "0.8"
    if "refine" in system_prompt.lower():
        # Stage 4: Refine the best candidate.
        return "Refined: optimized tool description"
    return "mock response"


async def _mock_provider_varied_scores(
    *, system_prompt: str, user_prompt: str, **kwargs: Any
) -> str:
    """Mock provider that returns different scores for different candidates."""
    if "generate" in system_prompt.lower() or "improved tool description" in system_prompt.lower():
        if "1 of" in user_prompt:
            return "Candidate A: basic description"
        if "2 of" in user_prompt:
            return "Candidate B: detailed description"
        return "Candidate C: verbose description"
    if "rate" in system_prompt.lower() or "evaluating" in system_prompt.lower():
        if "Candidate B" in user_prompt:
            return "0.95"
        if "Candidate A" in user_prompt:
            return "0.6"
        return "0.7"
    if "refine" in system_prompt.lower():
        # Should refine Candidate B (the highest scored).
        return "Refined: best description from B"
    return "mock"


async def _mock_provider_non_numeric(*, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
    """Mock provider that returns non-numeric evaluation scores."""
    if "generate" in system_prompt.lower() or "improved tool description" in system_prompt.lower():
        return "A candidate description"
    if "rate" in system_prompt.lower() or "evaluating" in system_prompt.lower():
        return "This description is quite good!"  # Non-numeric
    if "refine" in system_prompt.lower():
        return "Refined description"
    return "mock"


def _make_tool_operator(
    name: str = "search_tool",
    description: str = "Search the web for information.",
) -> ToolCallOperator:
    return ToolCallOperator(name, _mock_tool_fn, tool_description=description)


def _make_eval_cases(
    *,
    n_pass: int = 2,
    n_fail: int = 2,
) -> list[dict[str, Any]]:
    """Build a mix of passing and failing evaluation cases."""
    cases: list[dict[str, Any]] = []
    for i in range(n_pass):
        cases.append(
            {
                "input": f"query-{i}",
                "output": f"result-{i}",
                "expected": f"result-{i}",
                "score": 1.0,
                "status": "success",
            }
        )
    for i in range(n_fail):
        cases.append(
            {
                "input": f"hard-query-{i}",
                "output": f"wrong-{i}",
                "expected": f"correct-{i}",
                "score": 0.2,
                "status": "failed",
            }
        )
    return cases


# ---------------------------------------------------------------------------
# Init & properties
# ---------------------------------------------------------------------------


class TestToolOptimizerInit:
    def test_basic_init(self) -> None:
        op = _make_tool_operator()
        opt = ToolOptimizer("gpt-4", [op])
        assert opt.model == "gpt-4"
        assert len(opt.operators) == 1

    def test_multiple_operators(self) -> None:
        ops = [_make_tool_operator("tool-a"), _make_tool_operator("tool-b")]
        opt = ToolOptimizer("gpt-4", ops)
        assert len(opt.operators) == 2

    def test_with_llm_fn(self) -> None:
        opt = ToolOptimizer("gpt-4", [_make_tool_operator()], llm_fn=_mock_provider)
        assert opt.model == "gpt-4"

    def test_operators_are_copies(self) -> None:
        ops = [_make_tool_operator()]
        opt = ToolOptimizer("gpt-4", ops)
        assert opt.operators is not ops


# ---------------------------------------------------------------------------
# optimize() — no llm_fn
# ---------------------------------------------------------------------------


class TestOptimizeWithoutLLM:
    async def test_returns_original_descriptions(self) -> None:
        op = _make_tool_operator("my-tool", "Original description.")
        opt = ToolOptimizer("gpt-4", [op])
        result = await opt.optimize(_make_eval_cases())
        assert result == {"my-tool": "Original description."}

    async def test_empty_operators(self) -> None:
        opt = ToolOptimizer("gpt-4", [])
        result = await opt.optimize(_make_eval_cases())
        assert result == {}

    async def test_multiple_operators_without_llm(self) -> None:
        ops = [
            _make_tool_operator("tool-a", "Desc A"),
            _make_tool_operator("tool-b", "Desc B"),
        ]
        opt = ToolOptimizer("gpt-4", ops)
        result = await opt.optimize(_make_eval_cases())
        assert result == {"tool-a": "Desc A", "tool-b": "Desc B"}


# ---------------------------------------------------------------------------
# optimize() — with mock provider
# ---------------------------------------------------------------------------


class TestOptimizeWithProvider:
    async def test_returns_optimized_descriptions(self) -> None:
        op = _make_tool_operator("search", "Search the web.")
        opt = ToolOptimizer("gpt-4", [op], llm_fn=_mock_provider)
        result = await opt.optimize(_make_eval_cases())
        assert "search" in result
        assert result["search"] != "Search the web."
        # The refined description from our mock.
        assert "Refined" in result["search"]

    async def test_operator_state_updated(self) -> None:
        op = _make_tool_operator("search", "Search the web.")
        opt = ToolOptimizer("gpt-4", [op], llm_fn=_mock_provider)
        await opt.optimize(_make_eval_cases())
        # Operator's description was updated via load_state.
        assert op.get_state()["tool_description"] != "Search the web."

    async def test_multiple_operators_optimized(self) -> None:
        ops = [
            _make_tool_operator("tool-a", "Description A"),
            _make_tool_operator("tool-b", "Description B"),
        ]
        opt = ToolOptimizer("gpt-4", ops, llm_fn=_mock_provider)
        result = await opt.optimize(_make_eval_cases())
        assert "tool-a" in result
        assert "tool-b" in result

    async def test_beam_width_controls_candidates(self) -> None:
        """Verify beam_width parameter affects the number of LLM calls."""
        call_count = 0

        async def _counting_provider(*, system_prompt: str, user_prompt: str, **kw: Any) -> str:
            nonlocal call_count
            call_count += 1
            if "rate" in system_prompt.lower() or "evaluating" in system_prompt.lower():
                return "0.7"
            if "refine" in system_prompt.lower():
                return "refined desc"
            return f"candidate-{call_count}"

        op = _make_tool_operator()
        opt = ToolOptimizer("gpt-4", [op], llm_fn=_counting_provider)
        await opt.optimize(_make_eval_cases(), beam_width=5)
        # 5 generate + 5 evaluate + 1 refine = 11 calls
        assert call_count == 11

    async def test_default_beam_width_is_3(self) -> None:
        call_count = 0

        async def _counting_provider(*, system_prompt: str, user_prompt: str, **kw: Any) -> str:
            nonlocal call_count
            call_count += 1
            if "rate" in system_prompt.lower() or "evaluating" in system_prompt.lower():
                return "0.7"
            if "refine" in system_prompt.lower():
                return "refined desc"
            return f"candidate-{call_count}"

        op = _make_tool_operator()
        opt = ToolOptimizer("gpt-4", [op], llm_fn=_counting_provider)
        await opt.optimize(_make_eval_cases())
        # 3 generate + 3 evaluate + 1 refine = 7 calls
        assert call_count == 7

    async def test_model_passed_to_llm_fn(self) -> None:
        received_models: list[str | None] = []

        async def _capture(*, system_prompt: str, user_prompt: str, **kw: Any) -> str:
            received_models.append(kw.get("model"))
            if "rate" in system_prompt.lower() or "evaluating" in system_prompt.lower():
                return "0.8"
            if "refine" in system_prompt.lower():
                return "refined"
            return "candidate"

        opt = ToolOptimizer("claude-3", [_make_tool_operator()], llm_fn=_capture)
        await opt.optimize(_make_eval_cases())
        assert all(m == "claude-3" for m in received_models)


# ---------------------------------------------------------------------------
# Beam search selection
# ---------------------------------------------------------------------------


class TestBeamSearchSelection:
    async def test_selects_highest_scored_candidate(self) -> None:
        op = _make_tool_operator("search", "Basic search tool.")
        opt = ToolOptimizer("gpt-4", [op], llm_fn=_mock_provider_varied_scores)
        result = await opt.optimize(_make_eval_cases())
        # The refined result should be based on Candidate B (highest scored).
        assert "best description from B" in result["search"]

    async def test_non_numeric_scores_fallback(self) -> None:
        """Non-numeric evaluation responses should fall back to 0.5."""
        op = _make_tool_operator("search", "Search tool.")
        opt = ToolOptimizer("gpt-4", [op], llm_fn=_mock_provider_non_numeric)
        result = await opt.optimize(_make_eval_cases())
        # Should still produce a result (all candidates score 0.5).
        assert "search" in result


# ---------------------------------------------------------------------------
# _summarise_cases helper
# ---------------------------------------------------------------------------


class TestSummariseCases:
    def test_empty_cases(self) -> None:
        result = ToolOptimizer._summarise_cases([])
        assert result == "(no cases)"

    def test_basic_summary(self) -> None:
        cases = [{"input": "q1", "output": "a1", "score": 0.9}]
        result = ToolOptimizer._summarise_cases(cases)
        assert "q1" in result
        assert "a1" in result
        assert "0.9" in result

    def test_caps_at_max_cases(self) -> None:
        cases = [{"input": f"i{i}", "output": f"o{i}"} for i in range(10)]
        result = ToolOptimizer._summarise_cases(cases, max_cases=5)
        assert "i4" in result
        assert "i5" not in result
        assert "5 more" in result

    def test_no_overflow_message_when_within_limit(self) -> None:
        cases = [{"input": "a", "output": "b"}]
        result = ToolOptimizer._summarise_cases(cases)
        assert "more" not in result


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    async def test_empty_eval_cases_with_provider(self) -> None:
        op = _make_tool_operator("tool", "desc")
        opt = ToolOptimizer("gpt-4", [op], llm_fn=_mock_provider)
        result = await opt.optimize([])
        # Still runs the pipeline even with empty cases.
        assert "tool" in result

    async def test_single_beam_width(self) -> None:
        op = _make_tool_operator("tool", "desc")
        opt = ToolOptimizer("gpt-4", [op], llm_fn=_mock_provider)
        result = await opt.optimize(_make_eval_cases(), beam_width=1)
        assert "tool" in result

    async def test_score_clamped_to_range(self) -> None:
        """Scores outside 0.0-1.0 should be clamped."""

        async def _extreme_scorer(*, system_prompt: str, user_prompt: str, **kw: Any) -> str:
            if "rate" in system_prompt.lower() or "evaluating" in system_prompt.lower():
                return "5.0"  # Way above 1.0
            if "refine" in system_prompt.lower():
                return "refined"
            return "candidate"

        op = _make_tool_operator()
        opt = ToolOptimizer("gpt-4", [op], llm_fn=_extreme_scorer)
        # Should not raise — score gets clamped to 1.0.
        result = await opt.optimize(_make_eval_cases())
        assert len(result) == 1
