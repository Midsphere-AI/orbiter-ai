"""Optimizers for LLM prompts and tool descriptions."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from typing import Any

from exo.train.evolution import (  # pyright: ignore[reportMissingImports]
    EvolutionStrategy,
)
from exo.train.operator.llm_call import (  # pyright: ignore[reportMissingImports]
    LLMCallOperator,
)
from exo.train.operator.tool_call import (  # pyright: ignore[reportMissingImports]
    ToolCallOperator,
)


class InstructionOptimizer(EvolutionStrategy):
    """Optimizer that improves system prompts via textual gradients.

    Analyzes evaluation cases (especially failures) to generate natural-
    language "gradients" describing how each operator's prompts should
    change, then applies those gradients to produce improved prompts.

    Implements :class:`EvolutionStrategy` so it can be plugged directly
    into an :class:`EvolutionPipeline`.

    Args:
        model: Model identifier (e.g., ``"gpt-4"``).
        operators: LLM call operators whose prompts will be optimized.
        llm_fn: Async callable with signature
            ``(*, system_prompt: str, user_prompt: str, **kw) -> str``
            used for meta-LLM calls (gradient generation and prompt
            rewriting).  If *None*, the optimizer produces placeholder
            gradients and does not rewrite prompts.
    """

    __slots__ = ("_gradients", "_llm_fn", "_model", "_operators")

    def __init__(
        self,
        model: str,
        operators: list[LLMCallOperator],
        *,
        llm_fn: Any = None,
    ) -> None:
        self._model = model
        self._operators = list(operators)
        self._llm_fn = llm_fn
        self._gradients: list[str] = []

    @property
    def model(self) -> str:
        """Model identifier used for meta-LLM calls."""
        return self._model

    @property
    def operators(self) -> list[LLMCallOperator]:
        """Operators whose prompts are being optimized."""
        return list(self._operators)

    @property
    def gradients(self) -> list[str]:
        """Most recently computed textual gradients."""
        return list(self._gradients)

    # ------------------------------------------------------------------
    # Core optimization API
    # ------------------------------------------------------------------

    async def backward(self, evaluated_cases: Sequence[dict[str, Any]]) -> list[str]:
        """Generate textual gradients from evaluated cases.

        Analyses failures and low-scoring cases to produce natural-language
        descriptions of how each operator's prompts should change.

        Args:
            evaluated_cases: Dicts with at least ``input``, ``output``,
                and optionally ``expected``, ``score``, ``status``.

        Returns:
            One gradient string per operator.
        """
        if not evaluated_cases or not self._operators:
            self._gradients = []
            return []

        # Identify weak cases (score < 1.0 or status == "failed").
        weak = [
            c for c in evaluated_cases if c.get("score", 1.0) < 1.0 or c.get("status") == "failed"
        ]
        if not weak:
            # No clear failures — use all cases for analysis.
            weak = list(evaluated_cases)

        if self._llm_fn is not None:
            system_prompt = (
                "You are an expert prompt engineer. Analyse the "
                "following evaluation failures and describe concisely "
                "how the system prompt should be improved."
            )

            async def _gen_gradient(op: Any) -> str:
                state = op.get_state()
                prompt_text = self._build_backward_prompt(state, weak)
                raw = await self._llm_fn(
                    system_prompt=system_prompt,
                    user_prompt=prompt_text,
                    model=self._model,
                )
                return str(raw)

            gradients = list(await asyncio.gather(*[_gen_gradient(op) for op in self._operators]))
        else:
            gradients = [
                f"Improve prompt for '{op.name}': {len(weak)} weak case(s) detected"
                for op in self._operators
            ]

        self._gradients = gradients
        return list(gradients)

    async def step(self) -> dict[str, str]:
        """Apply accumulated gradients to produce improved prompts.

        Uses the meta-LLM to rewrite each operator's ``system_prompt``
        based on the textual gradients from :meth:`backward`.  Template
        variables (``{{...}}``) are preserved across rewrites.

        Returns:
            Mapping of operator name → new system prompt.
        """
        if not self._gradients:
            return {}

        improved: dict[str, str] = {}
        for op, gradient in zip(self._operators, self._gradients, strict=False):
            state = op.get_state()
            current = state.get("system_prompt", "")

            if self._llm_fn is not None:
                rewrite_prompt = self._build_step_prompt(current, gradient)
                raw = await self._llm_fn(
                    system_prompt=(
                        "You are an expert prompt engineer. Rewrite the "
                        "system prompt incorporating the suggested "
                        "improvements. Keep it concise and preserve all "
                        "template variables exactly as they appear."
                    ),
                    user_prompt=rewrite_prompt,
                    model=self._model,
                )
                new_prompt = str(raw)
            else:
                new_prompt = current

            # Guarantee template variables survive the rewrite.
            new_prompt = self._preserve_templates(current, new_prompt)

            # Apply the updated prompt to the operator.
            op.load_state({**state, "system_prompt": new_prompt})
            improved[op.name] = new_prompt

        self._gradients = []
        return improved

    # ------------------------------------------------------------------
    # EvolutionStrategy interface
    # ------------------------------------------------------------------

    async def synthesize(
        self,
        agent: Any,
        data: Sequence[dict[str, Any]],
        epoch: int,
    ) -> list[dict[str, Any]]:
        """Pass-through — InstructionOptimizer does not synthesize data."""
        return list(data)

    async def train(
        self,
        agent: Any,
        data: Sequence[dict[str, Any]],
        epoch: int,
    ) -> float:
        """Run backward + step as the 'training' phase."""
        gradients = await self.backward(list(data))
        if gradients:
            await self.step()
        return 0.0

    async def evaluate(
        self,
        agent: Any,
        data: Sequence[dict[str, Any]],
        epoch: int,
    ) -> float:
        """Stub — always returns 0.0.

        InstructionOptimizer does not perform its own evaluation.
        Wire up a real eval function in the EvolutionPipeline caller, or
        use ``OperatorTrainer`` which accepts an explicit ``eval_fn``.
        Do not rely on this method for the EVALUATION phase of a pipeline.
        """
        return 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_backward_prompt(
        state: dict[str, Any],
        weak_cases: Sequence[dict[str, Any]],
    ) -> str:
        """Build the analysis prompt for gradient generation."""
        lines = [
            f"Current system_prompt: {state.get('system_prompt', '')}",
            f"Current user_prompt: {state.get('user_prompt', '')}",
            "",
            "Failure cases:",
        ]
        for i, case in enumerate(weak_cases[:10]):  # cap at 10
            lines.append(
                f"  {i + 1}. input={case.get('input', '')!r} "
                f"output={case.get('output', '')!r} "
                f"expected={case.get('expected', '')!r} "
                f"score={case.get('score', 'N/A')}"
            )
        lines.append("")
        lines.append("Describe how the system prompt should be changed.")
        return "\n".join(lines)

    @staticmethod
    def _build_step_prompt(current_prompt: str, gradient: str) -> str:
        """Build the rewrite prompt for prompt improvement."""
        return (
            f"Current system prompt:\n{current_prompt}\n\n"
            f"Suggested improvements:\n{gradient}\n\n"
            "Rewrite the system prompt with these improvements applied. "
            "Preserve all template variables (e.g., {{query}}) exactly."
        )

    @staticmethod
    def _preserve_templates(original: str, rewritten: str) -> str:
        """Ensure all ``{{...}}`` template variables from *original* appear in *rewritten*."""
        templates = set(re.findall(r"\{\{[^}]+\}\}", original))
        for tmpl in templates:
            if tmpl not in rewritten:
                rewritten = f"{rewritten} {tmpl}"
        return rewritten


class ToolOptimizer:
    """Optimizer that improves tool descriptions via beam search.

    For each :class:`ToolCallOperator`, generates multiple candidate
    descriptions, evaluates them against the provided cases, selects the
    best, and optionally refines it — implementing a multi-stage beam
    search over description space.

    Args:
        model: Model identifier for meta-LLM calls.
        operators: Tool call operators whose descriptions will be optimized.
        llm_fn: Async callable with signature
            ``(*, system_prompt: str, user_prompt: str, **kw) -> str``.
            If *None*, the optimizer returns the original descriptions.
    """

    __slots__ = ("_llm_fn", "_model", "_operators")

    def __init__(
        self,
        model: str,
        operators: list[ToolCallOperator],
        *,
        llm_fn: Any = None,
    ) -> None:
        self._model = model
        self._operators = list(operators)
        self._llm_fn = llm_fn

    @property
    def model(self) -> str:
        """Model identifier used for meta-LLM calls."""
        return self._model

    @property
    def operators(self) -> list[ToolCallOperator]:
        """Operators whose descriptions are being optimized."""
        return list(self._operators)

    async def optimize(
        self,
        eval_cases: Sequence[dict[str, Any]],
        *,
        beam_width: int = 3,
    ) -> dict[str, str]:
        """Run beam search to find optimized tool descriptions.

        Multi-stage pipeline:
        1. **Generate** — produce *beam_width* candidate descriptions per
           operator.
        2. **Evaluate** — score each candidate against *eval_cases*.
        3. **Select** — pick the highest-scoring candidate.
        4. **Refine** — improve the selected candidate once more.

        The final description is applied to each operator via
        ``load_state()``.

        Args:
            eval_cases: Evaluation cases with ``input``, ``output``,
                ``expected``, ``score``, ``status``.
            beam_width: Number of candidate descriptions to generate per
                operator.

        Returns:
            Mapping of operator name → optimized description.
        """
        if not self._operators:
            return {}

        result: dict[str, str] = {}

        for op in self._operators:
            current = op.get_state().get("tool_description", "")

            if self._llm_fn is None:
                result[op.name] = current
                continue

            # Stage 1: Generate candidates.
            candidates = await self._generate_candidates(op.name, current, eval_cases, beam_width)

            # Stage 2: Evaluate candidates.
            scored = await self._evaluate_candidates(op.name, candidates, eval_cases)

            # Stage 3: Select best.
            best = max(scored, key=lambda pair: pair[1])[0]

            # Stage 4: Refine.
            refined = await self._refine(op.name, best, eval_cases)

            # Apply to operator.
            op.load_state({"tool_description": refined})
            result[op.name] = refined

        return result

    # ------------------------------------------------------------------
    # Internal stages
    # ------------------------------------------------------------------

    async def _generate_candidates(
        self,
        tool_name: str,
        current_desc: str,
        eval_cases: Sequence[dict[str, Any]],
        beam_width: int,
    ) -> list[str]:
        """Stage 1: Generate *beam_width* candidate descriptions in parallel."""
        assert self._llm_fn is not None
        case_summary = self._summarise_cases(eval_cases)
        system_prompt = (
            "You are an expert at writing tool descriptions for AI "
            "agents. Generate a single improved tool description. "
            "Be concise and specific."
        )

        async def _gen(i: int) -> str:
            raw = await self._llm_fn(
                system_prompt=system_prompt,
                user_prompt=(
                    f"Tool name: {tool_name}\n"
                    f"Current description: {current_desc}\n"
                    f"Candidate number: {i + 1} of {beam_width}\n\n"
                    f"Usage context:\n{case_summary}\n\n"
                    "Write ONE improved description."
                ),
                model=self._model,
            )
            return str(raw)

        return list(await asyncio.gather(*[_gen(i) for i in range(beam_width)]))

    async def _evaluate_candidates(
        self,
        tool_name: str,
        candidates: list[str],
        eval_cases: Sequence[dict[str, Any]],
    ) -> list[tuple[str, float]]:
        """Stage 2: Score each candidate description."""
        assert self._llm_fn is not None
        scored: list[tuple[str, float]] = []
        case_summary = self._summarise_cases(eval_cases)

        for candidate in candidates:
            raw = await self._llm_fn(
                system_prompt=(
                    "You are evaluating tool descriptions. Rate the "
                    "following description on a scale of 0.0 to 1.0 based "
                    "on clarity, specificity, and usefulness for the given "
                    "usage context. Reply with ONLY a number."
                ),
                user_prompt=(
                    f"Tool: {tool_name}\nDescription: {candidate}\n\nUsage context:\n{case_summary}"
                ),
                model=self._model,
            )
            try:
                score = float(str(raw).strip())
                score = max(0.0, min(1.0, score))
            except (ValueError, TypeError):
                score = 0.5  # fallback for non-numeric responses
            scored.append((candidate, score))

        return scored

    async def _refine(
        self,
        tool_name: str,
        description: str,
        eval_cases: Sequence[dict[str, Any]],
    ) -> str:
        """Stage 4: Refine the selected best candidate."""
        assert self._llm_fn is not None
        case_summary = self._summarise_cases(eval_cases)

        raw = await self._llm_fn(
            system_prompt=(
                "You are an expert at writing tool descriptions for AI "
                "agents. Refine the given tool description to be maximally "
                "clear and useful. Be concise."
            ),
            user_prompt=(
                f"Tool: {tool_name}\n"
                f"Description to refine: {description}\n\n"
                f"Usage context:\n{case_summary}\n\n"
                "Write the refined description."
            ),
            model=self._model,
        )
        return str(raw)

    @staticmethod
    def _summarise_cases(
        cases: Sequence[dict[str, Any]],
        *,
        max_cases: int = 5,
    ) -> str:
        """Build a compact summary of evaluation cases for prompts."""
        if not cases:
            return "(no cases)"
        lines: list[str] = []
        for i, case in enumerate(cases[:max_cases]):
            lines.append(
                f"  {i + 1}. input={case.get('input', '')!r} "
                f"output={case.get('output', '')!r} "
                f"score={case.get('score', 'N/A')}"
            )
        if len(cases) > max_cases:
            lines.append(f"  ... and {len(cases) - max_cases} more")
        return "\n".join(lines)
