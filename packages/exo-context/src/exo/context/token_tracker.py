"""TokenTracker — per-agent, per-step token tracking for cost analysis and budget enforcement.

Tracks prompt and output token counts per agent per LLM call step, enabling
trajectory analysis, cost aggregation, and budget enforcement.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TokenStep:
    """A single token usage observation for one LLM call.

    Parameters
    ----------
    agent_id:
        The agent that made the LLM call.
    step:
        Zero-based step index within the agent's trajectory.
    prompt_tokens:
        Number of tokens in the prompt (input).
    output_tokens:
        Number of tokens generated (output/completion).
    """

    agent_id: str
    step: int
    prompt_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        """Prompt + output tokens for this step."""
        return self.prompt_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class TokenUsageSummary:
    """Aggregated token usage across agents and steps."""

    prompt_tokens: int
    output_tokens: int
    total_tokens: int
    step_count: int


class TokenTracker:
    """Tracks per-agent, per-step token usage.

    Usage::

        tracker = TokenTracker()
        tracker.add_step("agent-a", prompt_tokens=100, output_tokens=50)
        tracker.add_step("agent-a", prompt_tokens=120, output_tokens=60)
        tracker.add_step("agent-b", prompt_tokens=200, output_tokens=80)

        # Per-agent trajectory
        trajectory = tracker.get_trajectory("agent-a")
        assert len(trajectory) == 2

        # Global aggregation
        usage = tracker.total_usage()
        assert usage.total_tokens == 610
    """

    __slots__ = (
        "_agent_indices",
        "_agent_prompt_totals",
        "_agent_step_counts",
        "_agent_totals",
        "_global_output_total",
        "_global_prompt_total",
        "_global_total",
        "_steps",
    )

    def __init__(self) -> None:
        self._steps: list[TokenStep] = []
        # Running counters — avoids O(n) scans in add_step debug logging.
        self._agent_totals: defaultdict[str, int] = defaultdict(int)
        self._agent_step_counts: defaultdict[str, int] = defaultdict(int)
        self._global_total: int = 0
        # Per-agent index lists for O(1) trajectory lookup.
        self._agent_indices: defaultdict[str, list[int]] = defaultdict(list)
        # Separate prompt/output running totals for O(1) total_usage().
        self._global_prompt_total: int = 0
        self._global_output_total: int = 0
        self._agent_prompt_totals: defaultdict[str, int] = defaultdict(int)

    def add_step(
        self,
        agent_id: str,
        *,
        prompt_tokens: int,
        output_tokens: int,
    ) -> TokenStep:
        """Record a token usage step for an agent.

        Parameters
        ----------
        agent_id:
            Identifier of the agent that made the LLM call.
        prompt_tokens:
            Number of prompt (input) tokens.
        output_tokens:
            Number of output (completion) tokens.

        Returns
        -------
        The created :class:`TokenStep`.
        """
        # Step index is per-agent O(1) via running counter.
        step_index = self._agent_step_counts[agent_id]
        token_step = TokenStep(
            agent_id=agent_id,
            step=step_index,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )
        total = prompt_tokens + output_tokens
        # Record the index before appending so we can build the per-agent index.
        step_list_index = len(self._steps)
        self._steps.append(token_step)
        self._agent_step_counts[agent_id] += 1
        self._agent_totals[agent_id] += total
        self._global_total += total
        # Maintain O(1) per-agent index and split prompt/output running totals.
        self._agent_indices[agent_id].append(step_list_index)
        self._global_prompt_total += prompt_tokens
        self._global_output_total += output_tokens
        self._agent_prompt_totals[agent_id] += prompt_tokens
        logger.debug(
            "token step recorded: agent=%r step=%d prompt=%d output=%d",
            agent_id,
            step_index,
            prompt_tokens,
            output_tokens,
        )
        logger.debug(
            "TokenTracker: agent_used=%d / global_total=%d (%.0f%%)",
            self._agent_totals[agent_id],
            self._global_total,
            100.0 * self._agent_totals[agent_id] / self._global_total
            if self._global_total > 0
            else 0.0,
        )
        return token_step

    def get_trajectory(self, agent_id: str) -> list[TokenStep]:
        """Get the ordered list of token steps for a specific agent.

        O(k) where k = number of steps for this agent (uses per-agent index).
        """
        return [self._steps[i] for i in self._agent_indices[agent_id]]

    def total_usage(self) -> TokenUsageSummary:
        """Aggregate token usage across all agents and steps.

        O(1) — uses running counters maintained by :meth:`add_step`.
        """
        return TokenUsageSummary(
            prompt_tokens=self._global_prompt_total,
            output_tokens=self._global_output_total,
            total_tokens=self._global_total,
            step_count=len(self._steps),
        )

    def agent_usage(self, agent_id: str) -> TokenUsageSummary:
        """Aggregate token usage for a specific agent.

        O(1) — uses running counters maintained by :meth:`add_step`.
        """
        prompt = self._agent_prompt_totals[agent_id]
        total = self._agent_totals[agent_id]
        output = total - prompt
        step_count = self._agent_step_counts[agent_id]
        return TokenUsageSummary(
            prompt_tokens=prompt,
            output_tokens=output,
            total_tokens=total,
            step_count=step_count,
        )

    def add_usage(self, agent_id: str, usage: Any) -> TokenStep:
        """Record token usage from a Usage object (e.g. ``exo.types.Usage``).

        Accepts any object with ``input_tokens`` and ``output_tokens``
        attributes (duck typing — no direct import of ``exo.types``).

        Parameters
        ----------
        agent_id:
            Identifier of the agent that made the LLM call.
        usage:
            A usage object with ``input_tokens`` and ``output_tokens`` fields.

        Returns
        -------
        The created :class:`TokenStep`.
        """
        return self.add_step(
            agent_id,
            prompt_tokens=getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
        )

    @property
    def agent_ids(self) -> list[str]:
        """List of unique agent IDs that have recorded steps, in first-seen order."""
        seen: dict[str, None] = {}
        for s in self._steps:
            seen.setdefault(s.agent_id, None)
        return list(seen)

    @property
    def steps(self) -> list[TokenStep]:
        """All recorded steps in order."""
        return list(self._steps)

    def __len__(self) -> int:
        return len(self._steps)

    def __repr__(self) -> str:
        agents = len(self.agent_ids)
        steps = len(self._steps)
        return f"TokenTracker(agents={agents}, steps={steps})"
