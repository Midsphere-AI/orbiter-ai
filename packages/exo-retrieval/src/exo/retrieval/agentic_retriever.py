"""Agentic retriever that iteratively refines queries via LLM feedback.

``AgenticRetriever`` wraps a base retriever and uses an LLM to judge
whether the retrieved results sufficiently answer the query.  If not,
the query is rewritten and retrieval is retried for up to *max_rounds*.
Results are deduplicated across rounds by chunk identity.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from exo.retrieval.query_rewriter import QueryRewriter  # pyright: ignore[reportMissingImports]
from exo.retrieval.retriever import Retriever  # pyright: ignore[reportMissingImports]
from exo.retrieval.types import RetrievalResult  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)

_SUFFICIENCY_PROMPT = """You are a retrieval quality judge. Given a query and retrieved passages, assess whether the passages sufficiently answer the query.

Query: {query}

Passages:
{passages}

Rate the sufficiency as a score between 0.0 and 1.0, where:
- 0.0 means the passages are completely irrelevant
- 0.5 means the passages are partially relevant but missing key information
- 1.0 means the passages fully and thoroughly answer the query

Return a JSON object with exactly two fields:
- "score": a float between 0.0 and 1.0
- "reason": a brief explanation

Return ONLY the JSON object, no other text."""


class AgenticRetriever(Retriever):
    """Multi-round LLM-driven retriever that refines queries until satisfied.

    Wraps a base retriever and iteratively rewrites the query, retrieves,
    and asks an LLM to judge result sufficiency.  Stops early when the
    sufficiency score meets the threshold.

    Args:
        base_retriever: The underlying retriever to delegate to.
        rewriter: A ``QueryRewriter`` for query refinement between rounds.
        model: Model string for the sufficiency judge, e.g. ``"openai:gpt-4o"``.
        max_rounds: Maximum retrieval rounds (default 3).
        sufficiency_threshold: Minimum score to accept results (default 0.7).
        provider_kwargs: Extra keyword arguments forwarded to
            ``get_provider()`` for the judge LLM.
    """

    def __init__(
        self,
        base_retriever: Retriever,
        rewriter: QueryRewriter,
        model: str,
        *,
        max_rounds: int = 3,
        sufficiency_threshold: float = 0.7,
        **provider_kwargs: Any,
    ) -> None:
        self.base_retriever = base_retriever
        self.rewriter = rewriter
        self.model = model
        self.max_rounds = max_rounds
        self.sufficiency_threshold = sufficiency_threshold
        self._provider_kwargs = provider_kwargs
        self._provider: Any = None  # lazily initialised on first judge call

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        """Retrieve with iterative refinement.

        Each round: rewrite the query, retrieve, judge sufficiency.
        Stops when the sufficiency threshold is met or max rounds are
        exhausted.  Returns deduplicated results from all rounds,
        sorted by score descending.

        Args:
            query: The search query text.
            top_k: Maximum number of results to return.
            **kwargs: Passed through to the base retriever.

        Returns:
            Deduplicated ``RetrievalResult`` list, highest score first.
        """
        all_results: dict[tuple[str, int], RetrievalResult] = {}

        for _round in range(self.max_rounds):
            # Always rewrite from the original query so that each round
            # independently expands/disambiguates rather than compounding
            # rewrites of rewrites.
            current_query = await self.rewriter.rewrite(query)

            # Retrieve using the base retriever
            round_results = await self.base_retriever.retrieve(current_query, top_k=top_k, **kwargs)

            # Deduplicate: keep the highest-scoring result per chunk
            for result in round_results:
                key = (result.chunk.document_id, result.chunk.index)
                existing = all_results.get(key)
                if existing is None or result.score > existing.score:
                    all_results[key] = result

            # Judge sufficiency of current round results
            if round_results:
                score = await self._judge_sufficiency(query, round_results)
                if score >= self.sufficiency_threshold:
                    break

        # Return deduplicated results sorted by score, limited to top_k
        ranked = sorted(all_results.values(), key=lambda r: r.score, reverse=True)
        return ranked[:top_k]

    async def _judge_sufficiency(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> float:
        """Ask the LLM to judge whether results sufficiently answer the query.

        Returns:
            A float score between 0.0 and 1.0.
        """
        from exo.models import get_provider  # pyright: ignore[reportMissingImports]
        from exo.types import UserMessage

        passages_text = "\n".join(f"[{i}] {r.chunk.content}" for i, r in enumerate(results))
        prompt = _SUFFICIENCY_PROMPT.format(query=query, passages=passages_text)

        if self._provider is None:
            self._provider = get_provider(self.model, **self._provider_kwargs)
        provider = self._provider
        try:
            response = await provider.complete([UserMessage(content=prompt)])
        except Exception as exc:
            logger.debug(
                "AgenticRetriever: sufficiency judge LLM call failed (%s); assuming score=0.0"
                " and continuing.",
                exc,
            )
            return 0.0

        return self._parse_sufficiency(response.content)

    @staticmethod
    def _parse_sufficiency(content: str) -> float:
        """Extract the sufficiency score from the LLM response.

        Falls back to 0.0 if parsing fails.
        """
        # Try JSON parse first
        match = re.search(r"\{[^}]+\}", content)
        if match:
            try:
                data = json.loads(match.group())
                score = float(data.get("score", 0.0))
                return max(0.0, min(1.0, score))
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        # Fallback: try to find a bare float
        float_match = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", content)
        if float_match:
            return float(float_match.group())

        logger.debug(
            "AgenticRetriever: could not parse sufficiency score from LLM response;"
            " falling back to 0.0. Response: %r",
            content[:200],
        )
        return 0.0
