"""Reranker abstractions for refining retrieval results.

A ``Reranker`` takes a query and a list of ``RetrievalResult`` objects and
returns a reordered (and optionally truncated) list. ``LLMReranker`` uses
an LLM provider to judge relevance and reorder results accordingly.
"""

from __future__ import annotations

import abc
import json
import logging
import re
from typing import Any

from exo.retrieval.types import RetrievalResult  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT = """You are a relevance judge. Given a query and a list of text passages, rank the passages by relevance to the query.

Query: {query}

Passages:
{passages}

Return a JSON array of passage indices ordered from most relevant to least relevant.
For example, if passages 2, 0, 1 are in order of relevance, return: [2, 0, 1]

Return ONLY the JSON array, no other text."""


class Reranker(abc.ABC):
    """Abstract base class for rerankers.

    Subclasses must implement ``rerank`` to reorder retrieval results
    by relevance to the query.
    """

    @abc.abstractmethod
    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        *,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Rerank retrieval results by relevance to the query.

        Args:
            query: The original search query.
            results: Retrieval results to rerank.
            top_k: Maximum number of results to return.

        Returns:
            A reordered list of ``RetrievalResult`` objects, most relevant first.
        """


class LLMReranker(Reranker):
    """Reranker that uses an LLM to judge passage relevance.

    Sends the query and passage texts to an LLM provider, asks for a
    relevance ranking, and reorders the results accordingly.

    Args:
        model: Model string, e.g. ``"openai:gpt-4o"``.
        prompt_template: Template with ``{query}`` and ``{passages}``
            placeholders. Defaults to a built-in relevance judging prompt.
        provider_kwargs: Extra keyword arguments forwarded to
            ``get_provider()`` (e.g. ``api_key``, ``base_url``).
    """

    def __init__(
        self,
        model: str,
        *,
        prompt_template: str | None = None,
        **provider_kwargs: Any,
    ) -> None:
        self.model = model
        self.prompt_template = prompt_template or _DEFAULT_PROMPT
        self._provider_kwargs = provider_kwargs
        self._provider: Any = None  # lazily initialised on first rerank call

    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        *,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Rerank results using an LLM relevance judgement.

        Args:
            query: The original search query.
            results: Retrieval results to rerank.
            top_k: Maximum number of results to return.

        Returns:
            A reordered list of ``RetrievalResult`` objects with updated scores.
        """
        if not results:
            return []

        from exo.models import get_provider  # pyright: ignore[reportMissingImports]
        from exo.retrieval.types import RetrievalError  # pyright: ignore[reportMissingImports]
        from exo.types import UserMessage

        # Build numbered passages text
        passages_text = "\n".join(f"[{i}] {r.chunk.content}" for i, r in enumerate(results))
        prompt = self.prompt_template.format(query=query, passages=passages_text)

        if self._provider is None:
            self._provider = get_provider(self.model, **self._provider_kwargs)
        provider = self._provider
        try:
            response = await provider.complete([UserMessage(content=prompt)])
        except Exception as exc:
            raise RetrievalError(
                f"Reranking LLM call failed: {exc}",
                context={"model": self.model, "operation": "rerank"},
                hint="Check the model string and API key for the reranking LLM.",
            ) from exc

        ranking = self._parse_ranking(response.content, len(results))

        # Build reranked results with new scores (highest score for rank 0)
        reranked: list[RetrievalResult] = []
        for rank, idx in enumerate(ranking[:top_k]):
            original = results[idx]
            score = 1.0 - (rank / len(ranking))
            reranked.append(
                RetrievalResult(
                    chunk=original.chunk,
                    score=score,
                    metadata={**original.metadata, "original_score": original.score},
                )
            )
        return reranked

    @staticmethod
    def _parse_ranking(content: str, num_results: int) -> list[int]:
        """Parse the LLM's ranking response into a list of indices.

        Falls back to original order if parsing fails.

        Args:
            content: Raw LLM response text.
            num_results: Number of results that were sent to the LLM.

        Returns:
            A list of integer indices representing the ranking.
        """
        # Try to extract a JSON array from the response
        match = re.search(r"\[[\s\d,]+\]", content)
        if match:
            try:
                indices = json.loads(match.group())
                # Validate: all must be ints in range
                if all(isinstance(i, int) and 0 <= i < num_results for i in indices):
                    # Deduplicate while preserving order
                    seen: set[int] = set()
                    unique: list[int] = []
                    for i in indices:
                        if i not in seen:
                            seen.add(i)
                            unique.append(i)
                    # Append any missing indices at the end
                    for i in range(num_results):
                        if i not in seen:
                            unique.append(i)
                    return unique
            except (json.JSONDecodeError, TypeError):
                pass

        # Fallback: original order
        logger.debug(
            "LLMReranker could not parse ranking from LLM response; falling back to original order."
            " Response: %r",
            content[:200],
        )
        return list(range(num_results))
