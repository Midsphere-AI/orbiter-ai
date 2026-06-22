"""LLM-based query rewriting for improved retrieval quality.

``QueryRewriter`` rewrites user queries to expand synonyms, disambiguate
terms, and optionally incorporate conversation history for better
retrieval results.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_TEMPLATE = """You are a search query optimizer. Rewrite the following query to improve retrieval quality.

Expand the query with relevant synonyms, related terms, and disambiguations.
Keep the rewritten query concise (1-2 sentences max).

Original query: {query}

Return ONLY the rewritten query, no other text."""

_HISTORY_TEMPLATE = """You are a search query optimizer. Rewrite the following query to improve retrieval quality.

Consider the conversation history to resolve pronouns and references.
Expand the query with relevant synonyms, related terms, and disambiguations.
Keep the rewritten query concise (1-2 sentences max).

Conversation history:
{history}

Original query: {query}

Return ONLY the rewritten query, no other text."""


class QueryRewriter:
    """Rewrites queries via an LLM to improve retrieval quality.

    Uses an LLM provider to expand queries with synonyms, disambiguate
    terms, and optionally incorporate conversation history context.

    Args:
        model: Model string, e.g. ``"openai:gpt-4o"``.
        prompt_template: Template with ``{query}`` placeholder (and
            optional ``{history}`` placeholder). Defaults to a built-in
            query expansion prompt.
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
        self.prompt_template = prompt_template or _DEFAULT_TEMPLATE
        self._provider_kwargs = provider_kwargs
        self._provider: Any = None  # lazily initialised on first rewrite call

    async def rewrite(
        self,
        query: str,
        *,
        history: list[str] | None = None,
    ) -> str:
        """Rewrite a query for better retrieval.

        Args:
            query: The original user query.
            history: Optional conversation history for context resolution.

        Returns:
            The rewritten query string.
        """
        from exo.models import get_provider  # pyright: ignore[reportMissingImports]
        from exo.types import UserMessage

        # Choose template based on whether history is provided
        if history:
            template = (
                self.prompt_template if "{history}" in self.prompt_template else _HISTORY_TEMPLATE
            )
            history_text = "\n".join(f"- {h}" for h in history)
            prompt = template.format(query=query, history=history_text)
        else:
            prompt = self.prompt_template.format(query=query)

        if self._provider is None:
            self._provider = get_provider(self.model, **self._provider_kwargs)
        provider = self._provider
        try:
            response = await provider.complete([UserMessage(content=prompt)])
        except Exception as exc:
            from exo.retrieval.types import RetrievalError  # pyright: ignore[reportMissingImports]

            raise RetrievalError(
                f"Query rewrite failed: {exc}",
                context={"model": self.model},
                hint="Check the model string and API key for the rewrite LLM.",
            ) from exc

        # Strip whitespace and return; fall back to original on empty response
        rewritten = response.content.strip()
        return rewritten if rewritten else query
