"""PromptBuilder — compose neurons in priority order to build rich system prompts.

The builder collects neurons by name, resolves them from the neuron registry,
formats each in priority order, and joins the results into a single prompt
string.  Template variable resolution is supported via
:class:`DynamicVariableRegistry`.
"""

from __future__ import annotations

import logging
from typing import Any

from exo.types import ExoError  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)

from exo.context.context import Context  # pyright: ignore[reportMissingImports]
from exo.context.neuron import Neuron, neuron_registry  # pyright: ignore[reportMissingImports]
from exo.context.variables import (  # pyright: ignore[reportMissingImports]
    DynamicVariableRegistry,
)


class PromptBuilderError(ExoError):
    """Raised for prompt building failures."""


class PromptBuilder:
    """Composes neurons in priority order to build system prompts.

    Usage::

        builder = PromptBuilder(ctx)
        builder.add("task")
        builder.add("history")
        builder.add("system")
        prompt = await builder.build()

    Neurons are resolved from the global :data:`neuron_registry` by name.
    Each neuron's :meth:`~Neuron.format` is called with the context and any
    extra ``kwargs`` passed to :meth:`add`.

    An optional :class:`DynamicVariableRegistry` can be provided to resolve
    ``${path}`` template variables in the final prompt.

    Parameters
    ----------
    ctx:
        The context to pass to each neuron's ``format()``.
    variables:
        Optional variable registry for template resolution in the final prompt.
    separator:
        String used to join neuron outputs. Default ``"\\n\\n"``.
    """

    __slots__ = ("_ctx", "_entries", "_separator", "_variables")

    def __init__(
        self,
        ctx: Context,
        *,
        variables: DynamicVariableRegistry | None = None,
        separator: str = "\n\n",
    ) -> None:
        self._ctx = ctx
        self._variables = variables
        self._separator = separator
        self._entries: list[_NeuronEntry] = []

    @property
    def ctx(self) -> Context:
        """The context used for neuron formatting."""
        return self._ctx

    def add(self, neuron_name: str, **kwargs: Any) -> PromptBuilder:
        """Register a neuron by name for inclusion in the prompt.

        The neuron is resolved from :data:`neuron_registry` immediately.
        Extra *kwargs* are passed to the neuron's ``format()`` call.

        Returns ``self`` for method chaining.

        Raises
        ------
        PromptBuilderError
            If *neuron_name* is not found in the registry.
        """
        try:
            neuron = neuron_registry.get(neuron_name)
        except Exception as exc:
            msg = f"Neuron {neuron_name!r} not found in registry"
            logger.warning(msg)
            raise PromptBuilderError(msg) from exc
        self._entries.append(_NeuronEntry(neuron=neuron, kwargs=kwargs))
        return self

    def add_neuron(self, neuron: Neuron, **kwargs: Any) -> PromptBuilder:
        """Register a neuron instance directly (bypassing the registry).

        Returns ``self`` for method chaining.
        """
        self._entries.append(_NeuronEntry(neuron=neuron, kwargs=kwargs))
        return self

    async def build(self) -> str:
        """Resolve all neurons in priority order and compose the final prompt.

        Steps:
        1. Sort entries by neuron priority (ascending — lower = earlier).
        2. Call each neuron's ``format(ctx, **kwargs)``.
        3. Filter out empty results.
        4. Join non-empty fragments with the separator.
        5. If a variable registry is set, resolve ``${path}`` templates.

        Returns
        -------
        str
            The assembled prompt string.
        """
        if not self._entries:
            return ""

        # Sort by priority (stable sort preserves insertion order for ties)
        sorted_entries = sorted(self._entries, key=lambda e: e.neuron.priority)

        fragments: list[str] = []
        for entry in sorted_entries:
            fragment = await entry.neuron.format(self._ctx, **entry.kwargs)
            if fragment:
                fragments.append(fragment)

        prompt = self._separator.join(fragments)

        # Template variable resolution
        if self._variables is not None and prompt:
            prompt = self._variables.resolve_template(prompt, self._ctx.state)

        logger.debug(
            "prompt built: %d neurons, %d fragments, %d chars",
            len(sorted_entries),
            len(fragments),
            len(prompt),
        )
        return prompt

    def clear(self) -> None:
        """Remove all registered neuron entries."""
        self._entries.clear()

    def __len__(self) -> int:
        """Number of registered neuron entries."""
        return len(self._entries)

    def __repr__(self) -> str:
        names = [e.neuron.name for e in self._entries]
        return f"PromptBuilder(neurons={names})"


class _NeuronEntry:
    """Internal: pairs a neuron with its format kwargs."""

    __slots__ = ("kwargs", "neuron")

    def __init__(self, *, neuron: Neuron, kwargs: dict[str, Any]) -> None:
        self.neuron = neuron
        self.kwargs = kwargs
