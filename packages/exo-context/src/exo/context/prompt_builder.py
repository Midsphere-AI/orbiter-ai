"""PromptBuilder — compose sections in priority order to build rich system prompts.

The builder collects sections by name, resolves them from the section registry,
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
from exo.context.neuron import (  # pyright: ignore[reportMissingImports]
    # Deprecated aliases kept for internal backward-compat
    PromptSection,
    section_registry,
)
from exo.context.variables import (  # pyright: ignore[reportMissingImports]
    DynamicVariableRegistry,
)


class PromptBuilderError(ExoError):
    """Raised for prompt building failures."""


class PromptBuilder:
    """Composes sections in priority order to build system prompts.

    Usage::

        builder = PromptBuilder(ctx)
        builder.add("task")
        builder.add("history")
        builder.add("system")
        prompt = await builder.build()

    Sections are resolved from the global :data:`section_registry` by name.
    Each section's :meth:`~PromptSection.format` is called with the context and any
    extra ``kwargs`` passed to :meth:`add`.

    An optional :class:`DynamicVariableRegistry` can be provided to resolve
    ``${path}`` template variables in the final prompt.

    Parameters
    ----------
    ctx:
        The context to pass to each section's ``format()``.
    variables:
        Optional variable registry for template resolution in the final prompt.
    separator:
        String used to join section outputs. Default ``"\\n\\n"``.
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
        self._entries: list[_SectionEntry] = []

    @property
    def ctx(self) -> Context:
        """The context used for section formatting."""
        return self._ctx

    def add(self, section_name: str, **kwargs: Any) -> PromptBuilder:
        """Register a section by name for inclusion in the prompt.

        The section is resolved from :data:`section_registry` immediately.
        Extra *kwargs* are passed to the section's ``format()`` call.

        Returns ``self`` for method chaining.

        Raises
        ------
        PromptBuilderError
            If *section_name* is not found in the registry.
        """
        try:
            section = section_registry.get(section_name)
        except Exception as exc:
            msg = f"PromptSection {section_name!r} not found in registry"
            logger.warning(msg)
            raise PromptBuilderError(msg) from exc
        self._entries.append(_SectionEntry(section=section, kwargs=kwargs))
        return self

    def add_section(self, section: PromptSection, **kwargs: Any) -> PromptBuilder:
        """Register a section instance directly (bypassing the registry).

        Returns ``self`` for method chaining.
        """
        self._entries.append(_SectionEntry(section=section, kwargs=kwargs))
        return self

    def add_neuron(self, neuron: PromptSection, **kwargs: Any) -> PromptBuilder:
        """Deprecated alias for :meth:`add_section`.

        .. deprecated::
            Use :meth:`add_section` instead.
        """
        return self.add_section(neuron, **kwargs)

    async def build(self) -> str:
        """Resolve all sections in priority order and compose the final prompt.

        Steps:
        1. Sort entries by section priority (ascending — lower = earlier).
        2. Call each section's ``format(ctx, **kwargs)``.
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
        sorted_entries = sorted(self._entries, key=lambda e: e.section.priority)

        fragments: list[str] = []
        for entry in sorted_entries:
            fragment = await entry.section.format(self._ctx, **entry.kwargs)
            if fragment:
                fragments.append(fragment)

        prompt = self._separator.join(fragments)

        # Template variable resolution
        if self._variables is not None and prompt:
            prompt = self._variables.resolve_template(prompt, self._ctx.state)

        logger.debug(
            "prompt built: %d sections, %d fragments, %d chars",
            len(sorted_entries),
            len(fragments),
            len(prompt),
        )
        return prompt

    def clear(self) -> None:
        """Remove all registered section entries."""
        self._entries.clear()

    def __len__(self) -> int:
        """Number of registered section entries."""
        return len(self._entries)

    def __repr__(self) -> str:
        names = [e.section.name for e in self._entries]
        return f"PromptBuilder(neurons={names})"


class _SectionEntry:
    """Internal: pairs a section with its format kwargs."""

    __slots__ = ("kwargs", "section")

    def __init__(self, *, section: PromptSection, kwargs: dict[str, Any]) -> None:
        self.section = section
        self.kwargs = kwargs


# Deprecated alias
_NeuronEntry = _SectionEntry
