"""Reflection framework with LLM-powered analysis, insight extraction, and suggestion generation."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from exo.eval.llm_scorer import extract_json  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ReflectionType(StrEnum):
    """Category of a reflection result."""

    SUCCESS = "success"
    FAILURE = "failure"
    OPTIMIZATION = "optimization"
    PATTERN = "pattern"
    INSIGHT = "insight"


class ReflectionLevel(StrEnum):
    """Depth of reflection analysis."""

    SHALLOW = "shallow"
    MEDIUM = "medium"
    DEEP = "deep"
    META = "meta"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReflectionResult:
    """Output of a single reflection pass."""

    reflection_type: ReflectionType
    level: ReflectionLevel
    summary: str
    key_findings: list[str] = field(default_factory=list)
    root_causes: list[str] = field(default_factory=list)
    insights: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReflectionHistory:
    """Tracks reflection results over time with statistics."""

    _entries: list[ReflectionResult] = field(default_factory=list)
    total_count: int = 0
    success_count: int = 0
    failure_count: int = 0

    def add(self, result: ReflectionResult) -> None:
        """Append a result and update counters."""
        self._entries.append(result)
        self.total_count += 1
        if result.reflection_type == ReflectionType.SUCCESS:
            self.success_count += 1
        elif result.reflection_type == ReflectionType.FAILURE:
            self.failure_count += 1

    def get_recent(self, n: int = 5) -> list[ReflectionResult]:
        """Return the *n* most recent reflections."""
        return list(self._entries[-n:])

    def get_by_type(self, rtype: ReflectionType) -> list[ReflectionResult]:
        """Return all reflections matching *rtype*."""
        return [r for r in self._entries if r.reflection_type == rtype]

    def summarize(self) -> dict[str, Any]:
        """Return aggregate statistics."""
        return {
            "total": self.total_count,
            "success": self.success_count,
            "failure": self.failure_count,
            "types": {t.value: len(self.get_by_type(t)) for t in ReflectionType},
        }


# ---------------------------------------------------------------------------
# Reflector ABC
# ---------------------------------------------------------------------------


class Reflector(ABC):
    """Abstract reflector with a three-step template: analyze → insight → suggest."""

    __slots__ = ("level", "name", "reflection_type")

    def __init__(
        self,
        name: str = "reflector",
        reflection_type: ReflectionType = ReflectionType.INSIGHT,
        level: ReflectionLevel = ReflectionLevel.MEDIUM,
    ) -> None:
        self.name = name
        self.reflection_type = reflection_type
        self.level = level

    async def reflect(self, context: dict[str, Any]) -> ReflectionResult:
        """Run the full three-step reflection pipeline."""
        analysis = await self.analyze(context)
        derived = await self.insight(analysis)
        actions = await self.suggest(derived)
        return ReflectionResult(
            reflection_type=self.reflection_type,
            level=self.level,
            summary=analysis.get("summary", ""),
            key_findings=analysis.get("key_findings", []),
            root_causes=analysis.get("root_causes", []),
            insights=derived.get("insights", []),
            suggestions=actions.get("suggestions", []),
            metadata={"reflector": self.name},
        )

    @abstractmethod
    async def analyze(self, context: dict[str, Any]) -> dict[str, Any]:
        """Step 1: Extract facts and key findings from the execution context."""

    async def insight(self, analysis: dict[str, Any]) -> dict[str, Any]:
        """Step 2: Derive insights from the analysis (override for custom logic).

        The default implementation passes through both ``"insights"`` and
        ``"suggestions"`` from the analysis dict so that :meth:`suggest` —
        which receives the ``derived`` dict returned here — can read
        ``"suggestions"`` directly.
        """
        return {
            "insights": analysis.get("insights", []),
            "suggestions": analysis.get("suggestions", []),
        }

    async def suggest(self, insights: dict[str, Any]) -> dict[str, Any]:
        """Step 3: Generate actionable suggestions from insights (override for custom logic).

        *insights* is the dict returned by :meth:`insight` (the ``derived``
        dict), not the raw analysis.
        """
        return {"suggestions": insights.get("suggestions", [])}


# ---------------------------------------------------------------------------
# GeneralReflector — LLM-powered
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an expert AI execution analyst. Given the execution context, "
    "provide a structured reflection with five sections:\n"
    "1. Summary — one-sentence overview\n"
    "2. Key Findings — list of notable observations\n"
    "3. Root Causes — underlying reasons for outcomes\n"
    "4. Insights — patterns and learnings\n"
    "5. Suggestions — concrete, actionable improvements\n\n"
    "Return a JSON object:\n"
    '{"summary": "<str>", "key_findings": ["<str>", ...], '
    '"root_causes": ["<str>", ...], "insights": ["<str>", ...], '
    '"suggestions": ["<str>", ...]}'
)


class GeneralReflector(Reflector):
    """LLM-powered reflector using a judge callable ``(prompt: str) -> str``."""

    __slots__ = ("_judge", "_system_prompt", "_timeout")

    def __init__(
        self,
        judge: Any = None,
        *,
        system_prompt: str | None = None,
        name: str = "general_reflector",
        reflection_type: ReflectionType = ReflectionType.INSIGHT,
        level: ReflectionLevel = ReflectionLevel.DEEP,
        timeout: float = 0.0,
    ) -> None:
        super().__init__(name=name, reflection_type=reflection_type, level=level)
        self._judge = judge
        self._system_prompt = system_prompt or _SYSTEM_PROMPT
        self._timeout = timeout

    async def analyze(self, context: dict[str, Any]) -> dict[str, Any]:
        """Call the LLM judge with the execution context and parse the response."""
        if self._judge is None:
            return {"summary": "No judge callable provided", "error": True}

        prompt = self._build_prompt(context)
        if self._timeout > 0:
            response = await asyncio.wait_for(self._judge(prompt), timeout=self._timeout)
        else:
            response = await self._judge(prompt)
        return self._parse_response(str(response))

    async def insight(self, analysis: dict[str, Any]) -> dict[str, Any]:
        """Pass through insights and suggestions from the LLM analysis.

        Both keys are forwarded so that :meth:`suggest` can read
        ``"suggestions"`` from the ``derived`` dict it receives.
        """
        return {
            "insights": analysis.get("insights", []),
            "suggestions": analysis.get("suggestions", []),
        }

    async def suggest(self, insights: dict[str, Any]) -> dict[str, Any]:
        """Pass through suggestions from the LLM analysis (already in analyze)."""
        return {"suggestions": insights.get("suggestions", [])}

    def _build_prompt(self, context: dict[str, Any]) -> str:
        """Build the prompt from system prompt + context."""
        parts = [self._system_prompt, ""]
        if "input" in context:
            parts.append(f"[Input]\n{context['input']}")
        if "output" in context:
            parts.append(f"[Output]\n{context['output']}")
        if "error" in context:
            parts.append(f"[Error]\n{context['error']}")
        if "iteration" in context:
            parts.append(f"[Iteration] {context['iteration']}")
        return "\n".join(parts)

    @staticmethod
    def _parse_response(text: str) -> dict[str, Any]:
        """Extract JSON from the LLM response with fallback.

        Delegates to :func:`exo.eval.llm_scorer.extract_json` — the canonical
        brace-depth JSON extractor shared across this package.
        """
        data = extract_json(text)
        if data:
            return data
        return {"summary": text[:200] if text else "", "parse_error": True}
