"""UserInputGuardrail: detects prompt injection and jailbreak attempts."""

from __future__ import annotations

import re
from typing import Any

from exo.guardrail._helpers import (  # pyright: ignore[reportMissingImports]
    _extract_latest_user_message,
)
from exo.guardrail.base import BaseGuardrail  # pyright: ignore[reportMissingImports]
from exo.guardrail.types import (  # pyright: ignore[reportMissingImports]
    GuardrailBackend,
    RiskAssessment,
    RiskLevel,
)
from exo.hooks import HookPoint

# Default patterns targeting common prompt-injection and jailbreak phrases.
# Each tuple is ``(compiled_regex, risk_level, description)``.
_DEFAULT_PATTERNS: list[tuple[str, RiskLevel, str]] = [
    # Direct instruction overrides
    (
        r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)",
        RiskLevel.HIGH,
        "instruction_override",
    ),
    (
        r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)",
        RiskLevel.HIGH,
        "instruction_override",
    ),
    (
        r"forget\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)",
        RiskLevel.HIGH,
        "instruction_override",
    ),
    # Role impersonation / DAN-style jailbreaks
    (r"you\s+are\s+now\s+(in\s+)?(\w+\s+)*mode", RiskLevel.HIGH, "role_impersonation"),
    (
        r"(act|behave|pose|roleplay)\s+(as|like)\s+(if\s+you\s+(are|were)\s+)?"
        r"(a\s+)?(\w+\s+)*(unrestricted|unfiltered|evil|dan\b)",
        RiskLevel.HIGH,
        "role_impersonation",
    ),
    (r"\bdan\s+mode\b", RiskLevel.HIGH, "role_impersonation"),
    (
        r"pretend\s+you\s+(are|have|do\s+not\s+have)\s+"
        r"(no\s+|without\s+|absolutely\s+no\s+)?(restrictions|rules|limitations|guidelines)",
        RiskLevel.HIGH,
        "role_impersonation",
    ),
    # System prompt extraction
    (
        r"(reveal|show|print|display|output|repeat)\s+(your\s+)?(system\s+prompt|initial\s+instructions|hidden\s+instructions)",
        RiskLevel.MEDIUM,
        "system_prompt_extraction",
    ),
    (
        r"what\s+(are|is)\s+your\s+(system\s+prompt|instructions|rules)",
        RiskLevel.MEDIUM,
        "system_prompt_extraction",
    ),
    # Delimiter / context escape
    (r"```\s*(system|admin|root)\b", RiskLevel.HIGH, "delimiter_attack"),
    (r"\[INST\]|\[/INST\]|<<SYS>>|<\|im_start\|>|<\|im_end\|>", RiskLevel.HIGH, "delimiter_attack"),
    # Encoded / obfuscated injection
    (r"base64[-:\s]+(decode|encode)", RiskLevel.MEDIUM, "encoded_injection"),
    (r"eval\s*\(|exec\s*\(", RiskLevel.MEDIUM, "code_injection"),
]


class PatternBackend(GuardrailBackend):
    """Regex-based detection backend for prompt injection.

    Scans the latest user message in ``data['messages']`` against a
    configurable list of regex patterns.  Each pattern is associated with
    a risk level and description.

    Args:
        patterns: Custom patterns as ``(regex_str, RiskLevel, description)``
            tuples.  If ``None``, the built-in default set is used.
        extra_patterns: Additional patterns appended to the defaults (or
            to *patterns* when provided).
    """

    def __init__(
        self,
        patterns: list[tuple[str, RiskLevel, str]] | None = None,
        extra_patterns: list[tuple[str, RiskLevel, str]] | None = None,
    ) -> None:
        raw = patterns if patterns is not None else list(_DEFAULT_PATTERNS)
        if extra_patterns:
            raw = [*raw, *extra_patterns]

        self._patterns: list[tuple[re.Pattern[str], RiskLevel, str]] = [
            (re.compile(pat, re.IGNORECASE), level, desc) for pat, level, desc in raw
        ]

    async def analyze(self, data: dict[str, Any]) -> RiskAssessment:
        """Analyze the latest user message for injection patterns.

        Expects ``data["messages"]`` to be a list of message dicts, each
        with at least ``role`` and ``content`` keys.  Inspects the last
        message whose ``role`` is ``"user"``.

        Args:
            data: Hook data containing a ``messages`` list.

        Returns:
            A ``RiskAssessment`` — ``has_risk=True`` with the matched
            pattern's risk level on a hit, ``has_risk=False`` otherwise.
        """
        text = _extract_latest_user_message(data)
        if not text:
            return RiskAssessment(has_risk=False, risk_level=RiskLevel.SAFE)

        # Check all patterns, keep the highest-severity match.
        worst_level = RiskLevel.SAFE
        matched_patterns: list[str] = []

        for regex, level, desc in self._patterns:
            if regex.search(text):
                matched_patterns.append(desc)
                if _risk_rank(level) > _risk_rank(worst_level):
                    worst_level = level

        if worst_level == RiskLevel.SAFE:
            return RiskAssessment(has_risk=False, risk_level=RiskLevel.SAFE)

        n = len(matched_patterns)
        return RiskAssessment(
            has_risk=True,
            risk_level=worst_level,
            risk_type="prompt_injection",
            confidence=min(1.0, 0.7 + (n - 1) * 0.15),
            details={"matched_patterns": matched_patterns},
        )


class UserInputGuardrail(BaseGuardrail):
    """Guardrail that detects prompt injection and jailbreak attempts.

    Attaches to ``PRE_LLM_CALL`` by default and uses a
    :class:`PatternBackend` for regex-based detection.

    Args:
        patterns: Custom injection patterns.  See :class:`PatternBackend`.
        extra_patterns: Extra patterns added on top of defaults.
        backend: Explicit backend override.  When provided, *patterns*
            and *extra_patterns* are ignored.
        events: Hook points to monitor.  Each entry may be a
            :class:`~exo.hooks.HookPoint` enum member (recommended) or a
            plain string.  Defaults to ``[HookPoint.PRE_LLM_CALL]``.

    Example::

        from exo.hooks import HookPoint

        guard = UserInputGuardrail()
        guard.attach(agent)
        # Agent will now block prompt-injection attempts automatically.

        # Monitor additional hook points:
        guard = UserInputGuardrail(events=[HookPoint.PRE_LLM_CALL, HookPoint.PRE_TOOL_CALL])
    """

    def __init__(
        self,
        *,
        patterns: list[tuple[str, RiskLevel, str]] | None = None,
        extra_patterns: list[tuple[str, RiskLevel, str]] | None = None,
        backend: GuardrailBackend | None = None,
        events: list[HookPoint | str] | None = None,
    ) -> None:
        if backend is None:
            backend = PatternBackend(patterns=patterns, extra_patterns=extra_patterns)
        super().__init__(backend=backend, events=events or [HookPoint.PRE_LLM_CALL])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# _extract_latest_user_message is imported from _helpers and re-exported here
# so that existing test imports (from exo.guardrail.user_input import
# _extract_latest_user_message) continue to work.

_RISK_ORDER: dict[RiskLevel, int] = {
    RiskLevel.SAFE: 0,
    RiskLevel.LOW: 1,
    RiskLevel.MEDIUM: 2,
    RiskLevel.HIGH: 3,
    RiskLevel.CRITICAL: 4,
}


def _risk_rank(level: RiskLevel) -> int:
    return _RISK_ORDER.get(level, 0)
