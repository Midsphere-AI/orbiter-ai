"""Private helpers shared across guardrail backends."""

from __future__ import annotations

import unicodedata
from typing import Any


def _extract_latest_user_message(data: dict[str, Any]) -> str:
    """Return the text content of the last user message, or ``""``.

    Handles both plain dicts (``{"role": "user", "content": "..."}``),
    and Pydantic message objects with ``.role`` / ``.content`` attributes
    (e.g. ``UserMessage``).  List-format content (``list[ContentBlock]``)
    is supported for both dict parts and Pydantic ``TextBlock`` objects.

    The returned string is NFKC-normalised so that homoglyph substitutions
    (e.g. full-width ASCII, Unicode lookalike letters) do not bypass
    pattern-matching guardrails.
    """
    messages = data.get("messages")
    if not isinstance(messages, list):
        return ""
    for msg in reversed(messages):
        # Support both dict-like and object-like messages.
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role != "user":
            continue
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        if isinstance(content, str):
            return unicodedata.normalize("NFKC", content)
        # Handle list-format content:
        #   • dict parts:  {"type": "text", "text": "..."}
        #   • Pydantic TextBlock objects (and any object with a .text attribute)
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif hasattr(part, "text") and isinstance(part.text, str):
                    parts.append(part.text)
            return unicodedata.normalize("NFKC", " ".join(parts))
        return ""
    return ""
