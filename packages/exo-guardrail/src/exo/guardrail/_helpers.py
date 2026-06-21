"""Private helpers shared across guardrail backends."""

from __future__ import annotations

from typing import Any


def _extract_latest_user_message(data: dict[str, Any]) -> str:
    """Return the text content of the last user message, or ``""``.

    Handles both plain dicts (``{"role": "user", "content": "..."}``),
    and Pydantic message objects with ``.role`` / ``.content`` attributes
    (e.g. ``UserMessage``).
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
            return content
        # Handle list-format content (e.g. [{"type": "text", "text": "..."}])
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
            return " ".join(parts)
        return ""
    return ""
