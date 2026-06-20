"""Context state inspector API endpoints.

Provides endpoints for inspecting hierarchical context state during agent
execution.  Context state mirrors Exo's ContextState parent-child
hierarchy and tracks fork/merge events with token deltas.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends

from exo_web.database import get_db
from exo_web.routes.auth import get_current_user

router = APIRouter(prefix="/api/v1/context-state", tags=["context-state"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_tree(task_id: str = "root") -> dict[str, Any]:
    """Return a minimal empty context state tree."""
    return {
        "task_id": task_id,
        "parent_task_id": None,
        "local_entries": {},
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "token_delta": None,
        "fork_event": None,
        "merge_event": None,
        "children": [],
    }


def _parse_usage(usage_json: str | None) -> dict[str, int]:
    """Parse a usage_json string into a token-usage dict."""
    if not usage_json:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    try:
        raw = json.loads(usage_json)
        if isinstance(raw, dict):
            prompt = int(raw.get("prompt_tokens", 0) or 0)
            completion = int(raw.get("completion_tokens", 0) or 0)
            total = int(raw.get("total_tokens", 0) or (prompt + completion))
            return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/conversation/{conversation_id}")
async def get_context_state(
    conversation_id: str,
    step: int | None = None,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return the context state tree for a conversation.

    The ``step`` query param selects a specific step snapshot — when supplied,
    only messages up to and including step N (0-indexed) are included in the
    token aggregation.  When omitted the latest accumulated state is returned.

    The tree is built from the persisted messages table:
    - ``local_entries``: role → content mapping for each message
    - ``token_usage``: accumulated prompt/completion/total tokens drawn from
      each message's ``usage_json`` field (assistant messages carry usage)
    - ``children``: one child node per assistant turn, each carrying that
      turn's incremental token delta
    """
    async with get_db() as db:
        # Verify the conversation belongs to the requesting user.
        cur = await db.execute(
            "SELECT id FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user["id"]),
        )
        if await cur.fetchone() is None:
            # Return an empty tree rather than 404 so the frontend inspector
            # can render a "no data yet" state without special-casing.
            return {
                "conversation_id": conversation_id,
                "step": step,
                "tree": _empty_tree(conversation_id),
            }

        # Fetch all messages in chronological order.
        cur = await db.execute(
            "SELECT id, role, content, usage_json, created_at "
            "FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
            (conversation_id,),
        )
        rows = await cur.fetchall()

    messages = [dict(r) for r in rows]

    # Apply step-slice: keep only the first (step+1) messages when step is given.
    if step is not None and step >= 0:
        messages = messages[: step + 1]

    # Build root-level local_entries from the full message list.
    local_entries: dict[str, Any] = {}
    for idx, msg in enumerate(messages):
        key = f"{msg['role']}_{idx}"
        local_entries[key] = msg["content"]

    # Accumulate total token usage across all messages.
    total_prompt = 0
    total_completion = 0

    # Build child nodes — one per assistant turn with its per-turn delta.
    children: list[dict[str, Any]] = []
    assistant_idx = 0
    for _idx, msg in enumerate(messages):
        usage = _parse_usage(msg.get("usage_json"))
        total_prompt += usage["prompt_tokens"]
        total_completion += usage["completion_tokens"]

        if msg["role"] == "assistant":
            children.append(
                {
                    "task_id": f"turn_{assistant_idx}",
                    "parent_task_id": conversation_id,
                    "local_entries": {f"assistant_{assistant_idx}": msg["content"]},
                    "token_usage": usage,
                    "token_delta": usage["total_tokens"],
                    "fork_event": None,
                    "merge_event": None,
                    "children": [],
                }
            )
            assistant_idx += 1

    total_tokens = total_prompt + total_completion
    root_tree: dict[str, Any] = {
        "task_id": conversation_id,
        "parent_task_id": None,
        "local_entries": local_entries,
        "token_usage": {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_tokens,
        },
        "token_delta": None,
        "fork_event": None,
        "merge_event": None,
        "children": children,
    }

    return {
        "conversation_id": conversation_id,
        "step": step,
        "tree": root_tree,
    }
