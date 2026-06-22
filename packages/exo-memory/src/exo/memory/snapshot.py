"""Context snapshot persistence — serialize/deserialize processed msg_list.

Persists the windowed, summarized, hook-mutated message list at end of run
so the next run can load it directly instead of rebuilding from raw history.
Raw history remains intact (append-only) for restoration.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from typing import Any

from exo.memory.base import (  # pyright: ignore[reportMissingImports]
    MemoryCategory,
    MemoryItem,
    MemoryMetadata,
)

logger = logging.getLogger(__name__)

# Snapshot content format version — bump when serialization changes.
SNAPSHOT_FORMAT_VERSION = 1

# Prefix used for deterministic snapshot IDs.
_SNAPSHOT_ID_PREFIX = "snapshot_"

# Marker that identifies a summary SystemMessage (vs instruction SystemMessage).
_SUMMARY_MARKER = "[Conversation Summary]"


class SnapshotMemory(MemoryItem):
    """Persisted context snapshot from end of an agent run.

    The ``content`` field holds JSON-serialized processed messages
    (excluding instruction SystemMessages).  All other fields provide
    metadata for freshness checking and invalidation.

    Attributes:
        snapshot_version: Serialization format version.
        raw_item_count: Number of raw (non-snapshot) items at snapshot time.
        latest_raw_id: ID of the newest raw MemoryItem when snapshot was saved.
        latest_raw_created_at: ``created_at`` of that item — freshness anchor.
        config_hash: Hash of windowing config fields for invalidation on config change.
    """

    memory_type: str = "snapshot"
    snapshot_version: int = SNAPSHOT_FORMAT_VERSION
    raw_item_count: int = 0
    latest_raw_id: str = ""
    latest_raw_created_at: str = ""
    config_hash: str = ""


def snapshot_id(agent_name: str, conversation_id: str) -> str:
    """Build the deterministic ID for a snapshot.

    Only one active snapshot per (agent, conversation) pair.
    The separator ``::`` is chosen because it cannot appear in valid
    agent names or UUID-based conversation IDs, preventing collisions
    between ``("a_b", "c")`` and ``("a", "b_c")``.
    """
    return f"{_SNAPSHOT_ID_PREFIX}{agent_name}::{conversation_id}"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def serialize_msg_list(msg_list: list[Any]) -> str:
    """Serialize a processed msg_list to JSON.

    Instruction SystemMessages (those not starting with the conversation
    summary marker) are **excluded** — they are regenerated fresh by
    ``build_messages()`` on each run to support dynamic callable instructions.

    ``[Conversation Summary]`` SystemMessages **are** included — they
    represent summarized history and must persist.

    Args:
        msg_list: The processed message list from the end of a run.

    Returns:
        JSON string of serialized messages.
    """
    from exo.types import (  # pyright: ignore[reportMissingImports]
        AssistantMessage,
        SystemMessage,
        ToolResult,
        UserMessage,
    )

    entries: list[dict[str, Any]] = []
    for msg in msg_list:
        if isinstance(msg, SystemMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if content.startswith(_SUMMARY_MARKER):
                entries.append({"type": "summary_system", "content": content})
            # Instruction SystemMessages are intentionally skipped.
            continue
        if isinstance(msg, UserMessage):
            # UserMessage.content can be str or list[ContentBlock].
            if isinstance(msg.content, str):
                entries.append({"type": "user", "content": msg.content})
            else:
                entries.append(
                    {
                        "type": "user",
                        "content": [_content_block_to_dict(cb) for cb in msg.content],
                    }
                )
            continue
        if isinstance(msg, AssistantMessage):
            entry: dict[str, Any] = {"type": "assistant"}
            if isinstance(msg.content, str):
                entry["content"] = msg.content
            else:
                entry["content"] = [_content_block_to_dict(cb) for cb in msg.content]
            if msg.tool_calls:
                entry["tool_calls"] = [_tool_call_to_dict(tc) for tc in msg.tool_calls]
            entries.append(entry)
            continue
        if isinstance(msg, ToolResult):
            entry = {
                "type": "tool_result",
                "tool_call_id": msg.tool_call_id,
                "tool_name": msg.tool_name,
            }
            if msg.content is not None:
                if isinstance(msg.content, str):
                    entry["content"] = msg.content
                else:
                    entry["content"] = [_content_block_to_dict(cb) for cb in msg.content]
            if msg.error is not None:
                entry["error"] = msg.error
            entries.append(entry)
            continue
        # Unknown message type — skip with warning.
        logger.warning("snapshot: skipping unknown message type %s", type(msg).__name__)

    return json.dumps(entries, separators=(",", ":"))


def deserialize_msg_list(data: str) -> list[Any]:
    """Reconstruct Message objects from snapshot JSON.

    Args:
        data: JSON string produced by :func:`serialize_msg_list`.

    Returns:
        List of ``exo.types.Message`` objects.

    Raises:
        ValueError: If the JSON is malformed or contains unknown types.
    """
    from exo.types import (  # pyright: ignore[reportMissingImports]
        AssistantMessage,
        SystemMessage,
        ToolCall,
        ToolResult,
        UserMessage,
    )

    entries: list[dict[str, Any]] = json.loads(data)
    messages: list[Any] = []

    for entry in entries:
        msg_type = entry["type"]
        if msg_type == "summary_system":
            messages.append(SystemMessage(content=entry["content"]))
        elif msg_type == "user":
            content = _restore_content(entry["content"])
            messages.append(UserMessage(content=content))
        elif msg_type == "assistant":
            content = _restore_content(entry.get("content", ""))
            tool_calls = [
                ToolCall(
                    id=tc["id"],
                    name=tc["name"],
                    arguments=tc.get("arguments", ""),
                    thought_signature=_b64_decode(tc.get("thought_signature")),
                )
                for tc in entry.get("tool_calls", [])
            ]
            messages.append(AssistantMessage(content=content, tool_calls=tool_calls))
        elif msg_type == "tool_result":
            content = _restore_content(entry.get("content", ""))
            messages.append(
                ToolResult(
                    tool_call_id=entry.get("tool_call_id", ""),
                    tool_name=entry.get("tool_name", ""),
                    content=content if content else "",
                    error=entry.get("error"),
                )
            )
        else:
            logger.warning("snapshot: skipping unknown entry type %r", msg_type)

    return messages


# ---------------------------------------------------------------------------
# Config hashing
# ---------------------------------------------------------------------------


def compute_config_hash(context_config: Any) -> str:
    """Compute a deterministic hash of windowing-relevant config fields.

    If the config changes between runs the snapshot is invalidated.

    Args:
        context_config: A ``ContextConfig`` or similar object with
            ``history_rounds``, ``summary_threshold``, ``offload_threshold``.

    Returns:
        Hex digest string.
    """
    cfg = getattr(context_config, "config", context_config)
    parts = (
        str(getattr(cfg, "history_rounds", 20)),
        str(getattr(cfg, "summary_threshold", 10)),
        str(getattr(cfg, "offload_threshold", 50)),
        str(getattr(cfg, "token_budget_trigger", 0.8)),
    )
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def has_message_content(msg_list: list[Any], marker: str) -> bool:
    """Check if any message in *msg_list* contains *marker* in its content.

    Useful for idempotent ``PRE_LLM_CALL`` hooks that need to avoid
    duplicate injection when loading from a snapshot.

    Args:
        msg_list: List of ``exo.types.Message`` objects.
        marker: String to search for.

    Returns:
        True if any message's content contains *marker*.
    """
    for msg in msg_list:
        content = getattr(msg, "content", None)
        if isinstance(content, str) and marker in content:
            return True
        if isinstance(content, list):
            for block in content:
                text = getattr(block, "text", None) or getattr(block, "content", None)
                if isinstance(text, str) and marker in text:
                    return True
    return False


def make_snapshot_metadata(
    agent_name: str,
    conversation_id: str,
) -> MemoryMetadata:
    """Build scoped metadata for a snapshot item."""
    return MemoryMetadata(agent_id=agent_name, task_id=conversation_id)


def make_snapshot(
    agent_name: str,
    conversation_id: str,
    msg_list: list[Any],
    context_config: Any,
    raw_item_count: int = 0,
    latest_raw_id: str = "",
    latest_raw_created_at: str = "",
) -> SnapshotMemory:
    """Build a ``SnapshotMemory`` ready for persistence.

    Args:
        agent_name: Agent name for scoping.
        conversation_id: Conversation scope.
        msg_list: The processed message list to snapshot.
        context_config: Active context config (for config hash).
        raw_item_count: Count of raw items at snapshot time.
        latest_raw_id: ID of newest raw MemoryItem.
        latest_raw_created_at: Its ``created_at`` timestamp.

    Returns:
        A ``SnapshotMemory`` with deterministic ID.
    """
    return SnapshotMemory(
        id=snapshot_id(agent_name, conversation_id),
        content=serialize_msg_list(msg_list),
        category=MemoryCategory.SUMMARY,
        metadata=make_snapshot_metadata(agent_name, conversation_id),
        raw_item_count=raw_item_count,
        latest_raw_id=latest_raw_id,
        latest_raw_created_at=latest_raw_created_at,
        config_hash=compute_config_hash(context_config),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tool_call_to_dict(tc: Any) -> dict[str, Any]:
    """Serialize a ToolCall to a JSON-safe dict."""
    d: dict[str, Any] = {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
    if tc.thought_signature is not None:
        d["thought_signature"] = base64.b64encode(tc.thought_signature).decode("ascii")
    return d


def _b64_decode(val: str | None) -> bytes | None:
    """Decode a base64 string, or return None."""
    if val is None:
        return None
    return base64.b64decode(val)


def _content_block_to_dict(block: Any) -> dict[str, Any]:
    """Serialize a ContentBlock to a JSON-safe dict."""
    if hasattr(block, "model_dump"):
        return block.model_dump()
    return {"type": "text", "text": str(block)}


def _restore_content(raw: Any) -> Any:
    """Restore content from snapshot — string or list of content blocks."""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        # Return as-is for now — ContentBlock reconstruction happens at
        # the provider level; raw dicts are accepted by the message builder.
        return raw
    return str(raw) if raw else ""
