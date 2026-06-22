"""Automatically persist LLM responses and tool results to a memory store.

Registers POST_LLM_CALL and POST_TOOL_CALL hooks on an agent so that
conversation turns are saved without manual intervention.  The caller
saves a ``HumanMemory`` before calling ``run()`` / ``run.stream()``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from exo.hooks import HookPoint  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)
from exo.memory.base import (  # pyright: ignore[reportMissingImports]
    AIMemory,
    MemoryMetadata,
    MemoryStore,
    ToolMemory,
)

if TYPE_CHECKING:
    from exo.types import Message  # pyright: ignore[reportMissingImports]


class MemoryPersistence:
    """Hook-based auto-persistence for conversation memory.

    Attaches ``POST_LLM_CALL`` and ``POST_TOOL_CALL`` hooks to an agent
    that automatically save ``AIMemory`` and ``ToolMemory`` items to the
    provided store.

    Args:
        store: A :class:`MemoryStore` implementation to persist items to.
        metadata: Optional scoping metadata applied to every saved item.
    """

    def __init__(
        self,
        store: MemoryStore,
        metadata: MemoryMetadata | None = None,
        *,
        history_fetch_multiplier: int = 10,
    ) -> None:
        self.store = store
        self.metadata = metadata or MemoryMetadata()
        self._attached_agent_ids: set[int] = set()
        # Per-round fetch headroom: each conversation round can generate multiple
        # memory items (human + ai + tool messages), so fetch this many items per
        # requested round to avoid truncation.
        self._history_fetch_multiplier = history_fetch_multiplier

    async def _save_llm_response(self, *, agent: Any, response: Any, **_: Any) -> None:
        """POST_LLM_CALL hook — save an AIMemory item."""
        tool_calls = [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
            for tc in getattr(response, "tool_calls", []) or []
        ]
        content = getattr(response, "content", None)
        item = AIMemory(
            content=content if content is not None else "",
            metadata=self.metadata,
            tool_calls=tool_calls,
        )
        await self.store.add(item)
        logger.debug("persisted AIMemory id=%s tool_calls=%d", item.id, len(tool_calls))

    async def _save_tool_result(self, *, agent: Any, tool_name: str, result: Any, **_: Any) -> None:
        """POST_TOOL_CALL hook — save a ToolMemory item."""
        # Use explicit None checks so that empty-string content is not replaced
        # by the error field.
        content = result.content if result.content is not None else (result.error or "")
        item = ToolMemory(
            content=content,
            metadata=self.metadata,
            tool_call_id=result.tool_call_id,
            tool_name=result.tool_name,
            is_error=result.error is not None,
        )
        await self.store.add(item)
        logger.debug(
            "persisted ToolMemory id=%s tool=%s is_error=%s", item.id, tool_name, item.is_error
        )

    def attach(self, agent: Any) -> None:
        """Register persistence hooks on the given agent.

        Idempotent — a second call for the same agent object is a no-op.
        """
        agent_id = id(agent)
        if agent_id in self._attached_agent_ids:
            logger.debug("already attached to agent=%s, skipping", getattr(agent, "name", agent))
            return
        self._attached_agent_ids.add(agent_id)
        agent.hook_manager.add(HookPoint.POST_LLM_CALL, self._save_llm_response)
        agent.hook_manager.add(HookPoint.POST_TOOL_CALL, self._save_tool_result)
        logger.debug("attached memory persistence hooks to agent=%s", getattr(agent, "name", agent))

    def detach(self, agent: Any) -> None:
        """Remove persistence hooks from the given agent."""
        self._attached_agent_ids.discard(id(agent))
        agent.hook_manager.remove(HookPoint.POST_LLM_CALL, self._save_llm_response)
        agent.hook_manager.remove(HookPoint.POST_TOOL_CALL, self._save_tool_result)
        logger.debug(
            "detached memory persistence hooks from agent=%s", getattr(agent, "name", agent)
        )

    async def load_history(
        self,
        agent_name: str,
        conversation_id: str,
        rounds: int,
    ) -> list[Message]:
        """Load the last *rounds* message pairs for the given agent and conversation.

        Queries the short-term store scoped to *conversation_id* (mapped to
        ``metadata.task_id``) and *agent_name* (mapped to ``metadata.agent_id``),
        then returns chronologically-ordered messages converted to
        ``exo.types.Message`` objects ready for injection into an agent run.

        Args:
            agent_name: Name of the agent (stored as ``metadata.agent_id``).
            conversation_id: Conversation scope (stored as ``metadata.task_id``).
            rounds: Maximum number of human+AI message pairs to return.
        """
        from exo.types import (  # pyright: ignore[reportMissingImports]
            AssistantMessage,
            SystemMessage,
            ToolCall,
            ToolResult,
            UserMessage,
        )

        meta = MemoryMetadata(agent_id=agent_name, task_id=conversation_id)
        fetch_limit = max(rounds * self._history_fetch_multiplier, 100)
        items = await self.store.search(metadata=meta, limit=fetch_limit)

        # Normalise to chronological order (SQLite returns newest-first)
        items = sorted(items, key=lambda x: x.created_at)

        # Window: keep only the last `rounds` conversation rounds
        if rounds > 0:
            human_positions = [i for i, item in enumerate(items) if item.memory_type == "human"]
            if len(human_positions) > rounds:
                cut_index = human_positions[-rounds]
                # Always keep system messages before the cut
                system_msgs = [m for m in items[:cut_index] if m.memory_type == "system"]
                items = system_msgs + items[cut_index:]

        # Convert MemoryItem → exo.types.Message
        messages: list[Message] = []
        for item in items:
            if item.memory_type == "human":
                messages.append(UserMessage(content=item.content))
            elif item.memory_type == "system":
                messages.append(SystemMessage(content=item.content))
            elif item.memory_type == "ai":
                tcs = [ToolCall(**tc) for tc in getattr(item, "tool_calls", [])]
                messages.append(AssistantMessage(content=item.content, tool_calls=tcs))
            elif item.memory_type == "tool":
                is_error = bool(getattr(item, "is_error", False))
                messages.append(
                    ToolResult(
                        tool_call_id=getattr(item, "tool_call_id", ""),
                        tool_name=getattr(item, "tool_name", ""),
                        content="" if is_error else item.content,
                        error=item.content if is_error else None,
                    )
                )

        logger.debug(
            "load_history agent=%s conversation=%s rounds=%d -> %d messages",
            agent_name,
            conversation_id,
            rounds,
            len(messages),
        )
        return messages

    # -- Context snapshot persistence ------------------------------------------

    async def save_snapshot(
        self,
        agent_name: str,
        conversation_id: str,
        msg_list: list[Any],
        context_config: Any,
    ) -> None:
        """Persist the processed msg_list as a context snapshot.

        Saves a :class:`~exo.memory.snapshot.SnapshotMemory` with a
        deterministic ID so subsequent calls upsert (replace) the
        previous snapshot for this agent+conversation.

        Args:
            agent_name: Agent name for scoping.
            conversation_id: Conversation scope.
            msg_list: The processed message list to snapshot.
            context_config: Active context config (for config hash).
        """
        from exo.memory.snapshot import (  # pyright: ignore[reportMissingImports]
            make_snapshot,
        )

        # Determine the latest raw item for freshness anchoring.
        meta = MemoryMetadata(agent_id=agent_name, task_id=conversation_id)
        raw_items = await self.store.search(metadata=meta, limit=500)
        # Exclude snapshot items from the raw list.
        raw_items = [it for it in raw_items if it.memory_type != "snapshot"]
        raw_items = sorted(raw_items, key=lambda x: x.created_at)

        latest_raw_id = raw_items[-1].id if raw_items else ""
        latest_raw_created_at = raw_items[-1].created_at if raw_items else ""
        raw_item_count = len(raw_items)

        snap = make_snapshot(
            agent_name=agent_name,
            conversation_id=conversation_id,
            msg_list=msg_list,
            context_config=context_config,
            raw_item_count=raw_item_count,
            latest_raw_id=latest_raw_id,
            latest_raw_created_at=latest_raw_created_at,
        )

        await self.store.add(snap)
        logger.debug(
            "saved snapshot id=%s agent=%s conversation=%s raw_items=%d",
            snap.id,
            agent_name,
            conversation_id,
            raw_item_count,
        )

    async def load_snapshot(
        self,
        agent_name: str,
        conversation_id: str,
    ) -> Any | None:
        """Load the context snapshot for an agent+conversation.

        Returns:
            A :class:`~exo.memory.snapshot.SnapshotMemory` if one exists,
            otherwise ``None``.
        """
        from exo.memory.snapshot import (  # pyright: ignore[reportMissingImports]
            SnapshotMemory,
            snapshot_id,
        )

        sid = snapshot_id(agent_name, conversation_id)
        item = await self.store.get(sid)
        if item is not None and isinstance(item, SnapshotMemory):
            logger.debug("loaded snapshot id=%s agent=%s", sid, agent_name)
            return item
        # Fallback: search by metadata + type in case the store doesn't
        # support direct get (e.g., some vector stores).
        meta = MemoryMetadata(agent_id=agent_name, task_id=conversation_id)
        results = await self.store.search(metadata=meta, memory_type="snapshot", limit=1)
        if results and isinstance(results[0], SnapshotMemory):
            logger.debug("loaded snapshot via search agent=%s", agent_name)
            return results[0]
        return None

    async def is_snapshot_fresh(
        self,
        snapshot: Any,
        agent_name: str,
        conversation_id: str,
        context_config: Any | None = None,
    ) -> bool:
        """Check whether a snapshot is still fresh.

        A snapshot is stale if:
        1. The raw store has newer items than the snapshot's anchor.
        2. The context config hash has changed (windowing params changed).

        Args:
            snapshot: A ``SnapshotMemory`` instance.
            agent_name: Agent name for scoping.
            conversation_id: Conversation scope.
            context_config: Current context config for hash comparison.
                If ``None``, the config hash check is skipped.

        Returns:
            ``True`` if the snapshot is fresh and can be used.
        """
        from exo.memory.snapshot import (  # pyright: ignore[reportMissingImports]
            compute_config_hash,
        )

        # 1. Config hash check
        if context_config is not None:
            current_hash = compute_config_hash(context_config)
            if snapshot.config_hash and snapshot.config_hash != current_hash:
                logger.debug(
                    "snapshot stale: config_hash mismatch (snap=%s current=%s)",
                    snapshot.config_hash,
                    current_hash,
                )
                return False

        # 2. Raw item freshness check
        meta = MemoryMetadata(agent_id=agent_name, task_id=conversation_id)
        raw_items = await self.store.search(metadata=meta, limit=500)
        raw_items = [it for it in raw_items if it.memory_type != "snapshot"]
        raw_items = sorted(raw_items, key=lambda x: x.created_at)

        if not raw_items:
            # No raw history — snapshot is fresh (conversation may have
            # been cleared or never had raw items).
            return True

        latest = raw_items[-1]
        if latest.id != snapshot.latest_raw_id:
            logger.debug(
                "snapshot stale: latest_raw_id mismatch (snap=%s actual=%s)",
                snapshot.latest_raw_id,
                latest.id,
            )
            return False

        return True
