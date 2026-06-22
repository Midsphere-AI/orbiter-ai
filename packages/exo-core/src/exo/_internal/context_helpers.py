"""Shared context-windowing and token-tracking helpers.

These helpers are used by both ``exo.agent`` (the agent's own ``_run``
loop) and ``exo.runner`` (the public streaming path).  Keeping them in
``_internal`` removes the private cross-module coupling where ``runner``
previously imported ``_apply_context_windowing`` and friends directly
from ``exo.agent``.
"""

from __future__ import annotations

from typing import Any

from exo.hooks import HookPoint
from exo.observability.logging import get_logger  # pyright: ignore[reportMissingImports]
from exo.types import AssistantMessage, Message, SystemMessage, UserMessage

_log = get_logger(__name__)


class _ProviderSummarizer:
    """Wraps a model provider for use with exo-memory's generate_summary()."""

    def __init__(self, provider: Any) -> None:
        self._provider = provider

    async def summarize(self, prompt: str) -> str:
        """Call provider.complete() to generate a summary string."""
        try:
            response = await self._provider.complete(
                [UserMessage(content=prompt)],
                tools=None,
                temperature=0.3,
                max_tokens=512,
            )
            return str(response.content or "")
        except Exception as exc:
            _log.warning("Context summarization provider call failed: %s", exc)
            return ""


class _ContextAction:
    """Metadata about a context windowing action that was applied."""

    __slots__ = ("action", "after_count", "before_count", "details")

    def __init__(
        self,
        action: str,
        before_count: int,
        after_count: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.action = action
        self.before_count = before_count
        self.after_count = after_count
        self.details = details or {}


def _get_context_window_tokens(model_name: str) -> int | None:
    """Look up context window token count from the model registry.

    Returns ``None`` if exo-models is not installed or the model is unknown.
    """
    try:
        from exo.models.context_windows import (
            MODEL_CONTEXT_WINDOWS,  # pyright: ignore[reportMissingImports]
        )

        return MODEL_CONTEXT_WINDOWS.get(model_name)
    except ImportError:
        return None


def _update_system_token_info(
    msg_list: list[Message],
    used: int,
    total: int,
) -> list[Message]:
    """Insert/replace ``[Context: {used}/{total} tokens ({pct}% full)]`` in the system message.

    If a :class:`SystemMessage` is present it is updated in-place (replacing
    any prior context tag). If no system message is present a new one is
    inserted at position 0 with just the tag.

    Parameters
    ----------
    msg_list:
        Current message list (not mutated — a new list is returned).
    used:
        Number of tokens currently used (last LLM call's input_tokens).
    total:
        Context window capacity in tokens.

    Returns
    -------
    Updated message list with the context tag injected into the system message.
    """
    pct = round(100.0 * used / total) if total > 0 else 0
    tag = f"[Context: {used}/{total} tokens ({pct}% full)]"
    result: list[Message] = list(msg_list)
    for i, msg in enumerate(result):
        if isinstance(msg, SystemMessage):
            # Strip any prior [Context: ...] tag line then append new one
            lines = msg.content.splitlines()
            lines = [ln for ln in lines if not ln.startswith("[Context:")]
            base = "\n".join(lines).rstrip()
            content = f"{base}\n{tag}" if base else tag
            result[i] = SystemMessage(content=content)
            return result
    # No system message — insert one with just the tag
    result.insert(0, SystemMessage(content=tag))
    return result


async def _inject_long_term_knowledge(
    agent_memory: Any,
    user_input: str,
    msg_list: list[Message],
    limit: int = 5,
) -> list[Message]:
    """Search long-term memory and inject relevant results into the system message.

    When long_term is a VectorMemoryStore or ChromaVectorMemoryStore, uses
    vector/semantic search. When it's SQLiteMemoryStore or LongTermMemory,
    uses keyword search. Results are injected in KnowledgeNeuron <knowledge> format.
    """
    long_term = getattr(agent_memory, "long_term", None)
    if long_term is None:
        return msg_list

    try:
        items = await long_term.search(query=user_input, limit=limit)
    except Exception as exc:
        _log.debug("long-term search failed: %s", exc)
        return msg_list

    if not items:
        return msg_list

    # Format as KnowledgeNeuron <knowledge> block
    lines = ["<knowledge>"]
    for item in items:
        lines.append(f"  [long_term_memory]: {item.content}")
    lines.append("</knowledge>")
    knowledge_block = "\n".join(lines)

    # Inject into system message (append) or insert new SystemMessage at front
    new_msg_list = list(msg_list)
    sys_idx = next((i for i, m in enumerate(new_msg_list) if isinstance(m, SystemMessage)), None)
    if sys_idx is not None:
        existing = new_msg_list[sys_idx]
        existing_content = existing.content if isinstance(existing.content, str) else ""
        new_content = (
            f"{existing_content}\n\n{knowledge_block}" if existing_content else knowledge_block
        )
        new_msg_list[sys_idx] = SystemMessage(content=new_content)
    else:
        new_msg_list.insert(0, SystemMessage(content=knowledge_block))

    _log.debug("injected %d long-term memory items into system message", len(items))
    return new_msg_list


async def _apply_context_windowing(
    msg_list: list[Message],
    context: Any,
    provider: Any,
    *,
    force_summarize: bool = False,
    hook_manager: Any | None = None,
    agent: Any | None = None,
    step: int = -1,
    max_steps: int = 0,
    agent_name: str = "",
    model_name: str = "",
    context_window_tokens: int | None = None,
    last_usage: Any | None = None,
    token_tracker: Any | None = None,
) -> tuple[list[Message], list[_ContextAction]]:
    """Apply context windowing and optional summarization to *msg_list*.

    Behaviour depends on the ``overflow`` strategy:

    - **none**: no windowing at all — messages grow unbounded.
    - **truncate**: drop oldest non-system messages when count > limit.
    - **summarize** (default): three-stage cascade —
      1. Emergency offload when far over limit.
      2. LLM summarization when over threshold.
      3. Hard window to ``history_rounds``.

    When *force_summarize* is ``True`` (token pressure exceeded), summarization
    fires regardless of message count.

    Returns:
        ``(processed_msg_list, actions)`` — callers use *actions* to emit
        streaming ``ContextEvent`` instances.
    """
    # Resolve config attrs: supports both Context (has .config) and ContextConfig directly
    _cfg = getattr(context, "config", context)
    overflow_strategy: str = getattr(_cfg, "overflow", "summarize")
    history_rounds: int = getattr(_cfg, "_history_rounds", getattr(_cfg, "history_rounds", 20))
    summary_threshold: int = getattr(
        _cfg, "_summary_threshold", getattr(_cfg, "summary_threshold", 10)
    )
    offload_threshold: int = getattr(
        _cfg, "_offload_threshold", getattr(_cfg, "offload_threshold", 50)
    )
    keep_recent_cfg: int = getattr(_cfg, "keep_recent", 5)

    actions: list[_ContextAction] = []

    # ── overflow="hook" — delegate entirely to hooks ────────────────────
    if overflow_strategy == "hook":
        if hook_manager is not None:
            try:
                from exo.context.info import (  # pyright: ignore[reportMissingImports]
                    build_context_window_info,
                )
            except ImportError:
                _log.debug("exo-context not installed, skipping CONTEXT_WINDOW hook")
            else:
                _info = build_context_window_info(
                    msg_list,
                    _cfg,
                    step=step,
                    max_steps=max_steps,
                    agent_name=agent_name,
                    model=model_name,
                    context_window_tokens=context_window_tokens,
                    last_usage=last_usage,
                    token_tracker=token_tracker,
                    force=force_summarize,
                )
                await hook_manager.run(
                    HookPoint.CONTEXT_WINDOW,
                    agent=agent,
                    messages=msg_list,
                    info=_info,
                    provider=provider,
                    actions=actions,
                )
        return msg_list, actions

    # Separate system messages from conversation history
    system_msgs: list[Message] = [m for m in msg_list if isinstance(m, SystemMessage)]
    non_system: list[Message] = [m for m in msg_list if not isinstance(m, SystemMessage)]
    msg_count = len(non_system)

    # ── CONTEXT_WINDOW hook registered — bypass ALL built-in strategies ──
    # When a user registers a CONTEXT_WINDOW hook, it becomes the sole owner
    # of context reduction regardless of the configured overflow strategy.
    _has_ctx_hook = hook_manager is not None and hook_manager.has_hooks(HookPoint.CONTEXT_WINDOW)
    if _has_ctx_hook:
        result_list = system_msgs + non_system
        try:
            from exo.context.info import (  # pyright: ignore[reportMissingImports]
                build_context_window_info,
            )
        except ImportError:
            _log.debug("exo-context not installed, skipping CONTEXT_WINDOW hook")
        else:
            _info = build_context_window_info(
                result_list,
                _cfg,
                step=step,
                max_steps=max_steps,
                agent_name=agent_name,
                model=model_name,
                context_window_tokens=context_window_tokens,
                last_usage=last_usage,
                token_tracker=token_tracker,
                force=force_summarize,
            )
            await hook_manager.run(
                HookPoint.CONTEXT_WINDOW,
                agent=agent,
                messages=result_list,
                info=_info,
                provider=provider,
                actions=actions,
            )
        return result_list, actions

    # ── overflow="none" — no context management ──────────────────────────
    if overflow_strategy == "none":
        return system_msgs + non_system, actions

    # ── overflow="truncate" — simple drop of oldest messages ─────────────
    if overflow_strategy == "truncate":
        if msg_count > history_rounds:
            before = msg_count
            _log.debug(
                "context truncate: %d messages > limit=%d, dropping oldest",
                msg_count,
                history_rounds,
            )
            non_system = non_system[-history_rounds:]
            actions.append(
                _ContextAction(
                    "truncate",
                    before,
                    len(non_system),
                    {"limit": history_rounds},
                )
            )
    # ── overflow="summarize" — three-stage cascade ───────────────────────
    elif overflow_strategy == "summarize":
        # 1. Offload threshold: aggressive trim when far over limit
        if msg_count > offload_threshold:
            before = msg_count
            _log.debug(
                "context offload: %d messages > offload_threshold=%d, trimming to %d",
                msg_count,
                offload_threshold,
                summary_threshold,
            )
            non_system = non_system[-summary_threshold:]
            msg_count = len(non_system)
            actions.append(
                _ContextAction(
                    "offload",
                    before,
                    msg_count,
                    {"offload_threshold": offload_threshold},
                )
            )

        # 2. Summary threshold: attempt summarization via exo-memory.
        # Also fires when force_summarize=True (token budget exceeded) as long as
        # there are at least 2 messages to summarize.
        elif msg_count >= summary_threshold or (force_summarize and msg_count >= 2):
            try:
                from exo.memory.base import (  # pyright: ignore[reportMissingImports]
                    AIMemory,
                    HumanMemory,
                    MemoryItem,
                    ToolMemory,
                )
                from exo.memory.summary import (  # pyright: ignore[reportMissingImports]
                    SummaryConfig,
                    check_trigger,
                    generate_summary,
                )

                # Convert messages to MemoryItems for trigger check
                items: list[MemoryItem] = []
                for msg in non_system:
                    content = str(getattr(msg, "content", "") or "")
                    if isinstance(msg, UserMessage):
                        items.append(HumanMemory(content=content))
                    elif isinstance(msg, AssistantMessage):
                        items.append(AIMemory(content=content))
                    else:
                        items.append(ToolMemory(content=content))

                # When force_summarize=True, use a tighter keep_recent so that even
                # a small message list gets meaningfully compressed (keep half).
                if force_summarize:
                    keep_recent = max(2, msg_count // 2)
                else:
                    keep_recent = max(2, keep_recent_cfg)
                summary_cfg = SummaryConfig(
                    message_threshold=summary_threshold,
                    keep_recent=keep_recent,
                )

                # Bypass check_trigger() when force_summarize is set — the token
                # budget decision has already been made by the caller.
                should_summarize = force_summarize or check_trigger(items, summary_cfg).triggered

                if should_summarize and provider is not None:
                    before = msg_count
                    summarizer = _ProviderSummarizer(provider)
                    result = await generate_summary(items, summary_cfg, summarizer)
                    if result.summaries:
                        summary_text = "\n\n".join(result.summaries.values())
                        keep_count = len(result.compressed_items)
                        recent_msgs = non_system[-keep_count:] if keep_count > 0 else []
                        summary_msg = SystemMessage(
                            content=f"[Conversation Summary]\n{summary_text}"
                        )
                        non_system = [summary_msg, *recent_msgs]
                        msg_count = len(non_system)
                        _log.debug(
                            "context summarization applied: %d -> %d messages"
                            " (summary + %d recent)",
                            len(items),
                            msg_count,
                            keep_count,
                        )
                        actions.append(
                            _ContextAction(
                                "summarize",
                                before,
                                msg_count,
                                {
                                    "summary_threshold": summary_threshold,
                                    "keep_recent": keep_count,
                                    "forced": force_summarize,
                                },
                            )
                        )
            except ImportError:
                pass

        # 3. History windowing: keep last history_rounds messages
        if msg_count > history_rounds:
            before = msg_count
            _log.debug(
                "context windowing: trimming %d -> %d messages (history_rounds=%d)",
                msg_count,
                history_rounds,
                history_rounds,
            )
            non_system = non_system[-history_rounds:]
            actions.append(
                _ContextAction(
                    "window",
                    before,
                    history_rounds,
                    {"history_rounds": history_rounds},
                )
            )

    return system_msgs + non_system, actions
