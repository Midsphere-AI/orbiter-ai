"""ContextProcessor — event-driven context processing pipeline.

Processors intervene at specific points in the LLM execution cycle to
dynamically transform context.  Each processor declares which event type
it handles (e.g., ``"pre_llm_call"``, ``"post_tool_call"``).

The :class:`ProcessorPipeline` collects processors and fires them by event.
Processors are called sequentially in registration order for a given event.

Built-in processors:
- SummarizeProcessor    — ``pre_llm_call``: marks context for summarization
  when history exceeds the configured threshold.
- ToolResultOffloader   — ``post_tool_call``: offloads large tool results to
  workspace when they exceed a size threshold.
- MessageOffloader      — ``pre_llm_call``: replaces oversized messages with
  ``[[OFFLOAD: handle=<id>]]`` markers to keep context within budget.
- DialogueCompressor    — ``pre_llm_call``: compresses long tool-call chains
  into concise summaries.
- RoundWindowProcessor  — ``pre_llm_call``: keeps the last *N* conversation
  rounds (user → assistant, including tool calls) plus all system messages.
"""

from __future__ import annotations

import contextlib
import logging
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

from exo.types import ExoError  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)

from exo.context.context import Context  # pyright: ignore[reportMissingImports]


class ProcessorError(ExoError):
    """Raised for processor pipeline errors."""


# ── ABC ──────────────────────────────────────────────────────────────


class ContextProcessor(ABC):
    """Abstract base for context processors.

    Each processor handles a single event type (e.g., ``"pre_llm_call"``).
    The :meth:`process` method receives the context and an arbitrary payload
    dict containing event-specific data.

    Parameters
    ----------
    event:
        The event type this processor handles.
    name:
        Human-readable name for debugging.  Defaults to the class name.
    """

    __slots__ = ("_event", "_name")

    def __init__(self, event: str, *, name: str | None = None) -> None:
        if not event:
            msg = "event must be a non-empty string"
            raise ProcessorError(msg)
        self._event = event
        self._name = name or type(self).__name__

    @property
    def event(self) -> str:
        """The event type this processor handles."""
        return self._event

    @property
    def name(self) -> str:
        """Human-readable processor name."""
        return self._name

    @abstractmethod
    async def process(self, ctx: Context, payload: dict[str, Any]) -> None:
        """Process the event with context and payload.

        Implementations may mutate ``ctx.state`` to transform context.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(event={self._event!r}, name={self._name!r})"


# ── Pipeline ─────────────────────────────────────────────────────────


class ProcessorPipeline:
    """Registers and fires context processors by event type.

    Processors are called sequentially in registration order for each
    event.  Errors from processors propagate immediately.

    Usage::

        pipeline = ProcessorPipeline()
        pipeline.register(SummarizeProcessor())
        pipeline.register(ToolResultOffloader(max_size=5000))
        await pipeline.fire("pre_llm_call", ctx, {"messages": [...]})
    """

    __slots__ = ("_processors",)

    def __init__(self) -> None:
        self._processors: defaultdict[str, list[ContextProcessor]] = defaultdict(list)

    def register(self, processor: ContextProcessor) -> ProcessorPipeline:
        """Register a processor for its declared event type.

        Returns ``self`` for method chaining.
        """
        self._processors[processor.event].append(processor)
        logger.debug("registered processor %r for event %r", processor.name, processor.event)
        return self

    def unregister(self, processor: ContextProcessor) -> None:
        """Remove a processor.  Silently does nothing if not registered."""
        procs = self._processors.get(processor.event)
        if procs is not None:
            with contextlib.suppress(ValueError):
                procs.remove(processor)

    async def fire(self, event: str, ctx: Context, payload: dict[str, Any] | None = None) -> None:
        """Fire all processors registered for *event* in order.

        Parameters
        ----------
        event:
            The event type to fire.
        ctx:
            The context passed to each processor.
        payload:
            Optional event-specific data dict.  Defaults to ``{}``.
        """
        data = payload if payload is not None else {}
        processors = self._processors.get(event, [])
        if processors:
            logger.debug("Processing context with %d neurons", len(processors))
        for proc in processors:
            logger.debug("running processor %r for event %r", proc.name, event)
            await proc.process(ctx, data)

    def has_processors(self, event: str) -> bool:
        """Check whether any processors are registered for *event*."""
        return len(self._processors.get(event, [])) > 0

    def list_processors(self, event: str | None = None) -> list[ContextProcessor]:
        """List processors, optionally filtered by event."""
        if event is not None:
            return list(self._processors.get(event, []))
        result: list[ContextProcessor] = []
        for procs in self._processors.values():
            result.extend(procs)
        return result

    def clear(self) -> None:
        """Remove all processors."""
        self._processors.clear()

    def __len__(self) -> int:
        """Total number of registered processors across all events."""
        return sum(len(procs) for procs in self._processors.values())

    def __repr__(self) -> str:
        events = list(self._processors.keys())
        total = len(self)
        return f"ProcessorPipeline(events={events}, total={total})"


# ── Built-in processors ─────────────────────────────────────────────


class SummarizeProcessor(ContextProcessor):
    """Marks context for summarization when history exceeds a threshold.

    Fires on ``"pre_llm_call"``.  Checks the ``history`` list in
    ``ctx.state`` against ``ctx.config.summary_threshold``.  When
    exceeded, sets ``needs_summary=True`` in state and stores the
    excess messages under ``summary_candidates``.

    Parameters
    ----------
    name:
        Processor name.  Default ``"summarize"``.
    """

    def __init__(self, *, name: str = "summarize") -> None:
        super().__init__("pre_llm_call", name=name)

    async def process(self, ctx: Context, payload: dict[str, Any]) -> None:
        history: list[dict[str, Any]] | None = ctx.state.get("history")
        if not history:
            return

        threshold = ctx.config.summary_threshold
        if len(history) <= threshold:
            return

        # Mark for summarization and store candidates
        ctx.state.set("needs_summary", True)
        # Candidates are the oldest messages beyond the threshold
        excess_count = len(history) - threshold
        ctx.state.set("summary_candidates", history[:excess_count])
        logger.debug(
            "summarization triggered: %d messages exceed threshold %d, %d candidates",
            len(history),
            threshold,
            excess_count,
        )


class ToolResultOffloader(ContextProcessor):
    """Offloads large tool results to workspace.

    Fires on ``"post_tool_call"``.  When a tool result's content exceeds
    ``max_size`` characters, or when ``payload["large_output"]`` is ``True``,
    replaces it with a reference placeholder and stores the full content under
    ``offloaded_results`` in state.

    Parameters
    ----------
    max_size:
        Maximum character length before offloading.  Default 5000.
    name:
        Processor name.  Default ``"tool_result_offloader"``.
    """

    __slots__ = ("_event", "_max_size", "_name")

    def __init__(self, *, max_size: int = 5000, name: str = "tool_result_offloader") -> None:
        super().__init__("post_tool_call", name=name)
        self._max_size = max_size

    @property
    def max_size(self) -> int:
        """Maximum content size before offloading."""
        return self._max_size

    async def process(self, ctx: Context, payload: dict[str, Any]) -> None:
        tool_result = payload.get("tool_result")
        if tool_result is None:
            return

        content = str(tool_result)
        large_output: bool = bool(payload.get("large_output", False))

        if not large_output and len(content) <= self._max_size:
            return

        # Store full content in offloaded results
        offloaded: list[dict[str, Any]] = ctx.state.get("offloaded_results") or []
        tool_name = payload.get("tool_name", "unknown")
        tool_call_id = payload.get("tool_call_id", "unknown")
        artifact_id: str | None = payload.get("artifact_id")

        offloaded.append(
            {
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "content": content,
                "size": len(content),
                "artifact_id": artifact_id,
            }
        )
        ctx.state.set("offloaded_results", offloaded)

        # Replace tool result content with a reference
        if artifact_id is not None:
            reference = (
                f"[Result stored as artifact '{artifact_id}'. "
                f"Call retrieve_artifact('{artifact_id}') to access.]"
            )
        else:
            truncated = content[: self._max_size // 2]
            reference = (
                f"{truncated}...\n\n"
                f"[Result truncated — full content offloaded to workspace "
                f"({len(content)} chars)]"
            )
        payload["tool_result"] = reference
        logger.debug(
            "ToolResultOffloader: offloading %s result size=%d bytes artifact_id=%s",
            tool_name,
            len(content),
            artifact_id,
        )


class MessageOffloader(ContextProcessor):
    """Replaces oversized messages with ``[[OFFLOAD: handle=<id>]]`` markers.

    Fires on ``"pre_llm_call"``.  Scans ``ctx.state['history']`` and replaces
    any user, assistant, or tool message whose content exceeds
    ``max_message_size`` characters with a short marker.  The original content
    is stored in ``ctx.state['offloaded_messages']`` keyed by handle ID so it
    can be recovered later (e.g. via a reload tool).

    System messages are never offloaded.

    Parameters
    ----------
    max_message_size:
        Maximum character length per message before offloading.  Default 10000.
    name:
        Processor name.  Default ``"message_offloader"``.
    """

    __slots__ = ("_event", "_max_message_size", "_name")

    def __init__(self, *, max_message_size: int = 10000, name: str = "message_offloader") -> None:
        super().__init__("pre_llm_call", name=name)
        self._max_message_size = max_message_size

    @property
    def max_message_size(self) -> int:
        """Maximum message content size before offloading."""
        return self._max_message_size

    async def process(self, ctx: Context, payload: dict[str, Any]) -> None:
        history: list[dict[str, Any]] | None = ctx.state.get("history")
        if not history:
            return

        offloaded: dict[str, str] = ctx.state.get("offloaded_messages") or {}
        count = 0

        for msg in history:
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
            if role == "system":
                continue

            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
            if content is None:
                continue
            content_str = str(content)
            if len(content_str) <= self._max_message_size:
                continue

            handle_id = uuid.uuid4().hex[:12]
            offloaded[handle_id] = content_str
            marker = f"[[OFFLOAD: handle={handle_id}]]"
            if isinstance(msg, dict):
                msg["content"] = marker
            else:
                msg.content = marker  # type: ignore[attr-defined]
            count += 1

        ctx.state.set("offloaded_messages", offloaded)
        if count:
            logger.debug("MessageOffloader: offloaded %d oversized messages", count)


class DialogueCompressor(ContextProcessor):
    """Compresses long tool-call chains into concise summaries.

    Fires on ``"pre_llm_call"``.  Scans ``ctx.state['history']`` for
    consecutive sequences of assistant-with-tool-calls and tool-result
    messages.  When a chain length meets or exceeds ``min_tool_chain_length``,
    the entire chain is replaced by a single system-style summary.

    Parameters
    ----------
    min_tool_chain_length:
        Minimum number of tool-call/result message pairs before compression
        is triggered.  Default ``3``.
    model:
        Optional model identifier for LLM-based summarization.  When
        ``None``, uses simple concatenation fallback.
    name:
        Processor name.  Default ``"dialogue_compressor"``.
    """

    __slots__ = ("_event", "_min_tool_chain_length", "_model", "_name")

    def __init__(
        self,
        *,
        min_tool_chain_length: int = 3,
        model: str | None = None,
        name: str = "dialogue_compressor",
    ) -> None:
        super().__init__("pre_llm_call", name=name)
        self._min_tool_chain_length = min_tool_chain_length
        self._model = model

    @property
    def min_tool_chain_length(self) -> int:
        """Minimum chain length before compression."""
        return self._min_tool_chain_length

    @property
    def model(self) -> str | None:
        """Model identifier for LLM summarization, or ``None``."""
        return self._model

    async def process(self, ctx: Context, payload: dict[str, Any]) -> None:
        history: list[Any] | None = ctx.state.get("history")
        if not history:
            return

        chains = self._find_tool_chains(history)
        if not chains:
            return

        # Process chains in reverse order so indices stay valid
        for start, end in reversed(chains):
            chain_msgs = history[start:end]
            summary = self._summarize_chain(chain_msgs)
            summary_msg: dict[str, str] = {"role": "system", "content": summary}
            history[start:end] = [summary_msg]

        logger.debug("DialogueCompressor: compressed %d tool chains", len(chains))

    def _find_tool_chains(self, history: list[Any]) -> list[tuple[int, int]]:
        """Find consecutive tool-call/result chains that exceed the threshold."""
        chains: list[tuple[int, int]] = []
        i = 0
        n = len(history)

        while i < n:
            if self._is_tool_message(history[i]):
                chain_start = i
                while i < n and self._is_tool_message(history[i]):
                    i += 1
                chain_length = i - chain_start
                if chain_length >= self._min_tool_chain_length:
                    chains.append((chain_start, i))
            else:
                i += 1

        return chains

    @staticmethod
    def _is_tool_message(msg: Any) -> bool:
        """Check if a message is part of a tool-call chain."""
        if isinstance(msg, dict):
            role = msg.get("role")
            if role == "tool":
                return True
            if role == "assistant" and msg.get("tool_calls"):
                return True
            return False

        role = getattr(msg, "role", None)
        if role == "tool":
            return True
        if role == "assistant" and getattr(msg, "tool_calls", None):
            return True
        return False

    def _summarize_chain(self, chain: list[Any]) -> str:
        """Produce a text summary of a tool-call chain."""
        tool_names: list[str] = []
        results: list[str] = []

        for msg in chain:
            if isinstance(msg, dict):
                role = msg.get("role")
                if role == "assistant":
                    for tc in msg.get("tool_calls", []):
                        name = (
                            tc.get("name")
                            if isinstance(tc, dict)
                            else getattr(tc, "name", "unknown")
                        )
                        tool_names.append(str(name) if name else "unknown")
                elif role == "tool":
                    tool_name = msg.get("tool_name") or msg.get("name") or "tool"
                    content = str(msg.get("content", ""))
                    results.append(f"{tool_name}: {content[:100]}")
            else:
                role = getattr(msg, "role", None)
                if role == "assistant":
                    for tc in getattr(msg, "tool_calls", []):
                        tool_names.append(getattr(tc, "name", "unknown"))
                elif role == "tool":
                    tool_name = getattr(msg, "tool_name", None) or getattr(msg, "name", "tool")
                    content = str(getattr(msg, "content", ""))
                    results.append(f"{tool_name}: {content[:100]}")

        unique_tools = list(dict.fromkeys(tool_names))
        tools_str = ", ".join(unique_tools) if unique_tools else "tools"
        results_str = "; ".join(results) if results else "no results"

        return (
            f"[Tool chain compressed — called {tools_str} "
            f"({len(tool_names)} calls). Results: {results_str}]"
        )


class RoundWindowProcessor(ContextProcessor):
    """Keeps the last *N* conversation rounds plus all system messages.

    Fires on ``"pre_llm_call"``.  A *round* is a user message followed by
    everything up to (but not including) the next user message — typically
    an assistant reply and any interleaved tool-call/tool-result messages.

    System messages are always preserved regardless of windowing.

    Parameters
    ----------
    max_rounds:
        Maximum number of conversation rounds to retain.  Default ``20``.
    name:
        Processor name.  Default ``"round_window"``.
    """

    __slots__ = ("_event", "_max_rounds", "_name")

    def __init__(self, *, max_rounds: int = 20, name: str = "round_window") -> None:
        super().__init__("pre_llm_call", name=name)
        self._max_rounds = max_rounds

    @property
    def max_rounds(self) -> int:
        """Maximum number of conversation rounds to retain."""
        return self._max_rounds

    async def process(self, ctx: Context, payload: dict[str, Any]) -> None:
        history: list[Any] | None = ctx.state.get("history")
        if not history:
            return

        system_msgs: list[Any] = []
        rounds: list[list[Any]] = []
        current_round: list[Any] = []

        for msg in history:
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)

            if role == "system":
                system_msgs.append(msg)
                continue

            if role == "user":
                if current_round:
                    rounds.append(current_round)
                current_round = [msg]
            else:
                current_round.append(msg)

        if current_round:
            rounds.append(current_round)

        if len(rounds) <= self._max_rounds:
            return

        kept_rounds = rounds[-self._max_rounds :]
        result: list[Any] = list(system_msgs)
        for rnd in kept_rounds:
            result.extend(rnd)

        history[:] = result
        logger.debug(
            "RoundWindowProcessor: windowed from %d to %d rounds",
            len(rounds),
            self._max_rounds,
        )
