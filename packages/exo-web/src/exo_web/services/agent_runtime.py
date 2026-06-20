"""Bridge between database agent configs and live Exo Agent objects.

Converts stored agent configurations (model, tools, instructions, etc.) into
runnable ``exo.Agent`` instances backed by real ``ModelProvider`` objects.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from exo.agent import Agent
from exo.models.provider import ModelProvider, get_provider
from exo.models.types import ModelResponse, StreamChunk
from exo.runner import run as exo_run
from exo.tool import FunctionTool
from exo.types import ExoError, Message, TextEvent, Usage, UsageEvent, UserMessage
from exo_web.crypto import decrypt_api_key
from exo_web.database import get_db
from exo_web.services.sandbox import SandboxConfig, execute_code

_log = logging.getLogger(__name__)


class AgentRuntimeError(ExoError):
    """Raised when agent runtime operations fail."""


async def _load_agent_row(agent_id: str) -> dict[str, Any]:
    """Load an agent record from the database.

    Raises:
        AgentRuntimeError: If the agent is not found.
    """
    async with get_db() as db:
        cursor = await db.execute("SELECT * FROM agents WHERE id = ?", (agent_id,))
        row = await cursor.fetchone()
    if row is None:
        raise AgentRuntimeError(f"Agent not found: {agent_id}")
    return dict(row)


async def _resolve_provider(provider_type: str, model_name: str, user_id: str) -> ModelProvider:
    """Resolve a ModelProvider from DB provider config.

    Looks up the provider by type, decrypts the API key, and returns a
    configured ``ModelProvider`` instance.

    Raises:
        AgentRuntimeError: If no provider or API key is configured.
    """
    async with get_db() as db:
        # Find provider by type for this user
        cursor = await db.execute(
            """
            SELECT * FROM providers
            WHERE provider_type = ? AND user_id = ?
            AND (
                encrypted_api_key IS NOT NULL AND encrypted_api_key != ''
                OR EXISTS (
                    SELECT 1 FROM provider_keys pk
                    WHERE pk.provider_id = providers.id AND pk.status = 'active'
                )
            )
            LIMIT 1
            """,
            (provider_type, user_id),
        )
        provider_row = await cursor.fetchone()
        if provider_row is None:
            raise AgentRuntimeError(f"No {provider_type} provider configured for user {user_id}")
        provider = dict(provider_row)

        # Resolve API key
        api_key = ""
        if provider.get("encrypted_api_key"):
            api_key = decrypt_api_key(provider["encrypted_api_key"])
        else:
            cursor = await db.execute(
                "SELECT encrypted_key FROM provider_keys WHERE provider_id = ? AND status = 'active' LIMIT 1",
                (provider["id"],),
            )
            key_row = await cursor.fetchone()
            if key_row:
                api_key = decrypt_api_key(key_row["encrypted_key"])

    if not api_key:
        raise AgentRuntimeError(f"No API key configured for {provider_type} provider")

    model_string = f"{provider_type}:{model_name}"
    base_url = provider.get("base_url") or None
    return get_provider(model_string, api_key=api_key, base_url=base_url)


async def _resolve_tools(tools_json: str, project_id: str, user_id: str) -> list[FunctionTool]:
    """Resolve tool IDs from the agent's tools_json into FunctionTool objects.

    Tools stored in the DB have a schema and code. We create lightweight
    FunctionTool wrappers that execute the stored code.
    """
    try:
        tool_ids: list[str] = json.loads(tools_json)
    except (json.JSONDecodeError, TypeError):
        return []

    if not tool_ids:
        return []

    tools: list[FunctionTool] = []
    async with get_db() as db:
        placeholders = ", ".join("?" for _ in tool_ids)
        cursor = await db.execute(
            f"SELECT * FROM tools WHERE id IN ({placeholders}) AND user_id = ?",
            [*tool_ids, user_id],
        )
        rows = await cursor.fetchall()

    for row in rows:
        tool_data = dict(row)
        tool_name = tool_data["name"]
        description = tool_data.get("description", "")
        tool_code: str = tool_data.get("code", "") or ""
        tool_type: str = tool_data.get("tool_type", "function") or "function"

        # Parse schema for parameters
        try:
            schema = json.loads(tool_data.get("schema_json", "{}"))
        except (json.JSONDecodeError, TypeError):
            schema = {}

        if tool_code and tool_type == "function":
            # Execute the stored Python code in the sandbox.
            # Inject call arguments as top-level variables so the user code can
            # read them, then capture stdout as the result.
            async def _tool_sandbox(
                _name: str = tool_name,
                _code: str = tool_code,
                **kwargs: Any,
            ) -> str:
                # Build a preamble that binds each kwarg as a variable
                preamble_lines = []
                for k, v in kwargs.items():
                    preamble_lines.append(f"{k} = {v!r}")
                preamble = "\n".join(preamble_lines)
                full_code = f"{preamble}\n{_code}" if preamble else _code

                _log.debug("tool sandbox exec: tool=%s code_len=%d", _name, len(full_code))
                result = execute_code(full_code, SandboxConfig())

                if result.success:
                    output = result.stdout.strip()
                    return output if output else json.dumps({"status": "ok", "tool": _name})
                else:
                    _log.warning(
                        "tool sandbox failed: tool=%s error=%s stderr=%s",
                        _name,
                        result.error,
                        result.stderr[:200] if result.stderr else "",
                    )
                    return json.dumps(
                        {
                            "status": "error",
                            "tool": _name,
                            "error": result.error or "Execution failed",
                            "stderr": result.stderr[:500] if result.stderr else "",
                        }
                    )

            callable_fn = _tool_sandbox
        else:
            # No executable code — return a structured stub indicating the tool
            # type is not directly executable (e.g. 'http', 'schema', 'mcp').
            async def _tool_stub(
                _name: str = tool_name, _ttype: str = tool_type, **kwargs: Any
            ) -> str:
                _log.warning(
                    "tool %r (type=%r) has no executable code; returning stub result",
                    _name,
                    _ttype,
                )
                return json.dumps(
                    {
                        "status": "not_implemented",
                        "tool": _name,
                        "tool_type": _ttype,
                        "args": kwargs,
                    }
                )

            callable_fn = _tool_stub

        ft = FunctionTool(
            callable_fn,
            name=tool_name,
            description=description or f"Tool: {tool_name}",
        )
        # Override the auto-generated schema with the stored one
        if schema:
            ft.parameters = schema
        tools.append(ft)

    return tools


class AgentService:
    """Converts database agent config records into live Exo Agent objects."""

    async def build_agent(self, agent_id: str) -> Agent:
        """Build a live Agent instance from a database agent config.

        Args:
            agent_id: The database ID of the agent.

        Returns:
            A configured ``exo.Agent`` ready for execution.

        Raises:
            AgentRuntimeError: If the agent or its provider is not found.
        """
        row = await _load_agent_row(agent_id)

        provider_type = row.get("model_provider", "")
        model_name = row.get("model_name", "")
        if not provider_type or not model_name:
            raise AgentRuntimeError(f"Agent {agent_id} has no model configured")

        # Resolve tools
        tools = await _resolve_tools(
            row.get("tools_json", "[]"),
            row.get("project_id", ""),
            row.get("user_id", ""),
        )

        return Agent(
            name=row.get("name", agent_id),
            model=f"{provider_type}:{model_name}",
            instructions=row.get("instructions", ""),
            tools=tools or None,
            max_steps=row.get("max_steps") or 10,
            temperature=row.get("temperature") if row.get("temperature") is not None else 1.0,
            max_tokens=row.get("max_tokens"),
        )

    async def run_agent(
        self,
        agent_id: str,
        messages: list[Message],
    ) -> ModelResponse:
        """Execute a single agent turn and return the model response.

        Args:
            agent_id: The database ID of the agent.
            messages: Conversation history as Exo Message objects.

        Returns:
            The ``ModelResponse`` from the LLM.

        Raises:
            AgentRuntimeError: If the agent, provider, or model call fails.
        """
        _log.info("run_agent start: agent=%s messages=%d", agent_id, len(messages))
        row = await _load_agent_row(agent_id)
        provider_type = row.get("model_provider", "")
        model_name = row.get("model_name", "")
        if not provider_type or not model_name:
            raise AgentRuntimeError(f"Agent {agent_id} has no model configured")

        agent = await self.build_agent(agent_id)
        provider = await _resolve_provider(provider_type, model_name, row["user_id"])

        # Extract the last user message as input
        user_input = ""
        for msg in reversed(messages):
            if isinstance(msg, UserMessage):
                user_input = msg.content
                break

        try:
            output = await agent.run(
                input=user_input,
                messages=messages[:-1] if user_input else messages,
                provider=provider,
            )
        except Exception as exc:
            _log.error("run_agent failed for agent %s: %s", agent_id, exc, exc_info=True)
            raise AgentRuntimeError(f"Model call failed for agent {agent_id}: {exc}") from exc

        tool_call_count = len(output.tool_calls) if output.tool_calls else 0
        if tool_call_count:
            _log.debug("run_agent tool calls: agent=%s count=%d", agent_id, tool_call_count)

        return ModelResponse(
            content=output.text,
            tool_calls=output.tool_calls,
            usage=output.usage,
        )

    async def stream_agent(
        self,
        agent_id: str,
        messages: list[Message],
    ) -> AsyncIterator[StreamChunk]:
        """Stream agent execution using the full agent tool loop.

        Delegates to ``exo.runner.run.stream()`` so that tool calls
        emitted by the LLM are executed and their results fed back before
        the next streaming response.  Yields ``StreamChunk`` objects for
        backward compatibility with existing callers.

        Args:
            agent_id: The database ID of the agent.
            messages: Conversation history as Exo Message objects.

        Yields:
            ``StreamChunk`` objects — one per text delta, plus a final
            chunk with ``finish_reason="stop"`` and accumulated usage.

        Raises:
            AgentRuntimeError: If the agent, provider, or model call fails.
        """
        _log.info("stream_agent start: agent=%s messages=%d", agent_id, len(messages))
        row = await _load_agent_row(agent_id)
        provider_type = row.get("model_provider", "")
        model_name = row.get("model_name", "")
        if not provider_type or not model_name:
            raise AgentRuntimeError(f"Agent {agent_id} has no model configured")

        agent = await self.build_agent(agent_id)
        provider = await _resolve_provider(provider_type, model_name, row["user_id"])

        # Extract the last user message as input (same logic as run_agent)
        user_input = ""
        for msg in reversed(messages):
            if isinstance(msg, UserMessage):
                user_input = msg.content
                break

        history = messages[:-1] if user_input else list(messages)

        last_usage: Usage | None = None
        try:
            async for event in exo_run.stream(
                agent,
                user_input,
                messages=history,
                provider=provider,
                detailed=True,
            ):
                if isinstance(event, TextEvent):
                    yield StreamChunk(delta=event.text)
                elif isinstance(event, UsageEvent):
                    last_usage = event.usage
        except Exception as exc:
            _log.error("stream_agent failed for agent %s: %s", agent_id, exc, exc_info=True)
            raise AgentRuntimeError(f"Stream failed for agent {agent_id}: {exc}") from exc

        # Final chunk signals completion and carries accumulated token usage
        yield StreamChunk(finish_reason="stop", usage=last_usage or Usage())
