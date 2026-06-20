"""Workflow execution engine — runs DAGs in topological order."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from datetime import UTC, datetime
from typing import Any

from exo_web.database import get_db
from exo_web.routes.costs import check_budget

_log = logging.getLogger(__name__)

# Background queue-processing tasks (prevent GC).
_queue_tasks: set[asyncio.Task[Any]] = set()

# ---------------------------------------------------------------------------
# Node error-handling config defaults
# ---------------------------------------------------------------------------

_DEFAULT_RETRY_COUNT = 0
_DEFAULT_RETRY_DELAY_MS = 1000
_DEFAULT_ON_ERROR = "fail"  # "fail" | "skip" | "fallback"
_DEFAULT_TIMEOUT_MS = 30000

# ---------------------------------------------------------------------------
# Topological sort
# ---------------------------------------------------------------------------


def topological_sort(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[list[str]]:
    """Return nodes grouped into execution layers (parallel within each layer).

    Uses Kahn's algorithm. Each returned list is a set of node IDs that can
    execute concurrently because all their dependencies are satisfied.

    Raises ``ValueError`` if the graph contains a cycle.
    """
    node_ids = {n["id"] for n in nodes}
    in_degree: dict[str, int] = {nid: 0 for nid in node_ids}
    children: dict[str, list[str]] = defaultdict(list)

    for edge in edges:
        src, tgt = edge["source"], edge["target"]
        if src in node_ids and tgt in node_ids:
            in_degree[tgt] += 1
            children[src].append(tgt)

    queue: deque[str] = deque(nid for nid, deg in in_degree.items() if deg == 0)
    layers: list[list[str]] = []
    visited = 0

    while queue:
        layer = list(queue)
        queue.clear()
        layers.append(layer)
        visited += len(layer)
        for nid in layer:
            for child in children[nid]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

    if visited != len(node_ids):
        raise ValueError("Workflow graph contains a cycle")

    return layers


# ---------------------------------------------------------------------------
# Node executors
# ---------------------------------------------------------------------------

_AGENT_NODE_TYPES = {"agent_node", "sub_agent"}
_RETRIEVAL_NODE_TYPE = "knowledge_retrieval"
_APPROVAL_GATE_TYPE = "approval_gate"


def _build_incoming_edge_map(edges: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Build a map from node_id -> list of source node_ids (incoming edges).

    Building this once before iteration makes upstream-input gathering O(1)
    per node (amortized) instead of O(E) per node (L-16).
    """
    incoming: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        if src and tgt:
            incoming[tgt].append(src)
    return dict(incoming)


def _gather_upstream_inputs(
    node_id: str,
    edges: list[dict[str, Any]],
    variables: dict[str, Any],
    *,
    incoming_map: dict[str, list[str]] | None = None,
) -> str:
    """Collect text from all upstream nodes feeding into this node.

    Joins upstream outputs separated by newlines. Returns an empty string
    if no upstream data is available.

    When ``incoming_map`` is provided (pre-built via ``_build_incoming_edge_map``),
    the lookup is O(1) amortized; otherwise falls back to O(E) linear scan.
    """
    parts: list[str] = []
    if incoming_map is not None:
        for src_id in incoming_map.get(node_id, []):
            val = variables.get(src_id, "")
            if val:
                parts.append(str(val))
    else:
        for edge in edges:
            if edge.get("target") == node_id:
                src_id = edge.get("source", "")
                val = variables.get(src_id, "")
                if val:
                    parts.append(str(val))
    return "\n".join(parts)


async def _execute_agent_node(
    node: dict[str, Any],
    upstream_text: str,
    event_callback: Any | None = None,
) -> dict[str, Any]:
    """Execute an agent_node or sub_agent workflow node.

    Resolves the agent from DB (by agent_id or inline config), builds an
    Exo Agent, calls its tool loop, and returns the structured result.

    Supports streaming: if ``event_callback`` is provided, emits
    ``agent_token`` events as the agent streams its response.
    """
    from exo.types import UserMessage
    from exo_web.services.agent_runtime import (
        AgentRuntimeError,
        AgentService,
        _resolve_provider,
        _resolve_tools,
    )

    node_data = node.get("data", {})
    node_id = node["id"]
    agent_id = node_data.get("agent_id")
    is_inline = node_data.get("inline", False)

    user_text = upstream_text or ""
    messages = [UserMessage(content=user_text)]

    svc = AgentService()

    async def emit(event: dict[str, Any]) -> None:
        if event_callback is not None:
            try:
                await event_callback(event)
            except Exception as exc:
                _log.warning("WebSocket send failed: %s", exc)

    if agent_id and not is_inline:
        # --- Referenced agent from DB ---
        # Try streaming first for real-time token events.
        if event_callback is not None:
            try:
                collected_text = ""
                usage_dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                tool_calls_info: list[Any] = []
                async for chunk in svc.stream_agent(agent_id, messages):
                    if chunk.delta:
                        collected_text += chunk.delta
                        await emit(
                            {"type": "agent_token", "node_id": node_id, "token": chunk.delta}
                        )
                    if chunk.usage:
                        usage_dict = {
                            "prompt_tokens": chunk.usage.get("input_tokens", 0),
                            "completion_tokens": chunk.usage.get("output_tokens", 0),
                            "total_tokens": chunk.usage.get("total_tokens", 0),
                        }
                    if chunk.tool_calls:
                        for tc in chunk.tool_calls:
                            tool_calls_info.append(str(tc))

                return {
                    "node_id": node_id,
                    "node_type": node_data.get("nodeType", "agent_node"),
                    "label": node_data.get("label", ""),
                    "result": collected_text,
                    "logs": f"input: {user_text[:200]}\nresponse: {collected_text[:500]}\ntool_calls: {json.dumps(tool_calls_info)}",
                    "token_usage": usage_dict,
                }
            except AgentRuntimeError:
                raise
            except Exception:
                _log.debug("Streaming failed for agent node %s, falling back to run_agent", node_id)

        # Non-streaming fallback
        try:
            output = await svc.run_agent(agent_id, messages)
        except AgentRuntimeError as exc:
            _log.warning("Agent node %s failed: %s", node_id, exc)
            raise

        usage_dict = {
            "prompt_tokens": output.usage.input_tokens if output.usage else 0,
            "completion_tokens": output.usage.output_tokens if output.usage else 0,
            "total_tokens": output.usage.total_tokens if output.usage else 0,
        }
        tool_calls_info = []
        if output.tool_calls:
            for tc in output.tool_calls:
                tool_calls_info.append(
                    {"name": tc.function.name, "arguments": tc.function.arguments}
                    if hasattr(tc, "function")
                    else str(tc)
                )

        return {
            "node_id": node_id,
            "node_type": node_data.get("nodeType", "agent_node"),
            "label": node_data.get("label", ""),
            "result": output.content or "",
            "logs": f"input: {user_text[:200]}\nresponse: {(output.content or '')[:500]}\ntool_calls: {json.dumps(tool_calls_info)}",
            "token_usage": usage_dict,
        }

    elif is_inline:
        # --- Inline agent defined in node config ---
        from exo.agent import Agent
        from exo.tool import FunctionTool

        provider_type = node_data.get("inline_model_provider", "")
        model_name = node_data.get("inline_model_name", "")
        instructions = node_data.get("inline_instructions", "")
        inline_tool_ids = node_data.get("inline_tools", [])

        if not provider_type or not model_name:
            raise AgentRuntimeError(f"Inline agent on node {node_id} has no model configured")

        # Resolve the user_id from the workflow context (stored in node data
        # during execution, or fall back to default).
        user_id = node_data.get("_user_id", "default-user")

        provider = await _resolve_provider(provider_type, model_name, user_id)

        # Resolve inline tools if specified
        tools: list[FunctionTool] = []
        if inline_tool_ids:
            tools_json = json.dumps(inline_tool_ids)
            project_id = node_data.get("_project_id", "")
            tools = await _resolve_tools(tools_json, project_id, user_id)

        agent = Agent(
            name=node_data.get("inline_name", f"inline-agent-{node_id[:8]}"),
            model=f"{provider_type}:{model_name}",
            instructions=instructions,
            tools=tools or None,
            max_steps=10,
        )

        output = await agent.run(input=user_text, provider=provider)

        usage_dict = {
            "prompt_tokens": output.usage.input_tokens,
            "completion_tokens": output.usage.output_tokens,
            "total_tokens": output.usage.total_tokens,
        }

        return {
            "node_id": node_id,
            "node_type": node_data.get("nodeType", "agent_node"),
            "label": node_data.get("label", ""),
            "result": output.text or "",
            "logs": f"input: {user_text[:200]}\nmodel: {provider_type}:{model_name}\nresponse: {(output.text or '')[:500]}",
            "token_usage": usage_dict,
        }

    else:
        # No agent_id and not inline — missing configuration
        return {
            "node_id": node_id,
            "node_type": node_data.get("nodeType", "agent_node"),
            "label": node_data.get("label", ""),
            "result": "Error: agent node has no agent_id and is not configured inline",
            "logs": "No agent configuration found on this node",
        }


async def _execute_retrieval_node(
    node: dict[str, Any],
    query_text: str,
) -> dict[str, Any]:
    """Execute a knowledge_retrieval workflow node.

    Queries a knowledge base using the search endpoint logic and returns
    ranked chunks as the node result.
    """
    from exo_web.routes.knowledge_bases import _keyword_score

    node_data = node.get("data", {})
    node_id = node["id"]
    kb_id = node_data.get("knowledge_base_id", "")

    if not kb_id:
        return {
            "node_id": node_id,
            "node_type": _RETRIEVAL_NODE_TYPE,
            "label": node_data.get("label", ""),
            "result": json.dumps([]),
            "logs": "Error: no knowledge_base_id configured on retrieval node",
        }

    if not query_text.strip():
        return {
            "node_id": node_id,
            "node_type": _RETRIEVAL_NODE_TYPE,
            "label": node_data.get("label", ""),
            "result": json.dumps([]),
            "logs": "No query text provided",
        }

    async with get_db() as db:
        # Load KB retrieval settings
        cursor = await db.execute(
            "SELECT top_k, similarity_threshold FROM knowledge_bases WHERE id = ?",
            (kb_id,),
        )
        kb_row = await cursor.fetchone()
        if kb_row is None:
            return {
                "node_id": node_id,
                "node_type": _RETRIEVAL_NODE_TYPE,
                "label": node_data.get("label", ""),
                "result": json.dumps([]),
                "logs": f"Knowledge base {kb_id} not found",
            }

        top_k = kb_row["top_k"]
        similarity_threshold = kb_row["similarity_threshold"]

        # Keyword search
        query_terms = [t.lower() for t in query_text.split() if t.strip()]
        if not query_terms:
            return {
                "node_id": node_id,
                "node_type": _RETRIEVAL_NODE_TYPE,
                "label": node_data.get("label", ""),
                "result": json.dumps([]),
                "logs": "Empty query after tokenization",
            }

        like_clauses = " OR ".join("dc.content LIKE ?" for _ in query_terms)
        like_params = [f"%{term}%" for term in query_terms]

        cursor = await db.execute(
            f"""
            SELECT dc.id AS chunk_id, dc.document_id, d.filename,
                   dc.chunk_index, dc.content
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            WHERE dc.kb_id = ? AND ({like_clauses})
            """,
            [kb_id, *like_params],
        )
        rows = await cursor.fetchall()

    scored = []
    for row in rows:
        row_dict = dict(row)
        score = _keyword_score(query_terms, row_dict["content"])
        if score >= similarity_threshold:
            scored.append({**row_dict, "score": round(score, 4)})

    scored.sort(key=lambda x: x["score"], reverse=True)
    results = scored[:top_k]

    return {
        "node_id": node_id,
        "node_type": _RETRIEVAL_NODE_TYPE,
        "label": node_data.get("label", ""),
        "result": json.dumps(results),
        "logs": f"query: {query_text[:200]}\nresults: {len(results)} chunks returned",
    }


async def _execute_approval_gate(
    node: dict[str, Any],
    run_id: str,
    user_id: str,
    event_callback: Any | None = None,
    cancel_event: asyncio.Event | None = None,
) -> dict[str, Any]:
    """Execute an approval_gate node — pause until a human approves or rejects.

    Creates a workflow_approval record, updates run status to 'awaiting_approval',
    and polls until the approval is resolved or times out.
    """
    from exo_web.routes.approvals import create_approval, poll_approval

    node_data = node.get("data", {})
    node_id = node["id"]
    timeout_minutes = node_data.get("timeout_minutes", 60)

    async def emit(event: dict[str, Any]) -> None:
        if event_callback is not None:
            try:
                await event_callback(event)
            except Exception as exc:
                _log.warning("WebSocket send failed: %s", exc)

    # Create the approval record.
    approval_id = await create_approval(run_id, node_id, user_id, timeout_minutes)

    # Mark the run as awaiting approval.
    await _update_run_status(run_id, "awaiting_approval")

    # Create a notification in the alerts table for the user.
    notification_id = str(uuid.uuid4())
    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO alerts (id, rule_id, severity, message, acknowledged, user_id)
            VALUES (?, '', 'info', ?, 0, ?)
            """,
            (
                notification_id,
                f"Workflow approval required for node '{node_data.get('label', node_id)}'",
                user_id,
            ),
        )
        await db.commit()

    await emit(
        {
            "type": "approval_required",
            "node_id": node_id,
            "approval_id": approval_id,
            "timeout_minutes": timeout_minutes,
        }
    )

    # Poll for approval resolution.
    while True:
        if cancel_event is not None and cancel_event.is_set():
            return {
                "node_id": node_id,
                "node_type": _APPROVAL_GATE_TYPE,
                "label": node_data.get("label", ""),
                "result": "Cancelled",
                "logs": "Approval gate cancelled by run cancellation",
            }

        status = await poll_approval(approval_id)

        if status == "approved":
            # Restore run status to running.
            await _update_run_status(run_id, "running")
            return {
                "node_id": node_id,
                "node_type": _APPROVAL_GATE_TYPE,
                "label": node_data.get("label", ""),
                "result": "Approved",
                "logs": f"Approval gate approved (approval_id={approval_id})",
            }

        if status in ("rejected", "timed_out"):
            # Restore run status to running (the caller will handle the failure).
            await _update_run_status(run_id, "running")
            reason = "Rejected by user" if status == "rejected" else "Timed out"
            raise RuntimeError(f"Approval gate {status}: {reason}")

        # Still pending — wait before polling again.
        await asyncio.sleep(2)


async def _execute_llm_node(
    node: dict[str, Any],
    edges: list[dict[str, Any]],
    variables: dict[str, Any],
    event_callback: Any | None = None,
    *,
    user_id: str = "",
    incoming_map: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Execute an ``llm`` workflow node by calling a real model completion.

    Reads ``provider``, ``model``, and ``prompt`` from node config.  The prompt
    may reference ``{upstream}`` which is replaced with any upstream outputs.
    Falls back to the inline-agent path from ``_execute_agent_node`` so that
    provider resolution, API-key lookup, and token tracking all reuse the
    existing infrastructure.
    """
    from exo_web.services.agent_runtime import AgentRuntimeError, _resolve_provider

    node_data = node.get("data", {})
    node_id = node["id"]

    provider_type = node_data.get("provider", "")
    model_name = node_data.get("model", "")
    prompt_template = node_data.get("prompt", "")

    upstream_text = _gather_upstream_inputs(node_id, edges, variables, incoming_map=incoming_map)
    prompt = (
        prompt_template.replace("{upstream}", upstream_text) if upstream_text else prompt_template
    )

    if not provider_type or not model_name:
        return {
            "node_id": node_id,
            "node_type": "llm",
            "label": node_data.get("label", ""),
            "result": "Error: llm node has no provider/model configured",
            "logs": "Missing provider or model in node config",
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    try:
        from exo.agent import Agent

        provider = await _resolve_provider(provider_type, model_name, user_id)
        agent = Agent(
            name=f"llm-node-{node_id[:8]}",
            model=f"{provider_type}:{model_name}",
            instructions=node_data.get("system_prompt", ""),
            max_steps=1,
        )
        output = await agent.run(input=prompt, provider=provider)
        response_text = output.text or ""
        usage_dict = {
            "prompt_tokens": output.usage.input_tokens if output.usage else 0,
            "completion_tokens": output.usage.output_tokens if output.usage else 0,
            "total_tokens": output.usage.total_tokens if output.usage else 0,
        }
        return {
            "node_id": node_id,
            "node_type": "llm",
            "label": node_data.get("label", ""),
            "result": response_text,
            "logs": (
                f"provider: {provider_type}\nmodel: {model_name}\n"
                f"prompt: {prompt[:200]}\nresponse: {response_text[:500]}"
            ),
            "token_usage": usage_dict,
        }
    except AgentRuntimeError as exc:
        _log.warning("LLM node %s failed: %s", node_id, exc)
        return {
            "node_id": node_id,
            "node_type": "llm",
            "label": node_data.get("label", ""),
            "result": f"Error: {exc}",
            "logs": f"provider: {provider_type}\nmodel: {model_name}\nerror: {exc}",
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    except Exception as exc:
        _log.warning("LLM node %s unexpected error: %s", node_id, exc)
        raise


async def _execute_code_node(
    node: dict[str, Any],
    edges: list[dict[str, Any]],
    variables: dict[str, Any],
    *,
    incoming_map: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Execute a ``code`` workflow node via the existing sandbox service.

    Runs in a thread-pool executor so the blocking subprocess call does not
    block the event loop.  Upstream outputs are injected as the ``_upstream``
    variable in the sandbox environment by prepending a binding line.
    """
    import asyncio

    from exo_web.services.sandbox import SandboxConfig, execute_code

    node_data = node.get("data", {})
    node_id = node["id"]
    code = node_data.get("code", "")
    timeout_seconds = int(node_data.get("timeout_seconds", 30))

    upstream_text = _gather_upstream_inputs(node_id, edges, variables, incoming_map=incoming_map)

    # Inject upstream data as a Python variable so user code can access it.
    if upstream_text:
        injected = f"_upstream = {upstream_text!r}\n"
        full_code = injected + code
    else:
        full_code = code

    config = SandboxConfig(timeout_seconds=timeout_seconds)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, execute_code, full_code, config)

    if result.success:
        output_text = result.stdout or "(no output)"
        logs = (
            f"stdout: {result.stdout}\nstderr: {result.stderr}\n"
            f"execution_time_ms: {result.execution_time_ms}\ncode: {code[:200]}"
        )
        return {
            "node_id": node_id,
            "node_type": "code",
            "label": node_data.get("label", ""),
            "result": output_text,
            "logs": logs,
        }
    else:
        error_detail = result.error or result.stderr or "Unknown error"
        logs = (
            f"stdout: {result.stdout}\nstderr: {result.stderr}\n"
            f"error: {error_detail}\nexecution_time_ms: {result.execution_time_ms}\n"
            f"code: {code[:200]}"
        )
        return {
            "node_id": node_id,
            "node_type": "code",
            "label": node_data.get("label", ""),
            "result": f"Error: {error_detail}",
            "logs": logs,
        }


async def _execute_api_node(
    node: dict[str, Any],
    edges: list[dict[str, Any]],
    variables: dict[str, Any],
    *,
    incoming_map: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Execute an ``api`` workflow node by making a real HTTP request via httpx.

    Node config keys:
    - ``url``: target URL (required)
    - ``method``: HTTP method (default ``GET``)
    - ``headers``: dict of request headers (optional)
    - ``body``: request body string or JSON-serialisable object (optional)
    - ``timeout_seconds``: per-request timeout (default 30)
    """
    try:
        import httpx
    except ImportError as exc:
        node_data = node.get("data", {})
        node_id = node["id"]
        _log.error("httpx not available for api node %s: %s", node_id, exc)
        return {
            "node_id": node_id,
            "node_type": "api",
            "label": node_data.get("label", ""),
            "result": "Error: httpx is not installed",
            "logs": str(exc),
        }

    node_data = node.get("data", {})
    node_id = node["id"]

    url = node_data.get("url", "").strip()
    method = node_data.get("method", "GET").upper()
    headers: dict[str, str] = node_data.get("headers") or {}
    body = node_data.get("body")
    timeout_seconds = float(node_data.get("timeout_seconds", 30))

    if not url:
        return {
            "node_id": node_id,
            "node_type": "api",
            "label": node_data.get("label", ""),
            "result": "Error: api node has no url configured",
            "logs": "Missing url in node config",
        }

    # Normalise body — if it's a dict/list, serialise to JSON.
    content: str | bytes | None = None
    if body is not None:
        if isinstance(body, (dict, list)):
            content = json.dumps(body)
            headers.setdefault("Content-Type", "application/json")
        else:
            content = str(body)

    try:
        async with httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=True) as client:
            response = await client.request(
                method,
                url,
                headers=headers or None,
                content=content,
            )
        status_code = response.status_code
        try:
            response_body = response.text
        except Exception:
            response_body = ""
        result_text = f"{status_code} {response.reason_phrase}\n{response_body}"
        logs = (
            f"request: {method} {url}\nstatus: {status_code}\nresponse_body: {response_body[:500]}"
        )
        return {
            "node_id": node_id,
            "node_type": "api",
            "label": node_data.get("label", ""),
            "result": result_text,
            "logs": logs,
        }
    except httpx.TimeoutException as exc:
        _log.warning("API node %s timed out: %s", node_id, exc)
        return {
            "node_id": node_id,
            "node_type": "api",
            "label": node_data.get("label", ""),
            "result": f"Error: request timed out after {timeout_seconds}s",
            "logs": f"request: {method} {url}\nerror: {exc}",
        }
    except httpx.RequestError as exc:
        _log.warning("API node %s request error: %s", node_id, exc)
        return {
            "node_id": node_id,
            "node_type": "api",
            "label": node_data.get("label", ""),
            "result": f"Error: {exc}",
            "logs": f"request: {method} {url}\nerror: {exc}",
        }


async def _execute_node(
    node: dict[str, Any],
    edges: list[dict[str, Any]] | None = None,
    variables: dict[str, Any] | None = None,
    event_callback: Any | None = None,
    *,
    run_id: str = "",
    user_id: str = "",
    cancel_event: asyncio.Event | None = None,
    incoming_map: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Execute a single workflow node and return its output.

    For agent nodes (agent_node, sub_agent), delegates to ``_execute_agent_node``
    which calls the real Exo Agent runtime. Other node types have real
    implementations (llm, code, api) or fall through to a generic stub.

    ``incoming_map`` (pre-built via ``_build_incoming_edge_map``) accelerates
    upstream-input gathering from O(E) to O(1) per node (L-16).

    Args:
        node: The workflow node dict with ``id``, ``data``, etc.
        edges: All workflow edges (used to gather upstream inputs).
        variables: Outputs from already-executed upstream nodes.
        event_callback: Optional async callable for streaming events.
        incoming_map: Optional pre-built node_id -> [source_id, ...] map.

    Returns a dict with keys: node_id, node_type, label, result, and optionally
    logs (str) and token_usage (dict) for inspection support.
    """
    node_type = node.get("data", {}).get("nodeType", node.get("type", "unknown"))
    node_data = node.get("data", {})
    edges = edges or []
    variables = variables or {}

    # --- Agent nodes: real execution via AgentService ---
    if node_type in _AGENT_NODE_TYPES:
        upstream_text = _gather_upstream_inputs(
            node["id"], edges, variables, incoming_map=incoming_map
        )
        return await _execute_agent_node(node, upstream_text, event_callback)

    # --- Knowledge retrieval node ---
    if node_type == _RETRIEVAL_NODE_TYPE:
        query_text = _gather_upstream_inputs(
            node["id"], edges, variables, incoming_map=incoming_map
        )
        return await _execute_retrieval_node(node, query_text)

    # --- Approval gate node ---
    if node_type == _APPROVAL_GATE_TYPE:
        return await _execute_approval_gate(
            node,
            run_id=run_id,
            user_id=user_id,
            event_callback=event_callback,
            cancel_event=cancel_event,
        )

    # --- LLM node: real model/agent completion ---
    if node_type == "llm":
        return await _execute_llm_node(
            node, edges, variables, event_callback, user_id=user_id, incoming_map=incoming_map
        )

    # --- Code node: real sandboxed execution ---
    if node_type == "code":
        return await _execute_code_node(node, edges, variables, incoming_map=incoming_map)

    # --- API node: real HTTP request via httpx ---
    if node_type == "api":
        return await _execute_api_node(node, edges, variables, incoming_map=incoming_map)

    # --- Unknown / unimplemented node types ---
    await asyncio.sleep(0.01)
    return {
        "node_id": node["id"],
        "node_type": node_type,
        "label": node_data.get("label", ""),
        "result": f"Executed {node_type} node",
    }


# ---------------------------------------------------------------------------
# Retry / error-handling / timeout wrapper
# ---------------------------------------------------------------------------


def _get_node_error_config(node: dict[str, Any]) -> dict[str, Any]:
    """Extract retry/error/timeout config from node data."""
    data = node.get("data", {})
    return {
        "retry_count": min(max(int(data.get("retry_count", _DEFAULT_RETRY_COUNT)), 0), 5),
        "retry_delay_ms": int(data.get("retry_delay_ms", _DEFAULT_RETRY_DELAY_MS)),
        "on_error": data.get("on_error", _DEFAULT_ON_ERROR),
        "fallback_value": data.get("fallback_value", ""),
        "timeout_ms": int(data.get("timeout_ms", _DEFAULT_TIMEOUT_MS)),
    }


async def _execute_node_with_retry(
    node: dict[str, Any],
    edges: list[dict[str, Any]],
    variables: dict[str, Any],
    event_callback: Any | None = None,
    *,
    run_id: str = "",
    user_id: str = "",
    cancel_event: asyncio.Event | None = None,
    incoming_map: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Execute a node with retry logic, timeout, and on_error strategies.

    Returns the node result dict. For ``on_error='skip'``, returns a result
    with status 'skipped'. For ``on_error='fallback'``, returns a result
    using the configured fallback_value.

    Raises on ``on_error='fail'`` after exhausting retries.
    """
    cfg = _get_node_error_config(node)
    retry_count = cfg["retry_count"]
    retry_delay_ms = cfg["retry_delay_ms"]
    on_error = cfg["on_error"]
    fallback_value = cfg["fallback_value"]
    timeout_ms = cfg["timeout_ms"]

    node_id = node["id"]
    node_data = node.get("data", {})
    node_type = node_data.get("nodeType", node.get("type", "unknown"))

    last_error: Exception | None = None

    for attempt in range(retry_count + 1):
        try:
            # Apply per-node timeout
            timeout_s = timeout_ms / 1000.0
            result = await asyncio.wait_for(
                _execute_node(
                    node,
                    edges=edges,
                    variables=variables,
                    event_callback=event_callback,
                    run_id=run_id,
                    user_id=user_id,
                    cancel_event=cancel_event,
                    incoming_map=incoming_map,
                ),
                timeout=timeout_s,
            )
            # Tag result with retry info for logging
            result["_retry_attempt"] = attempt
            result["_max_retries"] = retry_count
            result["_on_error"] = on_error
            return result

        except TimeoutError:
            last_error = TimeoutError(
                f"Node {node_id} timed out after {timeout_ms}ms (attempt {attempt + 1}/{retry_count + 1})"
            )
            _log.warning("Node %s timed out (attempt %d/%d)", node_id, attempt + 1, retry_count + 1)

        except asyncio.CancelledError:
            raise  # Don't retry on cancellation

        except Exception as exc:
            last_error = exc
            _log.warning(
                "Node %s failed (attempt %d/%d): %s",
                node_id,
                attempt + 1,
                retry_count + 1,
                exc,
            )

        # Log the retry attempt
        if attempt < retry_count:
            delay_s = (retry_delay_ms / 1000.0) * (2**attempt)  # Exponential backoff
            _log.info(
                "Retrying node %s in %.1fs (attempt %d/%d)",
                node_id,
                delay_s,
                attempt + 2,
                retry_count + 1,
            )

            if event_callback is not None:
                try:
                    await event_callback(
                        {
                            "type": "node_retry",
                            "node_id": node_id,
                            "attempt": attempt + 2,
                            "max_attempts": retry_count + 1,
                            "delay_ms": int(delay_s * 1000),
                            "error": str(last_error),
                        }
                    )
                except Exception as exc:
                    _log.warning("WebSocket send failed: %s", exc)

            await asyncio.sleep(delay_s)

    # All retries exhausted — apply on_error strategy
    error_msg = str(last_error)

    if on_error == "skip":
        _log.info("Node %s: on_error=skip — marking as skipped", node_id)
        return {
            "node_id": node_id,
            "node_type": node_type,
            "label": node_data.get("label", ""),
            "result": None,
            "logs": f"Skipped after {retry_count + 1} attempt(s): {error_msg}",
            "_retry_attempt": retry_count,
            "_max_retries": retry_count,
            "_on_error": on_error,
            "_skipped": True,
        }

    if on_error == "fallback":
        _log.info("Node %s: on_error=fallback — using fallback_value", node_id)
        return {
            "node_id": node_id,
            "node_type": node_type,
            "label": node_data.get("label", ""),
            "result": fallback_value,
            "logs": f"Fallback after {retry_count + 1} attempt(s): {error_msg}\nfallback_value: {fallback_value}",
            "_retry_attempt": retry_count,
            "_max_retries": retry_count,
            "_on_error": on_error,
            "_fallback": True,
        }

    # on_error == "fail" (default) — re-raise the last error
    raise last_error  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Single-node execution
# ---------------------------------------------------------------------------


async def execute_single_node(
    run_id: str,
    workflow_id: str,
    user_id: str,
    node: dict[str, Any],
    mock_input: dict[str, Any],
) -> dict[str, Any]:
    """Execute a single node in isolation with mock input data.

    Creates a workflow_run record and a single workflow_run_logs entry,
    then returns the execution result.
    """
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    # Mark run as running.
    await _update_run_status(run_id, "running", started_at=now)

    # Merge mock_input into node data so the node executor sees it.
    node_with_input = {**node, "data": {**node.get("data", {}), **mock_input}}

    log_id = str(uuid.uuid4())
    input_snapshot = json.dumps(node_with_input.get("data", {}))

    async with get_db() as db:
        await db.execute(
            "INSERT INTO workflow_run_logs (id, run_id, node_id, status, started_at, input_json) VALUES (?, ?, ?, 'running', ?, ?)",
            (log_id, run_id, node["id"], now, input_snapshot),
        )
        await db.commit()

    try:
        output = await _execute_node(node_with_input, edges=[], variables={})
        completed_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        logs_text = output.pop("logs", None)
        token_usage = output.pop("token_usage", None)
        token_usage_json = json.dumps(token_usage) if token_usage else None

        async with get_db() as db:
            await db.execute(
                "UPDATE workflow_run_logs SET status = 'completed', output_json = ?, logs_text = ?, token_usage_json = ?, completed_at = ? WHERE run_id = ? AND node_id = ?",
                (json.dumps(output), logs_text, token_usage_json, completed_at, run_id, node["id"]),
            )
            await db.commit()

        aggregates = await _compute_run_aggregates(run_id)
        await _update_run_status(run_id, "completed", completed_at=completed_at, **aggregates)

        return {
            "run_id": run_id,
            "node_id": node["id"],
            "status": "completed",
            "output": output,
            "logs": logs_text,
            "token_usage": token_usage,
        }

    except Exception as exc:
        error_msg = str(exc)
        completed_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        async with get_db() as db:
            await db.execute(
                "UPDATE workflow_run_logs SET status = 'failed', error = ?, completed_at = ? WHERE run_id = ? AND node_id = ?",
                (error_msg, completed_at, run_id, node["id"]),
            )
            await db.commit()

        aggregates = await _compute_run_aggregates(run_id)
        await _update_run_status(
            run_id, "failed", completed_at=completed_at, error=error_msg, **aggregates
        )

        return {
            "run_id": run_id,
            "node_id": node["id"],
            "status": "failed",
            "error": error_msg,
        }


# ---------------------------------------------------------------------------
# Run manager — tracks active runs for cancellation
# ---------------------------------------------------------------------------

_active_runs: dict[str, asyncio.Event] = {}


def _register_run(run_id: str) -> asyncio.Event:
    """Register a run and return its cancellation event."""
    cancel_event = asyncio.Event()
    _active_runs[run_id] = cancel_event
    return cancel_event


def _unregister_run(run_id: str) -> None:
    _active_runs.pop(run_id, None)


def cancel_run(run_id: str) -> bool:
    """Signal cancellation for a run. Returns True if the run was found."""
    event = _active_runs.get(run_id)
    if event is None:
        return False
    event.set()
    return True


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


async def execute_workflow(
    run_id: str,
    workflow_id: str,
    user_id: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    event_callback: Any | None = None,
) -> str:
    """Execute a workflow DAG and persist results.

    ``event_callback`` is an optional async callable receiving event dicts
    for real-time streaming (e.g., over WebSocket).

    Returns the final status: 'completed', 'failed', or 'cancelled'.
    """

    cancel_event = _register_run(run_id)

    _log.info("Workflow %s starting: workflow=%s nodes=%d", run_id, workflow_id, len(nodes))

    async def emit(event: dict[str, Any]) -> None:
        if event_callback is not None:
            try:
                await event_callback(event)
            except Exception as exc:
                _log.warning("WebSocket send failed: %s", exc)

    # Budget check — warn at threshold, pause at 100%.
    budget_result = await check_budget(user_id)
    if budget_result is not None:
        if budget_result.status == "exceeded":
            error = f"Budget exceeded ({budget_result.percent_used:.1f}% of ${budget_result.budget_amount:.2f})"
            await _update_run_status(run_id, "failed", error=error)
            await emit({"type": "execution_completed", "status": "failed", "error": error})
            _unregister_run(run_id)
            return "failed"
        if budget_result.status == "warning":
            await emit(
                {
                    "type": "budget_warning",
                    "percent_used": budget_result.percent_used,
                    "budget_amount": budget_result.budget_amount,
                    "spent": budget_result.spent,
                }
            )

    try:
        layers = topological_sort(nodes, edges)
    except ValueError as exc:
        await _update_run_status(run_id, "failed", error=str(exc))
        await emit({"type": "execution_completed", "status": "failed", "error": str(exc)})
        _unregister_run(run_id)
        return "failed"

    node_map = {n["id"]: n for n in nodes}

    # Build incoming-edge map once (O(E)) so each node's upstream lookup is O(1) — L-16.
    incoming_map = _build_incoming_edge_map(edges)

    # Inject runtime context (_user_id, _workflow_id) into agent nodes so
    # the agent service can resolve providers and tools.
    for n in nodes:
        nt = n.get("data", {}).get("nodeType", "")
        if nt in _AGENT_NODE_TYPES:
            n.setdefault("data", {})["_user_id"] = user_id
            n["data"]["_workflow_id"] = workflow_id

    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    # Mark run as running.
    await _update_run_status(run_id, "running", started_at=now)

    final_status = "completed"
    variables: dict[str, Any] = {}

    for layer in layers:
        # Check cancellation before each layer.
        if cancel_event.is_set():
            final_status = "cancelled"
            break

        # Execute all nodes in this layer concurrently.
        tasks: dict[str, asyncio.Task[dict[str, Any]]] = {}
        node_start_times: dict[str, float] = {}
        for nid in layer:
            node = node_map.get(nid)
            if node is None:
                continue

            node_type = node.get("data", {}).get("nodeType", node.get("type", "unknown"))
            _log.debug("Executing node %s (type=%s)", nid, node_type)

            # Create log entry with input snapshot for inspection.
            log_id = str(uuid.uuid4())
            started = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
            input_snapshot = json.dumps(node.get("data", {}))
            cfg = _get_node_error_config(node)
            async with get_db() as db:
                await db.execute(
                    "INSERT INTO workflow_run_logs (id, run_id, node_id, status, started_at, input_json, retry_attempt, max_retries, on_error) VALUES (?, ?, ?, 'running', ?, ?, 0, ?, ?)",
                    (
                        log_id,
                        run_id,
                        nid,
                        started,
                        input_snapshot,
                        cfg["retry_count"],
                        cfg["on_error"],
                    ),
                )
                await db.commit()

            await emit({"type": "node_started", "node_id": nid})
            node_start_times[nid] = time.monotonic()
            tasks[nid] = asyncio.create_task(
                _execute_node_with_retry(
                    node,
                    edges=edges,
                    variables=variables,
                    event_callback=emit,
                    run_id=run_id,
                    user_id=user_id,
                    cancel_event=cancel_event,
                    incoming_map=incoming_map,
                )
            )

        # Gather results.
        for nid, task in tasks.items():
            try:
                if cancel_event.is_set():
                    task.cancel()
                    final_status = "cancelled"
                    continue

                output = await task
                elapsed_ms = (time.monotonic() - node_start_times.get(nid, time.monotonic())) * 1000
                completed_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

                logs_text = output.pop("logs", None)
                token_usage = output.pop("token_usage", None)
                token_usage_json = json.dumps(token_usage) if token_usage else None
                retry_attempt = output.pop("_retry_attempt", 0)
                max_retries = output.pop("_max_retries", 0)
                on_error = output.pop("_on_error", "fail")
                is_skipped = output.pop("_skipped", False)
                is_fallback = output.pop("_fallback", False)

                if is_skipped:
                    node_status = "skipped"
                elif is_fallback:
                    node_status = "completed"
                else:
                    node_status = "completed"

                async with get_db() as db:
                    await db.execute(
                        "UPDATE workflow_run_logs SET status = ?, output_json = ?, logs_text = ?, token_usage_json = ?, completed_at = ?, retry_attempt = ?, max_retries = ?, on_error = ? WHERE run_id = ? AND node_id = ?",
                        (
                            node_status,
                            json.dumps(output),
                            logs_text,
                            token_usage_json,
                            completed_at,
                            retry_attempt,
                            max_retries,
                            on_error,
                            run_id,
                            nid,
                        ),
                    )
                    await db.commit()

                # For skipped nodes, pass None as value to downstream
                variables[nid] = output.get("result", "") if not is_skipped else None

                _log.info("Node %s completed in %.1fms", nid, elapsed_ms)

                event_type = "node_skipped" if is_skipped else "node_completed"
                await emit(
                    {
                        "type": event_type,
                        "node_id": nid,
                        "output": output,
                        "variables": variables,
                        **({"retry_attempts": retry_attempt} if retry_attempt > 0 else {}),
                    }
                )

            except asyncio.CancelledError:
                async with get_db() as db:
                    await db.execute(
                        "UPDATE workflow_run_logs SET status = 'skipped' WHERE run_id = ? AND node_id = ?",
                        (run_id, nid),
                    )
                    await db.commit()
                final_status = "cancelled"

            except Exception as exc:
                error_msg = str(exc)
                completed_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

                _log.error("Node %s failed: %s", nid, exc, exc_info=True)

                async with get_db() as db:
                    await db.execute(
                        "UPDATE workflow_run_logs SET status = 'failed', error = ?, completed_at = ? WHERE run_id = ? AND node_id = ?",
                        (error_msg, completed_at, run_id, nid),
                    )
                    await db.commit()

                await emit({"type": "node_failed", "node_id": nid, "error": error_msg})
                final_status = "failed"
                break  # Stop on first failure (on_error='fail' after retries exhausted)

        if final_status in ("failed", "cancelled"):
            break

    # Finalize.
    completed_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    error_msg = None if final_status == "completed" else (f"Workflow {final_status}")
    aggregates = await _compute_run_aggregates(run_id)
    await _update_run_status(
        run_id, final_status, completed_at=completed_at, error=error_msg, **aggregates
    )

    # Update last_run_at on the workflow itself.
    async with get_db() as db:
        await db.execute(
            "UPDATE workflows SET last_run_at = ? WHERE id = ?",
            (completed_at, workflow_id),
        )
        await db.commit()

    await emit({"type": "execution_completed", "status": final_status, "variables": variables})
    _unregister_run(run_id)
    return final_status


# ---------------------------------------------------------------------------
# Debug execution — step-through mode
# ---------------------------------------------------------------------------

# Tracks active debug sessions: run_id -> command queue
_debug_sessions: dict[str, asyncio.Queue[dict[str, Any]]] = {}


def register_debug_session(run_id: str) -> asyncio.Queue[dict[str, Any]]:
    """Register a debug session and return its command queue."""
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    _debug_sessions[run_id] = q
    return q


def unregister_debug_session(run_id: str) -> None:
    _debug_sessions.pop(run_id, None)


def send_debug_command(run_id: str, command: dict[str, Any]) -> bool:
    """Send a command to a debug session. Returns True if session found."""
    q = _debug_sessions.get(run_id)
    if q is None:
        return False
    q.put_nowait(command)
    return True


async def execute_workflow_debug(
    run_id: str,
    workflow_id: str,
    user_id: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    command_queue: asyncio.Queue[dict[str, Any]],
    event_callback: Any | None = None,
) -> str:
    """Execute a workflow in debug mode — pauses before each node.

    The ``command_queue`` receives debug commands from the client:
    - ``{action: 'continue'}`` — execute the current node and advance
    - ``{action: 'skip'}`` — skip the current node and advance
    - ``{action: 'stop'}`` — abort execution
    - ``{action: 'set_breakpoint', node_id: str}`` — toggle breakpoint on a node
    - ``{action: 'set_variable', name: str, value: Any}`` — modify a variable

    Returns the final status: 'completed', 'failed', or 'cancelled'.
    """

    cancel_event = _register_run(run_id)
    breakpoints: set[str] = set()
    variables: dict[str, Any] = {}

    async def emit(event: dict[str, Any]) -> None:
        if event_callback is not None:
            try:
                await event_callback(event)
            except Exception as exc:
                _log.warning("WebSocket send failed: %s", exc)

    try:
        layers = topological_sort(nodes, edges)
    except ValueError as exc:
        await _update_run_status(run_id, "failed", error=str(exc))
        await emit({"type": "execution_completed", "status": "failed", "error": str(exc)})
        _unregister_run(run_id)
        return "failed"

    node_map = {n["id"]: n for n in nodes}

    # Build incoming-edge map once (O(E)) so each node's upstream lookup is O(1) — L-16.
    incoming_map = _build_incoming_edge_map(edges)

    # Inject runtime context into agent nodes.
    for n in nodes:
        nt = n.get("data", {}).get("nodeType", "")
        if nt in _AGENT_NODE_TYPES:
            n.setdefault("data", {})["_user_id"] = user_id
            n["data"]["_workflow_id"] = workflow_id

    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    await _update_run_status(run_id, "running", started_at=now)

    final_status = "completed"

    # Debug mode steps through layers rather than individual nodes.
    # Nodes within a layer are still executed concurrently (matching normal
    # execution semantics) — we only gate at layer boundaries.  This ensures
    # side-effecting nodes inside the same parallel layer behave identically
    # to non-debug mode (L-12).
    for layer in layers:
        if cancel_event.is_set():
            final_status = "cancelled"
            break

        layer_node_ids = [nid for nid in layer if node_map.get(nid) is not None]
        if not layer_node_ids:
            continue

        # Announce the layer so the client knows which nodes are about to run.
        await emit(
            {
                "type": "debug_paused",
                "layer": layer_node_ids,
                # Also include the first node_id for backward-compat with
                # single-node step UIs.
                "node_id": layer_node_ids[0],
                "variables": variables,
                "breakpoints": list(breakpoints),
            }
        )

        # Wait for a single layer-level command from the client.
        action = await _wait_for_debug_command(command_queue, cancel_event, breakpoints, variables)

        if action == "stop" or cancel_event.is_set():
            final_status = "cancelled"
            break

        if action == "skip":
            # Skip every node in this layer.
            for nid in layer_node_ids:
                log_id = str(uuid.uuid4())
                started = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
                async with get_db() as db:
                    await db.execute(
                        "INSERT INTO workflow_run_logs (id, run_id, node_id, status, started_at, completed_at)"
                        " VALUES (?, ?, ?, 'skipped', ?, ?)",
                        (log_id, run_id, nid, started, started),
                    )
                    await db.commit()
                await emit({"type": "node_skipped", "node_id": nid})
            continue

        # action == "continue" — execute all nodes in this layer concurrently
        # (same semantics as normal execution mode).
        tasks: dict[str, asyncio.Task[dict[str, Any]]] = {}
        node_start_times: dict[str, float] = {}

        for nid in layer_node_ids:
            node = node_map[nid]
            log_id = str(uuid.uuid4())
            started = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
            input_snapshot = json.dumps(node.get("data", {}))
            cfg = _get_node_error_config(node)
            async with get_db() as db:
                await db.execute(
                    "INSERT INTO workflow_run_logs"
                    " (id, run_id, node_id, status, started_at, input_json, retry_attempt, max_retries, on_error)"
                    " VALUES (?, ?, ?, 'running', ?, ?, 0, ?, ?)",
                    (
                        log_id,
                        run_id,
                        nid,
                        started,
                        input_snapshot,
                        cfg["retry_count"],
                        cfg["on_error"],
                    ),
                )
                await db.commit()

            await emit({"type": "node_started", "node_id": nid})
            node_start_times[nid] = time.monotonic()
            tasks[nid] = asyncio.create_task(
                _execute_node_with_retry(
                    node,
                    edges=edges,
                    variables=variables,
                    event_callback=emit,
                    run_id=run_id,
                    user_id=user_id,
                    cancel_event=cancel_event,
                    incoming_map=incoming_map,
                )
            )

        for nid, task in tasks.items():
            try:
                if cancel_event.is_set():
                    task.cancel()
                    final_status = "cancelled"
                    continue

                output = await task
                completed_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

                logs_text = output.pop("logs", None)
                token_usage = output.pop("token_usage", None)
                token_usage_json = json.dumps(token_usage) if token_usage else None
                retry_attempt = output.pop("_retry_attempt", 0)
                max_retries = output.pop("_max_retries", 0)
                on_error = output.pop("_on_error", "fail")
                is_skipped = output.pop("_skipped", False)
                output.pop("_fallback", False)

                node_status = "skipped" if is_skipped else "completed"

                async with get_db() as db:
                    await db.execute(
                        "UPDATE workflow_run_logs SET status = ?, output_json = ?, logs_text = ?,"
                        " token_usage_json = ?, completed_at = ?, retry_attempt = ?, max_retries = ?,"
                        " on_error = ? WHERE run_id = ? AND node_id = ?",
                        (
                            node_status,
                            json.dumps(output),
                            logs_text,
                            token_usage_json,
                            completed_at,
                            retry_attempt,
                            max_retries,
                            on_error,
                            run_id,
                            nid,
                        ),
                    )
                    await db.commit()

                # Store node output in variables for downstream inspection.
                variables[nid] = output.get("result", "") if not is_skipped else None

                event_type = "node_skipped" if is_skipped else "node_completed"
                await emit(
                    {
                        "type": event_type,
                        "node_id": nid,
                        "output": output,
                        "variables": variables,
                        **({"retry_attempts": retry_attempt} if retry_attempt > 0 else {}),
                    }
                )

            except asyncio.CancelledError:
                async with get_db() as db:
                    await db.execute(
                        "UPDATE workflow_run_logs SET status = 'skipped' WHERE run_id = ? AND node_id = ?",
                        (run_id, nid),
                    )
                    await db.commit()
                final_status = "cancelled"

            except Exception as exc:
                error_msg = str(exc)
                completed_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

                async with get_db() as db:
                    await db.execute(
                        "UPDATE workflow_run_logs SET status = 'failed', error = ?, completed_at = ?"
                        " WHERE run_id = ? AND node_id = ?",
                        (error_msg, completed_at, run_id, nid),
                    )
                    await db.commit()

                await emit({"type": "node_failed", "node_id": nid, "error": error_msg})
                final_status = "failed"
                break

        if final_status in ("failed", "cancelled"):
            break

    # Finalize.
    completed_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    error_msg = None if final_status == "completed" else f"Workflow {final_status}"
    aggregates = await _compute_run_aggregates(run_id)
    await _update_run_status(
        run_id, final_status, completed_at=completed_at, error=error_msg, **aggregates
    )

    async with get_db() as db:
        await db.execute(
            "UPDATE workflows SET last_run_at = ? WHERE id = ?",
            (completed_at, workflow_id),
        )
        await db.commit()

    await emit({"type": "execution_completed", "status": final_status, "variables": variables})
    _unregister_run(run_id)
    unregister_debug_session(run_id)
    return final_status


async def _wait_for_debug_command(
    command_queue: asyncio.Queue[dict[str, Any]],
    cancel_event: asyncio.Event,
    breakpoints: set[str],
    variables: dict[str, Any],
) -> str:
    """Wait for a continue/skip/stop command, processing side-effect commands inline.

    Side-effect commands (set_breakpoint, set_variable) are handled immediately
    and we keep waiting for a flow-control command.

    Returns: 'continue', 'skip', or 'stop'.
    """
    while True:
        if cancel_event.is_set():
            return "stop"

        try:
            cmd = await asyncio.wait_for(command_queue.get(), timeout=0.5)
        except TimeoutError:
            continue

        action = cmd.get("action", "")

        if action == "set_breakpoint":
            node_id = cmd.get("node_id", "")
            if node_id in breakpoints:
                breakpoints.discard(node_id)
            else:
                breakpoints.add(node_id)
            continue

        if action == "set_variable":
            name = cmd.get("name", "")
            value = cmd.get("value")
            if name:
                if name in variables:
                    _log.warning(
                        "Debug set_variable is overwriting existing node output '%s' — "
                        "this may corrupt workflow state",
                        name,
                    )
                variables[name] = value
            continue

        if action in ("continue", "skip", "stop"):
            return action

        # Unknown action — ignore and keep waiting.


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


async def _update_run_status(
    run_id: str,
    status: str,
    *,
    started_at: str | None = None,
    completed_at: str | None = None,
    error: str | None = None,
    step_count: int | None = None,
    total_tokens: int | None = None,
    total_cost: float | None = None,
) -> None:
    """Update the status (and optional timestamps/aggregates) of a workflow run."""
    fields = ["status = ?"]
    values: list[Any] = [status]

    if started_at:
        fields.append("started_at = ?")
        values.append(started_at)
    if completed_at:
        fields.append("completed_at = ?")
        values.append(completed_at)
    if error is not None:
        fields.append("error = ?")
        values.append(error)
    if step_count is not None:
        fields.append("step_count = ?")
        values.append(step_count)
    if total_tokens is not None:
        fields.append("total_tokens = ?")
        values.append(total_tokens)
    if total_cost is not None:
        fields.append("total_cost = ?")
        values.append(total_cost)

    values.append(run_id)

    async with get_db() as db:
        await db.execute(
            f"UPDATE workflow_runs SET {', '.join(fields)} WHERE id = ?",
            values,
        )
        await db.commit()

    # When a run reaches a terminal state, process the queue for waiting runs.
    if status in ("completed", "failed", "cancelled"):
        from exo_web.services.run_queue import process_queue

        task = asyncio.create_task(process_queue())
        _queue_tasks.add(task)
        task.add_done_callback(_queue_tasks.discard)


async def _compute_run_aggregates(run_id: str) -> dict[str, Any]:
    """Compute step_count and total_tokens from node execution logs."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT COUNT(*) as cnt FROM workflow_run_logs WHERE run_id = ?",
            (run_id,),
        )
        step_count = (await cursor.fetchone())["cnt"]

        cursor = await db.execute(
            "SELECT token_usage_json FROM workflow_run_logs WHERE run_id = ? AND token_usage_json IS NOT NULL",
            (run_id,),
        )
        rows = await cursor.fetchall()

    total_tokens = 0
    for row in rows:
        usage = json.loads(row["token_usage_json"])
        total_tokens += usage.get("total_tokens", 0)

    return {"step_count": step_count, "total_tokens": total_tokens, "total_cost": 0.0}
