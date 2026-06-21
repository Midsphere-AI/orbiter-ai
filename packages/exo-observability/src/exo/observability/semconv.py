"""Semantic conventions for agent, tool, LLM, and task observability.

Standardised attribute names used across spans, metrics, and log records
so that all Exo telemetry is consistent and queryable.

Ported from ``exo.trace.config`` with new cost conventions added.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# GenAI semantic conventions — gen_ai.*
# ---------------------------------------------------------------------------

# Standard OpenTelemetry GenAI semantic conventions.
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
GEN_AI_REQUEST_TOP_K = "gen_ai.request.top_k"
GEN_AI_REQUEST_FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
GEN_AI_REQUEST_PRESENCE_PENALTY = "gen_ai.request.presence_penalty"
GEN_AI_REQUEST_STOP_SEQUENCES = "gen_ai.request.stop_sequences"
GEN_AI_REQUEST_STREAMING = "gen_ai.request.streaming"
GEN_AI_PROMPT = "gen_ai.prompt"
GEN_AI_COMPLETION = "gen_ai.completion"
GEN_AI_DURATION = "gen_ai.duration"
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
GEN_AI_RESPONSE_ID = "gen_ai.response.id"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_SERVER_ADDRESS = "gen_ai.server.address"

# ---------------------------------------------------------------------------
# Agent conventions — exo.agent.*
# ---------------------------------------------------------------------------

AGENT_ID = "exo.agent.id"
AGENT_NAME = "exo.agent.name"
AGENT_TYPE = "exo.agent.type"
AGENT_MODEL = "exo.agent.model"
AGENT_STEP = "exo.agent.step"
AGENT_MAX_STEPS = "exo.agent.max_steps"
AGENT_RUN_SUCCESS = "exo.agent.run.success"

# ---------------------------------------------------------------------------
# Tool conventions — exo.tool.*
# ---------------------------------------------------------------------------

TOOL_NAME = "exo.tool.name"
TOOL_CALL_ID = "exo.tool.call_id"
TOOL_ARGUMENTS = "exo.tool.arguments"
TOOL_RESULT = "exo.tool.result"
TOOL_ERROR = "exo.tool.error"
TOOL_DURATION = "exo.tool.duration"
TOOL_STEP_SUCCESS = "exo.tool.step.success"

# ---------------------------------------------------------------------------
# Task / session / user conventions
# ---------------------------------------------------------------------------

TASK_ID = "exo.task.id"
TASK_INPUT = "exo.task.input"
SESSION_ID = "exo.session.id"
USER_ID = "exo.user.id"
TRACE_ID = "exo.trace.id"

# ---------------------------------------------------------------------------
# Cost conventions (new) — exo.cost.*
# ---------------------------------------------------------------------------

COST_INPUT_TOKENS = "exo.cost.input_tokens"
COST_OUTPUT_TOKENS = "exo.cost.output_tokens"
COST_TOTAL_USD = "exo.cost.total_usd"

# ---------------------------------------------------------------------------
# Streaming event metric names and attributes
# ---------------------------------------------------------------------------

METRIC_STREAM_EVENTS_EMITTED = "stream_events_emitted"

STREAM_EVENT_TYPE = "exo.stream.event_type"

# ---------------------------------------------------------------------------
# Span name prefixes
# ---------------------------------------------------------------------------

SPAN_PREFIX_AGENT = "agent."
SPAN_PREFIX_TOOL = "tool."
SPAN_PREFIX_LLM = "llm."
SPAN_PREFIX_TASK = "task."
