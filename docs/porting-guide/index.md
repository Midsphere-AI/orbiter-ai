# Porting Guide: agent-core (openJiuwen) → Exo

## 1. Introduction

**agent-core** is the core SDK of [openJiuwen](https://github.com/OpenJiuwen), a Chinese-origin open-source AI Agent framework (Apache 2.0, Python 3.11+). It provides a comprehensive, monolithic toolkit for building self-optimizing agents with features spanning LLM abstraction, multi-agent orchestration, Pregel-based graph execution, enterprise-grade RAG pipelines, context engineering, multi-dimensional memory, and security guardrails.

**Exo** is a modular Python monorepo that reimplements and extends agent-core's most valuable features as independent, composable packages. Rather than porting agent-core wholesale, Exo selectively adapted the strongest patterns — typed rails, context processors, memory evolution, retrieval pipelines — while replacing others with simpler alternatives (e.g., Swarm transfers instead of Pregel channels).

This guide documents every architectural decision made during the port, providing migration paths for developers familiar with either framework.

For a detailed feature-by-feature comparison, see [agent-core-vs-exo-analysis.md](../../agent-core-vs-exo-analysis.md).

## 2. Architecture Comparison

### agent-core: Single-Package Monolith

agent-core ships as a single `openjiuwen` package (~540+ files) where all features are bundled together:

```
openjiuwen/
├── llm/                # LLM abstraction, prompts, KV stores
├── tool/               # Tool system, vector store tools
├── agent/              # ReActAgent, ControllerAgent
├── agent_group/        # Multi-agent coordination
├── workflow/           # Pregel-based DAG execution
├── memory/             # 5-type taxonomy, encryption, deduplication
├── context_engine/     # Offloading, compression, windowing
├── retrieval/          # Embeddings, vector stores, hybrid search, reranking
├── session/            # Session lifecycle, checkpointing, streaming
├── runner/             # Global resource registry, callbacks
├── security/           # Guardrails, risk assessment
├── agent_evolution/    # LLM-driven optimization of prompts/tools/memory
├── deep_agent/         # Autonomous task execution (file/shell/code/todo)
└── extensions/         # Redis, Pulsar, context evolution algorithms
```

### Exo: Modular Monorepo

Exo splits these concerns across 18 independently installable packages:

```
packages/
├── exo-core/         # Agent, Swarm, Tool, HookManager, Runner
├── exo-models/       # ModelProvider ABC, 11 provider adapters
├── exo-memory/       # Typed memory hierarchy, stores, search
├── exo-context/      # ProcessorPipeline, token budgeting
├── exo-guardrail/    # RiskAssessment, guardrail backends
├── exo-retrieval/    # Embeddings, vector stores, retrievers
├── exo-train/        # Operator, Optimizer, textual gradient tuning
├── exo-sandbox/      # Isolated code execution
├── exo-eval/         # Evaluation framework
├── exo-mcp/          # Model Context Protocol integration
├── exo-a2a/          # Agent-to-Agent protocol
├── exo-server/       # FastAPI server runtime
├── exo-cli/          # Command-line interface
├── exo-distributed/  # Redis task queue, workers
├── exo-observability/ # OpenTelemetry logging, tracing, metrics
├── exo-search/   # Search-focused agent example
└── exo/              # Meta-package (convenience re-exports)
```

### Package Mapping Diagram

```
agent-core (openjiuwen/)            Exo (packages/)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

llm/, tool/, agent/          ──→    exo-core + exo-models
    runner/                  ──→    exo-core (Runner)
workflow/                    ──→    exo-core (Swarm, BranchNode, LoopNode)
memory/                      ──→    exo-memory
context_engine/              ──→    exo-context
security/                    ──→    exo-guardrail
retrieval/                   ──→    exo-retrieval
agent_evolution/             ──→    exo-train
deep_agent/                  ──→    exo-sandbox + exo-core (Tools)
session/                     ──→    exo-core (state) + exo-observability
agent_group/                 ──→    exo-core (Swarm handoffs)
extensions/redis             ──→    exo-distributed
extensions/context_evolver   ──→    exo-memory (evolution strategies)
```

## 3. Design Philosophy Differences

| Dimension | agent-core | Exo |
|-----------|-----------|---------|
| **Package structure** | Single package, all-or-nothing | Modular monorepo, install what you need |
| **Resource management** | Global `Runner` singleton with registry | Dependency injection through package hierarchy |
| **Agent composition** | Inheritance-heavy (`ReActAgent` → `ControllerAgent`) | Composition via Swarm transfers + HookManager |
| **Workflow execution** | Pregel graph with super-steps, channels, barriers | Swarm DSL (`"a >> b >> c"`) + topological sort |
| **Context lifecycle** | Stateful `ContextManager` orchestrates all processing | Event-driven `ProcessorPipeline` — processors are independent |
| **Agent lifecycle guards** | `AgentRail` callbacks with untyped context dict | Typed `Rail` ABC with Pydantic event models |
| **Memory model** | Flat `MemoryItem` with 5 string types | Typed hierarchy (`SystemMemory`, `HumanMemory`, `AIMemory`, `ToolMemory`) |
| **Security** | Pluggable guardrail backends on callback framework | `BaseGuardrail` integrates via `HookManager` |
| **Evolution/optimization** | LLM-driven Operators (prompt, tool, memory tuning) | Textual gradient optimization + genetic algorithms |
| **Observability** | Session tracing with spans | Full OpenTelemetry (logging, tracing, metrics, cost tracking) |
| **Provider support** | 4 providers (OpenAI-compatible, DashScope, SiliconFlow) | 11 providers (OpenAI, Anthropic, Gemini, Vertex, Ollama, etc.) |

## 4. Package Mapping Table

| agent-core Directory | Purpose | Exo Package(s) |
|---------------------|---------|-------------------|
| `openjiuwen/llm/` | LLM abstraction, prompt templates | `exo-models` (ModelProvider ABC, adapters) |
| `openjiuwen/tool/` | Tool definitions, execution | `exo-core` (Tool, ToolResult) |
| `openjiuwen/agent/` | ReActAgent, ControllerAgent | `exo-core` (Agent, Swarm) |
| `openjiuwen/agent_group/` | Multi-agent coordination | `exo-core` (Swarm handoffs) |
| `openjiuwen/workflow/` | Pregel-based DAG execution | `exo-core` (Swarm, BranchNode, LoopNode, GroupNode) |
| `openjiuwen/memory/` | 5-type memory, encryption, dedup | `exo-memory` (MemoryCategory, typed stores, SearchManager) |
| `openjiuwen/context_engine/` | Offloading, compression, windowing | `exo-context` (ProcessorPipeline, processors) |
| `openjiuwen/retrieval/` | Embeddings, vector stores, search | `exo-retrieval` (EmbeddingProvider, VectorStore, Retriever) |
| `openjiuwen/security/` | Guardrails, risk assessment | `exo-guardrail` (BaseGuardrail, RiskAssessment) |
| `openjiuwen/session/` | Session lifecycle, checkpointing | `exo-core` (WorkflowState, checkpointing) |
| `openjiuwen/runner/` | Global resource registry, callbacks | `exo-core` (Runner, HookManager) |
| `openjiuwen/agent_evolution/` | LLM-driven prompt/tool optimization | `exo-train` (Operator, BaseOptimizer, OperatorTrainer) |
| `openjiuwen/deep_agent/` | Autonomous file/shell/code/todo tools | `exo-sandbox` + `exo-core` (Tool definitions) |
| `openjiuwen/extensions/redis/` | Redis checkpointer, distributed | `exo-distributed` (Redis task queue, workers) |
| `openjiuwen/extensions/context_evolver/` | ACE, ReasoningBank, ReMe algorithms | `exo-memory` (MemoryEvolutionStrategy subclasses) |

## 5. Per-Epic Porting Guides

| Epic | Guide | Summary |
|------|-------|---------|
| 1 — Security Guardrails | [guardrails.md](guardrails.md) | Event-driven content moderation with RiskAssessment models and multiple detection backends |
| 2 — Advanced Workflow Engine | [workflow-engine.md](workflow-engine.md) | Pregel graph engine replaced by Swarm orchestrator with flow DSL, branch/loop/group nodes |
| 3 — RAG/Retrieval Pipeline | [rag-retrieval.md](rag-retrieval.md) | Complete RAG pipeline mapping: embeddings, vector stores, 5 retriever types, reranking |
| 4 — Context Engine | [context-engine.md](context-engine.md) | Monolithic ContextManager decomposed into composable ProcessorPipeline with pluggable processors |
| 5 — Enhanced Memory System | [memory-system.md](memory-system.md) | 5-type memory taxonomy with AES-256 encryption mapped to typed hierarchy with pluggable stores |
| 6 — Typed Agent Rails | [rails.md](rails.md) | Rail lifecycle guards with 10 callback events mapped to typed Rails with Pydantic models |
| 7 — Task Management | [task-management.md](task-management.md) | TaskManager and TaskScheduler mapped to Pydantic-based TaskManager with mid-execution steering |
| 8 — Deep Agent Toolkit | [deep-agent-toolkit.md](deep-agent-toolkit.md) | Autonomous toolkit (file I/O, shell, code, todo) distributed across Exo's tool system and sandbox |
| 9 — Context Evolution | [context-evolver.md](context-evolver.md) | Three memory evolution strategies (ACE, ReasoningBank, ReMe) mapped to composable strategy subclasses |
| 10 — Operator & Self-Optimization | [operator-optimization.md](operator-optimization.md) | Iterative optimization system mapped to exo-train with Operator ABC and textual gradient tuning |
| 11 — DeepAgent Example Port | [../design/deepagent-port-plan.md](../design/deepagent-port-plan.md) | Multi-agent ReAct system ported from openjiuwen to Exo (Agent, Swarm, MCP) |

## 6. Acknowledgments

This porting effort builds extensively on the work of the [openJiuwen](https://github.com/OpenJiuwen) project and its contributors. The agent-core SDK represents a thoughtful and comprehensive approach to AI agent development, and many of Exo's features — particularly the typed rail system, context engine processors, memory evolution strategies, and retrieval pipeline architecture — were directly inspired by or adapted from openJiuwen's designs.

We are grateful to the openJiuwen team for releasing their work under the Apache 2.0 license, making this kind of cross-project knowledge transfer possible.
