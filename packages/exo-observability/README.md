# exo-observability

> Unified observability for Exo agents: structured logging, distributed tracing, metrics, and LLM cost tracking in one lightweight package.

exo-observability is the instrumentation layer shared by every Exo package. It provides structured logging with JSON and text formatters, an optional OpenTelemetry tracing integration that degrades gracefully to no-ops when OTel is absent, an in-process metrics collector with pre-defined agent and tool counters, and a cost tracker that estimates USD spend per LLM call from a built-in pricing table. It has no required heavy dependencies — OTel support is optional and activated automatically when `opentelemetry-sdk` is installed.

## Installation

```bash
pip install exo-observability
# or
uv add exo-observability
```

## Quick start

```python
import asyncio
from exo.observability import configure, get_logger, traced, span, get_tracker

configure(log_level="INFO", log_format="json")
logger = get_logger(__name__)

@traced("my_operation")
async def process(text: str) -> str:
    async with span("validate") as s:
        s.set_attribute("input_length", len(text))
        logger.info("processing input", extra={"length": len(text)})
    return text.upper()

async def main() -> None:
    result = await process("hello")
    tracker = get_tracker()
    print(f"Total cost so far: ${tracker.total_cost:.6f}")

asyncio.run(main())
```

## What's inside

- **`get_logger` / `configure_logging`** — structured logger factory with `TextFormatter` and `JsonFormatter`; `LogContext` for attaching request-scoped fields
- **`traced` / `span` / `aspan`** — decorator and context managers for distributed tracing; no-op when OpenTelemetry is not installed
- **`MetricsCollector`** — in-process metrics with pre-built counters and histograms for agent runs (`METRIC_AGENT_RUN_DURATION`, `METRIC_AGENT_TOKEN_USAGE`) and tool steps (`METRIC_TOOL_STEP_DURATION`)
- **`CostTracker` / `get_tracker`** — per-call USD cost estimation from a built-in model pricing table; thread-safe aggregation with `CostEntry` records
- **`HealthRegistry` / `HealthCheck`** — composable health checks (`MemoryUsageCheck`, `EventLoopCheck`) with `get_health_summary()`
- **`AlertManager` / `AlertRule`** — threshold-based alerting with configurable `AlertSeverity` and callback routing

## Part of [Exo](https://github.com/midsphere-ai/exo)

Shared instrumentation used across all Exo packages. Get the full framework with `pip install exo-ai`.

---

MIT © Midsphere AI
