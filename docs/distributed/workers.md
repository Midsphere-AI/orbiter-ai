# Worker Setup and Scaling Guide

This guide covers configuring, deploying, and scaling distributed workers for `exo-distributed`.

## Prerequisites

- **Redis 7+** — Required for Redis Streams with consumer groups
- **Python 3.13+** — Required by exo packages
- **exo-distributed** — `pip install exo-distributed`
- **exo-cli** (optional) — `pip install exo-cli` for the `exo` command

## Starting a Worker

### Via CLI

```bash
exo start worker --redis-url redis://localhost:6379
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--redis-url` | `EXO_REDIS_URL` env var | Redis connection URL |
| `--concurrency` | `1` | Number of concurrent task executions per worker process |
| `--queue` | `exo:tasks` | Redis Streams queue name |
| `--worker-id` | Auto-generated | Unique worker identifier (`{hostname}-{pid}-{random}`) |

The startup banner displays the worker ID, masked Redis URL, queue name, and concurrency level.

### Via Python

```python
import asyncio
from exo.distributed.worker import Worker

worker = Worker(
    "redis://localhost:6379",
    concurrency=4,
    queue_name="exo:tasks",
    heartbeat_ttl=30,
)

asyncio.run(worker.start())
```

### Worker Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `redis_url` | `str` | (required) | Redis connection URL |
| `worker_id` | `str \| None` | Auto-generated | Unique worker identifier |
| `concurrency` | `int` | `1` | Concurrent task execution slots |
| `queue_name` | `str` | `"exo:tasks"` | Redis Streams queue name |
| `heartbeat_ttl` | `int` | `30` | Heartbeat key TTL in seconds |
| `executor` | `Literal["local", "temporal"]` | `"local"` | Execution backend |
| `provider_factory` | `Callable[[str], Any] \| None` | `None` | Custom provider factory (receives model string, returns provider) |

## Provider Factory

By default, the worker auto-resolves an LLM provider from the agent's model string using the model registry. For scenarios requiring token refresh, custom endpoints, or per-request credentials, pass a `provider_factory` callable:

```python
worker = Worker(
    "redis://localhost:6379",
    provider_factory=lambda model: create_provider(model, api_key=get_fresh_token()),
)
```

The factory receives the model string (e.g., `"openai:gpt-4o"`) and must return a provider with an async `stream()` method. When `provider_factory` is `None`, standard auto-resolution is used.

## Post-Task Callback (on_task_done)

Override `on_task_done()` in a `Worker` subclass to run cleanup logic after every task:

```python
from exo.distributed.worker import Worker
from exo.distributed.models import TaskPayload, TaskStatus

class MyWorker(Worker):
    async def on_task_done(self, task, status, result, error):
        """Called after every task, regardless of outcome."""
        await record_billing(task.task_id, status)
```

**Signature:**

```python
async def on_task_done(
    self,
    task: TaskPayload,
    status: TaskStatus,    # COMPLETED, FAILED, or CANCELLED
    result: str | None,    # final output text (on success)
    error: str | None,     # error message (on failure)
) -> None: ...
```

The callback fires in the `finally` block of task execution, so it runs on every outcome. Exceptions in `on_task_done` are logged but never crash the worker.

## Memory Hydration

Workers can automatically create a memory store, persist conversation data, and load prior history for multi-turn sessions. Enable this by including a `memory` key in the task metadata:

```python
handle = await distributed(
    agent,
    "Continue our conversation",
    metadata={
        "memory": {
            "backend": "short_term",     # "short_term", "sqlite", or "postgres"
            "dsn": "sqlite:///memory.db", # only for sqlite/postgres backends
            "scope": {
                "user_id": "u-1",
                "session_id": "s-1",
            },
        },
    },
)
```

**What the worker does when a memory config is present:**

1. Creates a `MemoryStore` from the backend config (short-term, SQLite, or Postgres)
2. Attaches `MemoryPersistence` hooks to auto-save `AIMemory` and `ToolMemory` during execution
3. Saves the user input as `HumanMemory`
4. Loads prior conversation history from the store and prepends it to the message list
5. Tears down the store in the `finally` block (even on failure)

This requires the `exo-memory` package: `pip install exo-distributed[memory]`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EXO_REDIS_URL` | None | Default Redis URL when `--redis-url` is not provided |
| `TEMPORAL_HOST` | `localhost:7233` | Temporal server address (only for `executor="temporal"`) |
| `TEMPORAL_NAMESPACE` | `default` | Temporal namespace (only for `executor="temporal"`) |

## Concurrency Tuning

The `--concurrency` flag controls how many tasks a single worker process executes in parallel. Each concurrent slot runs an independent `_claim_loop` coroutine using `asyncio.create_task()`.

**Guidelines:**

- **CPU-bound agents** (heavy tool execution): Keep concurrency low (1-2). Python's GIL limits true parallelism.
- **I/O-bound agents** (LLM API calls, network tools): Increase concurrency (4-16). Async I/O benefits from overlapping waits.
- **Memory considerations**: Each concurrent task reconstructs an agent from its config and streams events. Monitor memory usage per process.

```bash
# Low concurrency for CPU-heavy agents
exo start worker --concurrency 1

# Higher concurrency for I/O-bound LLM agents
exo start worker --concurrency 8
```

**Horizontal vs. vertical scaling:**

- Increase `--concurrency` to use more of a single machine's resources (vertical)
- Run multiple worker processes across machines for true horizontal scaling (see [Multi-Worker Deployment](#multi-worker-deployment))
- All workers in the same consumer group share the queue — Redis distributes tasks evenly

## Multi-Worker Deployment

Multiple worker processes can connect to the same Redis queue. Redis Streams consumer groups ensure each task is claimed by exactly one worker.

### Multiple processes on one machine

```bash
# Terminal 1
exo start worker --redis-url redis://redis:6379 --concurrency 4

# Terminal 2
exo start worker --redis-url redis://redis:6379 --concurrency 4
```

Each process auto-generates a unique worker ID (`{hostname}-{pid}-{random}`).

### Process manager (systemd)

```ini
# /etc/systemd/system/exo-worker@.service
[Unit]
Description=Exo Distributed Worker %i
After=network.target redis.service

[Service]
Type=simple
User=exo
Environment=EXO_REDIS_URL=redis://localhost:6379
ExecStart=/usr/local/bin/exo start worker --concurrency 4
Restart=always
RestartSec=5
KillSignal=SIGTERM
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```

Start multiple instances:

```bash
sudo systemctl enable --now exo-worker@1
sudo systemctl enable --now exo-worker@2
```

## Graceful Shutdown

Workers handle `SIGINT` and `SIGTERM` for graceful shutdown:

1. The shutdown event is set, stopping the claim loop from accepting new tasks
2. Currently executing tasks run to completion
3. The heartbeat loop stops
4. Redis connections (broker, store, publisher) are closed

Press `Ctrl+C` or send `SIGTERM` to stop a worker cleanly. Avoid `SIGKILL` — it skips cleanup and may leave tasks in a `RUNNING` state without acknowledgment.

## Heartbeat and Health Monitoring

Workers publish health data to Redis every `heartbeat_ttl / 3` seconds (default: every 10 seconds).

**Health data published to `exo:workers:{worker_id}` Redis hash:**

| Field | Description |
|-------|-------------|
| `status` | Worker status (`running`) |
| `tasks_processed` | Total successfully completed tasks |
| `tasks_failed` | Total failed tasks |
| `current_task_id` | Currently executing task ID (empty if idle) |
| `started_at` | Worker start timestamp |
| `last_heartbeat` | Last heartbeat timestamp |
| `concurrency` | Configured concurrency |
| `hostname` | Machine hostname |

The heartbeat key has a TTL equal to `heartbeat_ttl` (default 30s). If a worker crashes without graceful shutdown, the key expires and the worker is considered dead after 60 seconds.

### Checking worker health

**Via CLI:**

```bash
# List all active workers
exo worker list

# Output:
# ┌────────────────────────┬────────┬──────────┬───────┬────────┬──────────────┬─────────────┬─────────────────────────┐
# │ Worker ID              │ Status │ Hostname │ Tasks │ Failed │ Current Task │ Concurrency │ Last Heartbeat          │
# ├────────────────────────┼────────┼──────────┼───────┼────────┼──────────────┼─────────────┼─────────────────────────┤
# │ web01-1234-a1b2c3d4    │ running│ web01    │ 42    │ 2      │ -            │ 4           │ 2026-02-17 10:30:15 UTC │
# │ web02-5678-e5f6g7h8    │ running│ web02    │ 38    │ 1      │ abc123def    │ 4           │ 2026-02-17 10:30:12 UTC │
# └────────────────────────┴────────┴──────────┴───────┴────────┴──────────────┴─────────────┴─────────────────────────┘
```

**Via Python:**

```python
from exo.distributed.health import get_worker_fleet_status, WorkerHealthCheck

# Fleet-wide status
workers = await get_worker_fleet_status("redis://localhost:6379")
for w in workers:
    print(f"{w.worker_id}: {w.status} (alive={w.alive})")

# Individual worker health check (sync, implements HealthCheck protocol)
check = WorkerHealthCheck("redis://localhost:6379", "web01-1234-a1b2c3d4")
result = check.check()
print(f"{result.status}: {result.message}")
```

Workers with heartbeat older than 60 seconds are marked as `alive=False` (dead) in fleet status.

## Task Retry Behavior

The `TaskBroker` supports automatic retries with `max_retries=3` by default.

**Retry flow:**

1. Task execution fails with an exception
2. Worker sets task status to `FAILED` and records the error
3. Worker checks current retry count against `max_retries`
4. If retries remain: status set to `RETRYING`, task is nacked (re-enqueued for any worker)
5. If retries exhausted: task is acknowledged and remains in `FAILED` state

**Monitoring retries:**

```bash
# Check retry count for a specific task
exo task status <task_id>

# List all retrying tasks
exo task list --status retrying
```

## Execution Backends

### Local Execution (default)

The worker reconstructs the agent from its serialized config and runs `run.stream()` directly. Suitable for most use cases.

```bash
exo start worker --redis-url redis://localhost:6379
```

### Temporal Execution (durable)

For tasks that must survive worker crashes, use the Temporal execution backend. Temporal wraps agent execution in durable workflows with heartbeating activities.

**Requirements:**

- Temporal server running
- `temporalio` installed: `pip install exo-distributed[temporal]`

```bash
TEMPORAL_HOST=localhost:7233 \
TEMPORAL_NAMESPACE=default \
exo start worker --redis-url redis://localhost:6379
```

Or via Python:

```python
worker = Worker(
    "redis://localhost:6379",
    executor="temporal",
    concurrency=4,
)
asyncio.run(worker.start())
```

**How it works:**

1. Worker claims task from Redis queue (same as local mode)
2. Instead of executing directly, submits an `AgentExecutionWorkflow` to Temporal
3. Temporal activity (`execute_agent_activity`) reconstructs the agent and runs `run.stream()`
4. The activity heartbeats progress (`{"step", "chars"}`) every 10 events for liveness
   detection, and stops promptly when Temporal cancels the run
5. If the worker crashes, Temporal reassigns the activity to another worker
6. Failures are mapped onto Temporal `ApplicationError`s whose `non_retryable` flag respects
   exo's error taxonomy — deterministic errors (config/parse/validation) are not retried,
   transient ones (rate limits, network) are
7. `timeout_seconds` from `TaskPayload` sets the activity `start_to_close_timeout` unless an
   explicit `TimeoutConfig` is supplied

!!! note "Crash-resumable, not auto-checkpoint-restore"
    The whole agent run executes inside a **single activity**. Temporal re-runs a crashed
    activity **from the start** (it does not replay it step-by-step), so this tier is
    *crash-resumable via heartbeat checkpointing* — a dead worker's task is picked up by
    another — but it is **not** Temporal's automatic per-step checkpoint-restore. That
    requires lifting the agent loop into the workflow and is a later phase.

#### Production connection, encryption, retries, and tuning

Construct a `TemporalExecutor` directly to configure Temporal Cloud (TLS / mTLS / API key),
encrypt payloads at rest, and set default retry/timeout/tuning policies. All configuration is
grouped and additive — flat `host=`/`namespace=` still work, or pass a `ConnectionConfig`.

```python
from exo.distributed.temporal import (
    AESGCMPayloadCodec,
    ConnectionConfig,
    RetryConfig,
    TemporalExecutor,
    TimeoutConfig,
    WorkerTuningConfig,
)

executor = TemporalExecutor(
    # Temporal Cloud over TLS with an API key:
    connection=ConnectionConfig(
        host="my-ns.acct.tmprl.cloud:7233",
        namespace="my-ns.acct",
        api_key="…",          # implies TLS
        # or mTLS: client_cert=…, client_private_key=…, server_root_ca_cert=…
    ),
    # Encrypt agent payloads (prompts, configs, outputs) at rest in Temporal history:
    codec=AESGCMPayloadCodec.from_env("EXO_TEMPORAL_CODEC_KEY"),
    # Default per-task activity retry (deterministic errors marked non-retryable by default):
    retry=RetryConfig(maximum_attempts=5),
    # Activity- and workflow-level timeouts:
    timeouts=TimeoutConfig(start_to_close_seconds=300, execution_timeout_seconds=3600),
    # Map worker concurrency onto Temporal's tuner:
    tuning=WorkerTuningConfig.from_concurrency(8),
    # OTel + exo-baggage interceptors give workflow→activity trace continuity (on by default):
    interceptors=True,
)
```

`ConnectionConfig.from_env()` reads `TEMPORAL_HOST`, `TEMPORAL_NAMESPACE`, `TEMPORAL_API_KEY`,
`TEMPORAL_TLS`, and the `TEMPORAL_TLS_*_PATH` cert paths. Serialization uses Temporal's pydantic
data converter, so `TaskPayload`/configs/events round-trip cleanly. Large tool outputs can be
offloaded with the claim-check codec (`ClaimCheckCodec` + a `BlobStore`, e.g. `RedisBlobStore`).

## Redis Configuration

### Recommended settings

```conf
# redis.conf

# Memory — tune based on queue depth and event retention
maxmemory 1gb
maxmemory-policy noeviction

# Persistence — enable AOF for durability
appendonly yes
appendfsync everysec

# Streams — tune consumer group settings
# Default stream consumer group lag is fine for most workloads

# Connections — increase if running many workers
maxclients 10000

# Timeout — keep connections alive for workers
timeout 0
tcp-keepalive 300
```

### Memory considerations

| Redis Key | Estimated Size | Retention |
|-----------|---------------|-----------|
| Task queue (`exo:tasks`) | ~1KB per pending task | Until consumed |
| Task hashes (`exo:task:{id}`) | ~500B per task | TTL: 24 hours |
| Event streams (`exo:stream:{id}`) | ~200B per event | TTL: 1 hour |
| Worker heartbeats (`exo:workers:{id}`) | ~200B per worker | TTL: 30 seconds |
| Task index (`exo:task:index`) | ~40B per task ID | Persistent |

### Redis Sentinel / Cluster

For high availability, use Redis Sentinel or Cluster. Pass the appropriate URL:

```bash
# Sentinel
exo start worker --redis-url redis+sentinel://sentinel1:26379,sentinel2:26379/mymaster

# Standard HA
exo start worker --redis-url redis://redis-primary:6379
```

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install exo packages
RUN pip install exo-distributed exo-cli

# Copy application code (for tool resolution via importable paths)
COPY . .
RUN pip install -e .

# Default command: start a worker
CMD ["exo", "start", "worker", "--concurrency", "4"]
```

### docker-compose.yml

```yaml
version: "3.8"

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

  worker:
    build: .
    environment:
      - EXO_REDIS_URL=redis://redis:6379
    depends_on:
      redis:
        condition: service_healthy
    deploy:
      replicas: 2
    restart: unless-stopped
    stop_grace_period: 30s
    stop_signal: SIGTERM

  # Optional: Temporal server for durable execution
  temporal:
    image: temporalio/auto-setup:latest
    ports:
      - "7233:7233"
    environment:
      - DB=sqlite
    profiles:
      - temporal

  temporal-worker:
    build: .
    environment:
      - EXO_REDIS_URL=redis://redis:6379
      - TEMPORAL_HOST=temporal:7233
      - TEMPORAL_NAMESPACE=default
    command: >
      exo start worker
      --concurrency 4
    depends_on:
      redis:
        condition: service_healthy
      temporal:
        condition: service_started
    profiles:
      - temporal
    restart: unless-stopped

volumes:
  redis-data:
```

**Usage:**

```bash
# Redis-only mode (2 worker replicas)
docker compose up -d

# With Temporal for durable execution
docker compose --profile temporal up -d

# Scale workers
docker compose up -d --scale worker=4
```

## Monitoring and Alerting

### Built-in alert rules

Register pre-defined alert rules for distributed system health:

```python
from exo.distributed.alerts import register_distributed_alerts

register_distributed_alerts()
```

**Alert rules:**

| Rule | Threshold | Severity |
|------|-----------|----------|
| Queue depth high | > 100 tasks | WARNING |
| Queue depth critical | > 500 tasks | CRITICAL |
| Task failure rate | > 10% | WARNING |
| Worker count zero | = 0 workers | CRITICAL |
| Task wait time high | > 60 seconds | WARNING |

### Metrics

Workers automatically record metrics using the `exo-observability` infrastructure (OpenTelemetry when available, in-memory fallback otherwise):

| Metric | Type | Description |
|--------|------|-------------|
| `exo.distributed.tasks.submitted` | Counter | Tasks submitted to the queue |
| `exo.distributed.tasks.completed` | Counter | Successfully completed tasks |
| `exo.distributed.tasks.failed` | Counter | Failed tasks |
| `exo.distributed.tasks.cancelled` | Counter | Cancelled tasks |
| `exo.distributed.queue.depth` | Gauge | Current queue depth |
| `exo.distributed.task.duration` | Histogram | Task execution duration (seconds) |
| `exo.distributed.task.wait_time` | Histogram | Time from submission to execution start |

### Task management CLI

```bash
# List all tasks
exo task list

# Filter by status
exo task list --status running
exo task list --status failed

# Get detailed status for a specific task
exo task status <task_id>

# Cancel a running task
exo task cancel <task_id>
```

## Scaling Recommendations

| Workload | Workers | Concurrency | Notes |
|----------|---------|-------------|-------|
| Development | 1 | 1 | Single process for debugging |
| Small production | 2 | 4 | Basic HA with moderate throughput |
| Medium production | 4-8 | 4-8 | Balance across machines |
| High throughput | 10+ | 8-16 | Monitor Redis memory and connections |

**Key scaling factors:**

1. **Queue depth** — If queue depth grows consistently, add more workers
2. **Task wait time** — If tasks wait >60s before execution, add capacity
3. **Worker failure rate** — High failure rates may indicate resource constraints, not a need for more workers
4. **Redis connections** — Each worker uses 3+ Redis connections (broker, store, publisher, heartbeat). At scale, tune `maxclients` in Redis

## Troubleshooting

### Worker not claiming tasks

- Verify Redis is reachable: `redis-cli -u $EXO_REDIS_URL ping`
- Check the queue name matches between client and worker (`--queue` flag)
- Verify the consumer group exists: `redis-cli XINFO GROUPS exo:tasks`

### Tasks stuck in RUNNING

- Check if the worker crashed: `exo worker list` — dead workers show in red
- Tasks from dead workers remain in the pending entries list (PEL). They can be reclaimed via `XCLAIM` or will expire based on task hash TTL (24h)

### High memory usage

- Reduce event stream TTL (default 1 hour)
- Reduce task hash TTL (default 24 hours)
- Monitor queue depth — a growing queue means workers can't keep up

### Worker heartbeat expired

- The worker may be blocked on a long-running synchronous operation
- Increase `heartbeat_ttl` if tasks legitimately take a long time
- Ensure agents don't have blocking synchronous code in tools
