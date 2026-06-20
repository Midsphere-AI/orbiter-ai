# exo-distributed

> Distributed execution for Exo agents: Redis task queue, workers, and event streaming.

`exo-distributed` offloads agent execution from the calling process to a pool of worker processes connected through Redis. Submit an agent with `distributed()`, get back a `TaskHandle`, and either await the result or subscribe to live streaming events — without changing the agent definition itself. It sits above `exo-core` and `exo-models` in the Exo stack, adding horizontal scale and optional durable execution via Temporal.

## Installation

```bash
pip install exo-distributed
# or
uv add exo-distributed
```

For durable execution with Temporal:

```bash
pip install exo-distributed[temporal]
```

## Quick start

Start Redis, then run a worker in one terminal:

```bash
export EXO_REDIS_URL=redis://localhost:6379
python -m exo.distributed.worker
```

Submit and await a task from your application:

```python
from exo import Agent
from exo.distributed import distributed

agent = Agent(name="assistant", model="openai:gpt-4o-mini",
              instructions="You are a helpful assistant.")

handle = await distributed(agent, "What is the capital of France?")
result = await handle.result()
print(result["output"])
```

### Streaming live events

```python
handle = await distributed(agent, "Summarize quantum computing", detailed=True)

async for event in handle.stream():
    if event.type == "text":
        print(event.text, end="", flush=True)
```

### Custom worker

```python
from exo.distributed import Worker
from exo.distributed.models import TaskPayload, TaskStatus

class BillingWorker(Worker):
    async def on_task_done(self, task: TaskPayload, status: TaskStatus,
                           result: str | None, error: str | None) -> None:
        if status == TaskStatus.COMPLETED:
            await bill_user(task.metadata.get("user_id"), result)

worker = BillingWorker("redis://localhost:6379", concurrency=4)
await worker.start()
```

## What's inside

- **`distributed()`** — submit an agent or swarm to the queue; returns a `TaskHandle`
- **`TaskHandle`** — awaitable handle with `.result()`, `.stream()`, `.cancel()`, and `.status()`
- **`Worker`** — claims tasks from Redis, reconstructs agents, executes via `run.stream()`, publishes events; subclass and override `on_task_done()` for custom post-task logic
- **`TaskBroker`** — Redis Streams-backed queue with consumer groups, ack/nack, and cancellation via Pub/Sub
- **`EventPublisher`** / **`EventSubscriber`** — dual-channel event delivery: Pub/Sub for live streaming, Streams for replay
- **`TaskStore`** — Redis hash-backed task status with TTL cleanup
- **`TemporalExecutor`** — optional durable execution backend (requires `temporalio`)
- **`CancellationToken`** — cooperative cancellation propagated from broker to worker to agent

## Part of [Exo](https://github.com/midsphere-ai/exo)

Get the full framework with `pip install exo-ai`.

---

MIT © Midsphere AI
