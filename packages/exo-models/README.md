# exo-models

> LLM provider abstractions for OpenAI, Anthropic, and compatible APIs.

`exo-models` is the provider layer of the Exo stack. It wraps OpenAI, Anthropic, Google Gemini, and Vertex AI behind a single async interface so the rest of the framework never imports a vendor SDK directly. Any package in the stack that needs to call an LLM goes through `get_provider()` and speaks only `ModelResponse` and `StreamChunk`.

## Installation

```bash
pip install exo-models
# or
uv add exo-models
```

## Quick start

```python
import asyncio
from exo.models import get_provider, ModelResponse

async def main() -> None:
    provider = get_provider("openai:gpt-4o-mini")

    response: ModelResponse = await provider.complete(
        [{"role": "user", "content": "What is 2 + 2?"}]
    )
    print(response.content)

    # Stream the same request
    async for chunk in await provider.stream(
        [{"role": "user", "content": "Count to five."}]
    ):
        print(chunk.delta, end="", flush=True)

asyncio.run(main())
```

Set the matching environment variable before running:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."          # Gemini
export GOOGLE_CLOUD_PROJECT="..."    # Vertex AI
```

## What's inside

- **`get_provider`** — factory that parses `"provider:model"` strings and returns a configured `ModelProvider` instance
- **`ModelProvider`** — abstract base class; subclass it to add a new provider (`complete()` + `stream()`)
- **`ModelResponse`** — immutable Pydantic model returned by `complete()`; carries `content`, `tool_calls`, `usage`, and `finish_reason`
- **`StreamChunk`** — incremental chunk yielded by `stream()`; carries `delta` text and `tool_call_deltas`
- **`ModelError`** — exception raised on provider failures, with `model` and `code` fields
- **`model_registry`** — global `Registry` mapping provider names to their `ModelProvider` subclasses; extend it to register custom providers

Concrete providers already registered: `OpenAIProvider`, `AnthropicProvider`, `GeminiProvider`, `VertexProvider`.

## Part of [Exo](https://github.com/midsphere-ai/exo)

`exo-models` is the provider layer; agents and tools live upstream. Get the full framework with `pip install exo-ai`.

---

MIT © Midsphere AI
