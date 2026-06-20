# exo-search

> AI-powered search engine with deep research, citations, and multi-turn chat

exo-search is a production-ready AI search pipeline that goes beyond keyword retrieval. It classifies queries, runs parallel research agents against live web sources, verifies citations, and produces grounded answers with inline references. Three quality modes let you trade latency for depth — from sub-second responses to full sequential deep research. It integrates cleanly with the rest of the Exo stack as a standalone library or as a backend service.

## Installation

```bash
pip install exo-search
# or
uv add exo-search
```

## Quick start

```python
import asyncio
from exo.search import search, search_with_details, stream, SearchConfig

async def main():
    # Simple answer string
    answer = await search("What is quantum entanglement?", mode="balanced")
    print(answer)

    # Full response with sources and follow-up suggestions
    result = await search_with_details("latest advances in fusion energy", mode="quality")
    print(result.answer)
    for source in result.sources:
        print(f"  [{source.title}] {source.url}")

    # Streaming — yields PipelineEvent and text chunks as they arrive
    async for event in stream("explain large language models", mode="speed"):
        from exo.search import PipelineEvent
        if isinstance(event, PipelineEvent):
            print(f"[{event.stage}] {event.status}")

asyncio.run(main())
```

**Quality modes** — pass as the `mode` argument:

| Mode | Behaviour |
|---|---|
| `"speed"` | Snippet-only, fastest response, no page enrichment |
| `"balanced"` | Enriches top sources, parallel research agents (default) |
| `"quality"` | Full-page enrichment, claim-first writing, citation verification |

Environment variables: `EXO_SEARCH_MODEL`, `EXO_SEARCH_FAST_MODEL`, `SEARXNG_URL`.

## What's inside

- **`search(query, mode)`** — returns an answer string with citations inline
- **`search_with_details(query, mode)`** — returns a `SearchResponse` with `sources`, `suggestions`, and optional `verification` stats
- **`stream(query, mode)`** — async generator yielding `PipelineEvent`, text events, and a final `SearchResponse`
- **`SearchConfig`** — dataclass for model selection, SearXNG URL, Jina/Serper keys, and per-mode tuning
- **`ConversationManager`** — multi-turn session state for follow-up queries
- **`ResearchMode`** — `StrEnum` of `SPEED`, `BALANCED`, `QUALITY`, `DEEP`

## Part of [Exo](https://github.com/midsphere-ai/exo)

Get the full framework with `pip install exo-ai`.

---

MIT © Midsphere AI
