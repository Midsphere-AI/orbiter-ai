# exo-retrieval

> Retrieval framework: embeddings, vector stores, and RAG pipeline for Exo agents

exo-retrieval is a full-stack retrieval toolkit for building retrieval-augmented generation (RAG) pipelines. It covers the entire ingestion-to-query path — parsing and chunking raw documents, embedding text with OpenAI, Vertex AI, or any HTTP endpoint, storing and searching vectors, and reranking results with an LLM judge. It sits between exo-core and the LLM layer, giving agents accurate grounding without coupling them to any single backend.

## Installation

```bash
pip install exo-retrieval
# or
uv add exo-retrieval
```

## Quick start

```python
import asyncio
from exo.retrieval import (
    Document,
    CharacterChunker,
    OpenAIEmbeddings,
    InMemoryVectorStore,
    VectorRetriever,
)

async def main():
    doc = Document(id="doc-1", content="Exo is a modular multi-agent framework...")
    chunks = CharacterChunker(chunk_size=200, chunk_overlap=20).chunk(doc)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    store = InMemoryVectorStore()

    vectors = await asyncio.gather(*[embeddings.embed(c.content) for c in chunks])
    await store.add(chunks, vectors)

    retriever = VectorRetriever(embeddings, store, score_threshold=0.5)
    results = await retriever.retrieve("multi-agent framework", top_k=3)

    for r in results:
        print(f"{r.score:.3f}  {r.chunk.content[:80]}")

asyncio.run(main())
```

## What's inside

- **`VectorRetriever`** — dense semantic search: embeds a query, searches a `VectorStore`, filters by score threshold
- **`HybridRetriever`** — combines dense and sparse retrieval with configurable weighting
- **`AgenticRetriever`** — LLM-driven retrieval loop that refines queries until sufficient context is gathered
- **`CharacterChunker` / `ParagraphChunker` / `TokenChunker`** — document splitting strategies with overlap control
- **`OpenAIEmbeddings` / `VertexEmbeddings` / `HTTPEmbeddings`** — pluggable embedding providers
- **`LLMReranker`** — post-retrieval reranking using an LLM judge for precision
- **`index_tool` / `retrieve_tool`** — pre-built `@tool`-compatible functions for drop-in use inside Exo agents

## Part of [Exo](https://github.com/midsphere-ai/exo)

Get the full framework with `pip install exo-ai`.

---

MIT © Midsphere AI
