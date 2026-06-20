"""Tests for retrieval tools: retrieve_tool and index_tool."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

from exo.agent import Agent
from exo.models.types import ModelResponse
from exo.retrieval.chunker import CharacterChunker
from exo.retrieval.embeddings import Embeddings
from exo.retrieval.retriever import Retriever, VectorRetriever
from exo.retrieval.tools import index_tool, retrieve_tool
from exo.retrieval.types import Chunk, RetrievalResult
from exo.retrieval.vector_store import InMemoryVectorStore
from exo.tool import FunctionTool
from exo.types import ToolCall, Usage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(
    document_id: str = "doc-1",
    index: int = 0,
    content: str = "hello",
) -> Chunk:
    return Chunk(
        document_id=document_id,
        index=index,
        content=content,
        start=0,
        end=len(content),
    )


class MockEmbeddings(Embeddings):
    """Deterministic embeddings for testing (2-d vectors)."""

    def __init__(self, mapping: dict[str, list[float]] | None = None) -> None:
        self._mapping = mapping or {}

    async def embed(self, text: str) -> list[float]:
        if text in self._mapping:
            return self._mapping[text]
        val = float(sum(ord(c) for c in text) % 100) / 100.0
        return [val, 1.0 - val]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]

    @property
    def dimension(self) -> int:
        return 2


class StubRetriever(Retriever):
    """Returns a fixed list of results for any query."""

    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        return self._results[:top_k]


def _mock_provider(
    content: str = "Done!",
    tool_calls: list[ToolCall] | None = None,
) -> AsyncMock:
    """Create a mock provider that returns a fixed ModelResponse."""
    resp = ModelResponse(
        content=content,
        tool_calls=tool_calls or [],
        usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
    )
    provider = AsyncMock()
    provider.complete = AsyncMock(return_value=resp)
    return provider


# ---------------------------------------------------------------------------
# retrieve_tool
# ---------------------------------------------------------------------------


class TestRetrieveTool:
    def test_returns_function_tool(self) -> None:
        t = retrieve_tool(StubRetriever([]))
        assert isinstance(t, FunctionTool)
        assert t.name == "retrieve"

    def test_custom_name(self) -> None:
        t = retrieve_tool(StubRetriever([]), name="search_docs")
        assert t.name == "search_docs"

    def test_schema_has_query_and_top_k(self) -> None:
        t = retrieve_tool(StubRetriever([]))
        schema = t.to_schema()
        params = schema["function"]["parameters"]
        assert "query" in params["properties"]
        assert params["properties"]["query"]["type"] == "string"
        assert "top_k" in params["properties"]
        assert params["properties"]["top_k"]["type"] == "integer"
        # query is required, top_k has default
        assert "query" in params["required"]
        assert "top_k" not in params.get("required", [])

    async def test_execute_returns_formatted_results(self) -> None:
        results = [
            RetrievalResult(chunk=_chunk(content="first result"), score=0.95),
            RetrievalResult(chunk=_chunk(content="second result", index=1), score=0.80),
        ]
        t = retrieve_tool(StubRetriever(results))
        output = await t.execute(query="test query")
        assert isinstance(output, str)
        assert "0.9500" in output
        assert "first result" in output
        assert "0.8000" in output
        assert "second result" in output

    async def test_execute_empty_results(self) -> None:
        t = retrieve_tool(StubRetriever([]))
        output = await t.execute(query="nothing")
        assert output == "No results found."

    async def test_execute_respects_top_k(self) -> None:
        results = [
            RetrievalResult(chunk=_chunk(content=f"r{i}", index=i), score=0.9 - i * 0.1)
            for i in range(5)
        ]
        t = retrieve_tool(StubRetriever(results))
        output = await t.execute(query="test", top_k=2)
        # StubRetriever slices to top_k, so only 2 results
        assert "[1]" in output
        assert "[2]" in output
        assert "[3]" not in output


# ---------------------------------------------------------------------------
# index_tool
# ---------------------------------------------------------------------------


class TestIndexTool:
    def test_returns_function_tool(self) -> None:
        t = index_tool(CharacterChunker(chunk_size=100), InMemoryVectorStore(), MockEmbeddings())
        assert isinstance(t, FunctionTool)
        assert t.name == "index_document"

    def test_custom_name(self) -> None:
        t = index_tool(
            CharacterChunker(chunk_size=100),
            InMemoryVectorStore(),
            MockEmbeddings(),
            name="ingest",
        )
        assert t.name == "ingest"

    def test_schema_has_content_and_document_id(self) -> None:
        t = index_tool(CharacterChunker(chunk_size=100), InMemoryVectorStore(), MockEmbeddings())
        schema = t.to_schema()
        params = schema["function"]["parameters"]
        assert "content" in params["properties"]
        assert params["properties"]["content"]["type"] == "string"
        assert "document_id" in params["properties"]
        assert params["properties"]["document_id"]["type"] == "string"
        # content is required, document_id has default
        assert "content" in params["required"]
        assert "document_id" not in params.get("required", [])

    async def test_execute_indexes_document(self) -> None:
        store = InMemoryVectorStore()
        embeddings = MockEmbeddings()
        chunker = CharacterChunker(chunk_size=50, chunk_overlap=0)

        t = index_tool(chunker, store, embeddings)
        output = await t.execute(content="A" * 100, document_id="test-doc")

        assert isinstance(output, str)
        assert "chunk(s)" in output
        assert "test-doc" in output
        # Verify chunks are actually in the store
        vec = await embeddings.embed("search")
        results = await store.search(vec, top_k=10)
        assert len(results) > 0

    async def test_execute_empty_content(self) -> None:
        t = index_tool(CharacterChunker(chunk_size=100), InMemoryVectorStore(), MockEmbeddings())
        output = await t.execute(content="")
        assert output == "No chunks produced from the document."


# ---------------------------------------------------------------------------
# Agent integration — tools work with Agent via MockProvider
# ---------------------------------------------------------------------------


class TestAgentIntegration:
    async def test_agent_with_retrieve_tool(self) -> None:
        """Agent can be configured with a retrieve_tool and execute it."""
        results = [
            RetrievalResult(chunk=_chunk(content="relevant info"), score=0.92),
        ]
        rt = retrieve_tool(StubRetriever(results))

        # First call: LLM decides to call the tool
        tc = ToolCall(id="tc-1", name="retrieve", arguments='{"query": "test", "top_k": 3}')
        resp_tool = ModelResponse(
            content="",
            tool_calls=[tc],
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        # Second call: LLM produces final answer using tool result
        resp_text = ModelResponse(
            content="Based on the search, I found relevant info.",
            usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
        )
        provider = AsyncMock()
        provider.complete = AsyncMock(side_effect=[resp_tool, resp_text])

        agent = Agent(name="search-agent", tools=[rt])
        output = await agent.run("Find relevant information", provider=provider)

        assert output.text == "Based on the search, I found relevant info."
        assert provider.complete.await_count == 2

    async def test_agent_with_index_tool(self) -> None:
        """Agent can be configured with an index_tool and execute it."""
        store = InMemoryVectorStore()
        it = index_tool(CharacterChunker(chunk_size=50, chunk_overlap=0), store, MockEmbeddings())

        tc = ToolCall(
            id="tc-1",
            name="index_document",
            arguments='{"content": "Some document text to index", "document_id": "d1"}',
        )
        resp_tool = ModelResponse(
            content="",
            tool_calls=[tc],
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        resp_text = ModelResponse(
            content="Document indexed successfully.",
            usage=Usage(input_tokens=15, output_tokens=5, total_tokens=20),
        )
        provider = AsyncMock()
        provider.complete = AsyncMock(side_effect=[resp_tool, resp_text])

        agent = Agent(name="indexer", tools=[it])
        output = await agent.run("Index this document", provider=provider)

        assert output.text == "Document indexed successfully."
        # Verify the document was actually indexed in the store
        vec = await MockEmbeddings().embed("search")
        results = await store.search(vec, top_k=10)
        assert len(results) > 0

    async def test_agent_with_both_tools(self) -> None:
        """Agent can have both retrieve and index tools registered."""
        store = InMemoryVectorStore()
        emb = MockEmbeddings()
        rt = retrieve_tool(VectorRetriever(emb, store))
        it = index_tool(CharacterChunker(chunk_size=200), store, emb)

        agent = Agent(name="rag-agent", tools=[rt, it], memory=None, context=None)

        assert "retrieve" in agent.tools
        assert "index_document" in agent.tools
        schemas = agent.get_tool_schemas()
        tool_names = {s["function"]["name"] for s in schemas}
        assert tool_names == {"retrieve", "index_document"}
