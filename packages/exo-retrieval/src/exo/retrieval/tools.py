"""Retrieval tools for Agent integration.

Factory functions that wrap retrievers and indexing pipelines as
``exo.tool.Tool`` instances, ready to be added to an Agent.
"""

from __future__ import annotations

from exo.models.embeddings import Embeddings  # pyright: ignore[reportMissingImports]
from exo.retrieval.chunker import Chunker  # pyright: ignore[reportMissingImports]
from exo.retrieval.retriever import Retriever  # pyright: ignore[reportMissingImports]
from exo.retrieval.types import (  # pyright: ignore[reportMissingImports]
    Document,
    RetrievalResult,
)
from exo.retrieval.vector_store import VectorStore  # pyright: ignore[reportMissingImports]
from exo.tool import FunctionTool  # pyright: ignore[reportMissingImports]


def _format_results(results: list[RetrievalResult]) -> str:
    """Format retrieval results into a human-readable string."""
    if not results:
        return "No results found."

    parts: list[str] = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] (score: {r.score:.4f}) {r.chunk.content}")
    return "\n\n".join(parts)


def retrieve_tool(retriever: Retriever, *, name: str = "retrieve") -> FunctionTool:
    """Create a tool that searches a knowledge base via a retriever.

    Args:
        retriever: The retriever instance to wrap.
        name: Override the tool name (default ``"retrieve"``).

    Returns:
        A ``FunctionTool`` that agents can use to search documents.
    """

    async def _retrieve(query: str, top_k: int = 5) -> str:
        """Search a knowledge base for relevant documents.

        Args:
            query: The search query.
            top_k: Maximum number of results to return.
        """
        results = await retriever.retrieve(query, top_k=top_k)
        return _format_results(results)

    return FunctionTool(
        _retrieve, name=name, description="Search a knowledge base for relevant documents."
    )


def index_tool(
    chunker: Chunker,
    store: VectorStore,
    embeddings: Embeddings,
    *,
    name: str = "index_document",
) -> FunctionTool:
    """Create a tool that indexes new documents into a vector store.

    Args:
        chunker: The chunker for splitting documents.
        store: The vector store to add chunks to.
        embeddings: The embeddings provider for vectorising chunks.
        name: Override the tool name (default ``"index_document"``).

    Returns:
        A ``FunctionTool`` that agents can use to index documents.
    """

    async def _index(content: str, document_id: str = "doc") -> str:
        """Index a document into the knowledge base.

        Args:
            content: The document text content to index.
            document_id: An identifier for the document.
        """
        doc = Document(id=document_id, content=content)
        chunks = chunker.chunk(doc)
        if not chunks:
            return "No chunks produced from the document."
        vecs = await embeddings.embed_batch([c.content for c in chunks])
        await store.add(chunks, vecs)
        return f"Indexed {len(chunks)} chunk(s) from document '{document_id}'."

    return FunctionTool(_index, name=name, description="Index a document into the knowledge base.")
