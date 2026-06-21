"""Embedding-based vector memory store for semantic search.

Embedding infrastructure (the :class:`Embeddings` ABC and concrete providers)
is provided by ``exo-models`` and re-exported here.  This module also provides
the :class:`VectorMemoryStore` (in-process semantic search) and the
:class:`ChromaVectorMemoryStore` (ChromaDB-backed persistent semantic *agent
memory* — distinct from ``exo-retrieval``'s document-chunk vector store).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Protocol, runtime_checkable

from exo.memory.base import (  # pyright: ignore[reportMissingImports]
    MemoryItem,
    MemoryMetadata,
    MemoryStatus,
)

# ---------------------------------------------------------------------------
# Re-export canonical embedding types from exo-models
# ---------------------------------------------------------------------------
from exo.models import (  # pyright: ignore[reportMissingImports]
    Embeddings,
    OpenAIEmbeddings,  # noqa: F401 — re-exported for backward compat
    VertexEmbeddings,  # noqa: F401 — re-exported for backward compat
    cosine_similarity,
)

logger = logging.getLogger(__name__)

# Backward-compat alias — use exo.models.cosine_similarity directly when possible.
_cosine_similarity = cosine_similarity


# ---------------------------------------------------------------------------
# VectorMemoryStore
# ---------------------------------------------------------------------------


class VectorMemoryStore:
    """In-memory vector store backed by a canonical :class:`exo.models.Embeddings` provider.

    Stores memory items alongside their embedding vectors and supports
    semantic (cosine similarity) search via the ``search()`` method.

    Implements the MemoryStore protocol.

    The *embeddings* argument must satisfy the :class:`exo.models.Embeddings`
    interface (async ``embed(text) -> list[float]``).
    """

    __slots__ = ("_embeddings", "_items", "_vectors")

    def __init__(self, embeddings: Embeddings) -> None:
        self._embeddings = embeddings
        self._items: dict[str, MemoryItem] = {}
        self._vectors: dict[str, list[float]] = {}

    @property
    def embeddings(self) -> Embeddings:
        """Return the underlying embeddings provider."""
        return self._embeddings

    # -- MemoryStore protocol -------------------------------------------------

    async def add(self, item: MemoryItem) -> None:
        """Persist a memory item and compute its embedding."""
        vec = await self._embeddings.embed(item.content)
        self._items[item.id] = item
        self._vectors[item.id] = vec
        logger.debug("added item id=%s dim=%d", item.id, len(vec))

    async def get(self, item_id: str) -> MemoryItem | None:
        """Retrieve a memory item by ID."""
        return self._items.get(item_id)

    async def search(
        self,
        *,
        query: str = "",
        metadata: MemoryMetadata | None = None,
        memory_type: str | None = None,
        status: MemoryStatus | None = None,
        limit: int = 10,
    ) -> list[MemoryItem]:
        """Semantic search: embed query, rank by cosine similarity.

        Metadata, memory_type, and status filters are applied as
        post-filters on the candidate set before ranking.
        """
        candidates = list(self._items.values())

        # Apply filters
        if memory_type:
            candidates = [c for c in candidates if c.memory_type == memory_type]
        if status:
            candidates = [c for c in candidates if c.status == status]
        if metadata:
            candidates = [c for c in candidates if _matches_metadata(c, metadata)]

        if not candidates:
            return []

        # Semantic ranking
        if query:
            query_vec = await self._embeddings.embed(query)
            scored = [
                (item, cosine_similarity(query_vec, self._vectors[item.id]))
                for item in candidates
                if item.id in self._vectors
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            results = scored[:limit]
            logger.debug(
                "semantic search query=%r results=%d top_score=%.4f",
                query,
                len(results),
                results[0][1] if results else 0.0,
            )
            return [item for item, _ in results]

        # No query — return newest first
        candidates.sort(key=lambda c: c.created_at, reverse=True)
        logger.debug("search query=%r results=%d", query, min(len(candidates), limit))
        return candidates[:limit]

    async def clear(
        self,
        *,
        metadata: MemoryMetadata | None = None,
    ) -> int:
        """Remove memory items matching the filter. Returns count."""
        if metadata is None:
            count = len(self._items)
            self._items.clear()
            self._vectors.clear()
            logger.debug("cleared all items count=%d", count)
            return count

        to_remove = [
            item_id for item_id, item in self._items.items() if _matches_metadata(item, metadata)
        ]
        for item_id in to_remove:
            del self._items[item_id]
            self._vectors.pop(item_id, None)
        logger.debug("cleared filtered items count=%d", len(to_remove))
        return len(to_remove)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        dim = self._embeddings.dimension
        return f"VectorMemoryStore(items={len(self._items)}, dimension={dim})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _matches_metadata(item: MemoryItem, metadata: MemoryMetadata) -> bool:
    """Check whether an item's metadata matches the given filter."""
    m = item.metadata
    if metadata.user_id and m.user_id != metadata.user_id:
        return False
    if metadata.session_id and m.session_id != metadata.session_id:
        return False
    if metadata.task_id and m.task_id != metadata.task_id:
        return False
    return not (metadata.agent_id and m.agent_id != metadata.agent_id)




@runtime_checkable
class EmbeddingProvider(Protocol):
    """Runtime-checkable protocol for embedding providers.

    Any object with an ``async embed(text: str) -> list[float]`` method
    satisfies this protocol without explicit inheritance.
    """

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text."""
        ...


class OpenAIEmbeddingProvider:
    """OpenAI embedding provider using text-embedding-3-small.

    Reads ``OPENAI_API_KEY`` from the environment when *api_key* is not
    supplied explicitly.

    Requires the ``openai`` package to be installed.
    """

    __slots__ = ("_api_key", "_model")

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ) -> None:
        import os

        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")

    async def embed(self, text: str) -> list[float]:
        """Embed *text* asynchronously via the OpenAI API."""
        return await asyncio.to_thread(self._embed_sync, text)

    def _embed_sync(self, text: str) -> list[float]:
        import openai  # lazy import

        client = openai.OpenAI(api_key=self._api_key)
        resp = client.embeddings.create(input=text, model=self._model)
        return list(resp.data[0].embedding)


class SentenceTransformerEmbeddingProvider:
    """Local embedding provider using sentence-transformers.

    Does not require an API key; embeddings are generated locally.  Useful
    as a free fallback when no OpenAI key is available.

    Requires the ``sentence-transformers`` package to be installed
    (optional dep).
    """

    __slots__ = ("_model_name", "_st_model")

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import (
            SentenceTransformer,  # type: ignore[import-untyped]
        )

        self._model_name = model_name
        self._st_model = SentenceTransformer(model_name)

    async def embed(self, text: str) -> list[float]:
        """Embed *text* asynchronously using the local model."""
        return await asyncio.to_thread(self._embed_sync, text)

    def _embed_sync(self, text: str) -> list[float]:
        result = self._st_model.encode(text)
        return list(result.tolist())  # numpy → list[float]


# ---------------------------------------------------------------------------
# Embeddings ABC
# ---------------------------------------------------------------------------





class ChromaVectorMemoryStore:
    """ChromaDB-backed vector memory store for persistent semantic search.

    Uses an :class:`EmbeddingProvider` to generate embeddings and stores
    items in a ChromaDB collection.  Requires ``chromadb>=0.6``.

    .. code-block:: python

        store = ChromaVectorMemoryStore(OpenAIEmbeddingProvider())
        await store.add(HumanMemory(content="hello"))
        results = await store.search(query="greeting", limit=5)
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        *,
        collection_name: str = "exo_memory",
        path: str | None = None,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._collection_name = collection_name
        self._path = path
        self._client: Any = None
        self._collection: Any = None

    # -- internal helpers -----------------------------------------------------

    def _ensure_collection(self) -> Any:
        """Lazily create the ChromaDB client and collection."""
        if self._collection is None:
            import chromadb  # pyright: ignore[reportMissingImports]  # lazy import — optional dep

            if self._path:
                self._client = chromadb.PersistentClient(path=self._path)
            else:
                self._client = chromadb.EphemeralClient()
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                embedding_function=None,  # we supply our own embeddings
            )
        return self._collection

    # -- MemoryStore protocol -------------------------------------------------

    async def add(self, item: MemoryItem) -> None:
        """Embed and persist a memory item to ChromaDB."""
        collection = self._ensure_collection()
        embedding = await self._embedding_provider.embed(item.content)
        chroma_meta = _to_chroma_metadata(item)
        await asyncio.to_thread(
            collection.add,
            ids=[item.id],
            documents=[item.content],
            embeddings=[embedding],
            metadatas=[chroma_meta],
        )
        logger.debug("ChromaVectorMemoryStore: added item id=%s", item.id)

    async def get(self, item_id: str) -> MemoryItem | None:
        """Retrieve a single memory item by ID."""
        collection = self._ensure_collection()
        result = await asyncio.to_thread(
            collection.get,
            ids=[item_id],
            include=["documents", "metadatas"],
        )
        if not result["ids"]:
            return None
        return _from_chroma_row(result["ids"][0], result["documents"][0], result["metadatas"][0])

    async def search(
        self,
        *,
        query: str = "",
        metadata: MemoryMetadata | None = None,
        memory_type: str | None = None,
        status: MemoryStatus | None = None,
        limit: int = 10,
    ) -> list[MemoryItem]:
        """Search memory items.  When *query* is given, uses vector similarity."""
        collection = self._ensure_collection()
        where = _build_where_filter(metadata, memory_type, status)

        if query:
            embedding = await self._embedding_provider.embed(query)
            kwargs: dict[str, Any] = {
                "query_embeddings": [embedding],
                "n_results": limit,
                "include": ["documents", "metadatas"],
            }
            if where:
                kwargs["where"] = where
            results = await asyncio.to_thread(collection.query, **kwargs)
            items = [
                _from_chroma_row(id_, doc, meta)
                for id_, doc, meta in zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    strict=False,
                )
            ]
            logger.debug("ChromaVectorMemoryStore: vector search returned %d items", len(items))
            return items

        # No query — return newest first
        kwargs = {"include": ["documents", "metadatas"]}
        if where:
            kwargs["where"] = where
        results = await asyncio.to_thread(collection.get, **kwargs)
        items = [
            _from_chroma_row(id_, doc, meta)
            for id_, doc, meta in zip(
                results["ids"],
                results["documents"],
                results["metadatas"],
                strict=False,
            )
        ]
        items.sort(key=lambda x: x.created_at, reverse=True)
        logger.debug("ChromaVectorMemoryStore: get returned %d items", len(items[:limit]))
        return items[:limit]

    async def clear(
        self,
        *,
        metadata: MemoryMetadata | None = None,
    ) -> int:
        """Remove memory items.  Returns the number of items removed."""
        collection = self._ensure_collection()

        if metadata is None:
            # Count all, then wipe by dropping + recreating the collection
            result = await asyncio.to_thread(collection.get, include=[])
            count = len(result["ids"])
            client = self._client
            await asyncio.to_thread(client.delete_collection, self._collection_name)
            self._collection = await asyncio.to_thread(
                client.get_or_create_collection,
                name=self._collection_name,
                embedding_function=None,
            )
            logger.debug("ChromaVectorMemoryStore: cleared all count=%d", count)
            return count

        where = _build_where_filter(metadata, None, None)
        if not where:
            return await self.clear()

        result = await asyncio.to_thread(collection.get, where=where, include=[])
        count = len(result["ids"])
        if count > 0:
            await asyncio.to_thread(collection.delete, ids=result["ids"])
        logger.debug("ChromaVectorMemoryStore: cleared filtered count=%d", count)
        return count

    async def get_recent(
        self,
        n: int = 10,
        *,
        metadata: MemoryMetadata | None = None,
    ) -> list[MemoryItem]:
        """Return the *n* most recently added items, newest first."""
        collection = self._ensure_collection()
        where = _build_where_filter(metadata, None, None)
        kwargs: dict[str, Any] = {"include": ["documents", "metadatas"]}
        if where:
            kwargs["where"] = where
        results = await asyncio.to_thread(collection.get, **kwargs)
        items = [
            _from_chroma_row(id_, doc, meta)
            for id_, doc, meta in zip(
                results["ids"],
                results["documents"],
                results["metadatas"],
                strict=False,
            )
        ]
        items.sort(key=lambda x: x.created_at, reverse=True)
        return items[:n]

    def __repr__(self) -> str:
        return f"ChromaVectorMemoryStore(collection={self._collection_name!r})"


# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------


def _to_chroma_metadata(item: MemoryItem) -> dict[str, Any]:
    """Flatten a MemoryItem into a ChromaDB metadata dict."""
    extra: dict[str, Any] = {}
    if hasattr(item, "tool_calls"):
        extra["tool_calls"] = item.tool_calls  # type: ignore[attr-defined]
    if hasattr(item, "tool_call_id"):
        extra["tool_call_id"] = item.tool_call_id  # type: ignore[attr-defined]
    if hasattr(item, "tool_name"):
        extra["tool_name"] = item.tool_name  # type: ignore[attr-defined]
    if hasattr(item, "is_error"):
        extra["is_error"] = item.is_error  # type: ignore[attr-defined]

    return {
        "memory_type": item.memory_type,
        "status": item.status.value,
        "created_at": item.created_at,
        "updated_at": item.updated_at,
        "user_id": item.metadata.user_id or "",
        "session_id": item.metadata.session_id or "",
        "task_id": item.metadata.task_id or "",
        "agent_id": item.metadata.agent_id or "",
        "extra_json": json.dumps(extra),
    }


def _from_chroma_row(item_id: str, document: str, chroma_meta: dict[str, Any]) -> MemoryItem:
    """Reconstruct a MemoryItem from a ChromaDB result row."""
    from exo.memory.base import (  # pyright: ignore[reportMissingImports]
        AIMemory,
        HumanMemory,
        SystemMemory,
        ToolMemory,
    )

    extra = json.loads(chroma_meta.get("extra_json", "{}"))
    kwargs: dict[str, Any] = {
        "id": item_id,
        "content": document,
        "memory_type": chroma_meta["memory_type"],
        "status": MemoryStatus(chroma_meta["status"]),
        "metadata": MemoryMetadata(
            user_id=chroma_meta.get("user_id") or None,
            session_id=chroma_meta.get("session_id") or None,
            task_id=chroma_meta.get("task_id") or None,
            agent_id=chroma_meta.get("agent_id") or None,
        ),
        "created_at": chroma_meta["created_at"],
        "updated_at": chroma_meta["updated_at"],
    }

    memory_type = chroma_meta["memory_type"]
    if memory_type == "system":
        return SystemMemory(**kwargs)
    if memory_type == "human":
        return HumanMemory(**kwargs)
    if memory_type == "ai":
        kwargs["tool_calls"] = extra.get("tool_calls", [])
        return AIMemory(**kwargs)
    if memory_type == "tool":
        kwargs["tool_call_id"] = extra.get("tool_call_id", "")
        kwargs["tool_name"] = extra.get("tool_name", "")
        kwargs["is_error"] = extra.get("is_error", False)
        return ToolMemory(**kwargs)
    return MemoryItem(**kwargs)


def _build_where_filter(
    metadata: MemoryMetadata | None,
    memory_type: str | None,
    status: MemoryStatus | None,
) -> dict[str, Any] | None:
    """Build a ChromaDB ``where`` filter dict from optional filter criteria."""
    clauses: list[dict[str, Any]] = []

    if memory_type:
        clauses.append({"memory_type": {"$eq": memory_type}})
    if status:
        clauses.append({"status": {"$eq": status.value}})
    if metadata:
        if metadata.user_id:
            clauses.append({"user_id": {"$eq": metadata.user_id}})
        if metadata.session_id:
            clauses.append({"session_id": {"$eq": metadata.session_id}})
        if metadata.task_id:
            clauses.append({"task_id": {"$eq": metadata.task_id}})
        if metadata.agent_id:
            clauses.append({"agent_id": {"$eq": metadata.agent_id}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
