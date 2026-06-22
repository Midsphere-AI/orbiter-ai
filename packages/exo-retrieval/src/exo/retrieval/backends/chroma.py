"""ChromaDB vector store backend.

Requires the ``chromadb`` package::

    pip install exo-retrieval[chroma]
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from exo.retrieval.types import (  # pyright: ignore[reportMissingImports]
    Chunk,
    RetrievalError,
    RetrievalResult,
)
from exo.retrieval.vector_store import VectorStore  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)

try:
    import chromadb  # type: ignore[import-untyped]
except ImportError as exc:
    msg = (
        "chromadb is required for ChromaVectorStore. "
        "Install it with: pip install exo-retrieval[chroma]"
    )
    raise ImportError(msg) from exc

_DEFAULT_COLLECTION = "exo_vectors"

# Flat scalar metadata fields stored directly on ChromaDB metadata records.
# These support exact-match filtering via ChromaDB's $eq operator.
_FLAT_FILTER_FIELDS = {"document_id", "chunk_index", "start_offset", "end_offset"}


class ChromaVectorStore(VectorStore):
    """ChromaDB vector store for local persistent or ephemeral vector search.

    Wraps the ChromaDB ``Collection`` API for similarity search using cosine
    distance.

    Args:
        collection_name: Name of the ChromaDB collection.
        path: Directory path for persistent storage.  When *None*, an
            ephemeral (in-memory) client is used.
        client: Optional pre-existing ``chromadb.ClientAPI`` instance.

    Notes:
        ``add()`` uses ChromaDB's ``upsert`` semantics: adding a chunk with the
        same ``(document_id, chunk_index)`` identity as an existing record will
        overwrite it rather than insert a duplicate.  This matches the behaviour
        of most other backends and is the recommended contract for ``VectorStore``
        implementations.
    """

    def __init__(
        self,
        collection_name: str = _DEFAULT_COLLECTION,
        *,
        path: str | None = None,
        client: chromadb.ClientAPI | None = None,
    ) -> None:
        self._collection_name = collection_name
        self._path = path

        if client is not None:
            self._client = client
        elif path is not None:
            self._client = chromadb.PersistentClient(path=path)
        else:
            self._client = chromadb.EphemeralClient()

        try:
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise RetrievalError(
                f"ChromaDB collection init failed: {exc}",
                context={"collection": collection_name, "path": path},
                hint=(
                    "Check that the collection name is valid, the path is writable,"
                    " and chromadb is installed correctly."
                ),
            ) from exc

    async def add(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Add (upsert) chunks with their embedding vectors.

        Uses ChromaDB's ``upsert`` so that re-adding a chunk with the same
        ``(document_id, chunk_index)`` ID overwrites the existing record.
        """
        if len(chunks) != len(embeddings):
            msg = f"Number of chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must match"
            raise RetrievalError(
                msg,
                context={"collection": self._collection_name, "operation": "add"},
                hint="Ensure embed_batch() is called on the same chunk list before calling add().",
            )

        if not chunks:
            return

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for chunk in chunks:
            chunk_id = f"{chunk.document_id}:{chunk.index}"
            ids.append(chunk_id)
            documents.append(chunk.content)
            # ChromaDB metadata must be flat (str/int/float/bool values).
            # Store full chunk info so we can reconstruct on retrieval.
            # The four scalar fields (document_id, chunk_index, start_offset,
            # end_offset) are stored as flat values so they can be used as
            # exact-match ($eq) filter targets.
            meta: dict[str, Any] = {
                "document_id": chunk.document_id,
                "chunk_index": chunk.index,
                "start_offset": chunk.start,
                "end_offset": chunk.end,
                "chunk_metadata": json.dumps(chunk.metadata),
            }
            metadatas.append(meta)

        collection = self._collection
        await asyncio.to_thread(
            collection.upsert,
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Search for similar chunks using ChromaDB cosine distance.

        Returns results ranked by similarity (highest score first).
        ChromaDB returns cosine distances; we convert to similarity via
        ``1 - distance``.

        Filtering uses ChromaDB's ``$eq`` operator on the flat scalar metadata
        fields (``document_id``, ``chunk_index``, ``start_offset``,
        ``end_offset``).  Filtering on arbitrary chunk metadata keys stored
        inside the ``chunk_metadata`` JSON blob is not supported; add those
        fields as top-level document metadata to enable filtering.
        """
        where: dict[str, Any] | None = None
        if filter:
            conditions: list[dict[str, Any]] = []
            for key, value in filter.items():
                if key not in _FLAT_FILTER_FIELDS:
                    logger.warning(
                        "ChromaVectorStore: filter key %r is not a flat metadata field "
                        "(%s); it will be ignored. Only %s support filtering.",
                        key,
                        key,
                        sorted(_FLAT_FILTER_FIELDS),
                    )
                    continue
                conditions.append({key: {"$eq": value}})
            if len(conditions) == 1:
                where = conditions[0]
            elif len(conditions) > 1:
                where = {"$and": conditions}

        collection = self._collection
        query_result = await asyncio.to_thread(
            collection.query,
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        results: list[RetrievalResult] = []

        if not query_result["ids"] or not query_result["ids"][0]:
            return results

        ids_list = query_result["ids"][0]
        documents_list = (
            query_result["documents"][0] if query_result["documents"] else [None] * len(ids_list)
        )
        metadatas_list = (
            query_result["metadatas"][0] if query_result["metadatas"] else [{}] * len(ids_list)
        )
        distances_list = (
            query_result["distances"][0] if query_result["distances"] else [0.0] * len(ids_list)
        )

        for doc, meta, distance in zip(
            documents_list, metadatas_list, distances_list, strict=False
        ):
            chunk_metadata = {}
            raw_meta = meta.get("chunk_metadata", "{}") if meta else "{}"
            if isinstance(raw_meta, str):
                chunk_metadata = json.loads(raw_meta)

            chunk = Chunk(
                document_id=meta.get("document_id", "") if meta else "",
                index=meta.get("chunk_index", 0) if meta else 0,
                content=doc or "",
                start=meta.get("start_offset", 0) if meta else 0,
                end=meta.get("end_offset", 0) if meta else 0,
                metadata=chunk_metadata,
            )
            # ChromaDB cosine distance: 0 = identical, 2 = opposite.
            # Convert to similarity: 1 - distance.
            score = 1.0 - float(distance)
            results.append(RetrievalResult(chunk=chunk, score=score))

        return results

    async def delete(self, document_id: str) -> None:
        """Delete all chunks belonging to a document."""
        collection = self._collection
        await asyncio.to_thread(
            collection.delete,
            where={"document_id": document_id},
        )

    async def clear(self) -> None:
        """Remove all stored chunks and embeddings.

        Deletes and re-creates the collection to clear all data.
        """
        client = self._client
        collection_name = self._collection_name

        def _clear() -> chromadb.Collection:  # type: ignore[return]
            client.delete_collection(collection_name)
            return client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        self._collection = await asyncio.to_thread(_clear)

    async def close(self) -> None:
        """No-op: ChromaDB clients do not require explicit cleanup.

        Provided for API symmetry with ``PgVectorStore.close()``.
        """
