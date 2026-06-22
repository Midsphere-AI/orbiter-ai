"""Graph-based retriever that expands results via knowledge graph traversal.

``GraphRetriever`` wraps a base retriever and expands its results by
traversing knowledge graph triples.  Entities found in the initial
retrieval results are used to discover related triples, and the chunks
those triples originate from are included in the expanded result set.
A configurable beam search controls the breadth and depth of expansion.
"""

from __future__ import annotations

from typing import Any

from exo.retrieval.retriever import Retriever  # pyright: ignore[reportMissingImports]
from exo.retrieval.triple_extractor import Triple  # pyright: ignore[reportMissingImports]
from exo.retrieval.types import (  # pyright: ignore[reportMissingImports]
    Chunk,
    RetrievalResult,
)

# Default beam search parameters.
_DEFAULT_BEAM_WIDTH = 3
_DEFAULT_MAX_HOPS = 2

# Score decay per hop — expansion results score lower than direct hits.
_HOP_DECAY = 0.8


class GraphRetriever(Retriever):
    """Retriever that expands results via knowledge graph traversal.

    After an initial retrieval from the *base_retriever*, entities in the
    returned chunks are matched against the supplied ``triples``.  Matching
    triples provide new entities that are expanded further up to
    *max_hops*.  At each hop the *beam_width* highest-confidence triples
    are kept.

    Args:
        base_retriever: The underlying retriever used for the initial query.
        triples: Pre-extracted knowledge graph triples to traverse.
        beam_width: Maximum number of triples to expand per entity per hop.
        max_hops: Maximum traversal depth.
    """

    def __init__(
        self,
        base_retriever: Retriever,
        triples: list[Triple],
        *,
        beam_width: int = _DEFAULT_BEAM_WIDTH,
        max_hops: int = _DEFAULT_MAX_HOPS,
    ) -> None:
        self.base_retriever = base_retriever
        self.triples = triples
        self.beam_width = beam_width
        self.max_hops = max_hops
        # Index: entity (lowercased) → list of triples where it appears as subject or object.
        # Built once here so _find_triples() is O(1) per lookup instead of O(n).
        self._entity_index: dict[str, list[Triple]] = {}
        for triple in triples:
            for entity in (triple.subject.lower(), triple.object.lower()):
                self._entity_index.setdefault(entity, []).append(triple)

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        """Retrieve chunks, then expand via graph traversal.

        Args:
            query: The search query text.
            top_k: Maximum number of results to return.
            **kwargs: Passed through to the base retriever.

        Returns:
            A list of ``RetrievalResult`` objects ranked by score,
            including graph-expanded results with traversal metadata.
        """
        base_results = await self.base_retriever.retrieve(query, top_k=top_k, **kwargs)

        if not base_results or not self.triples:
            return base_results[:top_k]

        # Collect all results keyed by (document_id, index) for dedup.
        seen: dict[tuple[str, int], RetrievalResult] = {}
        for r in base_results:
            key = (r.chunk.document_id, r.chunk.index)
            seen[key] = r

        # Extract seed entities from initial results.
        seed_entities = self._extract_entities(base_results)

        # Beam search expansion over hops.
        frontier = seed_entities
        for hop in range(1, self.max_hops + 1):
            decay = _HOP_DECAY**hop
            next_frontier: set[str] = set()

            for entity in frontier:
                matching = self._find_triples(entity)
                # Keep top beam_width by confidence.
                matching.sort(key=lambda t: t.confidence, reverse=True)
                matching = matching[: self.beam_width]

                for triple in matching:
                    # The chunk that sourced this triple is a candidate.
                    chunk_key = self._parse_source_chunk_id(triple.source_chunk_id)
                    if chunk_key is not None and chunk_key not in seen:
                        doc_id, idx = chunk_key
                        expanded_chunk = Chunk(
                            document_id=doc_id,
                            index=idx,
                            content=f"{triple.subject} {triple.predicate} {triple.object}",
                            start=0,
                            end=0,
                            metadata={},
                        )
                        score = triple.confidence * decay
                        seen[chunk_key] = RetrievalResult(
                            chunk=expanded_chunk,
                            score=score,
                            metadata={
                                "graph_hop": hop,
                                "graph_triple": {
                                    "subject": triple.subject,
                                    "predicate": triple.predicate,
                                    "object": triple.object,
                                },
                                "graph_source_entity": entity,
                            },
                        )

                    # Discover new entities for the next hop.
                    other = (
                        triple.object
                        if triple.subject.lower() == entity.lower()
                        else triple.subject
                    )
                    next_frontier.add(other.lower())

            # Remove entities already explored.
            frontier = next_frontier - seed_entities
            seed_entities |= next_frontier

            if not frontier:
                break

        # Sort all results by score descending, return top_k.
        all_results = sorted(seen.values(), key=lambda r: r.score, reverse=True)
        return all_results[:top_k]

    def _extract_entities(self, results: list[RetrievalResult]) -> set[str]:
        """Extract entity names from triples whose source chunks match results."""
        result_chunks = {f"{r.chunk.document_id}:{r.chunk.index}" for r in results}
        entities: set[str] = set()
        for triple in self.triples:
            if triple.source_chunk_id in result_chunks:
                entities.add(triple.subject.lower())
                entities.add(triple.object.lower())
        return entities

    def _find_triples(self, entity: str) -> list[Triple]:
        """Find triples where entity appears as subject or object.

        Uses the pre-built ``_entity_index`` for O(1) lookup rather than
        scanning the full triple list on every call.
        """
        return list(self._entity_index.get(entity.lower(), []))

    @staticmethod
    def _parse_source_chunk_id(
        source_chunk_id: str,
    ) -> tuple[str, int] | None:
        """Parse 'document_id:index' into a (document_id, index) tuple."""
        parts = source_chunk_id.rsplit(":", 1)
        if len(parts) != 2:
            return None
        try:
            return (parts[0], int(parts[1]))
        except ValueError:
            return None
