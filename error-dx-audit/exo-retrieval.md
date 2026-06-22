# exo-retrieval — Error DX & Resilience Audit

## Counts
- raise sites: 15
- error classes total / not inheriting ExoError: 1 / 0 (`RetrievalError` correctly inherits `ExoError`; `EmbeddingError` in exo-models at `packages/exo-models/src/exo/models/embeddings.py:56` inherits bare `Exception` — re-exported by this package as a public API item)
- `except Exception` sites: 2 (`retriever.py:92`, `retriever.py:97`); swallow-and-pass: 0; drop-cause: 0
- CancelledError handlers: 0 (no `except BaseException` or bare except — safe)
- I/O call sites lacking timeout/retry: 7 (pgvector pool create, pgvector queries ×4, all LLM provider calls in QueryRewriter/LLMReranker/TripleExtractor/AgenticRetriever)

## Findings (prioritized)

| # | Priority | Checklist | File:Line | Problem | Concrete Fix |
|---|----------|-----------|-----------|---------|--------------|
| 1 | P1 | #2 #3 | `retriever.py:93` | `RetrievalError("Embedding failed: {exc}")` gives no context — which embeddings provider, which query | Add `context={"provider": type(self.embeddings).__name__}` and `hint="Check the embedding provider credentials and that the model name is correct."` |
| 2 | P1 | #2 #3 | `retriever.py:98` | `RetrievalError("Vector store search failed: {exc}")` — no store type, no query text, no hint | Add `context={"store": type(self.store).__name__}` and `hint="Check the vector store is initialized and reachable (run initialize() for PgVectorStore)."` |
| 3 | P1 | #1 | `packages/exo-models/src/exo/models/embeddings.py:56` | `EmbeddingError(Exception)` — not an `ExoError` subclass; re-exported from `exo.retrieval.__init__` as part of this package's public API | Change to `EmbeddingError(ExoError)` in exo-models; add `hint` in raise sites there |
| 4 | P1 | #1 #2 | `types.py:66-83` | `RetrievalError.__init__` bypasses `ExoError`'s `context`/`hint`/`doc` fields — takes `operation` and `details` instead; `super().__init__(message)` skips the structured kwargs | Remove `operation`/`details` params; use `super().__init__(message, context=..., hint=..., doc=...)` standard fields; update raise sites to use `context={"operation": ...}` |
| 5 | P1 | #2 | `query_rewriter.py:89-94` | LLM call `provider.complete(...)` is unguarded — raw provider exceptions (auth errors, rate limits, timeouts) bubble up unhandled with no context about which model or that this was a query-rewrite step | Wrap in try/except and raise `RetrievalError("Query rewrite failed", context={"model": self.model}, hint="Check the model string and API key for the rewrite LLM.")` |
| 6 | P1 | #2 | `reranker.py:110` | LLM call in `LLMReranker.rerank()` unguarded — provider errors bubble raw; no context that this was a reranking step | Wrap with `RetrievalError`, `context={"model": self.model, "operation": "rerank"}`, hint about model/key |
| 7 | P1 | #2 | `triple_extractor.py:102` | Sequential LLM calls per chunk in `TripleExtractor.extract()` are unguarded — a provider error mid-loop silently skips remaining chunks (nothing is actually caught here, the error escapes raw) | Wrap each `provider.complete()` call; log+continue on failure per chunk or raise `RetrievalError` with `context={"model": self.model, "chunk_id": chunk_id}` |
| 8 | P1 | #2 | `agentic_retriever.py:138` | `_judge_sufficiency` LLM call unguarded; failure silently returns `0.0` from `_parse_sufficiency` — but the exception actually escapes (no try/except wraps the `provider.complete()` call) | Wrap call; on failure return `0.0` explicitly with a log message, or re-raise as `RetrievalError` with model context |
| 9 | P1 | #8 | `backends/pgvector.py:60` | `asyncpg.create_pool(self._dsn)` — no timeout, no retry, no error wrapping; raw `asyncpg` connection errors (refused, bad DSN, auth) surface to callers with no actionable hint | Wrap in try/except; raise `RetrievalError("pgvector connection failed", context={"dsn_hint": self._dsn[:20]+"..."}, hint="Check DATABASE_URL is set, the server is running, and the pgvector extension is installed.")` |
| 10 | P1 | #8 | `backends/pgvector.py:65-83` | `initialize()` runs raw `asyncpg` DDL with no error handling — extension or schema creation failures (missing superuser, wrong dimensions) surface as raw `asyncpg` exceptions | Wrap in try/except and re-raise as `RetrievalError` with table/dimensions context |
| 11 | P1 | #8 | `backends/pgvector.py:98-118` | `add()` and `delete()` execute raw asyncpg queries with no error handling or context | Wrap; raise `RetrievalError` with `context={"table": self._table, "operation": "add/delete"}` |
| 12 | P1 | #8 | `backends/pgvector.py:131-156` | `search()` executes raw asyncpg query — connection errors, dimension mismatch errors surface raw | Same fix: wrap and raise `RetrievalError` with table/query context |
| 13 | P1 | #8 | `backends/chroma.py:54-61` | `ChromaVectorStore.__init__` calls `get_or_create_collection()` synchronously — ChromaDB errors (path permissions, collection name conflicts) leak raw to callers | Wrap in try/except; raise `RetrievalError` with `context={"collection": collection_name, "path": path}` |
| 14 | P2 | #2 | `chunker.py:48-52, 109, 209-213` | `ValueError("chunk_size must be positive")` — plain `ValueError`; no hint about valid ranges, no ExoError context | Convert to `RetrievalError` with `hint="chunk_size must be a positive integer; chunk_overlap must be less than chunk_size."` |
| 15 | P2 | #2 | `vector_store.py:95`, `backends/chroma.py:71`, `backends/pgvector.py:93` | `ValueError("Number of chunks (...) and embeddings (...) must match")` — correct message, but raw `ValueError` escapes the package boundary | Convert to `RetrievalError(..., hint="Ensure embed_batch() is called on the same chunk list before calling add().")` |
| 16 | P2 | #2 | `parsers.py:229` | `PDFParser` raises `ImportError` without context; the message is actionable but bypasses `RetrievalError` | OK to keep as ImportError (install-time); no change strictly needed, but could wrap as `RetrievalError` for consistency |
| 17 | P2 | #9 | `chunker.py:227` | `except ModuleNotFoundError: self._encoder = _WhitespaceEncoder()` — silently falls back to whitespace tokenizer when tiktoken is missing; no log or warning | Add a one-time log warning so users know they're using the fallback tokenizer |
| 18 | P2 | #3 | `agentic_retriever.py:155`, `reranker.py:160`, `triple_extractor.py:129` | Parse-failure fallbacks (`pass`, return empty list, return `0.0`) are completely silent — a malformed LLM response is silently discarded | Add a `logger.debug(...)` at each fallback so developers can diagnose unexpected LLM output formats |

## Resilience gaps

| File:Line | System | Gap |
|-----------|--------|-----|
| `backends/pgvector.py:60` | asyncpg pool | No connection timeout (`timeout` kwarg to `create_pool`); no retry on transient connection refusal; no cleanup if pool creation partially succeeds |
| `backends/pgvector.py:65-83` | asyncpg DDL | `initialize()` has no timeout on DDL execution; a hung DB will block indefinitely |
| `backends/pgvector.py:98-118, 131-156, 175-188` | asyncpg queries | All DML/queries unguarded — a lost connection mid-query raises raw `asyncpg.InterfaceError` or `asyncpg.PostgresConnectionError` |
| `backends/chroma.py:54-100` | ChromaDB | All ChromaDB calls (init, upsert, query, delete) are sync and unguarded; no timeout possible; Chroma disk I/O errors leak raw |
| `query_rewriter.py:89`, `reranker.py:110`, `agentic_retriever.py:138`, `triple_extractor.py:102` | LLM provider | Zero timeout/retry/backoff on any LLM call; a rate-limit or transient 503 from the provider is immediately fatal and surfaces as a raw provider SDK exception |
| `hybrid_retriever.py:65-68` | asyncio.gather | Uses `asyncio.gather` with no `return_exceptions=True` — a failure in either retriever cancels the other silently; errors surface as an unhandled `ExceptionGroup` |
| `parsers.py:52` | File I/O | `source.read_text(encoding="utf-8")` and `source.read_bytes()` — file not found / permission errors surface as raw `OSError`; no wrapping or hint |

## Effort estimate

**M** — About 2–3 focused sessions: the taxonomy/DX fixes (items 1–5, 14–15) are mechanical find-and-replace, the I/O resilience gaps (items 9–13, hybrid gather) need per-file wrapping with actionable `RetrievalError` raises, and the LLM call guards (items 5–8) are straightforward try/except additions; no architectural changes required.
