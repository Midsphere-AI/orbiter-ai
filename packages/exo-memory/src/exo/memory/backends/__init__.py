"""Exo Memory Backends: storage implementations.

Convenience imports (no optional deps required for sqlite):
  from exo.memory import SQLiteMemoryStore          # eager — always available

Lazy top-level access (fires import on first use; optional dep must be installed):
  from exo.memory import ChromaVectorMemoryStore    # needs: chromadb
  from exo.memory import PostgresMemoryStore        # needs: asyncpg

Or import directly from the backend modules:
  from exo.memory.backends.sqlite import SQLiteMemoryStore
  from exo.memory.backends.postgres import PostgresMemoryStore
  from exo.memory.backends.vector import (
      VectorMemoryStore, ChromaVectorMemoryStore,
      Embeddings, OpenAIEmbeddings, VertexEmbeddings,
      EmbeddingProvider, OpenAIEmbeddingProvider,
      SentenceTransformerEmbeddingProvider,
  )
"""
