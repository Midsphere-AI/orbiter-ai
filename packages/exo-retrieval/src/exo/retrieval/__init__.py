"""Exo Retrieval: Embeddings, vector stores, and RAG pipeline."""

from importlib.metadata import PackageNotFoundError, version
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

try:
    __version__: str = version("exo-retrieval")
except PackageNotFoundError:
    __version__ = "0.1.0"

from exo.models.embeddings import (  # pyright: ignore[reportMissingImports]
    EmbeddingError,
    Embeddings,
    HTTPEmbeddings,
    OpenAIEmbeddings,
    VertexEmbeddings,
    cosine_similarity,
)
from exo.retrieval.agentic_retriever import (
    AgenticRetriever,  # pyright: ignore[reportMissingImports]
)
from exo.retrieval.chunker import (  # pyright: ignore[reportMissingImports]
    CharacterChunker,
    Chunker,
    ParagraphChunker,
    TokenChunker,
)
from exo.retrieval.graph_retriever import (
    GraphRetriever,  # pyright: ignore[reportMissingImports]
)
from exo.retrieval.hybrid_retriever import (
    HybridRetriever,  # pyright: ignore[reportMissingImports]
)
from exo.retrieval.parsers import (  # pyright: ignore[reportMissingImports]
    JSONParser,
    MarkdownParser,
    Parser,
    PDFParser,
    TextParser,
)
from exo.retrieval.query_rewriter import QueryRewriter  # pyright: ignore[reportMissingImports]
from exo.retrieval.reranker import (  # pyright: ignore[reportMissingImports]
    LLMReranker,
    Reranker,
)
from exo.retrieval.retriever import (  # pyright: ignore[reportMissingImports]
    Retriever,
    VectorRetriever,
)
from exo.retrieval.sparse_retriever import (
    SparseRetriever,  # pyright: ignore[reportMissingImports]
)
from exo.retrieval.tools import (  # pyright: ignore[reportMissingImports]
    index_tool,
    retrieve_tool,
)
from exo.retrieval.triple_extractor import (  # pyright: ignore[reportMissingImports]
    Triple,
    TripleExtractor,
)
from exo.retrieval.types import (  # pyright: ignore[reportMissingImports]
    Chunk,
    Document,
    RetrievalError,
    RetrievalResult,
)
from exo.retrieval.vector_store import (  # pyright: ignore[reportMissingImports]
    InMemoryVectorStore,
    VectorStore,
)

__all__: list[str] = [
    "AgenticRetriever",
    "CharacterChunker",
    "Chunk",
    "Chunker",
    "Document",
    "EmbeddingError",
    "Embeddings",
    "GraphRetriever",
    "HTTPEmbeddings",
    "HybridRetriever",
    "InMemoryVectorStore",
    "JSONParser",
    "LLMReranker",
    "MarkdownParser",
    "OpenAIEmbeddings",
    "PDFParser",
    "ParagraphChunker",
    "Parser",
    "QueryRewriter",
    "Reranker",
    "RetrievalError",
    "RetrievalResult",
    "Retriever",
    "SparseRetriever",
    "TextParser",
    "TokenChunker",
    "Triple",
    "TripleExtractor",
    "VectorRetriever",
    "VectorStore",
    "VertexEmbeddings",
    "cosine_similarity",
    "index_tool",
    "retrieve_tool",
]
