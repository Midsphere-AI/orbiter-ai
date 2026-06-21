"""Embeddings ABC — re-exported from the canonical exo.models layer.

This module is a compatibility shim. The implementation now lives in
``exo.models.embeddings``. Import from there directly for new code.
"""

from exo.models.embeddings import Embeddings  # pyright: ignore[reportMissingImports]

__all__ = ["Embeddings"]
