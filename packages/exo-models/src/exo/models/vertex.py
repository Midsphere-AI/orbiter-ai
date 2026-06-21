"""Google Vertex AI LLM provider — compatibility shim.

``VertexProvider`` is an alias for ``GoogleProvider`` (defined in
``gemini.py``).  Both ``"gemini:..."`` and ``"vertex:..."`` model strings
resolve to the same unified provider class; the Vertex AI authentication
path is taken automatically when ``google_project`` / ``GOOGLE_CLOUD_PROJECT``
is present.

This module re-exports the shared Google helpers so that existing test imports
from ``exo.models.vertex`` continue to work without modification.
"""

from __future__ import annotations

from exo.models._google_common import (
    _build_config,
    _convert_tools,
    _map_finish_reason,
    _parse_response,
    _parse_stream_chunk,
    _to_google_contents,
)
from exo.models.gemini import GoogleProvider, _credentials_from_base64
from exo.models.provider import model_registry

# Public alias — VertexProvider IS GoogleProvider
VertexProvider = GoogleProvider

__all__ = [
    "VertexProvider",
    "_build_config",
    "_convert_tools",
    "_credentials_from_base64",
    "_map_finish_reason",
    "_parse_response",
    "_parse_stream_chunk",
    "_to_google_contents",
]

# ---------------------------------------------------------------------------
# Registration — "vertex" prefix → same unified GoogleProvider class
# ---------------------------------------------------------------------------

model_registry.register("vertex", GoogleProvider)
