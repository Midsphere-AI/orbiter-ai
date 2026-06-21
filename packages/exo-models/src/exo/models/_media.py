"""Backward-compatibility re-exports.

The multimodal content block helpers have been folded into ``_google_common``.
This module re-exports them so existing imports continue to work.
"""

from exo.models._google_common import _guess_mime_from_url, content_blocks_to_google

__all__ = ["_guess_mime_from_url", "content_blocks_to_google"]
