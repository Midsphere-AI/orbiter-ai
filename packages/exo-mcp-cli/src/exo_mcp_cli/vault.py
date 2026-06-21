"""Backward-compatible re-export of ``exo.mcp.vault``.

The canonical implementation now lives in ``exo-mcp`` so that the
framework's MCP client can reuse it without depending on the CLI package.

All public names are re-exported here to avoid breaking existing code that
imports from ``exo_mcp_cli.vault``.
"""

from exo.mcp.vault import (  # pyright: ignore[reportMissingImports]
    Vault,
    VaultError,
    derive_key,
)

__all__: list[str] = ["Vault", "VaultError", "derive_key"]
