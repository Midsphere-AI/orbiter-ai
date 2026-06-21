"""exo-mcp-cli: standalone CLI for MCP servers."""

from importlib.metadata import PackageNotFoundError, version

from exo.mcp.client import HttpConfig, StdioConfig  # pyright: ignore[reportMissingImports]
from exo_mcp_cli.config import MCPConfigError, ServerEntry

try:
    __version__: str = version("exo-mcp-cli")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__: list[str] = [
    "HttpConfig",
    "MCPConfigError",
    "ServerEntry",
    "StdioConfig",
    "__version__",
]
