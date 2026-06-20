"""exo-mcp-cli: standalone CLI for MCP servers."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("exo-mcp-cli")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__: list[str] = ["__version__"]
