"""Exo MCP: Model Context Protocol client and tools."""

__version__: str = "0.1.0"

from exo.mcp.client import (  # pyright: ignore[reportMissingImports]
    MCPClient,
    MCPClientError,
    MCPServerConfig,
    MCPServerConnection,
    MCPTransport,
)
from exo.mcp.execution import (  # pyright: ignore[reportMissingImports]
    MCPExecutionError,
    load_mcp_client,
    load_mcp_config,
    substitute_env_vars,
)
from exo.mcp.server import (  # pyright: ignore[reportMissingImports]
    MCPServerError,
    MCPServerRegistry,
    mcp_server,
)
from exo.mcp.tools import (  # pyright: ignore[reportMissingImports]
    MCPToolError,
    MCPToolFilter,
    MCPToolWrapper,
    convert_mcp_tools,
    extract_schema,
    namespace_tool_name,
    parse_namespaced_name,
)

__all__: list[str] = [
    "MCPClient",
    "MCPClientError",
    "MCPExecutionError",
    "MCPServerConfig",
    "MCPServerConnection",
    "MCPServerError",
    "MCPServerRegistry",
    "MCPToolError",
    "MCPToolFilter",
    "MCPToolWrapper",
    "MCPTransport",
    "convert_mcp_tools",
    "extract_schema",
    "load_mcp_client",
    "load_mcp_config",
    "mcp_server",
    "namespace_tool_name",
    "parse_namespaced_name",
    "substitute_env_vars",
]
