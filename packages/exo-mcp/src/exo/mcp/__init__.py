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
from exo.mcp.transport import (  # pyright: ignore[reportMissingImports]
    MCPTransportError,
    create_transport_streams,
)
from exo.mcp.vault import (  # pyright: ignore[reportMissingImports]
    Vault,
    VaultError,
    derive_key,
)

__all__: list[str] = [
    "MCPClient",
    "MCPClientError",
    "MCPExecutionError",
    "MCPServerConfig",
    "MCPServerConnection",
    "MCPServerError",
    "MCPToolError",
    "MCPToolFilter",
    "MCPToolWrapper",
    "MCPTransport",
    "MCPTransportError",
    "Vault",
    "VaultError",
    "convert_mcp_tools",
    "create_transport_streams",
    "derive_key",
    "extract_schema",
    "load_mcp_client",
    "load_mcp_config",
    "mcp_server",
    "namespace_tool_name",
    "parse_namespaced_name",
    "substitute_env_vars",
]
