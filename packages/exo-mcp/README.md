# exo-mcp

> MCP (Model Context Protocol) client and tool integration for the Exo framework.

`exo-mcp` provides a production-ready async client for connecting to MCP servers over stdio, SSE, and Streamable HTTP transports. It handles connection lifecycle, tool discovery, and result routing — and bridges the MCP protocol into Exo's native `Tool` format so agents can invoke MCP tools without extra plumbing.

## Installation

```bash
pip install exo-mcp
# or
uv add exo-mcp
```

## Quick start

```python
from exo.mcp import MCPClient, MCPServerConfig, MCPTransport, convert_mcp_tools

config = MCPServerConfig(
    name="my-server",
    transport=MCPTransport.STDIO,
    command="python",
    args=["-m", "my_mcp_server"],
)

client = MCPClient()
client.add_server(config)

async with client:
    tools = await client.list_tools("my-server")
    result = await client.call_tool("my-server", "search", {"query": "hello"})

# Convert MCP tools to Exo Tool objects for use with an Agent
exo_tools = await convert_mcp_tools(client, "my-server")
```

## What's inside

- **`MCPClient`** — high-level async client managing multiple named server connections with caching and reconnect
- **`MCPServerConfig`** — configuration dataclass for a server (transport, command/URL, timeouts, env, headers)
- **`MCPServerConnection`** — single live connection with `connect()`, `list_tools()`, `call_tool()`, and `cleanup()`
- **`MCPTransport`** — enum of supported transports: `STDIO`, `SSE`, `STREAMABLE_HTTP`
- **`convert_mcp_tools`** — converts MCP tool descriptors into Exo `Tool` objects ready for agent use
- **`MCPToolFilter`** — include/exclude filter applied to tool lists before conversion
- **`load_mcp_config`** — parse an `mcp.json` file into a list of `MCPServerConfig` objects

## Part of [Exo](https://github.com/midsphere-ai/exo)

Get the full framework with `pip install exo-ai`.

---

MIT © Midsphere AI
