# exo-mcp-cli

> Standalone CLI for interacting with MCP servers from the command line.

`exo-mcp-cli` is a self-contained terminal client for the Model Context Protocol. It lets you add, test, and remove server configurations, browse tools/resources/prompts, and call tools directly — all without writing any Python. Credentials are stored in an encrypted local vault and referenced via `${vault:NAME}` placeholders in `mcp.json`.

## Installation

```bash
pip install exo-mcp-cli
# or
uv add exo-mcp-cli
```

## Quick start

```bash
# Add a stdio server (API key is encrypted in the local vault automatically)
exo-mcp server add my-server --command python --arg -m --arg my_mcp_server

# Add an SSE server with a secret header
exo-mcp server add remote --transport sse --url https://api.example.com/mcp \
    --header Authorization=Bearer\ sk-abc123

# Check connectivity
exo-mcp server test my-server

# Explore available tools
exo-mcp tool list my-server

# Call a tool
exo-mcp tool call my-server search --arg query=hello

# Store a credential manually
exo-mcp auth set my-api-key sk-abc123

# List all stored secret names
exo-mcp auth list
```

## What's inside

- **`server list`** — show all servers configured in `mcp.json`
- **`server add`** — register a server (stdio/SSE/streamable_http); sensitive `--header`/`--env` values are auto-vaulted
- **`server remove`** — delete a server entry from config
- **`server test`** — connect, ping, and report tool count
- **`tool list`** — list tools exposed by a server (table or `--json`)
- **`tool call`** — invoke a tool with `--arg KEY=VALUE`, `--json`, or `--inject` flags
- **`resource list` / `resource read`** — browse and fetch MCP resources
- **`prompt list` / `prompt get`** — list and render MCP prompt templates
- **`auth set` / `auth list` / `auth remove`** — manage the encrypted credential vault

## Part of [Exo](https://github.com/midsphere-ai/exo)

Get the full framework with `pip install exo-ai`.

---

MIT © Midsphere AI
