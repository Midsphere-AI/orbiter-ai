"""MCP execution utilities — config loading and env var substitution."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from exo.mcp.client import (  # pyright: ignore[reportMissingImports]
    MCPClient,
    MCPClientError,
    MCPServerConfig,
    MCPTransport,
)

logger = logging.getLogger(__name__)


class MCPExecutionError(MCPClientError):
    """Error raised during MCP config loading or execution."""


# ---------------------------------------------------------------------------
# Environment variable substitution
# ---------------------------------------------------------------------------

_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def substitute_env_vars(value: str) -> str:
    """Replace ``${VAR_NAME}`` placeholders with environment variable values.

    Unset variables are replaced with empty strings.

    Args:
        value: String potentially containing ``${VAR_NAME}`` placeholders.

    Returns:
        String with placeholders replaced by environment variable values.
    """

    def _replace(match: re.Match[str]) -> str:
        return os.environ.get(match.group(1), "")

    return _ENV_PATTERN.sub(_replace, value)


def _substitute_recursive(obj: Any) -> Any:
    """Recursively substitute env vars in strings within dicts/lists."""
    if isinstance(obj, str):
        return substitute_env_vars(obj)
    if isinstance(obj, dict):
        return {k: _substitute_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute_recursive(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Config loading from mcp.json
# ---------------------------------------------------------------------------


def load_mcp_config(path: str | Path) -> list[MCPServerConfig]:
    """Load MCP server configurations from an ``mcp.json`` file.

    Environment variables in ``${VAR}`` format are substituted in all string
    values.  The file must contain a JSON object with a ``"mcpServers"`` key
    mapping server names to configuration objects.

    Example ``mcp.json``::

        {
            "mcpServers": {
                "my-server": {
                    "transport": "stdio",
                    "command": "${PYTHON_PATH}",
                    "args": ["-m", "my_module"]
                }
            }
        }

    Args:
        path: Path to the mcp.json file.

    Returns:
        List of MCPServerConfig instances.

    Raises:
        MCPExecutionError: If the file cannot be read or parsed.
    """
    path = Path(path)
    if not path.exists():
        raise MCPExecutionError(f"MCP config file not found: {path}")

    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (json.JSONDecodeError, OSError) as exc:
        raise MCPExecutionError(f"Failed to parse MCP config '{path}': {exc}") from exc

    servers_raw = data.get("mcpServers", {})
    if not isinstance(servers_raw, dict):
        raise MCPExecutionError(f"Expected 'mcpServers' to be a dict in '{path}'")

    configs: list[MCPServerConfig] = []
    for name, raw_cfg in servers_raw.items():
        cfg = _substitute_recursive(raw_cfg)
        transport = MCPTransport(cfg.get("transport", "stdio"))
        configs.append(
            MCPServerConfig(
                name=name,
                transport=transport,
                command=cfg.get("command"),
                args=cfg.get("args"),
                env=cfg.get("env"),
                cwd=cfg.get("cwd"),
                url=cfg.get("url"),
                headers=cfg.get("headers"),
                timeout=cfg.get("timeout", 30.0),
                cache_tools=cfg.get("cache_tools", False),
            )
        )

    return configs


def load_mcp_client(path: str | Path) -> MCPClient:
    """Create an MCPClient from an ``mcp.json`` config file.

    Convenience wrapper around :func:`load_mcp_config`.

    Args:
        path: Path to the mcp.json file.

    Returns:
        MCPClient with all servers from the config registered.
    """
    configs = load_mcp_config(path)
    client = MCPClient()
    for cfg in configs:
        client.add_server(cfg)
    return client
