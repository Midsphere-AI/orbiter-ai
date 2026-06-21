"""MCP server configuration loading and management.

Reads/writes ``mcp.json`` files in the standard format::

    {
        "mcpServers": {
            "name": {
                "transport": "stdio",
                "command": "python",
                "args": ["-m", "server"]
            }
        }
    }

Supports ``${ENV_VAR}`` and ``${vault:NAME}`` substitution in all
string values at connection time.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any

from exo.mcp.client import HttpConfig, StdioConfig  # pyright: ignore[reportMissingImports]
from exo.types import ExoError  # pyright: ignore[reportMissingImports]

_ENV_PATTERN = re.compile(r"\$\{([^}:]+)\}")
_DEFAULT_CONFIG_NAMES = ("mcp.json",)
_HOME_CONFIG_DIR = Path.home() / ".exo-mcp"


class MCPConfigError(ExoError):
    """Raised on config loading/saving failures."""


@dataclass
class ServerEntry:
    """Configuration for a single MCP server.

    Flat kwargs (original API, always supported):
        name: Human-readable server name.
        transport: Transport type string (``"stdio"``, ``"sse"``, ``"streamable_http"``).
        command: Executable for stdio transport.
        args: Command-line arguments for stdio transport.
        env: Extra environment variables for stdio transport.
        cwd: Working directory for stdio transport.
        url: Server URL for HTTP-based transports.
        headers: HTTP request headers for HTTP-based transports.
        timeout: Connection timeout in seconds.

    Grouped kwargs (namespace API — populate the flat attrs above):
        stdio: :class:`~exo.mcp.client.StdioConfig` grouping stdio-transport params.
        http: :class:`~exo.mcp.client.HttpConfig` grouping HTTP-transport params.

    Raises:
        MCPConfigError: If both a grouped config and the corresponding flat kwarg
            are provided simultaneously.
    """

    name: str
    transport: str = "stdio"
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    cwd: str | None = None
    url: str | None = None
    headers: dict[str, str] | None = None
    timeout: float = 30.0
    # InitVar fields for grouped namespace kwargs (not stored as attrs)
    stdio: InitVar[StdioConfig | None] = None
    http: InitVar[HttpConfig | None] = None

    def __post_init__(self, stdio: StdioConfig | None, http: HttpConfig | None) -> None:
        if stdio is not None:
            _stdio_flat = {
                "command": self.command,
                "args": self.args or None,
                "env": self.env,
                "cwd": self.cwd,
            }
            _conflicting = [k for k, v in _stdio_flat.items() if v is not None]
            if _conflicting:
                raise MCPConfigError(
                    f"Server '{self.name}': cannot mix stdio= with flat kwarg(s) "
                    f"{_conflicting!r}; use one style or the other."
                )
            self.command = stdio.command
            self.args = list(stdio.args)
            self.env = stdio.env
            self.cwd = stdio.cwd

        if http is not None:
            _http_flat = {"url": self.url, "headers": self.headers}
            _conflicting_http = [k for k, v in _http_flat.items() if v is not None]
            if _conflicting_http:
                raise MCPConfigError(
                    f"Server '{self.name}': cannot mix http= with flat kwarg(s) "
                    f"{_conflicting_http!r}; use one style or the other."
                )
            self.url = http.url
            self.headers = http.headers

    # ------------------------------------------------------------------
    # Grouped read accessors (namespace view)
    # ------------------------------------------------------------------

    @property
    def stdio_config(self) -> StdioConfig | None:
        """Return a :class:`StdioConfig` view of the stdio params, or ``None`` if no command."""
        if self.command is None:
            return None
        return StdioConfig(command=self.command, args=list(self.args), env=self.env, cwd=self.cwd)

    @property
    def http_config(self) -> HttpConfig | None:
        """Return an :class:`HttpConfig` view of the HTTP params, or ``None`` if no URL."""
        if self.url is None:
            return None
        return HttpConfig(url=self.url, headers=self.headers)

    def validate(self) -> None:
        """Validate transport-specific requirements."""
        if self.transport == "stdio" and not self.command:
            raise MCPConfigError(f"Server '{self.name}': stdio transport requires 'command'")
        if self.transport in ("sse", "streamable_http", "websocket") and not self.url:
            raise MCPConfigError(f"Server '{self.name}': {self.transport} transport requires 'url'")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict (for saving back to mcp.json)."""
        d: dict[str, Any] = {"transport": self.transport}
        if self.command is not None:
            d["command"] = self.command
        if self.args:
            d["args"] = self.args
        if self.env:
            d["env"] = self.env
        if self.cwd is not None:
            d["cwd"] = self.cwd
        if self.url is not None:
            d["url"] = self.url
        if self.headers:
            d["headers"] = self.headers
        if self.timeout != 30.0:
            d["timeout"] = self.timeout
        return d


# ---------------------------------------------------------------------------
# Environment variable substitution
# ---------------------------------------------------------------------------


def substitute_env_vars(value: str) -> str:
    """Replace ``${VAR}`` placeholders with environment variable values.

    Unset variables are replaced with empty strings.  Vault references
    (``${vault:...}``) are left untouched.
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
# Config file discovery
# ---------------------------------------------------------------------------


def find_config(explicit_path: str | Path | None = None) -> Path | None:
    """Find an mcp.json config file.

    Search order:
        1. Explicit path (if given)
        2. ``./mcp.json`` in the current directory
        3. ``~/.exo-mcp/mcp.json`` in the user home
    """
    if explicit_path:
        p = Path(explicit_path)
        if p.is_file():
            return p
        raise MCPConfigError(f"Config file not found: {p}")
    for name in _DEFAULT_CONFIG_NAMES:
        candidate = Path.cwd() / name
        if candidate.is_file():
            return candidate
    home_cfg = _HOME_CONFIG_DIR / "mcp.json"
    if home_cfg.is_file():
        return home_cfg
    return None


# ---------------------------------------------------------------------------
# Config loading / saving
# ---------------------------------------------------------------------------


def load_config(path: Path) -> dict[str, ServerEntry]:
    """Load MCP server configs from an mcp.json file.

    Environment variable substitution is applied to all string values.
    Vault references (``${vault:...}``) are preserved for later resolution.

    Returns:
        Dict mapping server name to ``ServerEntry``.
    """
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (json.JSONDecodeError, OSError) as exc:
        raise MCPConfigError(f"Failed to parse config '{path}': {exc}") from exc

    servers_raw = data.get("mcpServers", {})
    if not isinstance(servers_raw, dict):
        raise MCPConfigError(f"Expected 'mcpServers' to be an object in '{path}'")

    servers: dict[str, ServerEntry] = {}
    for name, cfg in servers_raw.items():
        cfg = _substitute_recursive(cfg)
        servers[name] = ServerEntry(
            name=name,
            transport=cfg.get("transport", "stdio"),
            command=cfg.get("command"),
            args=cfg.get("args", []),
            env=cfg.get("env"),
            cwd=cfg.get("cwd"),
            url=cfg.get("url"),
            headers=cfg.get("headers"),
            timeout=cfg.get("timeout", 30.0),
        )
    return servers


def save_config(path: Path, servers: dict[str, ServerEntry]) -> None:
    """Write server configs back to an mcp.json file."""
    data = {"mcpServers": {name: entry.to_dict() for name, entry in servers.items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_or_empty(path: Path) -> dict[str, ServerEntry]:
    """Load config from *path*, returning empty dict if the file doesn't exist."""
    if not path.exists():
        return {}
    return load_config(path)


def add_server(path: Path, entry: ServerEntry) -> None:
    """Add (or replace) a server entry in the config file."""
    servers = load_or_empty(path)
    servers[entry.name] = entry
    save_config(path, servers)


def remove_server(path: Path, name: str) -> bool:
    """Remove a server from the config. Returns ``True`` if it existed."""
    servers = load_or_empty(path)
    if name not in servers:
        return False
    del servers[name]
    save_config(path, servers)
    return True


def default_config_path() -> Path:
    """Return the default path for creating a new mcp.json (current directory)."""
    return Path.cwd() / "mcp.json"
