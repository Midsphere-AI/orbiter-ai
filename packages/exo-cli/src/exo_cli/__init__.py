"""Exo CLI: Command-line agent runner."""

__version__: str = "0.1.0"

from exo_cli.batch import (
    BatchError,
    BatchItem,
    BatchResult,
    InputFormat,
    ItemResult,
    load_batch_items,
    results_to_csv,
    results_to_jsonl,
)
from exo_cli.console import InteractiveConsole, format_agents_table, parse_command
from exo_cli.executor import ExecutionResult, ExecutorError, LocalExecutor
from exo_cli.loader import (
    AgentLoadError,
    discover_agent_files,
    load_markdown_agent,
    load_python_agent,
    load_yaml_agents,
    scan_directory,
    validate_agent,
)
from exo_cli.main import CLIError, app, find_config, load_config, resolve_config
from exo_cli.plugins import PluginError, PluginHook, PluginManager, PluginSpec

__all__: list[str] = [
    "AgentLoadError",
    "BatchError",
    "BatchItem",
    "BatchResult",
    "CLIError",
    "ExecutionResult",
    "ExecutorError",
    "InputFormat",
    "InteractiveConsole",
    "ItemResult",
    "LocalExecutor",
    "PluginError",
    "PluginHook",
    "PluginManager",
    "PluginSpec",
    "app",
    "discover_agent_files",
    "find_config",
    "format_agents_table",
    "load_batch_items",
    "load_config",
    "load_markdown_agent",
    "load_python_agent",
    "load_yaml_agents",
    "parse_command",
    "resolve_config",
    "results_to_csv",
    "results_to_jsonl",
    "scan_directory",
    "validate_agent",
]
