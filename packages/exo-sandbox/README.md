# exo-sandbox

> Sandboxed execution environments for safe tool running by Exo agents.

`exo-sandbox` lets agents run shell commands, execute Python code, and read or write files without trusting the host process. It provides a `Sandbox` abstraction with three backends — local subprocess, E2B cloud sandbox, and Kubernetes pod — and a set of ready-made `Tool` subclasses that route through whichever backend is active.

## Installation

```bash
pip install exo-sandbox
# or
uv add exo-sandbox

# Optional backends
pip install exo-sandbox[e2b]         # E2B cloud sandbox
pip install exo-sandbox[kubernetes]  # Kubernetes pod sandbox
```

## Quick start

```python
import asyncio
from exo.sandbox import LocalSandbox, code_tool, shell_tool

async def main() -> None:
    async with LocalSandbox(timeout=30.0) as sandbox:
        # Run Python code in a restricted subprocess
        ct = code_tool(sandbox=sandbox)
        result = await ct.execute(code="print(1 + 1)")
        print(result)  # "2\n"

        # Run allow-listed shell commands
        st = shell_tool(allowed_commands=["ls", "echo"], sandbox=sandbox)
        out = await st.execute(command="echo hello")
        print(out)

asyncio.run(main())
```

### E2B cloud sandbox

```python
import asyncio
from exo.sandbox import E2BSandbox

async def main() -> None:
    # Reads E2B_API_KEY from the environment
    async with E2BSandbox(timeout=300.0) as sandbox:
        result = await sandbox.run_tool("shell", {"command": "python3 --version"})
        print(result["stdout"])

asyncio.run(main())
```

## What's inside

- **`LocalSandbox`** — executes tools in the local process with state-machine lifecycle (`INIT → RUNNING → IDLE → CLOSED`)
- **`E2BSandbox`** — cloud-isolated execution via [E2B](https://e2b.dev); supports custom templates, reconnect to existing sandboxes, and custom tool handlers via `register_tool()`
- **`KubernetesSandbox`** — pod-based isolation for production deployments
- **`CodeTool`** / **`code_tool`** — sandboxed Python execution with a configurable blocked-names list and subprocess timeout
- **`ShellTool`** / **`shell_tool`** — allowlist-based shell execution (no shell expansion, output capped at 10,000 chars)
- **`FilesystemTool`** — read/write/list files within declared `allowed_directories`; path traversal is blocked
- **`TerminalTool`** — full command execution with a blacklist for destructive commands (`rm`, `shutdown`, etc.)
- **`SandboxBuilder`** — declarative builder for assembling sandbox + tool combinations

## Part of [Exo](https://github.com/midsphere-ai/exo)

`exo-sandbox` integrates with the Exo tool registry so agents can call sandbox tools the same way they call any other tool. Get the full framework with `pip install exo-ai`.

---

MIT © Midsphere AI
