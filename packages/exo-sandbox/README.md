# exo-sandbox

> Sandboxed execution environments for safe tool running by Exo agents.

`exo-sandbox` lets agents run shell commands, execute Python code, and read or write files with guardrails around what the host process exposes. It ships a set of ready-made `Tool` subclasses that run in a restricted local subprocess by default, plus a `Sandbox` abstraction with two isolated backends — E2B cloud sandbox and Kubernetes pod — that tools route through when one is supplied.

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

By default the tools execute locally — `code_tool` in a restricted subprocess, `shell_tool` against an allowlist — with no `Sandbox` required:

```python
import asyncio
from exo.sandbox import code_tool, shell_tool

async def main() -> None:
    # Run Python code in a restricted subprocess
    ct = code_tool(timeout=10.0)
    result = await ct.execute(code="print(1 + 1)")
    print(result)  # "2\n"

    # Run allow-listed shell commands
    st = shell_tool(allowed_commands=["ls", "echo"])
    out = await st.execute(command="echo hello")
    print(out)

asyncio.run(main())
```

To isolate execution further, pass a `Sandbox` backend (`code_tool(sandbox=...)` / `shell_tool(..., sandbox=...)`) and the tool forwards the call to the sandbox instead of running locally.

### E2B cloud sandbox

Spin up a throwaway cloud VM and work with it directly — upload files, run commands, read results — no raw SDK required:

```python
import asyncio
from exo.sandbox import E2BSandbox

async def main() -> None:
    # Reads E2B_API_KEY from the environment
    async with E2BSandbox(timeout=300.0) as sandbox:
        await sandbox.upload("data.csv", "/home/user/data.csv")   # upload a local file
        result = await sandbox.run("python3 analyze.py")          # run a command
        print(result["stdout"])
        await sandbox.download("/home/user/report.txt", "report.txt")

asyncio.run(main())
```

Convenience helpers: `upload` / `download` / `read_file` / `write_file` / `list_files` / `run`. For anything the helpers don't cover (background commands, PTY, watchers), reach the live E2B handle via `sandbox.e2b_sandbox` — it raises a clear error if the sandbox isn't running. To let an *agent* drive the sandbox, hand it `sandbox.shell_tool()` / `sandbox.code_tool()` / `sandbox.filesystem_tool()` instead.

## What's inside

- **`Sandbox`** — abstract backend with a state-machine lifecycle (`INIT → RUNNING → IDLE → CLOSED`); tools run locally when no sandbox is supplied
- **`E2BSandbox`** — cloud-isolated execution via [E2B](https://e2b.dev); first-class `upload`/`download`/`run`/`read_file`/`write_file`/`list_files` helpers, a `e2b_sandbox` escape hatch to the raw SDK, custom templates, reconnect to existing sandboxes, and custom tool handlers via `register_tool()`
- **`KubernetesSandbox`** — pod-based isolation for production deployments
- **`CodeTool`** / **`code_tool`** — sandboxed Python execution with a configurable blocked-names list and subprocess timeout
- **`ShellTool`** / **`shell_tool`** — allowlist-based shell execution (no shell expansion, output capped at 10,000 chars)
- **`FilesystemTool`** — read/write/list files within declared `allowed_directories`; path traversal is blocked
- **`TerminalTool`** — full command execution with a blacklist for destructive commands (`rm`, `shutdown`, etc.)

## Part of [Exo](https://github.com/midsphere-ai/exo)

`exo-sandbox` integrates with the Exo tool registry so agents can call sandbox tools the same way they call any other tool. Get the full framework with `pip install exo-ai`.

---

MIT © Midsphere AI
