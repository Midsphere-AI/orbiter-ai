# exo-sandbox — Error DX & Resilience Audit

## Counts
- raise sites: 22
- error classes total / not inheriting ExoError: 2 total (`SandboxError`, `ToolError`) / 0 (both inherit `ExoError` via `SandboxError(ExoError)` and `ToolError(ExoError)`) — **plus 1 offender**: bare `ValueError` at `tools.py:248`
- `except Exception` sites: 12 ; swallow-and-pass: 1 (`e2b.py:189`) ; drop-cause: 1 (`kubernetes.py:107`)
- CancelledError handlers: 2 (both in lifecycle `start()` — correctly re-raised)
- I/O call sites lacking timeout: 6 (all `asyncio.to_thread` calls for E2B file/shell ops in `e2b.py:275,286,296,305`; `_kill_sandbox` at `e2b.py:186`; pod poll loop in `kubernetes.py:144`)

---

## Findings (prioritized)

### P0 — Silent failure / cause loss / process leak

**[P0] | #5 | kubernetes.py:107** — First `except Exception` (for `load_incluster_config`) is silently swallowed with no `from exc` — the original failure is dropped entirely before falling through to `load_kube_config()`. If both loaders fail, only the second exception surfaces.

```python
# current (cause silently dropped)
except Exception:
    try:
        config.load_kube_config()
    except Exception as cfg_exc:
        raise SandboxError("Failed to load Kubernetes configuration") from cfg_exc
```

Fix: capture the first exception and chain it: `except Exception as first_exc:` then `raise SandboxError(...) from cfg_exc` with a message naming both attempts, or chain via `__context__`.

---

**[P0] | #8 | tools.py:381** — `_exec_allowlist()` timeout handler calls `proc.kill()` but never calls `await proc.wait()`. The zombie subprocess is leaked. Additionally, the orphaned process can keep stdout/stderr pipes open, potentially blocking other operations.

```python
except TimeoutError as exc:
    proc.kill()          # kill() called but...
    # ← missing: await proc.wait()
    raise ToolError(f"Command timed out after {self._timeout}s") from exc
```

Fix: add `await proc.wait()` after `proc.kill()` (mirror the correct pattern in `_exec_blacklist`).

---

**[P0] | #8 | tools.py:574** — `CodeTool.execute()` has the identical zombie-process leak: `proc.kill()` is called on timeout without `await proc.wait()`.

```python
except TimeoutError as exc:
    proc.kill()          # same pattern — no await proc.wait()
    raise ToolError(...)
```

Fix: same as above — add `await proc.wait()`.

---

**[P0] | #4 | e2b.py:185–190** — `_kill_sandbox()` silently swallows the kill failure. A leaked remote sandbox is a serious resource/cost issue, but the error is discarded:

```python
except Exception:
    logger.warning("Failed to kill E2B sandbox %s", self._e2b_sandbox_id)
```

No exception object is logged, so there is zero diagnostic signal. This is the one true `except: pass` equivalent in the package.

Fix: log the exception: `logger.warning("Failed to kill E2B sandbox %s: %s", self._e2b_sandbox_id, exc)` and consider re-raising if called from `stop()`/`cleanup()` rather than from a cancel handler.

---

**[P0] | #8 | kubernetes.py:140–149** — `_wait_for_pod()` ignores the sandbox `_timeout` and hardcodes a fixed 60 s ceiling (`_MAX_POLL_ATTEMPTS=30 × _POLL_INTERVAL=2.0`). Worse, the poll loop calls `asyncio.sleep()` repeatedly but has no `asyncio.wait_for()` wrapper, so an external cancellation is only caught between sleeps — the pod could be left running.

Fix: replace the manual poll loop with `asyncio.wait_for(_poll_loop(), timeout=self._timeout)` and catch `asyncio.TimeoutError` to raise a contextual `SandboxError`.

---

### P1 — Unactionable / missing context

**[P1] | #2,#3 | e2b.py:162** — `start()` wraps all E2B failures in a plain string format with no `context=` or `hint=`:

```python
msg = f"Failed to start E2B sandbox: {exc}"
raise SandboxError(msg) from exc
```

The developer doesn't know which template was tried, which sandbox_id was assigned, or whether the API key or template was the problem.

Fix:
```python
raise SandboxError(
    "Failed to start E2B sandbox.",
    context={"sandbox_id": self._sandbox_id, "template": self._template or "(default)"},
    hint="Check E2B_API_KEY is valid and the template ID exists in your E2B account.",
    doc="https://e2b.dev/docs",
) from exc
```

---

**[P1] | #2,#3 | kubernetes.py:174** — `start()` failure message is similarly context-free:

```python
msg = f"Failed to start Kubernetes sandbox: {exc}"
raise SandboxError(msg) from exc
```

Doesn't name the pod, namespace, or image.

Fix: add `context={"pod": self._pod_name, "namespace": self._namespace, "image": self._image}` and a `hint=` pointing to `kubectl get pods` or kubeconfig troubleshooting.

---

**[P1] | #2,#3 | kubernetes.py:112** — `_load_client()` raises `SandboxError("Failed to load Kubernetes configuration")` with no hint. The developer doesn't know which config paths were tried.

Fix: add `hint="Set KUBECONFIG env var to your kubeconfig path, or ensure in-cluster service account credentials are mounted."` and include `context={"kubeconfig": kubeconfig_path}`.

---

**[P1] | #2 | e2b.py:267–268** — `run_tool()` on registered handler raises `SandboxError(f"Registered tool {tool_name!r} failed: {exc}")` — no sandbox context, no hint.

Fix: use `.with_context(sandbox_id=self._sandbox_id, tool=tool_name)`.

---

**[P1] | #2 | e2b.py:320–322** — Built-in tool dispatch wraps all E2B I/O failures as `SandboxError(f"E2B tool {tool_name!r} failed: {exc}")`. Shell exit codes are silently returned as success (no check of `result.exit_code` against non-zero). A command that exits 127 ("not found") or 1 ("failed") silently looks like `{"status": "ok", "exit_code": 127}`.

Fix: after E2B `commands.run`, check `result.exit_code != 0` and surface a contextual error or at minimum surface the exit code prominently.

---

**[P1] | #2,#3 | kubernetes.py:238** — `run_tool()` local path raises `SandboxError(f"Tool {tool_name!r} failed: {exc}")` with no `context=` or `hint=`.

Fix: add `context={"sandbox_id": self._sandbox_id, "tool": tool_name, "pod": self._pod_name}`.

---

**[P1] | #1 | tools.py:248** — `ShellTool.__init__` raises a bare `ValueError`:

```python
raise ValueError(f"mode must be 'allowlist' or 'blacklist', got {mode!r}")
```

This escapes the package boundary as a non-`ExoError` with no hint.

Fix: `raise SandboxError(f"mode must be 'allowlist' or 'blacklist', got {mode!r}", hint="Valid modes: 'allowlist' (explicit permitted commands) or 'blacklist' (block dangerous commands).")` — or import/use `ToolError`.

---

**[P1] | #4 | kubernetes.py:196–197** — `_delete_resources()` swallows delete failures silently:

```python
except Exception:
    logger.warning("Failed to delete pod %s", self._pod_name)
```

No exception logged, no namespace, no sandbox_id.

Fix: `logger.warning("Failed to delete pod %s (namespace=%s, sandbox=%s): %s", self._pod_name, self._namespace, self._sandbox_id, exc)`.

---

**[P1] | #2 | tools.py:383–384** — `_exec_allowlist()` `FileNotFoundError` raises `ToolError(f"Command not found: {parts[0]!r}")`. Good message, but no hint.

Fix: add `hint=f"Ensure {parts[0]!r} is installed and on PATH, or add it to allowed_commands."`.

---

**[P1] | #2 | tools.py:432** — `_exec_blacklist()` timeout message is decent but drops `command` from the error context, making it hard to identify which command hung in a multi-tool workflow.

Fix: add `context={"command": command[:120]}` to the `ToolError`.

---

### P2 — Polish

**[P2] | #3 | base.py:112** — `_transition()` raises `SandboxError(msg)` with no `context=` or `hint=`. Useful but could carry `{"sandbox_id": self._sandbox_id, "from": self._status.value, "to": target.value}`.

---

**[P2] | #2 | e2b.py:216–218** — `register_tool()` duplicate raises `SandboxError(f"Tool {name!r} is already registered")` — fine but lacks `hint=` ("Call unregister_tool(name) first if you want to replace it.").

---

**[P2] | #9 | e2b.py:159** — `start()` sets `self._status = SandboxStatus.ERROR` directly instead of calling `self._transition(SandboxStatus.ERROR)`, bypassing the state machine and logging. Minor inconsistency; same in `kubernetes.py:173`.

---

**[P2] | #7 | tools.py:560** — `CodeTool.execute()` sandbox path catches bare `Exception` which is fine (CancelledError isn't `Exception`), but there is no explicit `asyncio.CancelledError` re-raise guard. Low risk since `CancelledError` propagates naturally, but an explicit comment or narrow guard is cleaner.

---

## Resilience gaps

| Location | Issue |
|---|---|
| `e2b.py:129,143,147` (`start`) | `asyncio.to_thread(e2b_mod.Sandbox.connect/create)` — no timeout wrapper. E2B SDK timeout param is passed to E2B infra, not to the Python call itself. A hung TCP connection can block the coroutine indefinitely. Wrap with `asyncio.wait_for(..., timeout=self._timeout)`. |
| `e2b.py:186` (`_kill_sandbox`) | `asyncio.to_thread(self._e2b_sandbox.kill)` — no timeout. A hung kill call during cancel/cleanup will stall the cleanup path forever. Add `asyncio.wait_for(..., timeout=10.0)`. |
| `e2b.py:275,286,296,305` (`run_tool`) | All four built-in dispatch branches (`commands.run`, `files.read`, `files.write`, `files.list`) call `asyncio.to_thread(...)` with no timeout. A slow or hung E2B sandbox blocks the agent indefinitely. Use `asyncio.wait_for(..., timeout=self._timeout)` on each. |
| `kubernetes.py:144` (`_wait_for_pod`) | Poll loop uses fixed constants, ignores `self._timeout`. Replace with `asyncio.wait_for` and `self._timeout`. |
| `kubernetes.py:161,194` (`start`/`_delete_resources`) | `asyncio.to_thread(api.create_namespaced_pod / delete_namespaced_pod)` — K8s API calls have no timeout guard. Wrap with `asyncio.wait_for`. |
| `tools.py:381` (`_exec_allowlist`) | `proc.kill()` without `await proc.wait()` on timeout — zombie process leak (P0 above). |
| `tools.py:574` (`CodeTool`) | Same zombie leak on timeout. |

---

## Effort estimate

**M** — Two P0 process leaks are one-liners; the E2B/K8s `asyncio.wait_for` wrapping is mechanical but touches 7 call sites; the `ValueError`→`SandboxError` fix and structured `context=/hint=` additions across ~8 raise sites are straightforward. No architectural changes required.
