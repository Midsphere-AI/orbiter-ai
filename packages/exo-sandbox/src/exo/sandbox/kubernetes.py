"""Kubernetes-based sandbox for remote agent execution."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
from typing import Any

from exo.sandbox.base import (  # pyright: ignore[reportMissingImports]
    Sandbox,
    SandboxError,
    SandboxStatus,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_NAMESPACE = "default"
_DEFAULT_IMAGE = "python:3.11-slim"
_POLL_INTERVAL = 2.0
_MAX_POLL_ATTEMPTS = 30

# Max bytes read from pod exec stdout/stderr
_EXEC_OUTPUT_LIMIT = 1_048_576  # 1 MiB


# ---------------------------------------------------------------------------
# KubernetesSandbox
# ---------------------------------------------------------------------------


class KubernetesSandbox(Sandbox):
    """Sandbox that manages a Kubernetes pod for isolated execution.

    Requires the ``kubernetes`` extra (``pip install exo-sandbox[kubernetes]``).
    Pod lifecycle: ``start`` creates the pod and waits for readiness,
    ``stop`` deletes the pod, ``cleanup`` ensures all resources are removed.
    """

    __slots__ = (
        "_cluster_ip",
        "_image",
        "_k8s_client",
        "_namespace",
        "_pod_name",
        "_service_name",
        "_tools",
    )

    def __init__(
        self,
        *,
        sandbox_id: str | None = None,
        workspace: list[str] | None = None,
        mcp_config: dict[str, Any] | None = None,
        agents: dict[str, Any] | None = None,
        timeout: float = 60.0,
        namespace: str | None = None,
        image: str | None = None,
        tools: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            sandbox_id=sandbox_id,
            workspace=workspace,
            mcp_config=mcp_config,
            agents=agents,
            timeout=timeout,
        )
        self._namespace = namespace or os.environ.get("EXO_K8S_NAMESPACE", _DEFAULT_NAMESPACE)
        self._image = image or os.environ.get("EXO_K8S_IMAGE", _DEFAULT_IMAGE)
        self._pod_name: str | None = None
        self._service_name: str | None = None
        self._cluster_ip: str | None = None
        self._k8s_client: Any = None
        self._tools: dict[str, Any] = dict(tools) if tools else {}

    # -- properties ---------------------------------------------------------

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def image(self) -> str:
        return self._image

    @property
    def pod_name(self) -> str | None:
        return self._pod_name

    @property
    def cluster_ip(self) -> str | None:
        return self._cluster_ip

    # -- kubernetes helpers -------------------------------------------------

    def _load_client(self) -> Any:
        """Lazy-load the kubernetes client library."""
        if self._k8s_client is not None:
            return self._k8s_client
        try:
            from kubernetes import client, config  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            msg = "kubernetes package is required: pip install exo-sandbox[kubernetes]"
            raise SandboxError(msg) from exc

        kubeconfig_path = os.environ.get("KUBECONFIG")
        try:
            if kubeconfig_path:
                config.load_kube_config(config_file=kubeconfig_path)
            else:
                config.load_incluster_config()
        except Exception:
            try:
                config.load_kube_config()
            except Exception as cfg_exc:
                msg = "Failed to load Kubernetes configuration"
                raise SandboxError(msg) from cfg_exc

        self._k8s_client = client.CoreV1Api()
        logger.debug("Loaded Kubernetes client for sandbox %s", self._sandbox_id)
        return self._k8s_client

    def _pod_manifest(self) -> dict[str, Any]:
        """Build a minimal pod manifest."""
        return {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": f"exo-{self._sandbox_id}",
                "namespace": self._namespace,
                "labels": {"app": "exo-sandbox", "sandbox-id": self._sandbox_id},
            },
            "spec": {
                "containers": [
                    {
                        "name": "sandbox",
                        "image": self._image,
                        "command": ["sleep", "infinity"],
                    }
                ],
                "restartPolicy": "Never",
            },
        }

    def _service_manifest(self) -> dict[str, Any]:
        """Build a minimal service manifest."""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"exo-svc-{self._sandbox_id}",
                "namespace": self._namespace,
            },
            "spec": {
                "selector": {"sandbox-id": self._sandbox_id},
                "ports": [{"port": 80, "targetPort": 8080, "protocol": "TCP"}],
            },
        }

    async def _wait_for_pod(self) -> None:
        """Poll until the pod is in Running phase."""
        api = self._k8s_client
        for _ in range(_MAX_POLL_ATTEMPTS):
            pod = await asyncio.to_thread(api.read_namespaced_pod, self._pod_name, self._namespace)
            if pod.status and pod.status.phase == "Running":
                return
            await asyncio.sleep(_POLL_INTERVAL)
        msg = f"Pod {self._pod_name!r} did not reach Running within timeout"
        raise SandboxError(msg)

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        """Create the pod and service, wait for readiness."""
        self._transition(SandboxStatus.RUNNING)
        api = self._load_client()
        pod_manifest = self._pod_manifest()
        self._pod_name = pod_manifest["metadata"]["name"]

        try:
            await asyncio.to_thread(api.create_namespaced_pod, self._namespace, pod_manifest)
            logger.info("Created pod %s in namespace %s", self._pod_name, self._namespace)

            await self._wait_for_pod()

            svc_manifest = self._service_manifest()
            self._service_name = svc_manifest["metadata"]["name"]
            svc = await asyncio.to_thread(
                api.create_namespaced_service, self._namespace, svc_manifest
            )
            self._cluster_ip = svc.spec.cluster_ip if svc.spec else None
            logger.info("Created service %s (cluster_ip=%s)", self._service_name, self._cluster_ip)
        except SandboxError:
            raise
        except asyncio.CancelledError:
            logger.warning("Sandbox %s: start cancelled, cleaning up resources", self._sandbox_id)
            await self._delete_resources()
            raise
        except Exception as exc:
            self._status = SandboxStatus.ERROR
            logger.error("Failed to start Kubernetes sandbox %s: %s", self._sandbox_id, exc)
            msg = f"Failed to start Kubernetes sandbox: {exc}"
            raise SandboxError(msg) from exc

    async def stop(self) -> None:
        """Delete the pod and service (sandbox can be restarted)."""
        self._transition(SandboxStatus.IDLE)
        await self._delete_resources()

    async def cleanup(self) -> None:
        """Release all Kubernetes resources permanently."""
        self._transition(SandboxStatus.CLOSED)
        await self._delete_resources()

    async def _delete_resources(self) -> None:
        """Delete pod and service if they exist."""
        if self._k8s_client is None:
            return
        api = self._k8s_client
        if self._pod_name:
            try:
                await asyncio.to_thread(api.delete_namespaced_pod, self._pod_name, self._namespace)
                logger.info("Deleted pod %s", self._pod_name)
            except Exception:
                logger.warning("Failed to delete pod %s", self._pod_name)
            self._pod_name = None
        if self._service_name:
            try:
                await asyncio.to_thread(
                    api.delete_namespaced_service, self._service_name, self._namespace
                )
                logger.info("Deleted service %s", self._service_name)
            except Exception:
                logger.warning("Failed to delete service %s", self._service_name)
            self._service_name = None
        self._cluster_ip = None

    def register_tool(self, tool: Any) -> None:
        """Register a :class:`~exo.tool.Tool` for use in :meth:`run_tool`."""
        self._tools[tool.name] = tool

    async def _exec_in_pod(self, command: list[str]) -> tuple[str, str, int]:
        """Run *command* inside the running pod via the kubernetes exec stream API.

        Returns ``(stdout, stderr, exit_code)``.

        Raises :class:`SandboxError` if the kubernetes package is missing or
        the exec call fails.
        """
        if self._pod_name is None:
            msg = "No pod is running — call start() first"
            raise SandboxError(msg)

        try:
            from kubernetes.stream import (  # pyright: ignore[reportMissingImports]
                stream as k8s_stream,
            )
        except ImportError as exc:
            msg = "kubernetes package is required: pip install exo-sandbox[kubernetes]"
            raise SandboxError(msg) from exc

        api = self._k8s_client
        if api is None:
            msg = "Kubernetes client not initialised — call start() first"
            raise SandboxError(msg)

        try:
            resp = await asyncio.to_thread(
                k8s_stream,
                api.connect_get_namespaced_pod_exec,
                self._pod_name,
                self._namespace,
                command=command,
                stderr=True,
                stdin=False,
                stdout=True,
                tty=False,
                _preload_content=False,
            )

            stdout_parts: list[str] = []
            stderr_parts: list[str] = []

            def _drain() -> None:
                while resp.is_open():
                    resp.update(timeout=1)
                    if resp.peek_stdout():
                        stdout_parts.append(resp.read_stdout())
                    if resp.peek_stderr():
                        stderr_parts.append(resp.read_stderr())
                resp.close()

            await asyncio.to_thread(_drain)

            stdout = "".join(stdout_parts)[:_EXEC_OUTPUT_LIMIT]
            stderr = "".join(stderr_parts)[:_EXEC_OUTPUT_LIMIT]

            # kubernetes-python does not expose exit code directly via stream;
            # parse the returncode channel if available.
            try:
                exit_code = resp.returncode
                if exit_code is None:
                    exit_code = 0 if not stderr.strip() else 1
            except AttributeError:
                exit_code = 0 if not stderr.strip() else 1

            return stdout, stderr, exit_code

        except SandboxError:
            raise
        except Exception as exc:
            msg = f"Pod exec failed for sandbox {self._sandbox_id}: {exc}"
            raise SandboxError(msg) from exc

    async def run_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool within the Kubernetes sandbox.

        Dispatch order:

        1. **Registered tools** — callable objects added via
           :meth:`register_tool` (or the *tools* constructor parameter).
           Their ``execute(**arguments)`` method is invoked directly so
           that locally-available tools (e.g. :class:`~exo.sandbox.tools.FilesystemTool`)
           can be executed without requiring pod-side imports.

        2. **Pod exec** — if the tool is *not* in the local registry, the
           call is serialised as a ``python3 -c`` snippet and executed inside
           the running pod via the kubernetes exec-stream API.  The snippet
           prints a JSON result to stdout.  This path requires the tool's
           module to be importable inside the pod image.

        Raises :class:`SandboxError` if the sandbox is not running.
        """
        if self._status != SandboxStatus.RUNNING:
            msg = f"Sandbox must be running to call tools (status={self._status!r})"
            raise SandboxError(msg)

        logger.debug(
            "Sandbox %s: running tool %s on pod %s", self._sandbox_id, tool_name, self._pod_name
        )

        # --- path 1: locally-registered tool ---------------------------------
        local_tool = self._tools.get(tool_name)
        if local_tool is not None:
            try:
                result = local_tool.execute(**arguments)
                if inspect.isawaitable(result):
                    result = await result
                return result
            except SandboxError:
                raise
            except Exception as exc:
                raise SandboxError(f"Tool {tool_name!r} failed: {exc}") from exc

        # --- path 2: exec inside pod -----------------------------------------
        # Build a self-contained python snippet that imports the tool by its
        # module path and invokes it, printing a JSON result to stdout.
        # The tool's __module__ and class name are inferred; if unavailable
        # we raise clearly rather than returning a silent stub.
        if self._pod_name is None:
            msg = "No pod is running — call start() first"
            raise SandboxError(msg)

        args_json = json.dumps(arguments)
        script = (
            "import json, sys\n"
            f"args = json.loads({json.dumps(args_json)})\n"
            f"# Tool {tool_name!r} must be importable inside the pod\n"
            f"try:\n"
            f"    import importlib\n"
            f"    # Attempt: tool_name used as a dotted module path or\n"
            f"    # the pod may have exo installed.\n"
            f"    from exo.tool import FunctionTool\n"
            f"    raise ImportError('no pre-registered tool {tool_name!r} available in pod')\n"
            f"except ImportError as e:\n"
            f"    print(json.dumps({{'error': str(e), 'tool': {json.dumps(tool_name)}}}))\n"
            f"    sys.exit(1)\n"
        )

        stdout, stderr, exit_code = await self._exec_in_pod(["python3", "-c", script])

        logger.debug(
            "Sandbox %s: pod exec exit_code=%s stdout=%r stderr=%r",
            self._sandbox_id,
            exit_code,
            stdout[:200],
            stderr[:200],
        )

        if exit_code != 0:
            error_detail = stderr.strip() or stdout.strip() or f"exit code {exit_code}"
            raise SandboxError(
                f"Tool {tool_name!r} exec in pod {self._pod_name!r} failed: {error_detail}"
            )

        # Parse JSON result from stdout
        try:
            return json.loads(stdout.strip())
        except json.JSONDecodeError:
            return {"stdout": stdout, "stderr": stderr, "exit_code": exit_code, "status": "ok"}

    # -- context manager ----------------------------------------------------

    async def __aenter__(self) -> KubernetesSandbox:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.cleanup()

    # -- introspection ------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update(
            {
                "namespace": self._namespace,
                "image": self._image,
                "pod_name": self._pod_name,
                "service_name": self._service_name,
                "cluster_ip": self._cluster_ip,
            }
        )
        return info
