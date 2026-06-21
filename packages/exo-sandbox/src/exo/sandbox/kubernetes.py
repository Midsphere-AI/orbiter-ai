"""Kubernetes-based sandbox for remote agent execution."""

from __future__ import annotations

import asyncio
import inspect
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
        "_image",
        "_k8s_client",
        "_namespace",
        "_pod_name",
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
        """Create the pod and wait for readiness."""
        self._transition(SandboxStatus.RUNNING)
        api = self._load_client()
        pod_manifest = self._pod_manifest()
        self._pod_name = pod_manifest["metadata"]["name"]

        try:
            await asyncio.to_thread(api.create_namespaced_pod, self._namespace, pod_manifest)
            logger.info("Created pod %s in namespace %s", self._pod_name, self._namespace)

            await self._wait_for_pod()
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
        """Delete the pod if it exists."""
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

    def register_tool(self, tool: Any) -> None:
        """Register a :class:`~exo.tool.Tool` for use in :meth:`run_tool`."""
        self._tools[tool.name] = tool

    async def run_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool within the Kubernetes sandbox.

        Only tools registered via :meth:`register_tool` (or the *tools*
        constructor parameter) are supported.  Their ``execute(**arguments)``
        method is invoked directly.

        Unregistered tools raise :class:`SandboxError` immediately — the pod
        runs a plain ``sleep infinity`` process with no tool dispatch layer, so
        there is no way to execute arbitrary tools inside it without a
        pod-side agent.

        Raises :class:`SandboxError` if the sandbox is not running or the tool
        is not registered.
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

        # --- path 2: unregistered tool ---------------------------------------
        # Tools must be registered locally via register_tool() or the
        # *tools* constructor parameter before they can be used.  The pod
        # runs a plain `sleep infinity` process with no tool dispatch layer,
        # so there is no way to execute an arbitrary tool inside it without
        # a pod-side agent.  Raise a clear error rather than silently failing.
        msg = (
            f"Tool {tool_name!r} is not registered in this sandbox. "
            "Register tools via KubernetesSandbox(tools=...) or register_tool() "
            "before calling run_tool()."
        )
        raise SandboxError(msg)

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
            }
        )
        return info
