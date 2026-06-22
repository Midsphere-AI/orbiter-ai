"""Kubernetes-based sandbox for remote agent execution."""

from __future__ import annotations

import asyncio
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
        # Seed the shared registry from the constructor `tools` mapping.  Use
        # register_tool() so the duplicate-key guard fires for callers who pass
        # the same name twice (e.g. via both positional and keyword args).
        if tools:
            for name, tool in tools.items():
                self.register_tool(name, self._tool_to_handler(tool))

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
            # In-cluster config failed (or an explicit KUBECONFIG was bad).
            # Fall back to the default kubeconfig.  Python's implicit exception
            # chaining preserves the first failure as __context__; the second
            # failure is chained explicitly via ``raise ... from``.
            try:
                config.load_kube_config()
            except Exception as cfg_exc:
                raise SandboxError(
                    "Failed to load Kubernetes configuration "
                    "(tried in-cluster credentials and default kubeconfig).",
                    context={"kubeconfig": kubeconfig_path or "(default)"},
                    hint=(
                        "Set KUBECONFIG env var to your kubeconfig path, "
                        "or ensure in-cluster service account credentials are mounted."
                    ),
                ) from cfg_exc

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
        """Poll until the pod is in Running phase.

        Polling is bounded by two independent limits:

        * **``_MAX_POLL_ATTEMPTS``** — maximum number of status polls before
          giving up.  Patching this constant in tests avoids needing a long
          wall-clock ``timeout``.
        * **``self._timeout``** — hard wall-clock deadline via
          ``asyncio.wait_for``, guarding against a blocking API call that
          never returns.
        """
        api = self._k8s_client

        async def _poll_loop() -> None:
            for _ in range(_MAX_POLL_ATTEMPTS):
                pod = await asyncio.to_thread(
                    api.read_namespaced_pod, self._pod_name, self._namespace
                )
                if pod.status and pod.status.phase == "Running":
                    return
                await asyncio.sleep(_POLL_INTERVAL)
            # Exceeded max attempts — raise as SandboxError (not TimeoutError)
            # so the outer except TimeoutError block doesn't swallow it.
            raise SandboxError(
                f"Pod {self._pod_name!r} did not reach Running after {_MAX_POLL_ATTEMPTS} polls.",
                context={
                    "pod": self._pod_name,
                    "namespace": self._namespace,
                    "poll_attempts": _MAX_POLL_ATTEMPTS,
                },
                hint=(
                    "Check pod events with `kubectl describe pod "
                    f"{self._pod_name} -n {self._namespace}` "
                    "or increase the sandbox timeout."
                ),
            )

        try:
            await asyncio.wait_for(_poll_loop(), timeout=self._timeout)
        except SandboxError:
            raise
        except TimeoutError as exc:
            raise SandboxError(
                f"Pod {self._pod_name!r} did not reach Running within {self._timeout}s.",
                context={
                    "pod": self._pod_name,
                    "namespace": self._namespace,
                    "timeout": self._timeout,
                },
                hint=(
                    "Check pod events with `kubectl describe pod "
                    f"{self._pod_name} -n {self._namespace}` "
                    "or increase the sandbox timeout."
                ),
            ) from exc

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
            raise SandboxError(
                "Failed to start Kubernetes sandbox.",
                context={
                    "pod": self._pod_name,
                    "namespace": self._namespace,
                    "image": self._image,
                },
                hint=(
                    "Run `kubectl get pods -n "
                    f"{self._namespace}` to check pod status, "
                    "or verify the kubeconfig is correct."
                ),
            ) from exc

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
            except Exception as exc:
                logger.warning(
                    "Failed to delete pod %s (namespace=%s, sandbox=%s): %s",
                    self._pod_name,
                    self._namespace,
                    self._sandbox_id,
                    exc,
                )
            self._pod_name = None

    # ``register_tool`` / ``unregister_tool`` / ``registered_tools`` are
    # inherited from :class:`~exo.sandbox.base.Sandbox`. The everyday form here
    # is ``sandbox.register_tool(my_tool)`` (a Tool with .name + .execute), but
    # ``register_tool("name", handler)`` works too — identical to every backend.

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
        handled, result = await self._run_registered(tool_name, arguments)
        if handled:
            return result

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
