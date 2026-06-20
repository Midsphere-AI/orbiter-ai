"""Tests for KubernetesSandbox."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from exo.sandbox.base import (  # pyright: ignore[reportMissingImports]
    SandboxError,
    SandboxStatus,
)
from exo.sandbox.kubernetes import (  # pyright: ignore[reportMissingImports]
    _DEFAULT_IMAGE,
    _DEFAULT_NAMESPACE,
    KubernetesSandbox,
)
from exo.tool import FunctionTool  # pyright: ignore[reportMissingImports]


def _make_echo_tool() -> FunctionTool:
    async def echo(message: str) -> dict[str, Any]:
        return {"echo": message}

    return FunctionTool(echo, name="echo")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_k8s_api() -> MagicMock:
    """Create a mock CoreV1Api."""
    api = MagicMock()

    # read_namespaced_pod returns a pod with Running phase
    pod = MagicMock()
    pod.status.phase = "Running"
    api.read_namespaced_pod.return_value = pod

    # create_namespaced_pod returns a pod object
    api.create_namespaced_pod.return_value = MagicMock()

    # create_namespaced_service returns a service with cluster_ip
    svc = MagicMock()
    svc.spec.cluster_ip = "10.0.0.42"
    api.create_namespaced_service.return_value = svc

    # delete operations succeed
    api.delete_namespaced_pod.return_value = MagicMock()
    api.delete_namespaced_service.return_value = MagicMock()

    return api


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------


class TestInit:
    def test_defaults(self) -> None:
        sb = KubernetesSandbox()
        assert sb.namespace == _DEFAULT_NAMESPACE
        assert sb.image == _DEFAULT_IMAGE
        assert sb.status == SandboxStatus.INIT
        assert sb.pod_name is None
        assert sb.cluster_ip is None

    def test_custom_values(self) -> None:
        sb = KubernetesSandbox(
            sandbox_id="test-123",
            namespace="prod",
            image="myimage:latest",
            timeout=120.0,
            workspace=["ws1"],
            mcp_config={"key": "val"},
            agents={"a": "b"},
        )
        assert sb.sandbox_id == "test-123"
        assert sb.namespace == "prod"
        assert sb.image == "myimage:latest"
        assert sb.timeout == 120.0
        assert sb.workspace == ["ws1"]
        assert sb.mcp_config == {"key": "val"}
        assert sb.agents == {"a": "b"}

    def test_env_var_namespace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXO_K8S_NAMESPACE", "staging")
        sb = KubernetesSandbox()
        assert sb.namespace == "staging"

    def test_env_var_image(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXO_K8S_IMAGE", "custom:v2")
        sb = KubernetesSandbox()
        assert sb.image == "custom:v2"


# ---------------------------------------------------------------------------
# TestLoadClient
# ---------------------------------------------------------------------------


class TestLoadClient:
    def test_missing_kubernetes_package(self) -> None:
        sb = KubernetesSandbox()
        with (
            patch.dict(
                "sys.modules",
                {"kubernetes": None, "kubernetes.client": None, "kubernetes.config": None},
            ),
            pytest.raises(SandboxError, match="kubernetes package is required"),
        ):
            sb._load_client()

    def test_load_client_caches(self) -> None:
        sb = KubernetesSandbox()
        mock_api = _mock_k8s_api()
        sb._k8s_client = mock_api
        result = sb._load_client()
        assert result is mock_api


# ---------------------------------------------------------------------------
# TestManifests
# ---------------------------------------------------------------------------


class TestManifests:
    def test_pod_manifest(self) -> None:
        sb = KubernetesSandbox(sandbox_id="abc", namespace="test-ns", image="myimg:v1")
        manifest = sb._pod_manifest()
        assert manifest["metadata"]["name"] == "exo-abc"
        assert manifest["metadata"]["namespace"] == "test-ns"
        assert manifest["metadata"]["labels"]["sandbox-id"] == "abc"
        assert manifest["spec"]["containers"][0]["image"] == "myimg:v1"
        assert manifest["spec"]["restartPolicy"] == "Never"

    def test_service_manifest(self) -> None:
        sb = KubernetesSandbox(sandbox_id="xyz", namespace="dev")
        manifest = sb._service_manifest()
        assert manifest["metadata"]["name"] == "exo-svc-xyz"
        assert manifest["metadata"]["namespace"] == "dev"
        assert manifest["spec"]["selector"]["sandbox-id"] == "xyz"


# ---------------------------------------------------------------------------
# TestStart
# ---------------------------------------------------------------------------


class TestStart:
    async def test_start_creates_pod_and_service(self) -> None:
        sb = KubernetesSandbox(sandbox_id="s1")
        api = _mock_k8s_api()
        sb._k8s_client = api

        await sb.start()

        assert sb.status == SandboxStatus.RUNNING
        assert sb.pod_name == "exo-s1"
        assert sb.cluster_ip == "10.0.0.42"
        api.create_namespaced_pod.assert_called_once()
        api.create_namespaced_service.assert_called_once()

    async def test_start_waits_for_pod_ready(self) -> None:
        sb = KubernetesSandbox(sandbox_id="s2")
        api = _mock_k8s_api()

        # First poll: Pending, second poll: Running
        pending_pod = MagicMock()
        pending_pod.status.phase = "Pending"
        running_pod = MagicMock()
        running_pod.status.phase = "Running"
        api.read_namespaced_pod.side_effect = [pending_pod, running_pod]

        sb._k8s_client = api
        await sb.start()

        assert sb.status == SandboxStatus.RUNNING
        assert api.read_namespaced_pod.call_count == 2

    async def test_start_error_sets_error_status(self) -> None:
        sb = KubernetesSandbox(sandbox_id="s3")
        api = _mock_k8s_api()
        api.create_namespaced_pod.side_effect = RuntimeError("API error")
        sb._k8s_client = api

        with pytest.raises(SandboxError, match="Failed to start"):
            await sb.start()
        assert sb.status == SandboxStatus.ERROR

    async def test_start_pod_timeout(self) -> None:
        sb = KubernetesSandbox(sandbox_id="s4")
        api = _mock_k8s_api()

        # Pod never becomes Running
        pending_pod = MagicMock()
        pending_pod.status.phase = "Pending"
        api.read_namespaced_pod.return_value = pending_pod

        sb._k8s_client = api

        # Patch _MAX_POLL_ATTEMPTS to avoid long test
        with (
            patch("exo.sandbox.kubernetes._MAX_POLL_ATTEMPTS", 2),
            patch("exo.sandbox.kubernetes._POLL_INTERVAL", 0.01),
            pytest.raises(SandboxError, match="did not reach Running"),
        ):
            await sb.start()


# ---------------------------------------------------------------------------
# TestStop
# ---------------------------------------------------------------------------


class TestStop:
    async def test_stop_deletes_resources(self) -> None:
        sb = KubernetesSandbox(sandbox_id="s5")
        api = _mock_k8s_api()
        sb._k8s_client = api
        await sb.start()

        await sb.stop()

        assert sb.status == SandboxStatus.IDLE
        assert sb.pod_name is None
        assert sb.cluster_ip is None
        api.delete_namespaced_pod.assert_called_once()
        api.delete_namespaced_service.assert_called_once()

    async def test_stop_no_client(self) -> None:
        sb = KubernetesSandbox(sandbox_id="s6")
        api = _mock_k8s_api()
        sb._k8s_client = api
        await sb.start()

        sb._k8s_client = None
        await sb.stop()
        assert sb.status == SandboxStatus.IDLE


# ---------------------------------------------------------------------------
# TestCleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    async def test_cleanup_closes(self) -> None:
        sb = KubernetesSandbox(sandbox_id="s7")
        api = _mock_k8s_api()
        sb._k8s_client = api
        await sb.start()

        await sb.cleanup()

        assert sb.status == SandboxStatus.CLOSED
        api.delete_namespaced_pod.assert_called_once()
        api.delete_namespaced_service.assert_called_once()

    async def test_cleanup_tolerates_delete_errors(self) -> None:
        sb = KubernetesSandbox(sandbox_id="s8")
        api = _mock_k8s_api()
        sb._k8s_client = api
        await sb.start()

        api.delete_namespaced_pod.side_effect = RuntimeError("gone")
        api.delete_namespaced_service.side_effect = RuntimeError("gone")

        await sb.cleanup()  # should not raise
        assert sb.status == SandboxStatus.CLOSED


# ---------------------------------------------------------------------------
# TestRunTool
# ---------------------------------------------------------------------------


class TestRunTool:
    async def test_run_tool_registered_tool(self) -> None:
        """Registered tool is executed locally, bypassing pod exec."""
        echo = _make_echo_tool()
        sb = KubernetesSandbox(sandbox_id="s9", tools={"echo": echo})
        api = _mock_k8s_api()
        sb._k8s_client = api
        await sb.start()

        result = await sb.run_tool("echo", {"message": "hi"})
        assert result == {"echo": "hi"}

    async def test_run_tool_register_method(self) -> None:
        """register_tool() adds a tool callable for run_tool dispatch."""
        echo = _make_echo_tool()
        sb = KubernetesSandbox(sandbox_id="s9b")
        api = _mock_k8s_api()
        sb._k8s_client = api
        await sb.start()
        sb.register_tool(echo)

        result = await sb.run_tool("echo", {"message": "via register"})
        assert result == {"echo": "via register"}

    async def test_run_tool_not_running(self) -> None:
        sb = KubernetesSandbox(sandbox_id="s10")
        with pytest.raises(SandboxError, match="Sandbox must be running"):
            await sb.run_tool("tool", {})

    async def test_run_tool_unregistered_pod_exec_fails_no_client(self) -> None:
        """Unregistered tool triggers pod exec path; raises if k8s unavailable."""
        sb = KubernetesSandbox(sandbox_id="s11")
        api = _mock_k8s_api()
        sb._k8s_client = api
        await sb.start()

        # Mock the kubernetes stream import to simulate missing package
        with (
            patch.dict(
                "sys.modules",
                {
                    "kubernetes.stream": None,
                    "kubernetes.stream.stream": None,
                },
            ),
            pytest.raises(SandboxError),
        ):
            await sb.run_tool("unknown_tool", {})


# ---------------------------------------------------------------------------
# TestContextManager
# ---------------------------------------------------------------------------


class TestContextManager:
    async def test_async_context_manager(self) -> None:
        api = _mock_k8s_api()
        sb = KubernetesSandbox(sandbox_id="cm1")
        sb._k8s_client = api

        async with sb as s:
            assert s.status == SandboxStatus.RUNNING
            assert s is sb

        assert sb.status == SandboxStatus.CLOSED


# ---------------------------------------------------------------------------
# TestDescribe
# ---------------------------------------------------------------------------


class TestDescribe:
    def test_describe_before_start(self) -> None:
        sb = KubernetesSandbox(sandbox_id="d1", namespace="ns", image="img:v1")
        info = sb.describe()
        assert info["sandbox_id"] == "d1"
        assert info["namespace"] == "ns"
        assert info["image"] == "img:v1"
        assert info["pod_name"] is None
        assert info["service_name"] is None
        assert info["cluster_ip"] is None

    async def test_describe_after_start(self) -> None:
        sb = KubernetesSandbox(sandbox_id="d2")
        api = _mock_k8s_api()
        sb._k8s_client = api
        await sb.start()

        info = sb.describe()
        assert info["pod_name"] == "exo-d2"
        assert info["service_name"] == "exo-svc-d2"
        assert info["cluster_ip"] == "10.0.0.42"

    def test_repr(self) -> None:
        sb = KubernetesSandbox(sandbox_id="r1")
        r = repr(sb)
        assert "KubernetesSandbox" in r
        assert "r1" in r
