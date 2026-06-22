"""Tests for the LangSmith tracing backend.

No network calls, no real API keys. The OTLP exporter is available in the test
environment (opentelemetry-exporter-otlp is installed); langsmith SDK is NOT
installed, so dataset_client() paths test the graceful-None branch.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reload_langsmith() -> Any:
    """Force a fresh import of the langsmith backend module."""
    mod_name = "exo.observability.backends.langsmith"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    return importlib.import_module(mod_name)


# ---------------------------------------------------------------------------
# Import & self-registration
# ---------------------------------------------------------------------------


class TestSelfRegistration:
    def test_module_imports_without_error(self) -> None:
        """Module-level import must never raise even with no env or extras."""
        import exo.observability.backends.langsmith  # pyright: ignore[reportMissingImports]  # noqa: F401

        # If we reach here, the import succeeded without raising.
        assert True

    def test_registered_in_registry(self) -> None:
        import exo.observability.backends.langsmith  # pyright: ignore[reportMissingImports]  # noqa: F401
        from exo.observability.backends import _REGISTRY  # pyright: ignore[reportMissingImports]

        assert "langsmith" in _REGISTRY

    def test_factory_returns_langsmith_backend(self) -> None:
        from exo.observability.backends import _REGISTRY  # pyright: ignore[reportMissingImports]

        backend = _REGISTRY["langsmith"]()
        assert backend.name == "langsmith"


# ---------------------------------------------------------------------------
# resolve_backends integration
# ---------------------------------------------------------------------------


class TestResolveBackends:
    def test_returns_empty_when_no_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No key -> is_available() is False -> resolve_backends skips it."""
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)

        # Re-register with a fresh backend that has no key set.
        _reload_langsmith()

        from exo.observability.backends import (
            resolve_backends,  # pyright: ignore[reportMissingImports]
        )

        result = resolve_backends("langsmith")
        assert result == []

    def test_returns_backend_when_key_set_and_otel_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Key present + OTLP exporter available -> resolve_backends returns a backend."""
        otel = pytest.importorskip(
            "opentelemetry.exporter.otlp.proto.http.trace_exporter",
            reason="opentelemetry-exporter-otlp not installed",
        )
        assert otel is not None

        monkeypatch.setenv("LANGSMITH_API_KEY", "test-key-xyz")

        _reload_langsmith()

        from exo.observability.backends import (
            resolve_backends,  # pyright: ignore[reportMissingImports]
        )

        result = resolve_backends("langsmith")
        assert len(result) == 1
        assert result[0].name == "langsmith"


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------


class TestIsAvailable:
    def test_false_when_no_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)

        from exo.observability.backends.langsmith import (
            LangSmithBackend,  # pyright: ignore[reportMissingImports]
        )

        backend = LangSmithBackend()
        assert backend.is_available() is False

    def test_false_when_key_empty_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGSMITH_API_KEY", "")
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)

        from exo.observability.backends.langsmith import (
            LangSmithBackend,  # pyright: ignore[reportMissingImports]
        )

        backend = LangSmithBackend()
        assert backend.is_available() is False

    def test_true_when_key_set_and_otel_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip(
            "opentelemetry.exporter.otlp.proto.http.trace_exporter",
            reason="opentelemetry-exporter-otlp not installed",
        )
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test_key")

        from exo.observability.backends.langsmith import (
            LangSmithBackend,  # pyright: ignore[reportMissingImports]
        )

        backend = LangSmithBackend()
        assert backend.is_available() is True

    def test_false_when_otel_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test_key")

        from exo.observability.backends.langsmith import (
            LangSmithBackend,  # pyright: ignore[reportMissingImports]
        )

        with patch.dict(
            sys.modules, {"opentelemetry.exporter.otlp.proto.http.trace_exporter": None}
        ):  # type: ignore[dict-item]
            backend = LangSmithBackend()
            assert backend.is_available() is False


# ---------------------------------------------------------------------------
# Endpoint & header construction
# ---------------------------------------------------------------------------


class TestEndpointAndHeaders:
    def test_default_endpoint(self) -> None:
        from exo.observability.backends.langsmith import (
            LangSmithBackend,  # pyright: ignore[reportMissingImports]
        )

        backend = LangSmithBackend(api_key="test-key")
        assert backend.endpoint == "https://api.smith.langchain.com/otel/v1/traces"

    def test_custom_endpoint(self) -> None:
        from exo.observability.backends.langsmith import (
            LangSmithBackend,  # pyright: ignore[reportMissingImports]
        )

        backend = LangSmithBackend(api_key="test-key", endpoint="https://custom.example.com")
        assert backend.endpoint == "https://custom.example.com/otel/v1/traces"

    def test_trailing_slash_stripped(self) -> None:
        from exo.observability.backends.langsmith import (
            LangSmithBackend,  # pyright: ignore[reportMissingImports]
        )

        backend = LangSmithBackend(api_key="test-key", endpoint="https://custom.example.com/")
        assert backend.endpoint == "https://custom.example.com/otel/v1/traces"

    def test_api_key_in_headers(self) -> None:
        from exo.observability.backends.langsmith import (
            LangSmithBackend,  # pyright: ignore[reportMissingImports]
        )

        backend = LangSmithBackend(api_key="my-secret-key")
        assert backend.headers["x-api-key"] == "my-secret-key"

    def test_project_in_headers_default(self) -> None:
        from exo.observability.backends.langsmith import (
            LangSmithBackend,  # pyright: ignore[reportMissingImports]
        )

        backend = LangSmithBackend(api_key="test-key")
        assert backend.headers["Langsmith-Project"] == "default"

    def test_project_in_headers_explicit(self) -> None:
        from exo.observability.backends.langsmith import (
            LangSmithBackend,  # pyright: ignore[reportMissingImports]
        )

        backend = LangSmithBackend(api_key="test-key", project="my-project")
        assert backend.headers["Langsmith-Project"] == "my-project"

    def test_env_api_key_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGSMITH_API_KEY", "env-key")
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)

        from exo.observability.backends.langsmith import (
            LangSmithBackend,  # pyright: ignore[reportMissingImports]
        )

        backend = LangSmithBackend()
        assert backend.headers["x-api-key"] == "env-key"

    def test_langchain_api_key_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        monkeypatch.setenv("LANGCHAIN_API_KEY", "lc-fallback-key")

        from exo.observability.backends.langsmith import (
            LangSmithBackend,  # pyright: ignore[reportMissingImports]
        )

        backend = LangSmithBackend()
        assert backend.headers["x-api-key"] == "lc-fallback-key"

    def test_env_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://eu.smith.langchain.com")

        from exo.observability.backends.langsmith import (
            LangSmithBackend,  # pyright: ignore[reportMissingImports]
        )

        backend = LangSmithBackend(api_key="test-key")
        assert backend.endpoint == "https://eu.smith.langchain.com/otel/v1/traces"

    def test_env_project(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGSMITH_PROJECT", "prod-evals")

        from exo.observability.backends.langsmith import (
            LangSmithBackend,  # pyright: ignore[reportMissingImports]
        )

        backend = LangSmithBackend(api_key="test-key")
        assert backend.headers["Langsmith-Project"] == "prod-evals"

    def test_langchain_project_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
        monkeypatch.setenv("LANGCHAIN_PROJECT", "lc-project")

        from exo.observability.backends.langsmith import (
            LangSmithBackend,  # pyright: ignore[reportMissingImports]
        )

        backend = LangSmithBackend(api_key="test-key")
        assert backend.headers["Langsmith-Project"] == "lc-project"

    def test_explicit_args_override_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGSMITH_API_KEY", "env-key")
        monkeypatch.setenv("LANGSMITH_PROJECT", "env-project")
        monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://env.example.com")

        from exo.observability.backends.langsmith import (
            LangSmithBackend,  # pyright: ignore[reportMissingImports]
        )

        backend = LangSmithBackend(
            api_key="explicit-key",
            endpoint="https://explicit.example.com",
            project="explicit-project",
        )
        assert backend.headers["x-api-key"] == "explicit-key"
        assert backend.headers["Langsmith-Project"] == "explicit-project"
        assert backend.endpoint == "https://explicit.example.com/otel/v1/traces"


# ---------------------------------------------------------------------------
# dataset_client — graceful None when SDK absent
# ---------------------------------------------------------------------------


class TestDatasetClient:
    def test_returns_none_when_sdk_not_installed(self) -> None:
        """langsmith SDK isn't installed in the test environment -> None, no raise."""
        from exo.observability.backends.langsmith import (
            LangSmithBackend,  # pyright: ignore[reportMissingImports]
        )

        backend = LangSmithBackend(api_key="test-key")

        # Mask out the langsmith package to simulate it not being installed.
        with patch.dict(sys.modules, {"langsmith": None}):  # type: ignore[dict-item]
            result = backend.dataset_client()

        assert result is None

    def test_returns_none_when_no_api_key(self) -> None:
        from exo.observability.backends.langsmith import (
            LangSmithBackend,  # pyright: ignore[reportMissingImports]
        )

        backend = LangSmithBackend()  # no key
        result = backend.dataset_client()
        assert result is None

    def test_score_client_mirrors_dataset_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """score_client() is an alias for dataset_client(); both return None here."""
        from exo.observability.backends.langsmith import (
            LangSmithBackend,  # pyright: ignore[reportMissingImports]
        )

        backend = LangSmithBackend(api_key="test-key")

        with patch.dict(sys.modules, {"langsmith": None}):  # type: ignore[dict-item]
            assert backend.score_client() is None


# ---------------------------------------------------------------------------
# build_span_processor smoke test (OTLP path)
# ---------------------------------------------------------------------------


class TestBuildSpanProcessor:
    def test_returns_processor_when_otel_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip(
            "opentelemetry.exporter.otlp.proto.http.trace_exporter",
            reason="opentelemetry-exporter-otlp not installed",
        )
        monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")

        from exo.observability.backends.langsmith import (
            LangSmithBackend,  # pyright: ignore[reportMissingImports]
        )

        backend = LangSmithBackend()
        processor = backend.build_span_processor()
        assert processor is not None

    def test_returns_none_when_otel_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")

        from exo.observability.backends.langsmith import (
            LangSmithBackend,  # pyright: ignore[reportMissingImports]
        )

        backend = LangSmithBackend()
        with patch.dict(
            sys.modules,
            {
                "opentelemetry.exporter.otlp.proto.http.trace_exporter": None,  # type: ignore[dict-item]
                "opentelemetry.sdk.trace.export": None,  # type: ignore[dict-item]
            },
        ):
            processor = backend.build_span_processor()
        assert processor is None
