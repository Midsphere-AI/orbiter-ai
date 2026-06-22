"""Tests for the Braintrust tracing backend.

All tests are offline (no network). OTLP-dependent paths are skipped via
``pytest.importorskip`` when the exporter extra is absent.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reload_braintrust() -> None:
    """Force re-import of the braintrust backend module so env patches take effect."""
    mod_name = "exo.observability.backends.braintrust"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    importlib.import_module(mod_name)


# ---------------------------------------------------------------------------
# Import & self-registration
# ---------------------------------------------------------------------------


class TestSelfRegistration:
    def test_import_does_not_raise(self) -> None:
        """Importing the module must never raise even with no deps/env."""
        import exo.observability.backends.braintrust  # pyright: ignore[reportMissingImports]  # noqa: F401

    def test_registered_after_import(self) -> None:
        """After import, 'braintrust' must appear in the registry."""
        import exo.observability.backends.braintrust  # pyright: ignore[reportMissingImports]  # noqa: F401
        from exo.observability.backends import _REGISTRY  # pyright: ignore[reportMissingImports]

        assert "braintrust" in _REGISTRY


# ---------------------------------------------------------------------------
# is_available()
# ---------------------------------------------------------------------------


class TestIsAvailable:
    def test_false_when_key_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)
        _reload_braintrust()
        from exo.observability.backends.braintrust import (  # pyright: ignore[reportMissingImports]
            BraintrustBackend,
        )

        backend = BraintrustBackend()
        assert backend.is_available() is False

    def test_false_when_key_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BRAINTRUST_API_KEY", "")
        _reload_braintrust()
        from exo.observability.backends.braintrust import (  # pyright: ignore[reportMissingImports]
            BraintrustBackend,
        )

        backend = BraintrustBackend()
        assert backend.is_available() is False

    def test_true_when_key_set_and_otel_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip(
            "opentelemetry.exporter.otlp.proto.http.trace_exporter",
            reason="OTLP exporter extra not installed",
        )
        monkeypatch.setenv("BRAINTRUST_API_KEY", "bt_test_key_123")
        _reload_braintrust()
        from exo.observability.backends.braintrust import (  # pyright: ignore[reportMissingImports]
            BraintrustBackend,
        )

        backend = BraintrustBackend()
        assert backend.is_available() is True

    def test_explicit_key_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)
        _reload_braintrust()
        from exo.observability.backends.braintrust import (  # pyright: ignore[reportMissingImports]
            BraintrustBackend,
        )

        # No env key, but explicit constructor arg — is_available depends on OTLP too,
        # so we just check that the key was accepted without error.
        backend = BraintrustBackend(api_key="explicit_key")
        # Resolved key must be populated regardless of OTLP presence.
        assert backend._resolved_key == "explicit_key"


# ---------------------------------------------------------------------------
# resolve_backends() integration
# ---------------------------------------------------------------------------


class TestResolveBackends:
    def test_returns_empty_when_key_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)
        _reload_braintrust()
        from exo.observability.backends import (
            resolve_backends,  # pyright: ignore[reportMissingImports]
        )

        result = resolve_backends("braintrust")
        assert result == []

    def test_returns_backend_when_key_set_and_otel_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip(
            "opentelemetry.exporter.otlp.proto.http.trace_exporter",
            reason="OTLP exporter extra not installed",
        )
        monkeypatch.setenv("BRAINTRUST_API_KEY", "bt_test_key_xyz")
        _reload_braintrust()
        from exo.observability.backends import (
            resolve_backends,  # pyright: ignore[reportMissingImports]
        )
        from exo.observability.backends.braintrust import (  # pyright: ignore[reportMissingImports]
            BraintrustBackend,
        )

        result = resolve_backends("braintrust")
        assert len(result) == 1
        assert isinstance(result[0], BraintrustBackend)


# ---------------------------------------------------------------------------
# Endpoint construction
# ---------------------------------------------------------------------------


class TestEndpointConstruction:
    def test_default_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BRAINTRUST_API_URL", raising=False)
        _reload_braintrust()
        from exo.observability.backends.braintrust import (  # pyright: ignore[reportMissingImports]
            BraintrustBackend,
        )

        backend = BraintrustBackend(api_key="key")
        assert backend.endpoint == "https://api.braintrust.dev/otel/v1/traces"

    def test_custom_api_url_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BRAINTRUST_API_URL", "https://custom.braintrust.example.com")
        _reload_braintrust()
        from exo.observability.backends.braintrust import (  # pyright: ignore[reportMissingImports]
            BraintrustBackend,
        )

        backend = BraintrustBackend(api_key="key")
        assert backend.endpoint == "https://custom.braintrust.example.com/otel/v1/traces"

    def test_custom_api_url_explicit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BRAINTRUST_API_URL", raising=False)
        _reload_braintrust()
        from exo.observability.backends.braintrust import (  # pyright: ignore[reportMissingImports]
            BraintrustBackend,
        )

        backend = BraintrustBackend(api_key="key", api_url="https://explicit.example.com/")
        # Trailing slash on api_url must be stripped.
        assert backend.endpoint == "https://explicit.example.com/otel/v1/traces"

    def test_trailing_slash_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BRAINTRUST_API_URL", raising=False)
        _reload_braintrust()
        from exo.observability.backends.braintrust import (  # pyright: ignore[reportMissingImports]
            BraintrustBackend,
        )

        backend = BraintrustBackend(api_key="key", api_url="https://api.braintrust.dev///")
        assert not backend.endpoint.startswith("https://api.braintrust.dev///")
        assert backend.endpoint.endswith("/otel/v1/traces")


# ---------------------------------------------------------------------------
# Header construction
# ---------------------------------------------------------------------------


class TestHeaderConstruction:
    def test_authorization_header(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)
        monkeypatch.delenv("BRAINTRUST_PARENT", raising=False)
        _reload_braintrust()
        from exo.observability.backends.braintrust import (  # pyright: ignore[reportMissingImports]
            BraintrustBackend,
        )

        backend = BraintrustBackend(api_key="my_secret_key")
        assert backend.headers.get("Authorization") == "Bearer my_secret_key"

    def test_xbt_parent_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BRAINTRUST_PARENT", raising=False)
        _reload_braintrust()
        from exo.observability.backends.braintrust import (  # pyright: ignore[reportMissingImports]
            BraintrustBackend,
        )

        backend = BraintrustBackend(api_key="key")
        assert backend.headers.get("x-bt-parent") == "project_name:default"

    def test_xbt_parent_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BRAINTRUST_PARENT", "project_name:my-project")
        _reload_braintrust()
        from exo.observability.backends.braintrust import (  # pyright: ignore[reportMissingImports]
            BraintrustBackend,
        )

        backend = BraintrustBackend(api_key="key")
        assert backend.headers.get("x-bt-parent") == "project_name:my-project"

    def test_xbt_parent_explicit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BRAINTRUST_PARENT", raising=False)
        _reload_braintrust()
        from exo.observability.backends.braintrust import (  # pyright: ignore[reportMissingImports]
            BraintrustBackend,
        )

        backend = BraintrustBackend(api_key="key", parent="experiment:abc123")
        assert backend.headers.get("x-bt-parent") == "experiment:abc123"

    def test_no_auth_header_when_key_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)
        _reload_braintrust()
        from exo.observability.backends.braintrust import (  # pyright: ignore[reportMissingImports]
            BraintrustBackend,
        )

        backend = BraintrustBackend()
        assert "Authorization" not in backend.headers


# ---------------------------------------------------------------------------
# score_client()
# ---------------------------------------------------------------------------


class TestScoreClient:
    def test_returns_none_when_sdk_not_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """score_client() must return None (not raise) when braintrust SDK is absent."""
        monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)
        _reload_braintrust()
        from exo.observability.backends.braintrust import (  # pyright: ignore[reportMissingImports]
            BraintrustBackend,
        )

        backend = BraintrustBackend(api_key="key")
        # Simulate missing SDK by patching import machinery.
        with patch.dict(sys.modules, {"braintrust": None}):
            client = backend.score_client()
        assert client is None

    def test_returns_none_when_key_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)
        _reload_braintrust()
        from exo.observability.backends.braintrust import (  # pyright: ignore[reportMissingImports]
            BraintrustBackend,
        )

        backend = BraintrustBackend()
        assert backend.score_client() is None

    def test_returns_client_when_sdk_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """score_client() returns a _BraintrustScoreClient when a mock SDK is injected."""
        _reload_braintrust()
        from exo.observability.backends.braintrust import (  # pyright: ignore[reportMissingImports]
            BraintrustBackend,
            _BraintrustScoreClient,
        )

        mock_bt = MagicMock()
        with patch.dict(sys.modules, {"braintrust": mock_bt}):
            backend = BraintrustBackend(api_key="real_key")
            client = backend.score_client()

        assert isinstance(client, _BraintrustScoreClient)

    def test_score_client_log_score_noop_when_sdk_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_BraintrustScoreClient.log_score() must not raise when SDK is missing."""
        _reload_braintrust()
        from exo.observability.backends.braintrust import (  # pyright: ignore[reportMissingImports]
            _BraintrustScoreClient,
        )

        client = _BraintrustScoreClient(api_key="key")
        with patch.dict(sys.modules, {"braintrust": None}):
            # Should not raise.
            client.log_score("span-1", "accuracy", 0.95)


# ---------------------------------------------------------------------------
# dataset_client()
# ---------------------------------------------------------------------------


class TestDatasetClient:
    def test_dataset_client_returns_none(self) -> None:
        from exo.observability.backends.braintrust import (  # pyright: ignore[reportMissingImports]
            BraintrustBackend,
        )

        backend = BraintrustBackend(api_key="key")
        assert backend.dataset_client() is None


# ---------------------------------------------------------------------------
# name attribute
# ---------------------------------------------------------------------------


class TestName:
    def test_name_is_braintrust(self) -> None:
        from exo.observability.backends.braintrust import (  # pyright: ignore[reportMissingImports]
            BraintrustBackend,
        )

        backend = BraintrustBackend()
        assert backend.name == "braintrust"
