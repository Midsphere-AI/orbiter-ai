"""Tests for the Langfuse tracing backend.

Covers: self-registration, endpoint/header construction, is_available() under
missing keys, and score_client() when the langfuse SDK is absent. No network
calls are made.
"""

from __future__ import annotations

import base64
import importlib

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _b64(s: str) -> str:
    return base64.b64encode(s.encode()).decode()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_import_registers_langfuse(self) -> None:
        """Importing the module must call register('langfuse', ...)."""
        # Ensure the module is imported (may already be cached).
        import exo.observability.backends.langfuse  # noqa: F401  # pyright: ignore[reportMissingImports]
        from exo.observability.backends import (  # pyright: ignore[reportMissingImports]
            _REGISTRY,
        )

        assert "langfuse" in _REGISTRY

    def test_resolve_returns_backend_when_keys_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """resolve_backends('langfuse') returns a LangfuseBackend when keys + OTLP present."""
        otel_available = _otel_available()

        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

        import exo.observability.backends.langfuse  # noqa: F401  # pyright: ignore[reportMissingImports]
        from exo.observability.backends import (  # pyright: ignore[reportMissingImports]
            resolve_backends,
        )
        from exo.observability.backends.langfuse import (  # pyright: ignore[reportMissingImports]
            LangfuseBackend,
        )

        backends = resolve_backends("langfuse")
        if otel_available:
            assert len(backends) == 1
            assert isinstance(backends[0], LangfuseBackend)
        else:
            # OTLP exporter not installed — backend is skipped, not an error.
            assert backends == []

    def test_resolve_returns_empty_when_keys_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """resolve_backends('langfuse') returns [] when credentials are missing."""
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

        import exo.observability.backends.langfuse  # noqa: F401  # pyright: ignore[reportMissingImports]
        from exo.observability.backends import (  # pyright: ignore[reportMissingImports]
            resolve_backends,
        )

        backends = resolve_backends("langfuse")
        assert backends == []


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------


class TestIsAvailable:
    def test_false_when_public_key_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

        from exo.observability.backends.langfuse import (  # pyright: ignore[reportMissingImports]
            LangfuseBackend,
        )

        backend = LangfuseBackend(secret_key="sk-test")
        assert backend.is_available() is False

    def test_false_when_secret_key_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

        from exo.observability.backends.langfuse import (  # pyright: ignore[reportMissingImports]
            LangfuseBackend,
        )

        backend = LangfuseBackend(public_key="pk-test")
        assert backend.is_available() is False

    def test_false_when_both_keys_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

        from exo.observability.backends.langfuse import (  # pyright: ignore[reportMissingImports]
            LangfuseBackend,
        )

        backend = LangfuseBackend()
        assert backend.is_available() is False

    def test_true_when_keys_and_otel_present(self) -> None:
        """True path — only runs when the OTLP exporter extra is installed."""
        if not _otel_available():
            pytest.skip("opentelemetry-exporter-otlp not installed")

        from exo.observability.backends.langfuse import (  # pyright: ignore[reportMissingImports]
            LangfuseBackend,
        )

        backend = LangfuseBackend(public_key="pk", secret_key="sk")
        assert backend.is_available() is True


# ---------------------------------------------------------------------------
# Endpoint and header construction
# ---------------------------------------------------------------------------


class TestEndpointAndHeaders:
    def test_default_endpoint(self) -> None:
        from exo.observability.backends.langfuse import (  # pyright: ignore[reportMissingImports]
            LangfuseBackend,
        )

        b = LangfuseBackend(public_key="pk", secret_key="sk")
        assert b.endpoint == "https://cloud.langfuse.com/api/public/otel/v1/traces"

    def test_custom_host_endpoint(self) -> None:
        from exo.observability.backends.langfuse import (  # pyright: ignore[reportMissingImports]
            LangfuseBackend,
        )

        b = LangfuseBackend(public_key="pk", secret_key="sk", host="https://my.langfuse.io")
        assert b.endpoint == "https://my.langfuse.io/api/public/otel/v1/traces"

    def test_trailing_slash_stripped_from_host(self) -> None:
        from exo.observability.backends.langfuse import (  # pyright: ignore[reportMissingImports]
            LangfuseBackend,
        )

        b = LangfuseBackend(public_key="pk", secret_key="sk", host="https://my.langfuse.io/")
        assert b.endpoint == "https://my.langfuse.io/api/public/otel/v1/traces"

    def test_basic_auth_header_correct(self) -> None:
        from exo.observability.backends.langfuse import (  # pyright: ignore[reportMissingImports]
            LangfuseBackend,
        )

        b = LangfuseBackend(public_key="pk", secret_key="sk")
        expected = f"Basic {_b64('pk:sk')}"
        assert b.headers["Authorization"] == expected

    def test_known_keys_header(self) -> None:
        """Assert exact base64 encoding for well-known key pair."""
        from exo.observability.backends.langfuse import (  # pyright: ignore[reportMissingImports]
            LangfuseBackend,
        )

        b = LangfuseBackend(public_key="pk-abc123", secret_key="sk-xyz789")
        raw = base64.b64encode(b"pk-abc123:sk-xyz789").decode()
        assert b.headers["Authorization"] == f"Basic {raw}"

    def test_no_auth_header_when_keys_missing(self) -> None:
        from exo.observability.backends.langfuse import (  # pyright: ignore[reportMissingImports]
            LangfuseBackend,
        )

        b = LangfuseBackend()
        assert "Authorization" not in b.headers

    def test_env_host_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGFUSE_HOST", "https://env.langfuse.io")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")

        from exo.observability.backends.langfuse import (  # pyright: ignore[reportMissingImports]
            LangfuseBackend,
        )

        b = LangfuseBackend()
        assert b.endpoint == "https://env.langfuse.io/api/public/otel/v1/traces"

    def test_explicit_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "env-pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "env-sk")

        from exo.observability.backends.langfuse import (  # pyright: ignore[reportMissingImports]
            LangfuseBackend,
        )

        b = LangfuseBackend(public_key="explicit-pk", secret_key="explicit-sk")
        expected = f"Basic {_b64('explicit-pk:explicit-sk')}"
        assert b.headers["Authorization"] == expected


# ---------------------------------------------------------------------------
# score_client
# ---------------------------------------------------------------------------


class TestScoreClient:
    def test_returns_none_when_sdk_not_installed(self) -> None:
        """score_client() must return None without raising when langfuse is absent."""
        # If the langfuse SDK happens to be installed in this env, skip this
        # particular assertion path so we don't give a false failure.
        try:
            import langfuse  # noqa: F401

            pytest.skip("langfuse SDK is installed — skipping 'absent SDK' test")
        except ImportError:
            pass

        from exo.observability.backends.langfuse import (  # pyright: ignore[reportMissingImports]
            LangfuseBackend,
        )

        b = LangfuseBackend(public_key="pk", secret_key="sk")
        result = b.score_client()
        assert result is None

    def test_returns_none_when_keys_absent(self) -> None:
        from exo.observability.backends.langfuse import (  # pyright: ignore[reportMissingImports]
            LangfuseBackend,
        )

        b = LangfuseBackend()
        assert b.score_client() is None

    def test_no_raise_on_import(self) -> None:
        """Importing langfuse.py with no extra packages installed must not raise."""
        # Force a clean reload to confirm the top-level import is safe.
        import exo.observability.backends.langfuse as mod  # pyright: ignore[reportMissingImports]

        importlib.reload(mod)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _otel_available() -> bool:
    try:
        import opentelemetry.exporter.otlp.proto.http.trace_exporter
        import opentelemetry.sdk.trace  # noqa: F401
    except ImportError:
        return False
    return True
