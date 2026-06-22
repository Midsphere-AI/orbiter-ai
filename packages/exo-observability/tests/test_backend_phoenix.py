"""Tests for the Arize Phoenix tracing backend (OTLP seam, no network).

Skips automatically when ``opentelemetry-exporter-otlp-proto-http`` is not
installed — the module self-registers and the OTLP exporter guards are
exercised without making any real network calls.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_otlp() -> Any:
    """Return the OTLP trace exporter module, or skip the test if absent."""
    return pytest.importorskip(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter",
        reason="opentelemetry-exporter-otlp-proto-http not installed",
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestPhoenixRegistration:
    def test_module_registers_phoenix(self) -> None:
        """Importing the module must register 'phoenix' in the backend registry."""
        import exo.observability.backends.phoenix  # noqa: F401  # pyright: ignore[reportMissingImports]
        from exo.observability.backends import (  # pyright: ignore[reportMissingImports]
            _REGISTRY,
        )

        assert "phoenix" in _REGISTRY

    def test_resolve_backends_returns_phoenix_when_otlp_available(self) -> None:
        """resolve_backends('phoenix') returns a PhoenixBackend when OTLP is importable."""
        _import_otlp()  # skip if exporter absent

        import exo.observability.backends.phoenix  # noqa: F401  # pyright: ignore[reportMissingImports]
        from exo.observability.backends import (  # pyright: ignore[reportMissingImports]
            resolve_backends,
        )
        from exo.observability.backends.phoenix import (  # pyright: ignore[reportMissingImports]
            PhoenixBackend,
        )

        backends = resolve_backends("phoenix")
        assert len(backends) == 1
        assert isinstance(backends[0], PhoenixBackend)

    def test_is_available_true_when_otlp_importable(self) -> None:
        _import_otlp()

        from exo.observability.backends.phoenix import (  # pyright: ignore[reportMissingImports]
            PhoenixBackend,
        )

        assert PhoenixBackend().is_available() is True

    def test_is_available_false_when_otlp_missing(self, monkeypatch: Any) -> None:
        """is_available() returns False when the OTLP exporter cannot be imported."""
        # Remove the OTLP module from sys.modules and block re-import.
        otlp_name = "opentelemetry.exporter.otlp.proto.http.trace_exporter"
        saved = sys.modules.pop(otlp_name, None)
        monkeypatch.setitem(sys.modules, otlp_name, None)  # type: ignore[arg-type]

        try:
            # Force the module to reload so is_available() re-checks
            import exo.observability.backends.phoenix as _mod  # pyright: ignore[reportMissingImports]

            importlib.reload(_mod)
            from exo.observability.backends.phoenix import (
                PhoenixBackend,  # pyright: ignore[reportMissingImports]
            )

            assert PhoenixBackend().is_available() is False
        finally:
            if saved is not None:
                sys.modules[otlp_name] = saved
            else:
                sys.modules.pop(otlp_name, None)
            # Reload again to restore normal state
            importlib.reload(importlib.import_module("exo.observability.backends.phoenix"))


# ---------------------------------------------------------------------------
# Endpoint construction
# ---------------------------------------------------------------------------


class TestPhoenixEndpoint:
    def test_default_endpoint(self, monkeypatch: Any) -> None:
        """No env vars → endpoint is http://localhost:6006/v1/traces."""
        monkeypatch.delenv("PHOENIX_COLLECTOR_ENDPOINT", raising=False)
        monkeypatch.delenv("PHOENIX_API_KEY", raising=False)
        monkeypatch.delenv("PHOENIX_CLIENT_HEADERS", raising=False)

        from exo.observability.backends.phoenix import (  # pyright: ignore[reportMissingImports]
            PhoenixBackend,
        )

        b = PhoenixBackend()
        assert b.endpoint == "http://localhost:6006/v1/traces"

    def test_custom_collector_via_env(self, monkeypatch: Any) -> None:
        """PHOENIX_COLLECTOR_ENDPOINT is honoured and /v1/traces is appended."""
        monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "https://my-phoenix.example.com")
        monkeypatch.delenv("PHOENIX_API_KEY", raising=False)
        monkeypatch.delenv("PHOENIX_CLIENT_HEADERS", raising=False)

        from exo.observability.backends.phoenix import (  # pyright: ignore[reportMissingImports]
            PhoenixBackend,
        )

        b = PhoenixBackend()
        assert b.endpoint == "https://my-phoenix.example.com/v1/traces"

    def test_trailing_slash_stripped(self, monkeypatch: Any) -> None:
        """Trailing slashes on the collector URL are stripped before appending."""
        monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://host:9999/")
        monkeypatch.delenv("PHOENIX_API_KEY", raising=False)
        monkeypatch.delenv("PHOENIX_CLIENT_HEADERS", raising=False)

        from exo.observability.backends.phoenix import (  # pyright: ignore[reportMissingImports]
            PhoenixBackend,
        )

        b = PhoenixBackend()
        assert b.endpoint == "http://host:9999/v1/traces"

    def test_explicit_constructor_arg_overrides_env(self, monkeypatch: Any) -> None:
        """constructor collector_endpoint wins over PHOENIX_COLLECTOR_ENDPOINT."""
        monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://env-host")
        monkeypatch.delenv("PHOENIX_API_KEY", raising=False)
        monkeypatch.delenv("PHOENIX_CLIENT_HEADERS", raising=False)

        from exo.observability.backends.phoenix import (  # pyright: ignore[reportMissingImports]
            PhoenixBackend,
        )

        b = PhoenixBackend(collector_endpoint="http://explicit-host")
        assert b.endpoint == "http://explicit-host/v1/traces"


# ---------------------------------------------------------------------------
# Header construction
# ---------------------------------------------------------------------------


class TestPhoenixHeaders:
    def test_no_api_key_no_auth_headers(self, monkeypatch: Any) -> None:
        """Without a key, no Authorization or api_key headers are injected."""
        monkeypatch.delenv("PHOENIX_API_KEY", raising=False)
        monkeypatch.delenv("PHOENIX_CLIENT_HEADERS", raising=False)
        monkeypatch.delenv("PHOENIX_COLLECTOR_ENDPOINT", raising=False)

        from exo.observability.backends.phoenix import (  # pyright: ignore[reportMissingImports]
            PhoenixBackend,
        )

        b = PhoenixBackend()
        assert "Authorization" not in b.headers
        assert "api_key" not in b.headers

    def test_api_key_env_sets_bearer_and_api_key(self, monkeypatch: Any) -> None:
        """PHOENIX_API_KEY env adds both Authorization:Bearer and api_key headers."""
        monkeypatch.setenv("PHOENIX_API_KEY", "secret-env-key")
        monkeypatch.delenv("PHOENIX_CLIENT_HEADERS", raising=False)
        monkeypatch.delenv("PHOENIX_COLLECTOR_ENDPOINT", raising=False)

        from exo.observability.backends.phoenix import (  # pyright: ignore[reportMissingImports]
            PhoenixBackend,
        )

        b = PhoenixBackend()
        assert b.headers["Authorization"] == "Bearer secret-env-key"
        assert b.headers["api_key"] == "secret-env-key"

    def test_explicit_api_key_sets_headers(self, monkeypatch: Any) -> None:
        """Explicit api_key constructor arg sets auth headers (overrides env)."""
        monkeypatch.delenv("PHOENIX_API_KEY", raising=False)
        monkeypatch.delenv("PHOENIX_CLIENT_HEADERS", raising=False)
        monkeypatch.delenv("PHOENIX_COLLECTOR_ENDPOINT", raising=False)

        from exo.observability.backends.phoenix import (  # pyright: ignore[reportMissingImports]
            PhoenixBackend,
        )

        b = PhoenixBackend(api_key="my-explicit-key")
        assert b.headers["Authorization"] == "Bearer my-explicit-key"
        assert b.headers["api_key"] == "my-explicit-key"

    def test_client_headers_env_parsed(self, monkeypatch: Any) -> None:
        """PHOENIX_CLIENT_HEADERS=a=1,b=2 is parsed into {'a':'1','b':'2'}."""
        monkeypatch.delenv("PHOENIX_API_KEY", raising=False)
        monkeypatch.setenv("PHOENIX_CLIENT_HEADERS", "a=1,b=2")
        monkeypatch.delenv("PHOENIX_COLLECTOR_ENDPOINT", raising=False)

        from exo.observability.backends.phoenix import (  # pyright: ignore[reportMissingImports]
            PhoenixBackend,
        )

        b = PhoenixBackend()
        assert b.headers["a"] == "1"
        assert b.headers["b"] == "2"

    def test_client_headers_with_spaces(self, monkeypatch: Any) -> None:
        """Whitespace around k=v tokens is trimmed."""
        monkeypatch.delenv("PHOENIX_API_KEY", raising=False)
        monkeypatch.setenv("PHOENIX_CLIENT_HEADERS", " x = hello , y = world ")
        monkeypatch.delenv("PHOENIX_COLLECTOR_ENDPOINT", raising=False)

        from exo.observability.backends.phoenix import (  # pyright: ignore[reportMissingImports]
            PhoenixBackend,
        )

        b = PhoenixBackend()
        assert b.headers["x"] == "hello"
        assert b.headers["y"] == "world"

    def test_explicit_headers_override(self, monkeypatch: Any) -> None:
        """headers= arg overrides auto-resolved API-key headers."""
        monkeypatch.setenv("PHOENIX_API_KEY", "env-key")
        monkeypatch.delenv("PHOENIX_CLIENT_HEADERS", raising=False)
        monkeypatch.delenv("PHOENIX_COLLECTOR_ENDPOINT", raising=False)

        from exo.observability.backends.phoenix import (  # pyright: ignore[reportMissingImports]
            PhoenixBackend,
        )

        b = PhoenixBackend(headers={"Authorization": "Bearer override-key"})
        assert b.headers["Authorization"] == "Bearer override-key"


# ---------------------------------------------------------------------------
# Header parsing helper
# ---------------------------------------------------------------------------


class TestParseClientHeaders:
    def test_basic(self) -> None:
        from exo.observability.backends.phoenix import (  # pyright: ignore[reportMissingImports]
            _parse_client_headers,
        )

        assert _parse_client_headers("a=1,b=2") == {"a": "1", "b": "2"}

    def test_empty_string(self) -> None:
        from exo.observability.backends.phoenix import (  # pyright: ignore[reportMissingImports]
            _parse_client_headers,
        )

        assert _parse_client_headers("") == {}

    def test_value_with_equals(self) -> None:
        """Values containing '=' are kept intact (only first '=' splits)."""
        from exo.observability.backends.phoenix import (  # pyright: ignore[reportMissingImports]
            _parse_client_headers,
        )

        result = _parse_client_headers("token=abc=def")
        assert result["token"] == "abc=def"

    def test_malformed_token_skipped(self) -> None:
        from exo.observability.backends.phoenix import (  # pyright: ignore[reportMissingImports]
            _parse_client_headers,
        )

        result = _parse_client_headers("good=ok,badtoken,another=fine")
        assert result == {"good": "ok", "another": "fine"}


# ---------------------------------------------------------------------------
# name attribute
# ---------------------------------------------------------------------------


class TestPhoenixName:
    def test_name_is_phoenix(self) -> None:
        from exo.observability.backends.phoenix import (  # pyright: ignore[reportMissingImports]
            PhoenixBackend,
        )

        assert PhoenixBackend().name == "phoenix"
