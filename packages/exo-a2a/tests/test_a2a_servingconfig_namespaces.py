"""Tests for ServingConfig grouped namespace constructors (BindConfig / AdvertiseConfig).

Verifies:
- grouped → flat population (bind= / advertise= sugar)
- flat kwargs still work unchanged
- conflict between grouped and flat raises ValueError
- read-accessor views (bind_config / advertise_config) are correct
- sub-models importable from package top-level
"""

from __future__ import annotations

import pytest

from exo.a2a import (  # pyright: ignore[reportMissingImports]
    AdvertiseConfig,
    BindConfig,
    ServingConfig,
)
from exo.a2a.types import TransportMode  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# BindConfig sub-model
# ---------------------------------------------------------------------------


class TestBindConfig:
    def test_defaults(self) -> None:
        bc = BindConfig()
        assert bc.host == "localhost"
        assert bc.port == 0
        assert bc.endpoint == "/"

    def test_custom(self) -> None:
        bc = BindConfig(host="0.0.0.0", port=9000, endpoint="/api")
        assert bc.host == "0.0.0.0"
        assert bc.port == 9000
        assert bc.endpoint == "/api"

    def test_frozen(self) -> None:
        from pydantic import ValidationError

        bc = BindConfig()
        with pytest.raises((ValidationError, TypeError)):
            bc.host = "other"  # type: ignore[misc]

    def test_serialization_roundtrip(self) -> None:
        bc = BindConfig(host="127.0.0.1", port=4000, endpoint="/v1")
        restored = BindConfig.model_validate(bc.model_dump())
        assert restored == bc


# ---------------------------------------------------------------------------
# AdvertiseConfig sub-model
# ---------------------------------------------------------------------------


class TestAdvertiseConfig:
    def test_defaults(self) -> None:
        ac = AdvertiseConfig()
        assert ac.streaming is False
        assert ac.version == "0.0.1"
        assert ac.transports == (TransportMode.JSONRPC,)
        assert ac.input_modes == ("text",)
        assert ac.output_modes == ("text",)

    def test_custom(self) -> None:
        ac = AdvertiseConfig(
            streaming=True,
            version="2.0.0",
            transports=["jsonrpc"],
            input_modes=["text", "json"],
            output_modes=["text"],
        )
        assert ac.streaming is True
        assert ac.version == "2.0.0"
        assert ac.transports == (TransportMode.JSONRPC,)
        assert ac.input_modes == ("text", "json")

    def test_list_coercion(self) -> None:
        ac = AdvertiseConfig(transports=["jsonrpc"], input_modes=["text"])
        assert ac.transports == (TransportMode.JSONRPC,)
        assert ac.input_modes == ("text",)

    def test_frozen(self) -> None:
        from pydantic import ValidationError

        ac = AdvertiseConfig()
        with pytest.raises((ValidationError, TypeError)):
            ac.streaming = True  # type: ignore[misc]

    def test_serialization_roundtrip(self) -> None:
        ac = AdvertiseConfig(streaming=True, version="1.0", transports=["jsonrpc"])
        restored = AdvertiseConfig.model_validate(ac.model_dump())
        assert restored == ac


# ---------------------------------------------------------------------------
# ServingConfig — flat kwargs still work (backward compat)
# ---------------------------------------------------------------------------


class TestServingConfigFlatStillWorks:
    def test_all_flat_fields(self) -> None:
        from exo.a2a.types import AgentSkill  # pyright: ignore[reportMissingImports]

        skill = AgentSkill(id="s1", name="Search")
        cfg = ServingConfig(
            host="0.0.0.0",
            port=8080,
            endpoint="/api",
            streaming=True,
            version="1.0.0",
            skills=[skill],
            input_modes=["text", "json"],
            output_modes=["text"],
            transports=["jsonrpc"],
            extra={"custom": "value"},
        )
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8080
        assert cfg.endpoint == "/api"
        assert cfg.streaming is True
        assert cfg.version == "1.0.0"
        assert cfg.skills == (skill,)
        assert cfg.input_modes == ("text", "json")
        assert cfg.output_modes == ("text",)
        assert cfg.transports == (TransportMode.JSONRPC,)
        assert cfg.extra == {"custom": "value"}

    def test_defaults_unchanged(self) -> None:
        cfg = ServingConfig()
        assert cfg.host == "localhost"
        assert cfg.port == 0
        assert cfg.endpoint == "/"
        assert cfg.streaming is False
        assert cfg.version == "0.0.1"
        assert cfg.skills == ()
        assert cfg.input_modes == ("text",)
        assert cfg.output_modes == ("text",)
        assert cfg.transports == (TransportMode.JSONRPC,)
        assert cfg.extra == {}


# ---------------------------------------------------------------------------
# ServingConfig — bind= namespace populates flat fields
# ---------------------------------------------------------------------------


class TestServingConfigBindNamespace:
    def test_bind_populates_flat_fields(self) -> None:
        cfg = ServingConfig(bind=BindConfig(host="0.0.0.0", port=9000, endpoint="/v2"))
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 9000
        assert cfg.endpoint == "/v2"
        # advertise fields get defaults
        assert cfg.streaming is False
        assert cfg.version == "0.0.1"

    def test_bind_partial(self) -> None:
        """Only override some bind fields; rest get defaults."""
        cfg = ServingConfig(bind=BindConfig(port=5000))
        assert cfg.host == "localhost"
        assert cfg.port == 5000
        assert cfg.endpoint == "/"

    def test_bind_as_dict(self) -> None:
        """bind= also accepts a plain dict (pydantic coercion)."""
        cfg = ServingConfig(bind={"host": "127.0.0.1", "port": 1234})
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 1234

    def test_bind_combined_with_other_flat_fields(self) -> None:
        """bind= can combine with non-overlapping flat fields (e.g., streaming=)."""
        cfg = ServingConfig(bind=BindConfig(host="0.0.0.0", port=8080), streaming=True)
        assert cfg.host == "0.0.0.0"
        assert cfg.streaming is True


# ---------------------------------------------------------------------------
# ServingConfig — advertise= namespace populates flat fields
# ---------------------------------------------------------------------------


class TestServingConfigAdvertiseNamespace:
    def test_advertise_populates_flat_fields(self) -> None:
        cfg = ServingConfig(
            advertise=AdvertiseConfig(
                streaming=True,
                version="2.0",
                transports=["jsonrpc"],
                input_modes=["text", "json"],
                output_modes=["text"],
            )
        )
        assert cfg.streaming is True
        assert cfg.version == "2.0"
        assert cfg.transports == (TransportMode.JSONRPC,)
        assert cfg.input_modes == ("text", "json")
        assert cfg.output_modes == ("text",)
        # bind fields get defaults
        assert cfg.host == "localhost"
        assert cfg.port == 0

    def test_advertise_combined_with_bind(self) -> None:
        cfg = ServingConfig(
            bind=BindConfig(host="0.0.0.0", port=8080),
            advertise=AdvertiseConfig(streaming=True, version="1.5"),
        )
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8080
        assert cfg.streaming is True
        assert cfg.version == "1.5"


# ---------------------------------------------------------------------------
# ServingConfig — conflict detection
# ---------------------------------------------------------------------------


class TestServingConfigConflict:
    def test_bind_and_flat_host_conflict(self) -> None:
        with pytest.raises((ValueError, Exception)):
            ServingConfig(bind=BindConfig(host="0.0.0.0"), host="127.0.0.1")

    def test_bind_and_flat_port_conflict(self) -> None:
        with pytest.raises((ValueError, Exception)):
            ServingConfig(bind=BindConfig(port=9000), port=8080)

    def test_bind_and_flat_endpoint_conflict(self) -> None:
        with pytest.raises((ValueError, Exception)):
            ServingConfig(bind=BindConfig(endpoint="/v2"), endpoint="/v1")

    def test_advertise_and_flat_streaming_conflict(self) -> None:
        with pytest.raises((ValueError, Exception)):
            ServingConfig(advertise=AdvertiseConfig(streaming=True), streaming=False)

    def test_advertise_and_flat_version_conflict(self) -> None:
        with pytest.raises((ValueError, Exception)):
            ServingConfig(advertise=AdvertiseConfig(version="2.0"), version="1.0")


# ---------------------------------------------------------------------------
# ServingConfig — read-accessor views
# ---------------------------------------------------------------------------


class TestServingConfigReadAccessors:
    def test_bind_config_view(self) -> None:
        cfg = ServingConfig(host="0.0.0.0", port=8080, endpoint="/api")
        bc = cfg.bind_config
        assert isinstance(bc, BindConfig)
        assert bc.host == "0.0.0.0"
        assert bc.port == 8080
        assert bc.endpoint == "/api"

    def test_advertise_config_view(self) -> None:
        cfg = ServingConfig(streaming=True, version="2.0", transports=["jsonrpc"])
        ac = cfg.advertise_config
        assert isinstance(ac, AdvertiseConfig)
        assert ac.streaming is True
        assert ac.version == "2.0"
        assert ac.transports == (TransportMode.JSONRPC,)

    def test_default_views(self) -> None:
        cfg = ServingConfig()
        assert cfg.bind_config == BindConfig()
        assert cfg.advertise_config == AdvertiseConfig()

    def test_roundtrip_via_namespace(self) -> None:
        """Grouped constructor → flat fields → read-accessor reproduces original namespace."""
        original_bind = BindConfig(host="10.0.0.1", port=7000, endpoint="/v3")
        original_adv = AdvertiseConfig(streaming=True, version="3.0")
        cfg = ServingConfig(bind=original_bind, advertise=original_adv)
        assert cfg.bind_config == original_bind
        assert cfg.advertise_config.streaming == original_adv.streaming
        assert cfg.advertise_config.version == original_adv.version

    def test_accessor_names_dont_collide_with_flat_fields(self) -> None:
        """bind_config and advertise_config are not flat storage fields."""
        # Ensure the fields stored on the model don't include the accessor names
        assert "bind_config" not in ServingConfig.model_fields
        assert "advertise_config" not in ServingConfig.model_fields


# ---------------------------------------------------------------------------
# ServingConfig — serialization still works
# ---------------------------------------------------------------------------


class TestServingConfigSerialization:
    def test_model_dump_excludes_accessors(self) -> None:
        """model_dump() returns flat fields only, no accessor artefacts."""
        cfg = ServingConfig(bind=BindConfig(host="0.0.0.0", port=9000), streaming=True)
        data = cfg.model_dump()
        assert "bind_config" not in data
        assert "advertise_config" not in data
        assert data["host"] == "0.0.0.0"
        assert data["port"] == 9000
        assert data["streaming"] is True

    def test_model_validate_roundtrip(self) -> None:
        cfg = ServingConfig(host="0.0.0.0", port=5000, streaming=True, version="1.2")
        restored = ServingConfig.model_validate(cfg.model_dump())
        assert restored == cfg

    def test_model_dump_json_roundtrip(self) -> None:
        cfg = ServingConfig(port=3000, streaming=True)
        restored = ServingConfig.model_validate_json(cfg.model_dump_json())
        assert restored == cfg


# ---------------------------------------------------------------------------
# Package top-level imports
# ---------------------------------------------------------------------------


class TestPackageImports:
    def test_bind_config_importable_from_package(self) -> None:
        import exo.a2a as pkg  # pyright: ignore[reportMissingImports]

        assert pkg.BindConfig is BindConfig

    def test_advertise_config_importable_from_package(self) -> None:
        import exo.a2a as pkg  # pyright: ignore[reportMissingImports]

        assert pkg.AdvertiseConfig is AdvertiseConfig

    def test_both_in_all(self) -> None:
        import exo.a2a as pkg  # pyright: ignore[reportMissingImports]

        assert "BindConfig" in pkg.__all__
        assert "AdvertiseConfig" in pkg.__all__
