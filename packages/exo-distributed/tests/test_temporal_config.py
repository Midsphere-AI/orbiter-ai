"""Tests for exo.distributed.temporal.config — ConnectionConfig, RetryConfig,
TimeoutConfig, and WorkerTuningConfig.

All tests are pure-Python and make no real network calls.  Since temporalio IS
installed in the dev environment, methods that build temporalio objects are
called directly so we can assert on the resulting types and field values.
"""

from __future__ import annotations

import os
from datetime import timedelta
from unittest.mock import patch

import pytest
from temporalio.client import TLSConfig
from temporalio.common import RetryPolicy

from exo.distributed.temporal._compat import (  # pyright: ignore[reportMissingImports]
    _DEFAULT_TEMPORAL_HOST,
    _DEFAULT_TEMPORAL_NAMESPACE,
)
from exo.distributed.temporal.config import (  # pyright: ignore[reportMissingImports]
    ConnectionConfig,
    RetryConfig,
    TimeoutConfig,
    WorkerTuningConfig,
)

# ===========================================================================
# ConnectionConfig
# ===========================================================================


class TestConnectionConfigDefaults:
    def test_default_host_and_namespace(self) -> None:
        cfg = ConnectionConfig()
        assert cfg.host == _DEFAULT_TEMPORAL_HOST
        assert cfg.namespace == _DEFAULT_TEMPORAL_NAMESPACE

    def test_defaults_no_tls(self) -> None:
        cfg = ConnectionConfig()
        assert cfg.api_key is None
        assert cfg.tls is False
        assert cfg.server_root_ca_cert is None
        assert cfg.client_cert is None
        assert cfg.client_private_key is None
        assert cfg.tls_domain is None

    def test_frozen(self) -> None:
        cfg = ConnectionConfig()
        with pytest.raises(Exception):  # noqa: B017
            cfg.host = "other:7233"  # type: ignore[misc]


class TestConnectionConfigToConnectKwargs:
    def test_plain_no_tls(self) -> None:
        cfg = ConnectionConfig(host="localhost:7233", namespace="default")
        kwargs = cfg.to_connect_kwargs()
        assert kwargs["target_host"] == "localhost:7233"
        assert kwargs["namespace"] == "default"
        assert "tls" not in kwargs
        assert "api_key" not in kwargs

    def test_api_key_forces_tls(self) -> None:
        cfg = ConnectionConfig(api_key="tok_abc123")
        kwargs = cfg.to_connect_kwargs()
        assert kwargs["api_key"] == "tok_abc123"
        # Temporal Cloud: API key → TLS must be True
        assert kwargs["tls"] is True

    def test_tls_true_explicit(self) -> None:
        cfg = ConnectionConfig(tls=True)
        kwargs = cfg.to_connect_kwargs()
        assert kwargs["tls"] is True

    def test_server_ca_only(self) -> None:
        ca_cert = b"-----BEGIN CERTIFICATE-----\nfake\n-----END CERTIFICATE-----\n"
        cfg = ConnectionConfig(server_root_ca_cert=ca_cert)
        kwargs = cfg.to_connect_kwargs()
        assert isinstance(kwargs["tls"], TLSConfig)
        assert kwargs["tls"].server_root_ca_cert == ca_cert
        assert kwargs["tls"].client_cert is None
        assert kwargs["tls"].client_private_key is None

    def test_mtls_full(self) -> None:
        ca_cert = b"ca-cert"
        client_cert = b"client-cert"
        client_key = b"client-key"
        cfg = ConnectionConfig(
            client_cert=client_cert,
            client_private_key=client_key,
            server_root_ca_cert=ca_cert,
            tls_domain="temporal.example.com",
        )
        kwargs = cfg.to_connect_kwargs()
        tls = kwargs["tls"]
        assert isinstance(tls, TLSConfig)
        assert tls.client_cert == client_cert
        assert tls.client_private_key == client_key
        assert tls.server_root_ca_cert == ca_cert
        assert tls.domain == "temporal.example.com"

    def test_mtls_no_ca(self) -> None:
        """mTLS without explicit server CA — no server_root_ca_cert field."""
        cfg = ConnectionConfig(
            client_cert=b"cert",
            client_private_key=b"key",
        )
        kwargs = cfg.to_connect_kwargs()
        tls = kwargs["tls"]
        assert isinstance(tls, TLSConfig)
        assert tls.client_cert == b"cert"
        assert tls.client_private_key == b"key"
        assert tls.server_root_ca_cert is None


class TestConnectionConfigFromEnv:
    def test_defaults_when_env_empty(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            cfg = ConnectionConfig.from_env()
        assert cfg.host == _DEFAULT_TEMPORAL_HOST
        assert cfg.namespace == _DEFAULT_TEMPORAL_NAMESPACE
        assert cfg.api_key is None
        assert cfg.tls is False

    def test_reads_host_and_namespace(self) -> None:
        env = {"TEMPORAL_HOST": "temporal.prod:7233", "TEMPORAL_NAMESPACE": "prod"}
        with patch.dict(os.environ, env, clear=True):
            cfg = ConnectionConfig.from_env()
        assert cfg.host == "temporal.prod:7233"
        assert cfg.namespace == "prod"

    def test_reads_api_key(self) -> None:
        with patch.dict(os.environ, {"TEMPORAL_API_KEY": "secret-key"}, clear=True):
            cfg = ConnectionConfig.from_env()
        assert cfg.api_key == "secret-key"

    @pytest.mark.parametrize("val", ["1", "true", "True", "TRUE", "yes", "on"])
    def test_tls_truthy_strings(self, val: str) -> None:
        with patch.dict(os.environ, {"TEMPORAL_TLS": val}, clear=True):
            cfg = ConnectionConfig.from_env()
        assert cfg.tls is True

    @pytest.mark.parametrize("val", ["0", "false", "no", "off", ""])
    def test_tls_falsy_strings(self, val: str) -> None:
        with patch.dict(os.environ, {"TEMPORAL_TLS": val}, clear=True):
            cfg = ConnectionConfig.from_env()
        assert cfg.tls is False

    def test_reads_cert_files(self, tmp_path: pytest.TempPathFactory) -> None:
        ca_file = tmp_path / "ca.pem"  # type: ignore[operator]
        cert_file = tmp_path / "client.pem"  # type: ignore[operator]
        key_file = tmp_path / "client.key"  # type: ignore[operator]
        ca_file.write_bytes(b"CA_CERT")
        cert_file.write_bytes(b"CLIENT_CERT")
        key_file.write_bytes(b"CLIENT_KEY")

        env = {
            "TEMPORAL_TLS_SERVER_CA_CERT_PATH": str(ca_file),
            "TEMPORAL_TLS_CLIENT_CERT_PATH": str(cert_file),
            "TEMPORAL_TLS_CLIENT_KEY_PATH": str(key_file),
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = ConnectionConfig.from_env()

        assert cfg.server_root_ca_cert == b"CA_CERT"
        assert cfg.client_cert == b"CLIENT_CERT"
        assert cfg.client_private_key == b"CLIENT_KEY"

    def test_missing_cert_file_raises(self, tmp_path: pytest.TempPathFactory) -> None:
        env = {"TEMPORAL_TLS_SERVER_CA_CERT_PATH": "/no/such/file.pem"}
        with (
            patch.dict(os.environ, env, clear=True),
            pytest.raises(ValueError, match="TEMPORAL_TLS_SERVER_CA_CERT_PATH"),
        ):
            ConnectionConfig.from_env()


class TestConnectionConfigRepr:
    def test_api_key_redacted(self) -> None:
        cfg = ConnectionConfig(api_key="super-secret")
        r = repr(cfg)
        assert "super-secret" not in r
        assert "***" in r

    def test_client_private_key_redacted(self) -> None:
        cfg = ConnectionConfig(client_private_key=b"private-key-bytes")
        r = repr(cfg)
        assert "private-key-bytes" not in r
        assert "***" in r

    def test_none_fields_not_shown_as_redacted(self) -> None:
        cfg = ConnectionConfig()
        r = repr(cfg)
        # Neither api_key nor client_private_key is set, so no *** should appear
        assert "***" not in r

    def test_host_visible(self) -> None:
        cfg = ConnectionConfig(host="myhost:7233", api_key="tok")
        assert "myhost:7233" in repr(cfg)


# ===========================================================================
# RetryConfig
# ===========================================================================


class TestRetryConfigDefaults:
    def test_default_values(self) -> None:
        rc = RetryConfig()
        assert rc.initial_interval_seconds == 1.0
        assert rc.backoff_coefficient == 2.0
        assert rc.maximum_interval_seconds is None
        assert rc.maximum_attempts == 0
        assert rc.non_retryable_error_types == []

    def test_frozen(self) -> None:
        rc = RetryConfig()
        with pytest.raises(Exception):  # noqa: B017
            rc.maximum_attempts = 5  # type: ignore[misc]


class TestRetryConfigToRetryPolicy:
    def test_returns_retry_policy_instance(self) -> None:
        rp = RetryConfig().to_retry_policy()
        assert isinstance(rp, RetryPolicy)

    def test_initial_interval_maps(self) -> None:
        rc = RetryConfig(initial_interval_seconds=5.0)
        rp = rc.to_retry_policy()
        assert rp.initial_interval == timedelta(seconds=5.0)

    def test_backoff_coefficient_maps(self) -> None:
        rc = RetryConfig(backoff_coefficient=3.0)
        rp = rc.to_retry_policy()
        assert rp.backoff_coefficient == 3.0

    def test_maximum_interval_maps(self) -> None:
        rc = RetryConfig(maximum_interval_seconds=60.0)
        rp = rc.to_retry_policy()
        assert rp.maximum_interval == timedelta(seconds=60.0)

    def test_maximum_interval_none(self) -> None:
        rc = RetryConfig(maximum_interval_seconds=None)
        rp = rc.to_retry_policy()
        assert rp.maximum_interval is None

    def test_maximum_attempts_maps(self) -> None:
        rc = RetryConfig(maximum_attempts=3)
        rp = rc.to_retry_policy()
        assert rp.maximum_attempts == 3

    def test_non_retryable_error_types_maps(self) -> None:
        rc = RetryConfig(non_retryable_error_types=["ValueError", "TypeError"])
        rp = rc.to_retry_policy()
        assert list(rp.non_retryable_error_types or []) == ["ValueError", "TypeError"]

    def test_empty_non_retryable_becomes_none(self) -> None:
        rc = RetryConfig(non_retryable_error_types=[])
        rp = rc.to_retry_policy()
        # Empty list → None in RetryPolicy (Temporal convention)
        assert rp.non_retryable_error_types is None


class TestRetryConfigRoundTrip:
    def test_to_dict_from_dict(self) -> None:
        rc = RetryConfig(
            initial_interval_seconds=2.5,
            backoff_coefficient=1.5,
            maximum_interval_seconds=30.0,
            maximum_attempts=5,
            non_retryable_error_types=["IOError"],
        )
        d = rc.to_dict()
        rc2 = RetryConfig.from_dict(d)
        assert rc == rc2

    def test_dict_is_json_serialisable(self) -> None:
        import json

        rc = RetryConfig(non_retryable_error_types=["MyError"])
        assert json.dumps(rc.to_dict())  # must not raise


# ===========================================================================
# TimeoutConfig
# ===========================================================================


class TestTimeoutConfigDefaults:
    def test_all_none(self) -> None:
        tc = TimeoutConfig()
        assert tc.schedule_to_close_seconds is None
        assert tc.start_to_close_seconds is None
        assert tc.schedule_to_start_seconds is None
        assert tc.heartbeat_seconds is None
        assert tc.execution_timeout_seconds is None
        assert tc.run_timeout_seconds is None
        assert tc.task_timeout_seconds is None

    def test_frozen(self) -> None:
        tc = TimeoutConfig()
        with pytest.raises(Exception):  # noqa: B017
            tc.heartbeat_seconds = 10.0  # type: ignore[misc]


class TestTimeoutConfigToActivityTimeouts:
    def test_empty_when_all_none(self) -> None:
        assert TimeoutConfig().to_activity_timeouts() == {}

    def test_keys_match_temporal_kwargs(self) -> None:
        tc = TimeoutConfig(
            schedule_to_close_seconds=30.0,
            start_to_close_seconds=20.0,
            schedule_to_start_seconds=5.0,
            heartbeat_seconds=10.0,
        )
        result = tc.to_activity_timeouts()
        assert set(result.keys()) == {
            "schedule_to_close_timeout",
            "start_to_close_timeout",
            "schedule_to_start_timeout",
            "heartbeat_timeout",
        }

    def test_values_are_timedeltas(self) -> None:
        tc = TimeoutConfig(schedule_to_close_seconds=60.0, heartbeat_seconds=15.0)
        result = tc.to_activity_timeouts()
        assert result["schedule_to_close_timeout"] == timedelta(seconds=60.0)
        assert result["heartbeat_timeout"] == timedelta(seconds=15.0)

    def test_omits_none_fields(self) -> None:
        tc = TimeoutConfig(start_to_close_seconds=45.0)
        result = tc.to_activity_timeouts()
        assert list(result.keys()) == ["start_to_close_timeout"]


class TestTimeoutConfigToWorkflowTimeouts:
    def test_empty_when_all_none(self) -> None:
        assert TimeoutConfig().to_workflow_timeouts() == {}

    def test_keys_match_temporal_kwargs(self) -> None:
        tc = TimeoutConfig(
            execution_timeout_seconds=3600.0,
            run_timeout_seconds=1800.0,
            task_timeout_seconds=10.0,
        )
        result = tc.to_workflow_timeouts()
        assert set(result.keys()) == {"execution_timeout", "run_timeout", "task_timeout"}

    def test_values_are_timedeltas(self) -> None:
        tc = TimeoutConfig(execution_timeout_seconds=7200.0)
        result = tc.to_workflow_timeouts()
        assert result["execution_timeout"] == timedelta(seconds=7200.0)

    def test_omits_none_fields(self) -> None:
        tc = TimeoutConfig(task_timeout_seconds=5.0)
        result = tc.to_workflow_timeouts()
        assert list(result.keys()) == ["task_timeout"]

    def test_workflow_fields_not_in_activity_output(self) -> None:
        """Workflow-only fields must not bleed into activity timeout dict."""
        tc = TimeoutConfig(execution_timeout_seconds=600.0, heartbeat_seconds=5.0)
        act = tc.to_activity_timeouts()
        wf = tc.to_workflow_timeouts()
        assert "execution_timeout" not in act
        assert "heartbeat_timeout" not in wf


class TestTimeoutConfigRoundTrip:
    def test_to_dict_from_dict(self) -> None:
        tc = TimeoutConfig(
            schedule_to_close_seconds=30.0,
            heartbeat_seconds=5.0,
            execution_timeout_seconds=3600.0,
        )
        tc2 = TimeoutConfig.from_dict(tc.to_dict())
        assert tc == tc2

    def test_dict_is_json_serialisable(self) -> None:
        import json

        tc = TimeoutConfig(start_to_close_seconds=10.0)
        assert json.dumps(tc.to_dict())


# ===========================================================================
# WorkerTuningConfig
# ===========================================================================


class TestWorkerTuningConfigDefaults:
    def test_all_none(self) -> None:
        wtc = WorkerTuningConfig()
        assert wtc.max_concurrent_activities is None
        assert wtc.max_concurrent_workflow_tasks is None
        assert wtc.max_concurrent_local_activities is None
        assert wtc.max_cached_workflows is None

    def test_frozen(self) -> None:
        wtc = WorkerTuningConfig()
        with pytest.raises(Exception):  # noqa: B017
            wtc.max_concurrent_activities = 10  # type: ignore[misc]


class TestWorkerTuningConfigFromConcurrency:
    def test_sets_max_concurrent_activities(self) -> None:
        wtc = WorkerTuningConfig.from_concurrency(16)
        assert wtc.max_concurrent_activities == 16

    def test_other_fields_remain_none(self) -> None:
        wtc = WorkerTuningConfig.from_concurrency(8)
        assert wtc.max_concurrent_workflow_tasks is None
        assert wtc.max_concurrent_local_activities is None
        assert wtc.max_cached_workflows is None


class TestWorkerTuningConfigToWorkerKwargs:
    def test_empty_when_all_none(self) -> None:
        assert WorkerTuningConfig().to_worker_kwargs() == {}

    def test_omits_none_fields(self) -> None:
        wtc = WorkerTuningConfig(max_concurrent_activities=10)
        result = wtc.to_worker_kwargs()
        assert set(result.keys()) == {"max_concurrent_activities"}
        assert result["max_concurrent_activities"] == 10

    def test_all_fields_included_when_set(self) -> None:
        wtc = WorkerTuningConfig(
            max_concurrent_activities=10,
            max_concurrent_workflow_tasks=5,
            max_concurrent_local_activities=20,
            max_cached_workflows=200,
        )
        result = wtc.to_worker_kwargs()
        assert result == {
            "max_concurrent_activities": 10,
            "max_concurrent_workflow_tasks": 5,
            "max_concurrent_local_activities": 20,
            "max_cached_workflows": 200,
        }

    def test_keys_match_temporalio_worker_params(self) -> None:
        """Keys must be accepted by temporalio.worker.Worker.__init__."""
        import inspect

        from temporalio.worker import Worker

        sig = inspect.signature(Worker.__init__)
        valid_params = set(sig.parameters.keys())

        wtc = WorkerTuningConfig(
            max_concurrent_activities=1,
            max_concurrent_workflow_tasks=1,
            max_concurrent_local_activities=1,
            max_cached_workflows=100,
        )
        for key in wtc.to_worker_kwargs():
            assert key in valid_params, f"{key!r} is not a valid Worker.__init__ param"
