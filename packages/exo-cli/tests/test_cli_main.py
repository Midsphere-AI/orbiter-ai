"""Tests for exo_cli.main — CLI entry point, arg parsing, config loading."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from exo_cli.main import (
    CLIError,
    _cli_run,
    _format_duration,
    _format_timestamp,
    _mask_redis_url,
    _status_color,
    app,
    find_config,
    load_config,
    resolve_config,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# CLIError
# ---------------------------------------------------------------------------


class TestCLIError:
    def test_is_exception(self) -> None:
        assert issubclass(CLIError, Exception)

    def test_message(self) -> None:
        err = CLIError("bad config")
        assert str(err) == "bad config"


# ---------------------------------------------------------------------------
# find_config
# ---------------------------------------------------------------------------


class TestFindConfig:
    def test_finds_exo_yaml(self, tmp_path: Path) -> None:
        (tmp_path / ".exo.yaml").write_text("agents: {}")
        result = find_config(tmp_path)
        assert result is not None
        assert result.name == ".exo.yaml"

    def test_finds_exo_config_yaml(self, tmp_path: Path) -> None:
        (tmp_path / "exo.config.yaml").write_text("agents: {}")
        result = find_config(tmp_path)
        assert result is not None
        assert result.name == "exo.config.yaml"

    def test_prefers_exo_yaml_over_config(self, tmp_path: Path) -> None:
        (tmp_path / ".exo.yaml").write_text("a: 1")
        (tmp_path / "exo.config.yaml").write_text("b: 2")
        result = find_config(tmp_path)
        assert result is not None
        assert result.name == ".exo.yaml"

    def test_returns_none_when_no_config(self, tmp_path: Path) -> None:
        assert find_config(tmp_path) is None

    def test_defaults_to_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".exo.yaml").write_text("x: 1")
        result = find_config()
        assert result is not None
        assert result.name == ".exo.yaml"


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_loads_valid_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("agents:\n  bot:\n    model: openai:gpt-4o\n")
        result = load_config(cfg_file)
        assert "agents" in result
        assert result["agents"]["bot"]["model"] == "openai:gpt-4o"

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(CLIError, match="Config file not found"):
            load_config(tmp_path / "nonexistent.yaml")

    def test_raises_on_invalid_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "bad.yaml"
        cfg_file.write_text("- just a list")
        with pytest.raises(CLIError, match="Invalid config"):
            load_config(cfg_file)

    def test_variable_substitution(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_MODEL", "openai:gpt-4o")
        cfg_file = tmp_path / "env.yaml"
        cfg_file.write_text("agents:\n  bot:\n    model: ${TEST_MODEL}\n")
        result = load_config(cfg_file)
        assert result["agents"]["bot"]["model"] == "openai:gpt-4o"

    def test_vars_section_substitution(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "vars.yaml"
        cfg_file.write_text("vars:\n  temp: 0.7\nagents:\n  bot:\n    temperature: ${vars.temp}\n")
        result = load_config(cfg_file)
        assert result["agents"]["bot"]["temperature"] == 0.7

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "str.yaml"
        cfg_file.write_text("key: value\n")
        result = load_config(str(cfg_file))
        assert result["key"] == "value"


# ---------------------------------------------------------------------------
# resolve_config
# ---------------------------------------------------------------------------


class TestResolveConfig:
    def test_explicit_path(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "explicit.yaml"
        cfg_file.write_text("agents: {}\n")
        result = resolve_config(str(cfg_file))
        assert result is not None
        assert "agents" in result

    def test_auto_discovery(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".exo.yaml").write_text("agents:\n  a:\n    model: test\n")
        result = resolve_config(None)
        assert result is not None
        assert "agents" in result

    def test_no_config_returns_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        assert resolve_config(None) is None

    def test_explicit_overrides_discovery(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".exo.yaml").write_text("source: auto\n")
        explicit = tmp_path / "custom.yaml"
        explicit.write_text("source: explicit\n")
        result = resolve_config(str(explicit))
        assert result is not None
        assert result["source"] == "explicit"


# ---------------------------------------------------------------------------
# CLI arg parsing — run command
# ---------------------------------------------------------------------------


class TestCLIRun:
    def test_no_args_shows_help(self) -> None:
        result = runner.invoke(app, [])
        # Typer returns exit code 0 or 2 depending on version for no_args_is_help
        assert result.exit_code in (0, 2)
        assert "Usage" in result.output

    def test_run_without_config_exits_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["run", "hello"])
        assert result.exit_code == 1
        assert "No config file found" in result.output

    def test_run_with_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        cfg = tmp_path / ".exo.yaml"
        cfg.write_text("agents:\n  bot:\n    model: test\n")
        mock_result = MagicMock()
        mock_result.output = "Hello!"
        mock_result.steps = 1
        mock_result.elapsed = 0.1
        mock_result.usage = {}
        with patch(
            "exo_cli.executor.LocalExecutor.execute",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = runner.invoke(app, ["run", "hello"])
        assert result.exit_code == 0
        assert "Running with input:" in result.output

    def test_run_with_explicit_config(self, tmp_path: Path) -> None:
        cfg = tmp_path / "custom.yaml"
        cfg.write_text("agents:\n  bot:\n    model: test\n")
        mock_result = MagicMock()
        mock_result.output = "OK"
        mock_result.steps = 1
        mock_result.elapsed = 0.1
        mock_result.usage = {}
        with patch(
            "exo_cli.executor.LocalExecutor.execute",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = runner.invoke(app, ["run", "--config", str(cfg), "hello"])
        assert result.exit_code == 0

    def test_run_with_model_flag(self, tmp_path: Path) -> None:
        cfg = tmp_path / "m.yaml"
        cfg.write_text("agents:\n  bot:\n    model: test\n")
        mock_result = MagicMock()
        mock_result.output = "OK"
        mock_result.steps = 1
        mock_result.elapsed = 0.1
        mock_result.usage = {}
        with patch(
            "exo_cli.executor.LocalExecutor.execute",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = runner.invoke(app, ["run", "-c", str(cfg), "-m", "openai:gpt-4o", "test"])
        assert result.exit_code == 0

    def test_run_with_stream_flag(self, tmp_path: Path) -> None:
        cfg = tmp_path / "s.yaml"
        cfg.write_text("agents:\n  bot:\n    model: test\n")

        async def _fake_stream(*_a: object, **_kw: object):
            yield "chunk1"
            yield "chunk2"

        with patch("exo_cli.executor.LocalExecutor.stream", side_effect=_fake_stream):
            result = runner.invoke(app, ["run", "-c", str(cfg), "--stream", "test"])
        assert result.exit_code == 0

    def test_run_verbose(self, tmp_path: Path) -> None:
        cfg = tmp_path / "v.yaml"
        cfg.write_text("agents:\n  a:\n    model: x\n")
        mock_result = MagicMock()
        mock_result.output = "Done"
        mock_result.steps = 1
        mock_result.elapsed = 0.2
        mock_result.usage = {}
        with patch(
            "exo_cli.executor.LocalExecutor.execute",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = runner.invoke(app, ["--verbose", "run", "-c", str(cfg), "test"])
        assert result.exit_code == 0
        assert "Loaded config" in result.output

    def test_run_invalid_config_path(self) -> None:
        result = runner.invoke(app, ["run", "-c", "/nonexistent/path.yaml", "hi"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# CLI arg parsing — help text
# ---------------------------------------------------------------------------


class TestCLIHelp:
    def test_help_flag(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "multi-agent" in result.output.lower() or "exo" in result.output.lower()

    def test_run_help(self) -> None:
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--model" in result.output
        assert "--stream" in result.output


# ---------------------------------------------------------------------------
# Config search order
# ---------------------------------------------------------------------------


class TestConfigPrecedence:
    def test_exo_yaml_found_first(self, tmp_path: Path) -> None:
        (tmp_path / ".exo.yaml").write_text("first: true\n")
        (tmp_path / "exo.config.yaml").write_text("second: true\n")
        result = find_config(tmp_path)
        assert result is not None
        config = load_config(result)
        assert config.get("first") is True

    def test_exo_config_yaml_fallback(self, tmp_path: Path) -> None:
        (tmp_path / "exo.config.yaml").write_text("fallback: true\n")
        result = find_config(tmp_path)
        assert result is not None
        config = load_config(result)
        assert config.get("fallback") is True

    def test_ignores_non_config_files(self, tmp_path: Path) -> None:
        (tmp_path / "random.yaml").write_text("ignored: true\n")
        assert find_config(tmp_path) is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_yaml_dict(self, tmp_path: Path) -> None:
        cfg = tmp_path / "empty.yaml"
        cfg.write_text("{}\n")
        result = load_config(cfg)
        assert result == {}

    def test_config_with_nested_vars(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OUTER", "hello")
        cfg = tmp_path / "nested.yaml"
        cfg.write_text("vars:\n  inner: world\ngreeting: ${OUTER} ${vars.inner}\n")
        result = load_config(cfg)
        assert result["greeting"] == "hello world"

    def test_find_config_with_path_object(self, tmp_path: Path) -> None:
        (tmp_path / ".exo.yaml").write_text("ok: true\n")
        result = find_config(Path(tmp_path))
        assert result is not None


# ---------------------------------------------------------------------------
# _mask_redis_url helper
# ---------------------------------------------------------------------------


class TestMaskRedisUrl:
    def test_masks_standard_url(self) -> None:
        assert _mask_redis_url("redis://localhost:6379/0") == "redis://localhost:6379/***"

    def test_masks_url_with_password(self) -> None:
        result = _mask_redis_url("redis://:secret@myhost:6380/2")
        assert result == "redis://myhost:6380/***"
        assert "secret" not in result

    def test_masks_url_default_port(self) -> None:
        result = _mask_redis_url("redis://myhost")
        assert result == "redis://myhost:6379/***"

    def test_handles_invalid_url(self) -> None:
        result = _mask_redis_url("not a url at all")
        assert "***" in result


# ---------------------------------------------------------------------------
# CLI: start worker command
# ---------------------------------------------------------------------------


class TestStartWorkerCommand:
    def test_start_worker_registered(self) -> None:
        """The 'start worker' subcommand is registered on the app."""
        result = runner.invoke(app, ["start", "--help"])
        assert result.exit_code == 0
        assert "worker" in result.output.lower()

    def test_start_worker_help(self) -> None:
        """The 'start worker' command shows expected options."""
        result = runner.invoke(app, ["start", "worker", "--help"])
        assert result.exit_code == 0
        assert "--redis-url" in result.output
        assert "--concurrency" in result.output
        assert "--queue" in result.output
        assert "--worker-id" in result.output

    def test_start_worker_no_redis_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Error when neither --redis-url nor EXO_REDIS_URL provided."""
        monkeypatch.delenv("EXO_REDIS_URL", raising=False)
        result = runner.invoke(app, ["start", "worker"])
        assert result.exit_code == 1
        assert "redis-url" in result.output.lower() or "EXO_REDIS_URL" in result.output

    def test_start_worker_uses_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Falls back to EXO_REDIS_URL env var when --redis-url not set."""
        monkeypatch.setenv("EXO_REDIS_URL", "redis://envhost:6379")
        mock_worker = MagicMock()
        mock_worker.worker_id = "test-worker-123"
        mock_worker.start = AsyncMock()

        with patch("exo.distributed.worker.Worker", return_value=mock_worker):
            result = runner.invoke(app, ["start", "worker"])
        assert result.exit_code == 0
        assert "envhost" in result.output

    def test_start_worker_with_redis_url_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """--redis-url flag is used when provided."""
        monkeypatch.delenv("EXO_REDIS_URL", raising=False)
        mock_worker = MagicMock()
        mock_worker.worker_id = "custom-worker"
        mock_worker.start = AsyncMock()

        with patch("exo.distributed.worker.Worker", return_value=mock_worker):
            result = runner.invoke(
                app,
                ["start", "worker", "--redis-url", "redis://flaghost:6379"],
            )
        assert result.exit_code == 0
        assert "flaghost" in result.output

    def test_start_worker_banner_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Startup banner shows worker ID, masked URL, queue, concurrency."""
        monkeypatch.delenv("EXO_REDIS_URL", raising=False)
        mock_worker = MagicMock()
        mock_worker.worker_id = "my-worker-abc"
        mock_worker.start = AsyncMock()

        with patch("exo.distributed.worker.Worker", return_value=mock_worker):
            result = runner.invoke(
                app,
                [
                    "start",
                    "worker",
                    "--redis-url",
                    "redis://:password@myredis:6380/1",
                    "--concurrency",
                    "4",
                    "--queue",
                    "custom:queue",
                    "--worker-id",
                    "my-worker-abc",
                ],
            )
        assert result.exit_code == 0
        assert "my-worker-abc" in result.output
        assert "myredis" in result.output
        assert "password" not in result.output
        assert "custom:queue" in result.output
        assert "4" in result.output

    def test_start_worker_passes_options_to_worker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CLI options are passed correctly to Worker constructor."""
        monkeypatch.delenv("EXO_REDIS_URL", raising=False)
        mock_worker = MagicMock()
        mock_worker.worker_id = "w1"
        mock_worker.start = AsyncMock()

        with patch("exo.distributed.worker.Worker", return_value=mock_worker) as mock_cls:
            runner.invoke(
                app,
                [
                    "start",
                    "worker",
                    "--redis-url",
                    "redis://localhost:6379",
                    "--concurrency",
                    "3",
                    "--queue",
                    "my:queue",
                    "--worker-id",
                    "w1",
                ],
            )
        mock_cls.assert_called_once_with(
            "redis://localhost:6379",
            worker_id="w1",
            concurrency=3,
            queue_name="my:queue",
        )


# ---------------------------------------------------------------------------
# Helper functions for task commands
# ---------------------------------------------------------------------------


class TestFormatTimestamp:
    def test_none_returns_dash(self) -> None:
        assert _format_timestamp(None) == "-"

    def test_formats_unix_timestamp(self) -> None:
        # 2024-01-15 12:30:00 UTC = 1705318200.0
        result = _format_timestamp(1705318200.0)
        assert "2024" in result
        assert "UTC" in result


class TestFormatDuration:
    def test_none_started_returns_dash(self) -> None:
        assert _format_duration(None, None) == "-"

    def test_none_completed_returns_running(self) -> None:
        assert _format_duration(1000.0, None) == "running..."

    def test_milliseconds(self) -> None:
        result = _format_duration(1000.0, 1000.5)
        assert "ms" in result

    def test_seconds(self) -> None:
        result = _format_duration(1000.0, 1005.3)
        assert "s" in result

    def test_minutes(self) -> None:
        result = _format_duration(1000.0, 1120.0)
        assert "m" in result


class TestStatusColor:
    def test_known_statuses(self) -> None:
        assert _status_color("pending") == "yellow"
        assert _status_color("running") == "blue"
        assert _status_color("completed") == "green"
        assert _status_color("failed") == "red"
        assert _status_color("cancelled") == "dim"
        assert _status_color("retrying") == "magenta"

    def test_unknown_status(self) -> None:
        assert _status_color("unknown") == "white"


# ---------------------------------------------------------------------------
# CLI: task subcommand group
# ---------------------------------------------------------------------------


class TestTaskCommandRegistered:
    def test_task_subcommand_registered(self) -> None:
        """The 'task' subcommand group is registered on the app."""
        result = runner.invoke(app, ["task", "--help"])
        assert result.exit_code == 0
        assert "status" in result.output.lower()
        assert "cancel" in result.output.lower()
        assert "list" in result.output.lower()

    def test_task_status_help(self) -> None:
        result = runner.invoke(app, ["task", "status", "--help"])
        assert result.exit_code == 0
        assert "--redis-url" in result.output
        assert "task-id" in result.output.lower() or "TASK_ID" in result.output

    def test_task_cancel_help(self) -> None:
        result = runner.invoke(app, ["task", "cancel", "--help"])
        assert result.exit_code == 0
        assert "--redis-url" in result.output

    def test_task_list_help(self) -> None:
        result = runner.invoke(app, ["task", "list", "--help"])
        assert result.exit_code == 0
        assert "--redis-url" in result.output
        assert "--status" in result.output
        assert "--limit" in result.output


class TestTaskStatusCommand:
    def test_no_redis_url_exits_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("EXO_REDIS_URL", raising=False)
        result = runner.invoke(app, ["task", "status", "abc123"])
        assert result.exit_code == 1
        assert "redis-url" in result.output.lower() or "EXO_REDIS_URL" in result.output

    def test_task_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("EXO_REDIS_URL", raising=False)
        mock_store = MagicMock()
        mock_store.connect = AsyncMock()
        mock_store.disconnect = AsyncMock()
        mock_store.get_status = AsyncMock(return_value=None)

        with patch("exo.distributed.store.TaskStore", return_value=mock_store):
            result = runner.invoke(
                app,
                ["task", "status", "nonexistent", "--redis-url", "redis://localhost:6379"],
            )
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_shows_task_details(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("EXO_REDIS_URL", raising=False)

        from exo.distributed.models import (  # pyright: ignore[reportMissingImports]
            TaskResult,
            TaskStatus,
        )

        task_result = TaskResult(
            task_id="abc123",
            status=TaskStatus.RUNNING,
            worker_id="worker-1",
            started_at=1705318200.0,
            completed_at=None,
            retries=0,
        )

        mock_store = MagicMock()
        mock_store.connect = AsyncMock()
        mock_store.disconnect = AsyncMock()
        mock_store.get_status = AsyncMock(return_value=task_result)

        with patch("exo.distributed.store.TaskStore", return_value=mock_store):
            result = runner.invoke(
                app,
                ["task", "status", "abc123", "--redis-url", "redis://localhost:6379"],
            )
        assert result.exit_code == 0
        assert "abc123" in result.output
        assert "running" in result.output.lower()
        assert "worker-1" in result.output

    def test_shows_error_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("EXO_REDIS_URL", raising=False)

        from exo.distributed.models import (  # pyright: ignore[reportMissingImports]
            TaskResult,
            TaskStatus,
        )

        task_result = TaskResult(
            task_id="fail1",
            status=TaskStatus.FAILED,
            error="Connection timeout",
            started_at=1000.0,
            completed_at=1005.0,
        )

        mock_store = MagicMock()
        mock_store.connect = AsyncMock()
        mock_store.disconnect = AsyncMock()
        mock_store.get_status = AsyncMock(return_value=task_result)

        with patch("exo.distributed.store.TaskStore", return_value=mock_store):
            result = runner.invoke(
                app,
                ["task", "status", "fail1", "--redis-url", "redis://localhost:6379"],
            )
        assert result.exit_code == 0
        assert "Connection timeout" in result.output

    def test_shows_result_preview(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("EXO_REDIS_URL", raising=False)

        from exo.distributed.models import (  # pyright: ignore[reportMissingImports]
            TaskResult,
            TaskStatus,
        )

        task_result = TaskResult(
            task_id="done1",
            status=TaskStatus.COMPLETED,
            result={"output": "hello world"},
            started_at=1000.0,
            completed_at=1002.0,
        )

        mock_store = MagicMock()
        mock_store.connect = AsyncMock()
        mock_store.disconnect = AsyncMock()
        mock_store.get_status = AsyncMock(return_value=task_result)

        with patch("exo.distributed.store.TaskStore", return_value=mock_store):
            result = runner.invoke(
                app,
                ["task", "status", "done1", "--redis-url", "redis://localhost:6379"],
            )
        assert result.exit_code == 0
        assert "hello world" in result.output

    def test_uses_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXO_REDIS_URL", "redis://envhost:6379")

        mock_store = MagicMock()
        mock_store.connect = AsyncMock()
        mock_store.disconnect = AsyncMock()
        mock_store.get_status = AsyncMock(return_value=None)

        with patch("exo.distributed.store.TaskStore", return_value=mock_store) as mock_cls:
            runner.invoke(app, ["task", "status", "abc123"])
        mock_cls.assert_called_once_with("redis://envhost:6379")


class TestTaskCancelCommand:
    def test_no_redis_url_exits_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("EXO_REDIS_URL", raising=False)
        result = runner.invoke(app, ["task", "cancel", "abc123"])
        assert result.exit_code == 1

    def test_cancels_task(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("EXO_REDIS_URL", raising=False)

        mock_broker = MagicMock()
        mock_broker.connect = AsyncMock()
        mock_broker.disconnect = AsyncMock()
        mock_broker.cancel = AsyncMock()

        with patch("exo.distributed.broker.TaskBroker", return_value=mock_broker):
            result = runner.invoke(
                app,
                ["task", "cancel", "task-to-cancel", "--redis-url", "redis://localhost:6379"],
            )
        assert result.exit_code == 0
        assert "cancelled" in result.output.lower()
        mock_broker.cancel.assert_called_once_with("task-to-cancel")

    def test_uses_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXO_REDIS_URL", "redis://envhost:6379")

        mock_broker = MagicMock()
        mock_broker.connect = AsyncMock()
        mock_broker.disconnect = AsyncMock()
        mock_broker.cancel = AsyncMock()

        with patch("exo.distributed.broker.TaskBroker", return_value=mock_broker) as mock_cls:
            runner.invoke(app, ["task", "cancel", "abc123"])
        mock_cls.assert_called_once_with("redis://envhost:6379")


class TestTaskListCommand:
    def test_no_redis_url_exits_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("EXO_REDIS_URL", raising=False)
        result = runner.invoke(app, ["task", "list"])
        assert result.exit_code == 1

    def test_empty_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("EXO_REDIS_URL", raising=False)

        mock_store = MagicMock()
        mock_store.connect = AsyncMock()
        mock_store.disconnect = AsyncMock()
        mock_store.list_tasks = AsyncMock(return_value=[])

        with patch("exo.distributed.store.TaskStore", return_value=mock_store):
            result = runner.invoke(
                app,
                ["task", "list", "--redis-url", "redis://localhost:6379"],
            )
        assert result.exit_code == 0
        assert "no tasks" in result.output.lower()

    def test_lists_tasks_in_table(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("EXO_REDIS_URL", raising=False)

        from exo.distributed.models import (  # pyright: ignore[reportMissingImports]
            TaskResult,
            TaskStatus,
        )

        tasks = [
            TaskResult(
                task_id="task-1",
                status=TaskStatus.COMPLETED,
                worker_id="w1",
                started_at=1000.0,
                completed_at=1005.0,
            ),
            TaskResult(
                task_id="task-2",
                status=TaskStatus.RUNNING,
                worker_id="w2",
                started_at=2000.0,
            ),
        ]

        mock_store = MagicMock()
        mock_store.connect = AsyncMock()
        mock_store.disconnect = AsyncMock()
        mock_store.list_tasks = AsyncMock(return_value=tasks)

        with patch("exo.distributed.store.TaskStore", return_value=mock_store):
            result = runner.invoke(
                app,
                ["task", "list", "--redis-url", "redis://localhost:6379"],
            )
        assert result.exit_code == 0
        assert "task-1" in result.output
        assert "task-2" in result.output
        assert "Distributed Tasks" in result.output

    def test_filters_by_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("EXO_REDIS_URL", raising=False)

        from exo.distributed.models import TaskStatus  # pyright: ignore[reportMissingImports]

        mock_store = MagicMock()
        mock_store.connect = AsyncMock()
        mock_store.disconnect = AsyncMock()
        mock_store.list_tasks = AsyncMock(return_value=[])

        with patch("exo.distributed.store.TaskStore", return_value=mock_store):
            runner.invoke(
                app,
                ["task", "list", "--status", "running", "--redis-url", "redis://localhost:6379"],
            )
        mock_store.list_tasks.assert_called_once_with(status=TaskStatus.RUNNING, limit=100)

    def test_invalid_status_exits_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("EXO_REDIS_URL", raising=False)
        result = runner.invoke(
            app,
            ["task", "list", "--status", "bogus", "--redis-url", "redis://localhost:6379"],
        )
        assert result.exit_code == 1
        assert "invalid status" in result.output.lower()

    def test_limit_option(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("EXO_REDIS_URL", raising=False)

        mock_store = MagicMock()
        mock_store.connect = AsyncMock()
        mock_store.disconnect = AsyncMock()
        mock_store.list_tasks = AsyncMock(return_value=[])

        with patch("exo.distributed.store.TaskStore", return_value=mock_store):
            runner.invoke(
                app,
                ["task", "list", "--limit", "50", "--redis-url", "redis://localhost:6379"],
            )
        mock_store.list_tasks.assert_called_once_with(status=None, limit=50)

    def test_uses_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXO_REDIS_URL", "redis://envhost:6379")

        mock_store = MagicMock()
        mock_store.connect = AsyncMock()
        mock_store.disconnect = AsyncMock()
        mock_store.list_tasks = AsyncMock(return_value=[])

        with patch("exo.distributed.store.TaskStore", return_value=mock_store) as mock_cls:
            runner.invoke(app, ["task", "list"])
        mock_cls.assert_called_once_with("redis://envhost:6379")


# ---------------------------------------------------------------------------
# CLI: worker subcommand group
# ---------------------------------------------------------------------------


class TestWorkerCommandRegistered:
    def test_worker_subcommand_registered(self) -> None:
        """The 'worker' subcommand group is registered on the app."""
        result = runner.invoke(app, ["worker", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output.lower()

    def test_worker_list_help(self) -> None:
        result = runner.invoke(app, ["worker", "list", "--help"])
        assert result.exit_code == 0
        assert "--redis-url" in result.output


class TestWorkerListCommand:
    def test_no_redis_url_exits_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("EXO_REDIS_URL", raising=False)
        result = runner.invoke(app, ["worker", "list"])
        assert result.exit_code == 1
        assert "redis-url" in result.output.lower() or "EXO_REDIS_URL" in result.output

    def test_empty_worker_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("EXO_REDIS_URL", raising=False)

        with patch(
            "exo.distributed.health.get_worker_fleet_status",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = runner.invoke(
                app,
                ["worker", "list", "--redis-url", "redis://localhost:6379"],
            )
        assert result.exit_code == 0
        assert "no active workers" in result.output.lower()

    def test_lists_workers_in_table(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("EXO_REDIS_URL", raising=False)

        from exo.distributed.health import WorkerHealth  # pyright: ignore[reportMissingImports]

        workers = [
            WorkerHealth(
                worker_id="w1",
                status="running",
                tasks_processed=5,
                tasks_failed=1,
                hostname="host1",
                concurrency=2,
                last_heartbeat=1705318200.0,
                alive=True,
            ),
            WorkerHealth(
                worker_id="w2",
                status="running",
                tasks_processed=3,
                tasks_failed=0,
                current_task_id="task-abc",
                hostname="host2",
                concurrency=1,
                last_heartbeat=1705318200.0,
                alive=True,
            ),
        ]

        with patch(
            "exo.distributed.health.get_worker_fleet_status",
            new_callable=AsyncMock,
            return_value=workers,
        ):
            result = runner.invoke(
                app,
                ["worker", "list", "--redis-url", "redis://localhost:6379"],
            )
        assert result.exit_code == 0
        assert "w1" in result.output
        assert "w2" in result.output
        assert "host1" in result.output
        assert "host2" in result.output
        assert "Distributed Workers" in result.output

    def test_dead_worker_shown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("EXO_REDIS_URL", raising=False)

        from exo.distributed.health import WorkerHealth  # pyright: ignore[reportMissingImports]

        workers = [
            WorkerHealth(
                worker_id="dead-w",
                status="running",
                hostname="gone",
                alive=False,
            ),
        ]

        with patch(
            "exo.distributed.health.get_worker_fleet_status",
            new_callable=AsyncMock,
            return_value=workers,
        ):
            result = runner.invoke(
                app,
                ["worker", "list", "--redis-url", "redis://localhost:6379"],
            )
        assert result.exit_code == 0
        assert "dead-w" in result.output
        assert "dead" in result.output.lower()

    def test_uses_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXO_REDIS_URL", "redis://envhost:6379")

        with patch(
            "exo.distributed.health.get_worker_fleet_status",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_fn:
            runner.invoke(app, ["worker", "list"])
        mock_fn.assert_called_once_with("redis://envhost:6379")


# ---------------------------------------------------------------------------
# CLI: chat command
# ---------------------------------------------------------------------------


class TestChatCommand:
    def test_chat_help(self) -> None:
        result = runner.invoke(app, ["chat", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--model" in result.output
        assert "--stream" in result.output

    def test_chat_no_config_exits_1(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["chat"])
        assert result.exit_code == 1
        assert "No config file found" in result.output

    def test_chat_with_config(self, tmp_path: Path) -> None:
        cfg = tmp_path / "c.yaml"
        cfg.write_text("agents:\n  bot:\n    model: test\n")
        mock_console = MagicMock()
        mock_console.start = AsyncMock()
        with patch("exo_cli.console.InteractiveConsole", return_value=mock_console):
            result = runner.invoke(app, ["chat", "-c", str(cfg)])
        assert result.exit_code == 0
        mock_console.start.assert_called_once()

    def test_chat_with_model_override(self, tmp_path: Path) -> None:
        cfg = tmp_path / "c.yaml"
        cfg.write_text("agents:\n  bot:\n    model: test\n")
        mock_console = MagicMock()
        mock_console.start = AsyncMock()
        with patch("exo_cli.console.InteractiveConsole", return_value=mock_console):
            result = runner.invoke(app, ["chat", "-c", str(cfg), "-m", "openai:gpt-4o"])
        assert result.exit_code == 0

    def test_chat_empty_agents_exits_1(self, tmp_path: Path) -> None:
        cfg = tmp_path / "c.yaml"
        cfg.write_text("agents: {}\n")
        result = runner.invoke(app, ["chat", "-c", str(cfg)])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# CLI: batch command
# ---------------------------------------------------------------------------


class TestBatchCommand:
    def test_batch_help(self) -> None:
        result = runner.invoke(app, ["batch", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--concurrency" in result.output

    def test_batch_no_config_exits_1(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        inputs = tmp_path / "in.jsonl"
        inputs.write_text('{"input": "hello"}\n')
        result = runner.invoke(app, ["batch", str(inputs)])
        assert result.exit_code == 1
        assert "No config file found" in result.output

    def test_batch_missing_inputs_file_exits_1(self, tmp_path: Path) -> None:
        cfg = tmp_path / "c.yaml"
        cfg.write_text("agents:\n  bot:\n    model: test\n")
        result = runner.invoke(app, ["batch", "-c", str(cfg), str(tmp_path / "missing.jsonl")])
        assert result.exit_code == 1
        assert "loading inputs" in result.output.lower()

    def test_batch_runs_and_prints_results(self, tmp_path: Path) -> None:
        cfg = tmp_path / "c.yaml"
        cfg.write_text("agents:\n  bot:\n    model: test\n")
        inputs = tmp_path / "in.jsonl"
        inputs.write_text('{"input": "hello"}\n{"input": "world"}\n')

        from exo_cli.batch import BatchResult, ItemResult

        mock_batch_result = BatchResult(
            results=[
                ItemResult(item_id="1", success=True, output="hi", elapsed=0.1),
                ItemResult(item_id="2", success=True, output="there", elapsed=0.1),
            ],
            total=2,
            succeeded=2,
            failed=0,
        )
        with patch(
            "exo_cli.batch.batch_execute", new_callable=AsyncMock, return_value=mock_batch_result
        ):
            result = runner.invoke(app, ["batch", "-c", str(cfg), str(inputs)])
        assert result.exit_code == 0
        assert "2 items" in result.output

    def test_batch_csv_output(self, tmp_path: Path) -> None:
        cfg = tmp_path / "c.yaml"
        cfg.write_text("agents:\n  bot:\n    model: test\n")
        inputs = tmp_path / "in.jsonl"
        inputs.write_text('{"input": "hello"}\n')

        from exo_cli.batch import BatchResult, ItemResult

        mock_batch_result = BatchResult(
            results=[ItemResult(item_id="1", success=True, output="hi", elapsed=0.1)],
            total=1,
            succeeded=1,
            failed=0,
        )
        with patch(
            "exo_cli.batch.batch_execute", new_callable=AsyncMock, return_value=mock_batch_result
        ):
            result = runner.invoke(
                app, ["batch", "-c", str(cfg), str(inputs), "--output-format", "csv"]
            )
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# _cli_run boundary
# ---------------------------------------------------------------------------


class TestCliRun:
    """Tests for the _cli_run shared async boundary helper.

    These tests are synchronous because _cli_run itself calls asyncio.run(),
    which cannot be called from an already-running event loop.
    """

    def test_exo_error_exits_1(self) -> None:
        """ExoError from the coroutine → exit code 1, clean message, no traceback."""
        from exo.types import ExoError  # pyright: ignore[reportMissingImports]

        async def _failing() -> None:
            raise ExoError("something went wrong", hint="Check your config.")

        with pytest.raises(typer.Exit) as exc_info:
            _cli_run(_failing())
        assert exc_info.value.exit_code == 1

    def test_keyboard_interrupt_exits_130(self) -> None:
        """KeyboardInterrupt → exit code 130."""

        async def _interrupted() -> None:
            raise KeyboardInterrupt

        with pytest.raises(typer.Exit) as exc_info:
            _cli_run(_interrupted())
        assert exc_info.value.exit_code == 130

    def test_exception_group_unwrapped_exits_1(self) -> None:
        """A single-child BaseExceptionGroup is unwrapped, exits 1."""

        async def _grouped() -> None:
            raise BaseExceptionGroup("gather", [ValueError("inner cause")])

        with pytest.raises(typer.Exit) as exc_info:
            _cli_run(_grouped())
        assert exc_info.value.exit_code == 1

    def test_exception_group_verbose_reraises(self) -> None:
        """With verbose=True, BaseExceptionGroup is re-raised after printing."""

        async def _grouped() -> None:
            raise BaseExceptionGroup("gather", [ValueError("inner cause")])

        with pytest.raises(BaseExceptionGroup):
            _cli_run(_grouped(), verbose=True)

    def test_success_does_not_raise(self) -> None:
        """Successful coroutine → no exception raised."""

        async def _ok() -> None:
            return

        _cli_run(_ok())  # should not raise
