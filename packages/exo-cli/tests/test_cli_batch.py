"""Tests for exo_cli.batch — batch execution module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exo_cli.batch import (
    BatchError,
    BatchItem,
    BatchResult,
    InputFormat,
    ItemResult,
    batch_execute,
    load_batch_items,
    results_to_csv,
    results_to_jsonl,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_agent(name: str = "batch-agent") -> MagicMock:
    agent = MagicMock()
    agent.name = name
    return agent


def _write_file(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# BatchError
# ---------------------------------------------------------------------------


class TestBatchError:
    def test_is_exception(self) -> None:
        exc = BatchError("fail")
        assert isinstance(exc, Exception)
        assert str(exc) == "fail"


# ---------------------------------------------------------------------------
# InputFormat
# ---------------------------------------------------------------------------


class TestInputFormat:
    def test_values(self) -> None:
        assert InputFormat.JSON == "json"
        assert InputFormat.CSV == "csv"
        assert InputFormat.JSONL == "jsonl"

    def test_is_str_enum(self) -> None:
        assert isinstance(InputFormat.JSON, str)


# ---------------------------------------------------------------------------
# BatchItem
# ---------------------------------------------------------------------------


class TestBatchItem:
    def test_creation(self) -> None:
        item = BatchItem(id="1", input="hello")
        assert item.id == "1"
        assert item.input == "hello"
        assert item.metadata == {}

    def test_with_metadata(self) -> None:
        item = BatchItem(id="2", input="hi", metadata={"label": "test"})
        assert item.metadata == {"label": "test"}

    def test_frozen(self) -> None:
        item = BatchItem(id="1", input="x")
        with pytest.raises(AttributeError):
            item.id = "2"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ItemResult
# ---------------------------------------------------------------------------


class TestItemResult:
    def test_creation(self) -> None:
        r = ItemResult(item_id="1", success=True, output="ok")
        assert r.item_id == "1"
        assert r.success is True
        assert r.output == "ok"
        assert r.elapsed == 0.0
        assert r.error == ""

    def test_failure(self) -> None:
        r = ItemResult(item_id="2", success=False, output="", error="timeout")
        assert r.success is False
        assert r.error == "timeout"


# ---------------------------------------------------------------------------
# BatchResult
# ---------------------------------------------------------------------------


class TestBatchResult:
    def test_defaults(self) -> None:
        br = BatchResult()
        assert br.results == []
        assert br.total == 0
        assert br.succeeded == 0
        assert br.failed == 0

    def test_summary(self) -> None:
        br = BatchResult(total=10, succeeded=8, failed=2)
        assert "10 items" in br.summary()
        assert "8 succeeded" in br.summary()
        assert "2 failed" in br.summary()


# ---------------------------------------------------------------------------
# load_batch_items — JSON
# ---------------------------------------------------------------------------


class TestLoadJSON:
    def test_basic(self, tmp_path: Path) -> None:
        data = [{"input": "hello"}, {"input": "world"}]
        p = _write_file(tmp_path, "data.json", json.dumps(data))
        items = load_batch_items(p)
        assert len(items) == 2
        assert items[0].input == "hello"
        assert items[1].input == "world"

    def test_with_id(self, tmp_path: Path) -> None:
        data = [{"id": "a", "input": "hi"}]
        p = _write_file(tmp_path, "data.json", json.dumps(data))
        items = load_batch_items(p)
        assert items[0].id == "a"

    def test_auto_id(self, tmp_path: Path) -> None:
        data = [{"input": "hi"}]
        p = _write_file(tmp_path, "data.json", json.dumps(data))
        items = load_batch_items(p)
        assert items[0].id == "1"

    def test_metadata(self, tmp_path: Path) -> None:
        data = [{"input": "hi", "label": "test", "score": 5}]
        p = _write_file(tmp_path, "data.json", json.dumps(data))
        items = load_batch_items(p)
        assert items[0].metadata == {"label": "test", "score": 5}

    def test_not_a_list(self, tmp_path: Path) -> None:
        p = _write_file(tmp_path, "data.json", '{"input": "hi"}')
        with pytest.raises(BatchError, match="list"):
            load_batch_items(p)

    def test_missing_input_key(self, tmp_path: Path) -> None:
        p = _write_file(tmp_path, "data.json", '[{"text": "hi"}]')
        with pytest.raises(BatchError, match="missing required key"):
            load_batch_items(p)


# ---------------------------------------------------------------------------
# load_batch_items — JSONL
# ---------------------------------------------------------------------------


class TestLoadJSONL:
    def test_basic(self, tmp_path: Path) -> None:
        content = '{"input": "a"}\n{"input": "b"}\n'
        p = _write_file(tmp_path, "data.jsonl", content)
        items = load_batch_items(p)
        assert len(items) == 2
        assert items[0].input == "a"

    def test_blank_lines(self, tmp_path: Path) -> None:
        content = '{"input": "a"}\n\n{"input": "b"}\n'
        p = _write_file(tmp_path, "data.jsonl", content)
        items = load_batch_items(p)
        assert len(items) == 2

    def test_invalid_json(self, tmp_path: Path) -> None:
        p = _write_file(tmp_path, "data.jsonl", "not-json\n")
        with pytest.raises(BatchError, match="Invalid JSON"):
            load_batch_items(p)

    def test_not_object(self, tmp_path: Path) -> None:
        p = _write_file(tmp_path, "data.jsonl", "[1,2,3]\n")
        with pytest.raises(BatchError, match="not a JSON object"):
            load_batch_items(p)


# ---------------------------------------------------------------------------
# load_batch_items — CSV
# ---------------------------------------------------------------------------


class TestLoadCSV:
    def test_basic(self, tmp_path: Path) -> None:
        content = "input,label\nhello,a\nworld,b\n"
        p = _write_file(tmp_path, "data.csv", content)
        items = load_batch_items(p)
        assert len(items) == 2
        assert items[0].input == "hello"
        assert items[0].metadata == {"label": "a"}

    def test_custom_input_key(self, tmp_path: Path) -> None:
        content = "query,label\nhello,a\n"
        p = _write_file(tmp_path, "data.csv", content)
        items = load_batch_items(p, input_key="query")
        assert items[0].input == "hello"


# ---------------------------------------------------------------------------
# load_batch_items — format detection
# ---------------------------------------------------------------------------


class TestFormatDetection:
    def test_unsupported_extension(self, tmp_path: Path) -> None:
        p = _write_file(tmp_path, "data.txt", "hi")
        with pytest.raises(BatchError, match="Unsupported"):
            load_batch_items(p)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(BatchError, match="not found"):
            load_batch_items(tmp_path / "missing.json")

    def test_forced_format(self, tmp_path: Path) -> None:
        content = '{"input": "hi"}\n'
        p = _write_file(tmp_path, "data.txt", content)
        items = load_batch_items(p, fmt=InputFormat.JSONL)
        assert len(items) == 1
        assert items[0].input == "hi"


# ---------------------------------------------------------------------------
# batch_execute
# ---------------------------------------------------------------------------


class TestBatchExecute:
    async def test_basic(self) -> None:
        agent = _mock_agent()
        items = [BatchItem(id="1", input="hello")]

        mock_result = MagicMock()
        mock_result.output = "world"
        mock_result.elapsed = 0.5

        with patch("exo_cli.batch.LocalExecutor") as mock_cls:
            instance = mock_cls.return_value
            instance.execute = AsyncMock(return_value=mock_result)
            result = await batch_execute(agent, items)

        assert result.total == 1
        assert result.succeeded == 1
        assert result.failed == 0
        assert result.results[0].output == "world"

    async def test_multiple(self) -> None:
        agent = _mock_agent()
        items = [BatchItem(id=str(i), input=f"q{i}") for i in range(3)]

        mock_result = MagicMock()
        mock_result.output = "ans"
        mock_result.elapsed = 0.1

        with patch("exo_cli.batch.LocalExecutor") as mock_cls:
            instance = mock_cls.return_value
            instance.execute = AsyncMock(return_value=mock_result)
            result = await batch_execute(agent, items, concurrency=2)

        assert result.total == 3
        assert result.succeeded == 3

    async def test_failure(self) -> None:
        agent = _mock_agent()
        items = [BatchItem(id="1", input="hi")]

        with patch("exo_cli.batch.LocalExecutor") as mock_cls:
            instance = mock_cls.return_value
            instance.execute = AsyncMock(side_effect=RuntimeError("boom"))
            result = await batch_execute(agent, items)

        assert result.total == 1
        assert result.failed == 1
        assert result.results[0].error == "boom"

    async def test_mixed(self) -> None:
        agent = _mock_agent()
        items = [BatchItem(id="1", input="ok"), BatchItem(id="2", input="fail")]

        ok_result = MagicMock()
        ok_result.output = "good"
        ok_result.elapsed = 0.1

        with patch("exo_cli.batch.LocalExecutor") as mock_cls:
            instance = mock_cls.return_value
            instance.execute = AsyncMock(side_effect=[ok_result, RuntimeError("err")])
            result = await batch_execute(agent, items, concurrency=1)

        assert result.succeeded == 1
        assert result.failed == 1

    async def test_invalid_concurrency(self) -> None:
        agent = _mock_agent()
        with pytest.raises(BatchError, match="concurrency"):
            await batch_execute(agent, [], concurrency=0)

    async def test_empty(self) -> None:
        agent = _mock_agent()
        with patch("exo_cli.batch.LocalExecutor"):
            result = await batch_execute(agent, [])
        assert result.total == 0

    async def test_provider_forwarded(self) -> None:
        agent = _mock_agent()
        items = [BatchItem(id="1", input="hi")]
        mock_result = MagicMock()
        mock_result.output = "ok"
        mock_result.elapsed = 0.0

        with patch("exo_cli.batch.LocalExecutor") as mock_cls:
            instance = mock_cls.return_value
            instance.execute = AsyncMock(return_value=mock_result)
            await batch_execute(agent, items, provider="my-provider")
            mock_cls.assert_called_once_with(agent=agent, provider="my-provider", timeout=0.0)


# ---------------------------------------------------------------------------
# results_to_jsonl
# ---------------------------------------------------------------------------


class TestResultsToJSONL:
    def test_basic(self) -> None:
        br = BatchResult(
            results=[
                ItemResult(item_id="1", success=True, output="ok", elapsed=0.5),
            ],
            total=1,
            succeeded=1,
            failed=0,
        )
        text = results_to_jsonl(br)
        parsed = json.loads(text.strip())
        assert parsed["id"] == "1"
        assert parsed["success"] is True
        assert parsed["output"] == "ok"

    def test_error_included(self) -> None:
        br = BatchResult(
            results=[
                ItemResult(item_id="1", success=False, output="", error="timeout"),
            ],
            total=1,
            succeeded=0,
            failed=1,
        )
        text = results_to_jsonl(br)
        parsed = json.loads(text.strip())
        assert parsed["error"] == "timeout"

    def test_no_error_field_when_empty(self) -> None:
        br = BatchResult(
            results=[
                ItemResult(item_id="1", success=True, output="ok"),
            ],
            total=1,
            succeeded=1,
            failed=0,
        )
        text = results_to_jsonl(br)
        parsed = json.loads(text.strip())
        assert "error" not in parsed

    def test_empty(self) -> None:
        br = BatchResult()
        assert results_to_jsonl(br) == ""


# ---------------------------------------------------------------------------
# results_to_csv
# ---------------------------------------------------------------------------


class TestResultsToCSV:
    def test_basic(self) -> None:
        br = BatchResult(
            results=[
                ItemResult(item_id="1", success=True, output="hello", elapsed=1.0),
            ],
            total=1,
            succeeded=1,
            failed=0,
        )
        text = results_to_csv(br)
        assert "id,success,output,elapsed,error,timed_out" in text
        assert "1,True,hello,1.0," in text

    def test_multiple(self) -> None:
        br = BatchResult(
            results=[
                ItemResult(item_id="1", success=True, output="a"),
                ItemResult(item_id="2", success=False, output="", error="e"),
            ],
            total=2,
            succeeded=1,
            failed=1,
        )
        text = results_to_csv(br)
        lines = text.strip().splitlines()
        assert len(lines) == 3  # header + 2 rows


# ---------------------------------------------------------------------------
# ItemResult.timed_out flag
# ---------------------------------------------------------------------------


class TestItemResultTimedOut:
    def test_default_not_timed_out(self) -> None:
        r = ItemResult(item_id="1", success=True, output="ok")
        assert r.timed_out is False

    def test_timed_out_flag(self) -> None:
        r = ItemResult(item_id="1", success=False, output="", error="timed out", timed_out=True)
        assert r.timed_out is True

    async def test_executor_error_timeout_sets_flag(self) -> None:
        """ExecutorError containing 'timed out' sets timed_out=True on ItemResult."""
        from exo_cli.executor import ExecutorError

        agent = _mock_agent()
        items = [BatchItem(id="1", input="slow")]

        with patch("exo_cli.batch.LocalExecutor") as mock_cls:
            instance = mock_cls.return_value
            instance.execute = AsyncMock(
                side_effect=ExecutorError("Execution timed out after 5.0s")
            )
            result = await batch_execute(agent, items, concurrency=1, timeout=5.0)

        assert result.results[0].timed_out is True
        assert result.timed_out == 1
        assert result.failed == 1

    async def test_non_timeout_error_not_flagged(self) -> None:
        """A generic ExecutorError does NOT set timed_out."""
        from exo_cli.executor import ExecutorError

        agent = _mock_agent()
        items = [BatchItem(id="1", input="boom")]

        with patch("exo_cli.batch.LocalExecutor") as mock_cls:
            instance = mock_cls.return_value
            instance.execute = AsyncMock(side_effect=ExecutorError("Agent execution failed: crash"))
            result = await batch_execute(agent, items, concurrency=1)

        assert result.results[0].timed_out is False
        assert result.timed_out == 0

    def test_summary_includes_timeout_count(self) -> None:
        br = BatchResult(
            results=[
                ItemResult(item_id="1", success=False, output="", timed_out=True),
                ItemResult(item_id="2", success=True, output="ok"),
            ],
            total=2,
            succeeded=1,
            failed=1,
            timed_out=1,
        )
        assert "timed out" in br.summary()

    def test_summary_no_timeout_mention_when_zero(self) -> None:
        br = BatchResult(
            results=[ItemResult(item_id="1", success=True, output="ok")],
            total=1,
            succeeded=1,
            failed=0,
            timed_out=0,
        )
        assert "timed out" not in br.summary()

    def test_jsonl_includes_timed_out_field(self) -> None:
        import json

        br = BatchResult(
            results=[ItemResult(item_id="1", success=False, output="", timed_out=True)],
            total=1,
            succeeded=0,
            failed=1,
        )
        text = results_to_jsonl(br)
        parsed = json.loads(text.strip())
        assert parsed["timed_out"] is True

    def test_jsonl_omits_timed_out_when_false(self) -> None:
        import json

        br = BatchResult(
            results=[ItemResult(item_id="1", success=True, output="ok")],
            total=1,
            succeeded=1,
            failed=0,
        )
        text = results_to_jsonl(br)
        parsed = json.loads(text.strip())
        assert "timed_out" not in parsed


# ---------------------------------------------------------------------------
# worker subcommand stubs
# ---------------------------------------------------------------------------


class TestWorkerSubcommands:
    def test_worker_status_help(self) -> None:
        from typer.testing import CliRunner

        from exo_cli.main import app

        r = CliRunner()
        result = r.invoke(app, ["worker", "status", "--help"])
        assert result.exit_code == 0
        assert "worker-id" in result.output.lower() or "WORKER_ID" in result.output

    def test_worker_stop_help(self) -> None:
        from typer.testing import CliRunner

        from exo_cli.main import app

        r = CliRunner()
        result = r.invoke(app, ["worker", "stop", "--help"])
        assert result.exit_code == 0
        assert "--redis-url" in result.output

    def test_worker_help_shows_all_subcommands(self) -> None:
        from typer.testing import CliRunner

        from exo_cli.main import app

        r = CliRunner()
        result = r.invoke(app, ["worker", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "status" in result.output
        assert "stop" in result.output
