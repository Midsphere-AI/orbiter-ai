"""Batch execution for Exo CLI.

Loads inputs from JSON, CSV, or JSONL files and runs an agent against
each input concurrently with configurable parallelism.  Results are
collected into :class:`BatchResult` for output or further processing.

Usage::

    loader = BatchLoader()
    items = loader.load("inputs.jsonl")
    result = await batch_execute(agent, items, concurrency=4)
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from exo.types import ExoError  # pyright: ignore[reportMissingImports]
from exo_cli.executor import (  # pyright: ignore[reportMissingImports]
    ExecutionResult,
    ExecutorError,
    LocalExecutor,
)

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class BatchError(ExoError):
    """Raised for batch-level errors."""


# ---------------------------------------------------------------------------
# Input format
# ---------------------------------------------------------------------------


class InputFormat(StrEnum):
    """Supported batch input formats."""

    JSON = "json"
    CSV = "csv"
    JSONL = "jsonl"


# ---------------------------------------------------------------------------
# Batch item
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BatchItem:
    """A single batch input item.

    Parameters:
        id: Unique item identifier (row number or explicit id).
        input: The text prompt sent to the agent.
        metadata: Extra columns / fields carried through from the source.
    """

    id: str
    input: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Item result
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ItemResult:
    """Result for one batch item.

    Parameters:
        item_id: Batch item identifier.
        success: Whether execution succeeded.
        output: Agent output text (or error message).
        elapsed: Execution time in seconds.
        error: Error message when ``success`` is ``False``.
        timed_out: ``True`` when the item failed specifically due to a timeout.
    """

    item_id: str
    success: bool
    output: str
    elapsed: float = 0.0
    error: str = ""
    timed_out: bool = False


# ---------------------------------------------------------------------------
# Batch result (aggregate)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BatchResult:
    """Aggregate result for a batch run.

    Parameters:
        results: Per-item results.
        total: Total items processed.
        succeeded: Count of successful items.
        failed: Count of failed items.
        timed_out: Count of items that failed due to timeout.
    """

    results: list[ItemResult] = field(default_factory=list)
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    timed_out: int = 0

    def summary(self) -> str:
        """Human-readable summary."""
        s = f"{self.total} items: {self.succeeded} succeeded, {self.failed} failed"
        if self.timed_out:
            s += f" ({self.timed_out} timed out)"
        return s


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _detect_format(path: Path) -> InputFormat:
    """Detect input format from file extension."""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return InputFormat.JSONL
    if suffix == ".csv":
        return InputFormat.CSV
    if suffix == ".json":
        return InputFormat.JSON
    raise BatchError(f"Unsupported file extension: {suffix}")


def _load_json(text: str) -> list[dict[str, Any]]:
    data = json.loads(text)
    if isinstance(data, list):
        return data
    raise BatchError("JSON input must be a list of objects")


def _load_jsonl(text: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for i, line in enumerate(text.strip().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BatchError(f"Invalid JSON on line {i}: {exc}") from exc
        if not isinstance(obj, dict):
            raise BatchError(f"Line {i} is not a JSON object")
        items.append(obj)
    return items


def _load_csv(text: str) -> list[dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(text))
    return list(reader)


def load_batch_items(
    path: str | Path,
    *,
    input_key: str = "input",
    id_key: str = "id",
    fmt: InputFormat | None = None,
) -> list[BatchItem]:
    """Load batch items from a file.

    Parameters:
        path: Path to the input file.
        input_key: Column/field name containing the agent input text.
        id_key: Column/field name for item IDs (falls back to row number).
        fmt: Force a specific format (auto-detected from extension if ``None``).

    Returns:
        List of :class:`BatchItem` instances.

    Raises:
        BatchError: On missing file, invalid format, or missing input key.
    """
    p = Path(path)
    if not p.exists():
        raise BatchError(f"File not found: {p}")

    detected = fmt or _detect_format(p)
    text = p.read_text(encoding="utf-8")

    loaders = {
        InputFormat.JSON: _load_json,
        InputFormat.JSONL: _load_jsonl,
        InputFormat.CSV: _load_csv,
    }
    rows = loaders[detected](text)

    items: list[BatchItem] = []
    for idx, row in enumerate(rows, 1):
        if input_key not in row:
            raise BatchError(f"Row {idx} missing required key '{input_key}'")
        item_id = str(row.get(id_key, idx))
        meta = {k: v for k, v in row.items() if k not in (input_key, id_key)}
        items.append(BatchItem(id=item_id, input=str(row[input_key]), metadata=meta))
    return items


# ---------------------------------------------------------------------------
# Batch execution
# ---------------------------------------------------------------------------


async def batch_execute(
    agent: Any,
    items: Sequence[BatchItem],
    *,
    provider: Any = None,
    concurrency: int = 4,
    timeout: float = 0.0,
) -> BatchResult:
    """Execute an agent against multiple inputs concurrently.

    Parameters:
        agent: Agent or Swarm instance.
        items: Batch items to process.
        provider: LLM provider (auto-resolved when ``None``).
        concurrency: Maximum concurrent executions.
        timeout: Per-item timeout in seconds (0 = no timeout).

    Returns:
        :class:`BatchResult` with per-item results.
    """
    import asyncio

    if concurrency < 1:
        raise BatchError(
            "concurrency must be >= 1",
            hint="Pass --concurrency with a value of 1 or greater.",
        )

    sem = asyncio.Semaphore(concurrency)
    executor = LocalExecutor(agent=agent, provider=provider, timeout=timeout)

    async def _run_one(item: BatchItem) -> ItemResult:
        async with sem:
            try:
                result: ExecutionResult = await executor.execute(item.input)
                return ItemResult(
                    item_id=item.id,
                    success=True,
                    output=result.output,
                    elapsed=result.elapsed,
                )
            except ExecutorError as exc:
                # Distinguish timeout failures from other LLM/execution errors.
                is_timeout = "timed out" in str(exc).lower()
                return ItemResult(
                    item_id=item.id,
                    success=False,
                    output="",
                    error=str(exc),
                    timed_out=is_timeout,
                )
            except Exception as exc:
                return ItemResult(
                    item_id=item.id,
                    success=False,
                    output="",
                    error=str(exc),
                )

    tasks = [_run_one(item) for item in items]
    results = await asyncio.gather(*tasks)

    succeeded = sum(1 for r in results if r.success)
    timed_out = sum(1 for r in results if r.timed_out)
    return BatchResult(
        results=list(results),
        total=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
        timed_out=timed_out,
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def results_to_jsonl(batch: BatchResult) -> str:
    """Serialize batch results to JSONL string."""
    lines: list[str] = []
    for r in batch.results:
        obj: dict[str, Any] = {
            "id": r.item_id,
            "success": r.success,
            "output": r.output,
            "elapsed": r.elapsed,
        }
        if r.error:
            obj["error"] = r.error
        if r.timed_out:
            obj["timed_out"] = True
        lines.append(json.dumps(obj))
    return "\n".join(lines) + "\n" if lines else ""


def results_to_csv(batch: BatchResult) -> str:
    """Serialize batch results to CSV string."""
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf, fieldnames=["id", "success", "output", "elapsed", "error", "timed_out"]
    )
    writer.writeheader()
    for r in batch.results:
        writer.writerow(
            {
                "id": r.item_id,
                "success": r.success,
                "output": r.output,
                "elapsed": r.elapsed,
                "error": r.error,
                "timed_out": r.timed_out,
            }
        )
    return buf.getvalue()
