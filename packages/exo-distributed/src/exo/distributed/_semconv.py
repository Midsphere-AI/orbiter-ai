"""Distributed-only semantic conventions.

Attribute names and metric identifiers used exclusively by exo-distributed
spans, metrics, and log records.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Distributed task attribute conventions — exo.distributed.*
# ---------------------------------------------------------------------------

DIST_TASK_ID = "exo.distributed.task_id"
DIST_WORKER_ID = "exo.distributed.worker_id"
DIST_QUEUE_NAME = "exo.distributed.queue_name"
DIST_TASK_STATUS = "exo.distributed.task_status"

# ---------------------------------------------------------------------------
# Distributed task metric names
# ---------------------------------------------------------------------------

METRIC_DIST_TASKS_SUBMITTED = "dist_tasks_submitted"
METRIC_DIST_TASKS_COMPLETED = "dist_tasks_completed"
METRIC_DIST_TASKS_FAILED = "dist_tasks_failed"
METRIC_DIST_TASKS_CANCELLED = "dist_tasks_cancelled"
METRIC_DIST_QUEUE_DEPTH = "dist_queue_depth"
METRIC_DIST_TASK_DURATION = "dist_task_duration"
METRIC_DIST_TASK_WAIT_TIME = "dist_task_wait_time"

# ---------------------------------------------------------------------------
# Streaming event metric names and attributes (distributed-only)
# ---------------------------------------------------------------------------

METRIC_STREAM_EVENT_PUBLISH_DURATION = "stream_event_publish_duration"
