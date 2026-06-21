"""Pre-defined alert rules for distributed system health monitoring.

Registers threshold-based alerts for queue depth, task failure rate,
worker availability, and task wait time using the global AlertManager.
"""

from __future__ import annotations

from exo.distributed._alerts import (  # pyright: ignore[reportMissingImports]
    AlertRule,
    AlertSeverity,
    Comparator,
    get_manager,
)
from exo.distributed._semconv import (  # pyright: ignore[reportMissingImports]
    METRIC_DIST_QUEUE_DEPTH,
    METRIC_DIST_TASK_WAIT_TIME,
    METRIC_DIST_TASKS_FAILED,
)

# Rule name constants for external reference and testing.
RULE_QUEUE_DEPTH_WARNING = "dist_queue_depth_warning"
RULE_QUEUE_DEPTH_CRITICAL = "dist_queue_depth_critical"
RULE_FAILURE_RATE_WARNING = "dist_failure_rate_warning"
RULE_WORKER_COUNT_CRITICAL = "dist_worker_count_critical"
RULE_WAIT_TIME_WARNING = "dist_wait_time_warning"

# Metric name for worker count (evaluated externally, not a recorded metric).
METRIC_WORKER_COUNT = "dist_worker_count"


def register_distributed_alerts() -> list[AlertRule]:
    """Register pre-defined alert rules for distributed system health.

    Rules:
    - Queue depth > 100: WARNING
    - Queue depth > 500: CRITICAL
    - Task failure rate > 10%: WARNING
    - Worker count = 0: CRITICAL
    - Task wait time > 60s: WARNING

    Returns the list of registered rules.
    """
    rules = [
        AlertRule(
            name=RULE_QUEUE_DEPTH_WARNING,
            metric_name=METRIC_DIST_QUEUE_DEPTH,
            threshold=100.0,
            comparator=Comparator.GT,
            severity=AlertSeverity.WARNING,
        ),
        AlertRule(
            name=RULE_QUEUE_DEPTH_CRITICAL,
            metric_name=METRIC_DIST_QUEUE_DEPTH,
            threshold=500.0,
            comparator=Comparator.GT,
            severity=AlertSeverity.CRITICAL,
        ),
        AlertRule(
            name=RULE_FAILURE_RATE_WARNING,
            metric_name=METRIC_DIST_TASKS_FAILED,
            threshold=0.1,
            comparator=Comparator.GT,
            severity=AlertSeverity.WARNING,
        ),
        AlertRule(
            name=RULE_WORKER_COUNT_CRITICAL,
            metric_name=METRIC_WORKER_COUNT,
            threshold=0.0,
            comparator=Comparator.EQ,
            severity=AlertSeverity.CRITICAL,
        ),
        AlertRule(
            name=RULE_WAIT_TIME_WARNING,
            metric_name=METRIC_DIST_TASK_WAIT_TIME,
            threshold=60.0,
            comparator=Comparator.GT,
            severity=AlertSeverity.WARNING,
        ),
    ]

    manager = get_manager()
    for rule in rules:
        manager.register_rule(rule)

    return rules
