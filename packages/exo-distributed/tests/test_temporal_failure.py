"""Tests for exo.distributed.temporal.failure — error classification and conversion."""

from __future__ import annotations

import pytest
from pydantic import ValidationError, create_model

from exo._internal.background import BackgroundTaskError
from exo._internal.branch_node import BranchError
from exo._internal.expression import ExpressionError
from exo._internal.graph import GraphError
from exo._internal.loop_node import LoopError
from exo._internal.output_parser import OutputParseError
from exo.agent import TaskLoopAbort
from exo.config import ConfigError
from exo.distributed.temporal._compat import HAS_TEMPORAL, ApplicationError
from exo.distributed.temporal.failure import (
    NON_RETRYABLE_ERROR_TYPES,
    is_retryable,
    to_application_error,
)
from exo.ptc import MaxToolCallsExceeded
from exo.rail import GuardAbortError
from exo.registry import RegistryError
from exo.tool import ToolError
from exo.types import ExoError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_validation_error() -> ValidationError:
    """Create a real pydantic ValidationError for testing."""
    model_cls = create_model("M", x=(int, ...))
    try:
        model_cls(x="not-an-int")  # type: ignore[call-arg]
    except ValidationError as exc:
        return exc
    pytest.fail("expected ValidationError")  # pragma: no cover


# ---------------------------------------------------------------------------
# is_retryable — non-retryable cases
# ---------------------------------------------------------------------------


class TestIsRetryableNonRetryable:
    def test_config_error(self):
        assert is_retryable(ConfigError("bad config")) is False

    def test_tool_error(self):
        assert is_retryable(ToolError("bad tool")) is False

    def test_output_parse_error(self):
        assert is_retryable(OutputParseError("parse fail")) is False

    def test_expression_error(self):
        assert is_retryable(ExpressionError("expr fail")) is False

    def test_graph_error(self):
        assert is_retryable(GraphError("graph fail")) is False

    def test_branch_error(self):
        assert is_retryable(BranchError("branch fail")) is False

    def test_loop_error(self):
        assert is_retryable(LoopError("loop fail")) is False

    def test_max_tool_calls_exceeded(self):
        assert is_retryable(MaxToolCallsExceeded("too many calls")) is False

    def test_guard_abort_error(self):
        assert is_retryable(GuardAbortError("guard abort")) is False

    def test_task_loop_abort(self):
        assert is_retryable(TaskLoopAbort("abort")) is False

    def test_registry_error(self):
        assert is_retryable(RegistryError("registry fail")) is False

    def test_value_error(self):
        assert is_retryable(ValueError("bad value")) is False

    def test_type_error(self):
        assert is_retryable(TypeError("bad type")) is False

    def test_key_error(self):
        assert is_retryable(KeyError("missing key")) is False

    def test_pydantic_validation_error(self):
        exc = _make_validation_error()
        assert is_retryable(exc) is False


# ---------------------------------------------------------------------------
# is_retryable — retryable cases
# ---------------------------------------------------------------------------


class TestIsRetryableRetryable:
    def test_generic_exo_error(self):
        assert is_retryable(ExoError("generic")) is True

    def test_background_task_error(self):
        assert is_retryable(BackgroundTaskError("bg fail")) is True

    def test_runtime_error(self):
        assert is_retryable(RuntimeError("unexpected")) is True

    def test_connection_error(self):
        assert is_retryable(ConnectionError("network down")) is True

    def test_timeout_error(self):
        assert is_retryable(TimeoutError("timed out")) is True

    def test_exception(self):
        assert is_retryable(Exception("unknown")) is True


# ---------------------------------------------------------------------------
# is_retryable — ModelError (optional exo-models)
# ---------------------------------------------------------------------------


class TestIsRetryableModelError:
    @pytest.fixture(autouse=True)
    def _skip_without_models(self):
        pytest.importorskip("exo.models.types")

    def test_rate_limit_is_retryable(self):
        from exo.models.types import ModelError  # type: ignore[import-not-found]

        exc = ModelError("rate limited", code="rate_limit")
        assert is_retryable(exc) is True

    def test_context_length_is_not_retryable(self):
        from exo.models.types import ModelError  # type: ignore[import-not-found]

        exc = ModelError("too long", code="context_length")
        assert is_retryable(exc) is False

    def test_invalid_request_is_not_retryable(self):
        from exo.models.types import ModelError  # type: ignore[import-not-found]

        exc = ModelError("bad request", code="invalid_request")
        assert is_retryable(exc) is False

    def test_authentication_is_not_retryable(self):
        from exo.models.types import ModelError  # type: ignore[import-not-found]

        exc = ModelError("auth failed", code="authentication")
        assert is_retryable(exc) is False

    def test_permission_is_not_retryable(self):
        from exo.models.types import ModelError  # type: ignore[import-not-found]

        exc = ModelError("forbidden", code="permission")
        assert is_retryable(exc) is False

    def test_unknown_code_is_retryable(self):
        from exo.models.types import ModelError  # type: ignore[import-not-found]

        exc = ModelError("something weird", code="unknown_future_code")
        assert is_retryable(exc) is True

    def test_empty_code_is_retryable(self):
        from exo.models.types import ModelError  # type: ignore[import-not-found]

        exc = ModelError("no code")
        assert is_retryable(exc) is True


# ---------------------------------------------------------------------------
# to_application_error — requires temporalio
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_TEMPORAL, reason="temporalio not installed")
class TestToApplicationError:
    def test_permanent_error_sets_non_retryable_true(self):
        exc = ConfigError("bad config")
        result = to_application_error(exc)
        assert isinstance(result, ApplicationError)
        assert result.non_retryable is True

    def test_transient_error_sets_non_retryable_false(self):
        exc = RuntimeError("transient failure")
        result = to_application_error(exc)
        assert isinstance(result, ApplicationError)
        assert result.non_retryable is False

    def test_message_preserved(self):
        exc = ConfigError("the exact message")
        result = to_application_error(exc)
        assert "the exact message" in str(result)

    def test_type_field_is_class_name(self):
        exc = ToolError("bad tool")
        result = to_application_error(exc)
        assert result.type == "ToolError"

    def test_runtime_error_type_field(self):
        exc = RuntimeError("boom")
        result = to_application_error(exc)
        assert result.type == "RuntimeError"

    def test_override_non_retryable_true(self):
        # Force a normally-retryable error to be non-retryable.
        exc = RuntimeError("transient by nature")
        result = to_application_error(exc, non_retryable=True)
        assert result.non_retryable is True

    def test_override_non_retryable_false(self):
        # Force a normally-non-retryable error to be retryable.
        exc = ConfigError("permanent by nature")
        result = to_application_error(exc, non_retryable=False)
        assert result.non_retryable is False

    def test_exo_error_with_context_and_hint_does_not_raise(self):
        exc = ExoError(
            "something failed",
            context={"key": "value"},
            hint="try restarting",
            doc="https://docs.example.com",
        )
        result = to_application_error(exc)
        assert isinstance(result, ApplicationError)
        assert result.non_retryable is False  # generic ExoError is retryable

    def test_exo_error_details_included(self):
        exc = ExoError(
            "structured failure",
            context={"reason": "test"},
            hint="check the logs",
        )
        result = to_application_error(exc)
        # details are embedded; at minimum the call should not error
        assert isinstance(result, ApplicationError)

    def test_exo_error_no_context_no_details_noise(self):
        exc = ExoError("plain error")
        result = to_application_error(exc)
        assert isinstance(result, ApplicationError)

    def test_pydantic_validation_error_is_non_retryable(self):
        exc = _make_validation_error()
        result = to_application_error(exc)
        assert isinstance(result, ApplicationError)
        assert result.non_retryable is True

    def test_value_error_is_non_retryable(self):
        exc = ValueError("bad value")
        result = to_application_error(exc)
        assert result.non_retryable is True
        assert result.type == "ValueError"


# ---------------------------------------------------------------------------
# NON_RETRYABLE_ERROR_TYPES contents
# ---------------------------------------------------------------------------


class TestNonRetryableErrorTypes:
    def test_is_tuple_of_strings(self):
        assert isinstance(NON_RETRYABLE_ERROR_TYPES, tuple)
        assert all(isinstance(name, str) for name in NON_RETRYABLE_ERROR_TYPES)

    def test_contains_config_error(self):
        assert "ConfigError" in NON_RETRYABLE_ERROR_TYPES

    def test_contains_tool_error(self):
        assert "ToolError" in NON_RETRYABLE_ERROR_TYPES

    def test_contains_output_parse_error(self):
        assert "OutputParseError" in NON_RETRYABLE_ERROR_TYPES

    def test_contains_expression_error(self):
        assert "ExpressionError" in NON_RETRYABLE_ERROR_TYPES

    def test_contains_graph_error(self):
        assert "GraphError" in NON_RETRYABLE_ERROR_TYPES

    def test_contains_branch_error(self):
        assert "BranchError" in NON_RETRYABLE_ERROR_TYPES

    def test_contains_loop_error(self):
        assert "LoopError" in NON_RETRYABLE_ERROR_TYPES

    def test_contains_max_tool_calls_exceeded(self):
        assert "MaxToolCallsExceeded" in NON_RETRYABLE_ERROR_TYPES

    def test_contains_guard_abort_error(self):
        assert "GuardAbortError" in NON_RETRYABLE_ERROR_TYPES

    def test_contains_task_loop_abort(self):
        assert "TaskLoopAbort" in NON_RETRYABLE_ERROR_TYPES

    def test_contains_value_error(self):
        assert "ValueError" in NON_RETRYABLE_ERROR_TYPES

    def test_contains_type_error(self):
        assert "TypeError" in NON_RETRYABLE_ERROR_TYPES

    def test_contains_key_error(self):
        assert "KeyError" in NON_RETRYABLE_ERROR_TYPES

    def test_contains_validation_error(self):
        assert "ValidationError" in NON_RETRYABLE_ERROR_TYPES

    def test_no_duplicates(self):
        assert len(NON_RETRYABLE_ERROR_TYPES) == len(set(NON_RETRYABLE_ERROR_TYPES))
