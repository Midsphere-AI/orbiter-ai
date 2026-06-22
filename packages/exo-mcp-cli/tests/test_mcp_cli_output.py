"""Tests for exo_mcp_cli.output — _render_exc and formatting helpers."""

from __future__ import annotations

from exo_mcp_cli.output import _render_exc


class TestRenderExc:
    def test_plain_exception_passes_through(self) -> None:
        exc = ValueError("something went wrong")
        assert _render_exc(exc) == "something went wrong"

    def test_single_child_exception_group_unwrapped(self) -> None:
        inner = RuntimeError("the real cause")
        group = ExceptionGroup("wrapper", [inner])
        assert _render_exc(group) == "the real cause"

    def test_nested_single_child_groups_fully_unwrapped(self) -> None:
        inner = OSError("disk full")
        group1 = ExceptionGroup("inner-group", [inner])
        group2 = ExceptionGroup("outer-group", [group1])
        assert _render_exc(group2) == "disk full"

    def test_multi_child_group_not_unwrapped(self) -> None:
        group = ExceptionGroup("multi", [ValueError("a"), ValueError("b")])
        result = _render_exc(group)
        # Should return the group's own str representation, not one of the children
        assert "a" in result or "multi" in result
        # Must not be just "a" or just "b" — the group string is longer
        assert result != "a"
        assert result != "b"

    def test_base_exception_group_unwrapped(self) -> None:
        inner = KeyboardInterrupt()
        group = BaseExceptionGroup("base-wrapper", [inner])
        result = _render_exc(group)
        assert isinstance(result, str)
        # The unwrapped exception is a KeyboardInterrupt, str() of which is ""
        assert result == str(KeyboardInterrupt())
