"""Tests for exo_cli.tool_commands — tool offloading CLI."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pytest
import typer
from typer.testing import CliRunner

from exo_cli.main import app
from exo_cli.tool_commands import (
    _build_arguments,
    _coerce_value,
    _discover,
    _format_result,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixture: write a temporary tools module
# ---------------------------------------------------------------------------


@pytest.fixture()
def tools_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample_tools.py"
    p.write_text(
        dedent("""\
        from exo import tool

        @tool
        def greet(name: str, greeting: str = "Hello") -> str:
            \"\"\"Greet someone.

            Args:
                name: Person name.
                greeting: Greeting word.
            \"\"\"
            return f"{greeting}, {name}!"

        @tool
        def add(a: int, b: int) -> str:
            \"\"\"Add two numbers.

            Args:
                a: First number.
                b: Second number.
            \"\"\"
            return str(a + b)

        @tool
        def echo_bool(flag: bool) -> str:
            \"\"\"Echo a boolean.

            Args:
                flag: The flag value.
            \"\"\"
            return str(flag)
        """)
    )
    return p


@pytest.fixture()
def tools_file_with_list(tmp_path: Path) -> Path:
    """Module that exposes tools via a ``tools`` list."""
    p = tmp_path / "list_tools.py"
    p.write_text(
        dedent("""\
        from exo import tool

        _hello = tool(lambda name="World": f"Hi {name}", name="hello", description="Say hi")

        tools = [_hello]
        """)
    )
    return p


# ---------------------------------------------------------------------------
# _coerce_value
# ---------------------------------------------------------------------------


class TestCoerceValue:
    def test_string(self) -> None:
        assert _coerce_value("hello", {"type": "string"}) == "hello"

    def test_integer(self) -> None:
        assert _coerce_value("42", {"type": "integer"}) == 42

    def test_integer_bad(self) -> None:
        assert _coerce_value("not_int", {"type": "integer"}) == "not_int"

    def test_number(self) -> None:
        assert _coerce_value("3.14", {"type": "number"}) == pytest.approx(3.14)

    def test_boolean_true(self) -> None:
        assert _coerce_value("true", {"type": "boolean"}) is True

    def test_boolean_false(self) -> None:
        assert _coerce_value("false", {"type": "boolean"}) is False

    def test_array(self) -> None:
        assert _coerce_value("[1,2,3]", {"type": "array"}) == [1, 2, 3]

    def test_object(self) -> None:
        assert _coerce_value('{"a":1}', {"type": "object"}) == {"a": 1}

    def test_unknown_type(self) -> None:
        assert _coerce_value("x", {}) == "x"


# ---------------------------------------------------------------------------
# _collect_tools_from_module
# ---------------------------------------------------------------------------


class TestCollectTools:
    def test_discovers_tool_instances(self, tools_file: Path) -> None:
        tools = _discover(str(tools_file))
        assert "greet" in tools
        assert "add" in tools
        assert len(tools) >= 2

    def test_discovers_from_tools_list(self, tools_file_with_list: Path) -> None:
        tools = _discover(str(tools_file_with_list))
        assert "hello" in tools

    def test_bad_source_raises(self) -> None:
        with pytest.raises(typer.BadParameter):
            _discover("nonexistent_module_xyz_12345")


# ---------------------------------------------------------------------------
# _build_arguments
# ---------------------------------------------------------------------------


class TestBuildArguments:
    def test_json_args(self, tools_file: Path) -> None:
        t = _discover(str(tools_file))["greet"]
        result = _build_arguments(t, None, '{"name": "Alice"}')
        assert result == {"name": "Alice"}

    def test_key_value_args(self, tools_file: Path) -> None:
        t = _discover(str(tools_file))["add"]
        result = _build_arguments(t, ["a=10", "b=20"], None)
        assert result == {"a": 10, "b": 20}

    def test_combined(self, tools_file: Path) -> None:
        t = _discover(str(tools_file))["greet"]
        result = _build_arguments(t, ["greeting=Hey"], '{"name": "Bob"}')
        assert result == {"name": "Bob", "greeting": "Hey"}

    def test_bad_json(self, tools_file: Path) -> None:
        t = _discover(str(tools_file))["greet"]
        with pytest.raises(typer.BadParameter):
            _build_arguments(t, None, "not json")

    def test_bad_kv_format(self, tools_file: Path) -> None:
        t = _discover(str(tools_file))["greet"]
        with pytest.raises(typer.BadParameter):
            _build_arguments(t, ["no_equals"], None)

    def test_inject_basic(self, tools_file: Path) -> None:
        t = _discover(str(tools_file))["greet"]
        result = _build_arguments(t, None, None, inject=["name=Injected"])
        assert result == {"name": "Injected"}

    def test_inject_overridden_by_arg(self, tools_file: Path) -> None:
        t = _discover(str(tools_file))["greet"]
        result = _build_arguments(t, ["name=Explicit"], None, inject=["name=Injected"])
        assert result == {"name": "Explicit"}

    def test_inject_overridden_by_json(self, tools_file: Path) -> None:
        t = _discover(str(tools_file))["greet"]
        result = _build_arguments(t, None, '{"name": "FromJSON"}', inject=["name=Injected"])
        assert result == {"name": "FromJSON"}

    def test_inject_with_type_coercion(self, tools_file: Path) -> None:
        t = _discover(str(tools_file))["add"]
        result = _build_arguments(t, ["b=20"], None, inject=["a=10"])
        assert result == {"a": 10, "b": 20}

    def test_inject_bad_format(self, tools_file: Path) -> None:
        t = _discover(str(tools_file))["greet"]
        with pytest.raises(typer.BadParameter):
            _build_arguments(t, None, None, inject=["no_equals"])


# ---------------------------------------------------------------------------
# _format_result
# ---------------------------------------------------------------------------


class TestFormatResult:
    def test_string(self) -> None:
        assert _format_result("hello") == "hello"

    def test_dict(self) -> None:
        result = _format_result({"key": "val"})
        assert '"key"' in result

    def test_list_of_strings(self) -> None:
        assert _format_result(["a", "b"]) == "a\nb"


# ---------------------------------------------------------------------------
# CLI integration tests via CliRunner
# ---------------------------------------------------------------------------


class TestToolListCLI:
    def test_list_table(self, tools_file: Path) -> None:
        result = runner.invoke(app, ["tool", "list", "--from", str(tools_file)])
        assert result.exit_code == 0
        assert "greet" in result.output
        assert "add" in result.output

    def test_list_json(self, tools_file: Path) -> None:
        result = runner.invoke(app, ["tool", "list", "--from", str(tools_file), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        names = {item["name"] for item in data}
        assert "greet" in names
        assert "add" in names

    def test_list_bad_source(self) -> None:
        result = runner.invoke(app, ["tool", "list", "--from", "nonexistent_xyz"])
        assert result.exit_code != 0


class TestToolCallCLI:
    def test_call_with_kv(self, tools_file: Path) -> None:
        result = runner.invoke(
            app, ["tool", "call", "greet", "--from", str(tools_file), "-a", "name=World"]
        )
        assert result.exit_code == 0
        assert "Hello, World!" in result.output

    def test_call_with_json(self, tools_file: Path) -> None:
        result = runner.invoke(
            app,
            [
                "tool",
                "call",
                "greet",
                "--from",
                str(tools_file),
                "-j",
                '{"name": "Alice", "greeting": "Hey"}',
            ],
        )
        assert result.exit_code == 0
        assert "Hey, Alice!" in result.output

    def test_call_integer_coercion(self, tools_file: Path) -> None:
        result = runner.invoke(
            app,
            ["tool", "call", "add", "--from", str(tools_file), "-a", "a=17", "-a", "b=25"],
        )
        assert result.exit_code == 0
        assert "42" in result.output

    def test_call_raw(self, tools_file: Path) -> None:
        result = runner.invoke(
            app,
            [
                "tool",
                "call",
                "greet",
                "--from",
                str(tools_file),
                "-a",
                "name=X",
                "--raw",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["result"] == "Hello, X!"

    def test_call_missing_tool(self, tools_file: Path) -> None:
        result = runner.invoke(app, ["tool", "call", "nope", "--from", str(tools_file)])
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "nope" in result.output

    def test_call_with_inject(self, tools_file: Path) -> None:
        result = runner.invoke(
            app,
            [
                "tool",
                "call",
                "greet",
                "--from",
                str(tools_file),
                "-i",
                "name=Injected",
                "-a",
                "greeting=Hey",
            ],
        )
        assert result.exit_code == 0
        assert "Hey, Injected!" in result.output

    def test_call_inject_overridden(self, tools_file: Path) -> None:
        result = runner.invoke(
            app,
            [
                "tool",
                "call",
                "greet",
                "--from",
                str(tools_file),
                "-i",
                "name=Injected",
                "-a",
                "name=Explicit",
            ],
        )
        assert result.exit_code == 0
        assert "Hello, Explicit!" in result.output

    def test_call_bool_coercion(self, tools_file: Path) -> None:
        result = runner.invoke(
            app,
            ["tool", "call", "echo_bool", "--from", str(tools_file), "-a", "flag=true"],
        )
        assert result.exit_code == 0
        assert "True" in result.output


class TestToolSourceEnvVar:
    def test_list_via_env(self, tools_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXO_TOOL_SOURCE", str(tools_file))
        result = runner.invoke(app, ["tool", "list"])
        assert result.exit_code == 0
        assert "greet" in result.output

    def test_call_via_env(self, tools_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXO_TOOL_SOURCE", str(tools_file))
        result = runner.invoke(app, ["tool", "call", "greet", "-a", "name=EnvTest"])
        assert result.exit_code == 0
        assert "Hello, EnvTest!" in result.output

    def test_schema_via_env(self, tools_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXO_TOOL_SOURCE", str(tools_file))
        result = runner.invoke(app, ["tool", "schema", "greet"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["function"]["name"] == "greet"

    def test_flag_overrides_env(self, tools_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXO_TOOL_SOURCE", "nonexistent_module_xyz")
        result = runner.invoke(app, ["tool", "list", "--from", str(tools_file)])
        assert result.exit_code == 0
        assert "greet" in result.output

    def test_error_when_no_source(self) -> None:
        result = runner.invoke(app, ["tool", "list"])
        assert result.exit_code == 1
        assert "EXO_TOOL_SOURCE" in result.output


class TestToolInjectEnvVar:
    def test_inject_from_env(self, tools_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXO_TOOL_INJECT", '{"name": "EnvInjected"}')
        result = runner.invoke(app, ["tool", "call", "greet", "--from", str(tools_file)])
        assert result.exit_code == 0
        assert "Hello, EnvInjected!" in result.output

    def test_flag_inject_overrides_env_inject(
        self, tools_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXO_TOOL_INJECT", '{"name": "EnvVal"}')
        result = runner.invoke(
            app, ["tool", "call", "greet", "--from", str(tools_file), "-i", "name=FlagVal"]
        )
        assert result.exit_code == 0
        assert "Hello, FlagVal!" in result.output

    def test_arg_overrides_env_inject(
        self, tools_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXO_TOOL_INJECT", '{"name": "EnvVal"}')
        result = runner.invoke(
            app, ["tool", "call", "greet", "--from", str(tools_file), "-a", "name=Explicit"]
        )
        assert result.exit_code == 0
        assert "Hello, Explicit!" in result.output

    def test_json_overrides_env_inject(
        self, tools_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXO_TOOL_INJECT", '{"name": "EnvVal"}')
        result = runner.invoke(
            app,
            ["tool", "call", "greet", "--from", str(tools_file), "-j", '{"name": "JsonVal"}'],
        )
        assert result.exit_code == 0
        assert "Hello, JsonVal!" in result.output

    def test_malformed_env_inject_ignored(
        self, tools_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXO_TOOL_INJECT", "not json")
        result = runner.invoke(
            app, ["tool", "call", "greet", "--from", str(tools_file), "-a", "name=Works"]
        )
        assert result.exit_code == 0
        assert "Hello, Works!" in result.output


class TestToolSchemaCLI:
    def test_schema_output(self, tools_file: Path) -> None:
        result = runner.invoke(app, ["tool", "schema", "greet", "--from", str(tools_file)])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["function"]["name"] == "greet"
        assert "name" in data["function"]["parameters"]["properties"]

    def test_schema_missing_tool(self, tools_file: Path) -> None:
        result = runner.invoke(app, ["tool", "schema", "nope", "--from", str(tools_file)])
        assert result.exit_code != 0
