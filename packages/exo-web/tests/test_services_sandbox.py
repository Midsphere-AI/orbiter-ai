"""Tests for exo_web.services.sandbox — focused on the B-1 escaping fix and execution
semantics that are NOT already covered by the existing test_sandbox.py file.

Specifically:
- repr()-based code embedding handles backslashes, multi-line strings, and full Unicode
- Execution success/failure round-trips (stdout capture, error message extraction)
- _collect_files base64 encoding
- SandboxConfig defaults and timeout plumbing
"""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path

from exo_web.services.sandbox import (
    SandboxConfig,
    SandboxResult,
    _build_runner_script,
    _collect_files,
    execute_code,
)

# ---------------------------------------------------------------------------
# _build_runner_script — escaping correctness (B-1 audit finding)
# ---------------------------------------------------------------------------


class TestBuildRunnerScriptEscaping:
    """Verify that repr() embedding means ALL dangerous chars are safe."""

    def test_backslash_sequences_are_safe(self) -> None:
        """Backslashes in code do not break the generated script."""
        code = r'path = "C:\Users\admin\Documents"' + "\nprint(path)"
        script = _build_runner_script(code, "/tmp", [])
        # repr() of the code must appear in the script
        assert repr(code) in script

    def test_carriage_return_embedded_safely(self) -> None:
        """\\r in source code is escaped by repr() so the script is valid Python."""
        code = "print('hello\\r')"
        script = _build_runner_script(code, "/tmp", [])
        # The generated script must be syntactically valid (no unescaped CR inside string)
        assert repr(code) in script
        # Compile to verify the generated script is syntactically valid Python
        compile(script, "<test>", "exec")

    def test_null_bytes_embedded_safely(self) -> None:
        """Null bytes in code are escaped by repr()."""
        code = "x = b'\\x00\\x01\\x02'\nprint(len(x))"
        script = _build_runner_script(code, "/tmp", [])
        assert repr(code) in script
        compile(script, "<test>", "exec")

    def test_unicode_code_points_embedded_safely(self) -> None:
        """Non-ASCII Unicode in code is embedded safely via repr()."""
        code = "print('héllo wörld 日本語')"
        script = _build_runner_script(code, "/tmp", [])
        assert repr(code) in script
        compile(script, "<test>", "exec")

    def test_mixed_quotes_embedded_safely(self) -> None:
        """Code mixing single and double quotes does not break the embedding."""
        code = 'x = "it\'s a \\"test\\""\nprint(x)'
        script = _build_runner_script(code, "/tmp", [])
        assert repr(code) in script
        compile(script, "<test>", "exec")

    def test_triple_double_quotes_in_code(self) -> None:
        """Triple double-quotes in user code are not confused with the embedding."""
        code = 'x = """triple\ndouble"""\nprint(x)'
        script = _build_runner_script(code, "/tmp", [])
        assert repr(code) in script
        compile(script, "<test>", "exec")

    def test_injection_attempt_embedded_as_literal(self) -> None:
        """A quote-escape injection payload is inert when repr() is used."""
        # Old vulnerability: ending the string early and injecting exec()
        injection = "'; import subprocess; subprocess.call(['echo','PWNED']); x = '"
        script = _build_runner_script(injection, "/tmp", [])
        # The repr must appear in the script — not the raw payload
        assert repr(injection) in script
        # The raw injection string must NOT appear unquoted
        assert f"= '{injection}'" not in script

    def test_workspace_dir_embedded_via_repr(self) -> None:
        """Workspace dir with spaces/quotes in name is safely embedded with !r."""
        code = "print('ok')"
        workspace = "/tmp/my dir with 'quotes'"
        script = _build_runner_script(code, workspace, [])
        # The workspace appears via !r format spec
        assert repr(workspace) in script


# ---------------------------------------------------------------------------
# execute_code — execution semantics
# ---------------------------------------------------------------------------


class TestExecuteCodeSemantics:
    def test_simple_print_captures_stdout(self) -> None:
        result = execute_code("print('hello sandbox')")
        assert result.success is True
        assert "hello sandbox" in result.stdout
        assert result.error is None

    def test_stderr_on_syntax_error(self) -> None:
        result = execute_code("def broken(:\n    pass")
        assert result.success is False
        assert result.stderr != ""
        assert result.error is not None

    def test_runtime_error_captured(self) -> None:
        result = execute_code("raise ValueError('oops')")
        assert result.success is False
        # The last line of stderr is reported as error
        assert "ValueError" in (result.error or "") or "ValueError" in result.stderr

    def test_execution_time_is_recorded(self) -> None:
        result = execute_code("x = 1 + 1")
        assert result.execution_time_ms >= 0

    def test_multiline_output(self) -> None:
        code = "for i in range(5):\n    print(i)"
        result = execute_code(code)
        assert result.success is True
        for i in range(5):
            assert str(i) in result.stdout

    def test_forbidden_import_raises_import_error(self) -> None:
        # 'socket' is not in the default allowed list
        result = execute_code("import socket")
        assert result.success is False
        assert "ImportError" in result.stderr or "not allowed" in result.stderr

    def test_allowed_import_math(self) -> None:
        result = execute_code("import math\nprint(math.floor(3.9))")
        assert result.success is True
        assert "3" in result.stdout

    def test_custom_config_allowed_libraries(self) -> None:
        cfg = SandboxConfig(allowed_libraries=["json"], timeout_seconds=10)
        result = execute_code("import json\nprint(json.dumps({'a': 1}))", config=cfg)
        assert result.success is True
        assert '"a"' in result.stdout

    def test_timeout_produces_failure(self) -> None:
        # Use a CPU-bound infinite loop rather than time.sleep (time is not in
        # the default allowed list, but the busy loop still triggers the timeout)
        cfg = SandboxConfig(timeout_seconds=1)
        result = execute_code("while True:\n    pass", config=cfg)
        assert result.success is False
        assert "timed out" in (result.error or "").lower()

    def test_print_unicode(self) -> None:
        result = execute_code("print('\\u00e9l\\u00e8ve')")
        assert result.success is True
        assert "élève" in result.stdout

    def test_backslash_in_string_literal(self) -> None:
        result = execute_code(r'print("back\\slash")')
        assert result.success is True
        assert "back" in result.stdout

    def test_code_with_newlines_and_indentation(self) -> None:
        code = "def add(a, b):\n    return a + b\nprint(add(2, 3))"
        result = execute_code(code)
        assert result.success is True
        assert "5" in result.stdout

    def test_empty_code_succeeds(self) -> None:
        result = execute_code("")
        assert result.success is True

    def test_sandbox_result_fields_populated(self) -> None:
        result = execute_code("print('test')")
        assert isinstance(result, SandboxResult)
        assert result.stdout != "" or result.stderr == ""
        assert result.generated_files == []  # no files written


# ---------------------------------------------------------------------------
# execute_code — file collection
# ---------------------------------------------------------------------------


class TestExecuteCodeFileCollection:
    def test_file_written_in_sandbox_is_collected(self) -> None:
        code = "with open('output.txt', 'w') as f:\n    f.write('result')"
        result = execute_code(code)
        assert result.success is True
        names = [f["name"] for f in result.generated_files]
        assert any("output.txt" in n for n in names)

    def test_generated_file_has_base64_content(self) -> None:
        code = "with open('data.txt', 'w') as f:\n    f.write('hello')"
        result = execute_code(code)
        assert result.success is True
        file_entry = next((f for f in result.generated_files if "data.txt" in f["name"]), None)
        assert file_entry is not None
        assert "content_base64" in file_entry
        decoded = base64.b64decode(file_entry["content_base64"]).decode()
        assert decoded == "hello"


# ---------------------------------------------------------------------------
# _collect_files — unit tests
# ---------------------------------------------------------------------------


class TestCollectFiles:
    def test_collect_skips_runner_script(self) -> None:
        with tempfile.TemporaryDirectory() as ws:
            runner = Path(ws) / "_runner.py"
            runner.write_text("# runner")
            files = _collect_files(ws)
        assert all(f["name"] != "_runner.py" for f in files)

    def test_collect_returns_empty_for_empty_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as ws:
            files = _collect_files(ws)
        assert files == []

    def test_collect_includes_size_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as ws:
            p = Path(ws) / "test.txt"
            p.write_bytes(b"abcde")
            files = _collect_files(ws)
        assert len(files) == 1
        assert files[0]["size_bytes"] == 5

    def test_collect_large_file_omits_content_base64(self) -> None:
        """Files larger than _MAX_FILE_BYTES should not have content_base64."""
        from exo_web.services.sandbox import _MAX_FILE_BYTES

        with tempfile.TemporaryDirectory() as ws:
            p = Path(ws) / "big.bin"
            p.write_bytes(b"x" * (_MAX_FILE_BYTES + 1))
            files = _collect_files(ws)
        assert len(files) == 1
        assert "content_base64" not in files[0]


# ---------------------------------------------------------------------------
# SandboxConfig defaults
# ---------------------------------------------------------------------------


class TestSandboxConfig:
    def test_default_timeout(self) -> None:
        cfg = SandboxConfig()
        assert cfg.timeout_seconds == 30

    def test_default_memory_limit(self) -> None:
        cfg = SandboxConfig()
        assert cfg.memory_limit_mb == 256

    def test_default_allowed_libraries_includes_math(self) -> None:
        cfg = SandboxConfig()
        assert "math" in cfg.allowed_libraries

    def test_custom_config_override(self) -> None:
        cfg = SandboxConfig(timeout_seconds=5, memory_limit_mb=64, allowed_libraries=["csv"])
        assert cfg.timeout_seconds == 5
        assert cfg.memory_limit_mb == 64
        assert cfg.allowed_libraries == ["csv"]
