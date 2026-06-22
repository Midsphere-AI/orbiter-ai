"""Tests for exo_cli.loader — agent discovery and loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from exo.agent import Agent
from exo_cli.loader import (
    AgentLoadError,
    _parse_front_matter,
    discover_agent_files,
    load_markdown_agent,
    load_python_agent,
    load_yaml_agents,
    scan_directory,
    validate_agent,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# AgentLoadError
# ---------------------------------------------------------------------------


class TestAgentLoadError:
    def test_is_exception(self) -> None:
        assert issubclass(AgentLoadError, Exception)

    def test_message(self) -> None:
        err = AgentLoadError("boom")
        assert str(err) == "boom"


# ---------------------------------------------------------------------------
# _parse_front_matter
# ---------------------------------------------------------------------------


class TestParseFrontMatter:
    def test_basic(self) -> None:
        text = "---\nname: hello\nmodel: openai:gpt-4o\n---\nBody text."
        meta, body = _parse_front_matter(text)
        assert meta["name"] == "hello"
        assert meta["model"] == "openai:gpt-4o"
        assert body == "Body text."

    def test_no_front_matter(self) -> None:
        text = "Just some text."
        meta, body = _parse_front_matter(text)
        assert meta == {}
        assert body == "Just some text."

    def test_unclosed_front_matter(self) -> None:
        text = "---\nname: hello\nStill open."
        meta, body = _parse_front_matter(text)
        assert meta == {}
        assert body == "---\nname: hello\nStill open."

    def test_empty_body(self) -> None:
        text = "---\nname: hello\n---"
        meta, body = _parse_front_matter(text)
        assert meta["name"] == "hello"
        assert body == ""

    def test_multiline_body(self) -> None:
        text = "---\nname: test\n---\nLine 1\n\nLine 3"
        meta, body = _parse_front_matter(text)
        assert meta["name"] == "test"
        assert body == "Line 1\n\nLine 3"

    def test_lowercases_keys(self) -> None:
        text = "---\nName: bob\nMODEL: gpt-4\n---\n"
        meta, _ = _parse_front_matter(text)
        assert "name" in meta
        assert "model" in meta

    def test_no_colon_line_skipped(self) -> None:
        text = "---\nname: hello\nno colon here\nmodel: gpt-4\n---\n"
        meta, _ = _parse_front_matter(text)
        assert len(meta) == 2
        assert "name" in meta
        assert "model" in meta


# ---------------------------------------------------------------------------
# load_markdown_agent
# ---------------------------------------------------------------------------


class TestLoadMarkdownAgent:
    def test_basic(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "agent.md", "---\nname: test_agent\n---\nHello world.")
        result = load_markdown_agent(p)
        assert "test_agent" in result
        agent = result["test_agent"]
        assert agent.name == "test_agent"
        assert agent.instructions == "Hello world."

    def test_defaults_name_to_stem(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "my_agent.md", "---\nmodel: openai:gpt-4o\n---\nBody.")
        result = load_markdown_agent(p)
        assert "my_agent" in result

    def test_explicit_instructions(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path / "a.md",
            "---\nname: a\ninstructions: Be helpful.\n---\nBody is ignored.",
        )
        result = load_markdown_agent(p)
        assert result["a"].instructions == "Be helpful."

    def test_model_field(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "a.md", "---\nname: a\nmodel: anthropic:claude-3\n---\n")
        result = load_markdown_agent(p)
        assert "anthropic" in result["a"].model or "claude" in result["a"].model

    def test_temperature(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "a.md", "---\nname: a\ntemperature: 0.5\n---\n")
        result = load_markdown_agent(p)
        assert result["a"].temperature == 0.5

    def test_max_tokens(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "a.md", "---\nname: a\nmax_tokens: 1024\n---\n")
        result = load_markdown_agent(p)
        assert result["a"].max_tokens == 1024

    def test_max_steps(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "a.md", "---\nname: a\nmax_steps: 5\n---\n")
        result = load_markdown_agent(p)
        assert result["a"].max_steps == 5

    def test_invalid_temperature(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "a.md", "---\nname: a\ntemperature: not_a_num\n---\n")
        with pytest.raises(AgentLoadError, match="Invalid temperature"):
            load_markdown_agent(p)

    def test_invalid_max_tokens(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "a.md", "---\nname: a\nmax_tokens: abc\n---\n")
        with pytest.raises(AgentLoadError, match="Invalid max_tokens"):
            load_markdown_agent(p)

    def test_invalid_max_steps(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "a.md", "---\nname: a\nmax_steps: xyz\n---\n")
        with pytest.raises(AgentLoadError, match="Invalid max_steps"):
            load_markdown_agent(p)

    def test_no_front_matter_uses_body(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "agent.md", "Just a body, no front matter.")
        result = load_markdown_agent(p)
        agent = result["agent"]
        assert agent.instructions == "Just a body, no front matter."


# ---------------------------------------------------------------------------
# load_python_agent
# ---------------------------------------------------------------------------


class TestLoadPythonAgent:
    def test_basic(self, tmp_path: Path) -> None:
        code = (
            'from exo.agent import Agent\ndef create_agent():\n    return Agent(name="py_agent")\n'
        )
        p = _write(tmp_path / "my_agent.py", code)
        result = load_python_agent(p)
        assert "py_agent" in result
        assert result["py_agent"].name == "py_agent"

    def test_dict_return(self, tmp_path: Path) -> None:
        code = (
            "from exo.agent import Agent\n"
            "def create_agent():\n"
            '    return {"a": Agent(name="a"), "b": Agent(name="b")}\n'
        )
        p = _write(tmp_path / "multi.py", code)
        result = load_python_agent(p)
        assert "a" in result
        assert "b" in result

    def test_no_create_agent(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "empty.py", "x = 1\n")
        with pytest.raises(AgentLoadError, match="No create_agent"):
            load_python_agent(p)

    def test_module_error(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "bad.py", "raise RuntimeError('boom')\n")
        with pytest.raises(AgentLoadError, match="Error executing"):
            load_python_agent(p)

    def test_factory_error(self, tmp_path: Path) -> None:
        code = "def create_agent():\n    raise ValueError('nope')\n"
        p = _write(tmp_path / "fail.py", code)
        with pytest.raises(AgentLoadError, match=r"create_agent.*raised"):
            load_python_agent(p)

    def test_single_agent_uses_name_attr(self, tmp_path: Path) -> None:
        code = 'from exo.agent import Agent\ndef create_agent():\n    return Agent(name="named")\n'
        p = _write(tmp_path / "whatever.py", code)
        result = load_python_agent(p)
        assert "named" in result


# ---------------------------------------------------------------------------
# load_yaml_agents
# ---------------------------------------------------------------------------


class TestLoadYamlAgents:
    def test_basic(self, tmp_path: Path) -> None:
        yaml_content = 'agents:\n  helper:\n    instructions: "Be helpful."\n'
        p = _write(tmp_path / "agents.yaml", yaml_content)
        result = load_yaml_agents(p)
        assert "helper" in result
        assert result["helper"].name == "helper"

    def test_missing_agents_section(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "bad.yaml", "foo: bar\n")
        with pytest.raises(AgentLoadError, match="YAML load failed"):
            load_yaml_agents(p)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(AgentLoadError, match="YAML load failed"):
            load_yaml_agents(tmp_path / "nonexistent.yaml")


# ---------------------------------------------------------------------------
# discover_agent_files
# ---------------------------------------------------------------------------


class TestDiscoverAgentFiles:
    def test_finds_supported_types(self, tmp_path: Path) -> None:
        _write(tmp_path / "agent.py", "")
        _write(tmp_path / "config.yaml", "")
        _write(tmp_path / "prompt.md", "")
        _write(tmp_path / "readme.txt", "")  # should be skipped
        files = discover_agent_files(tmp_path)
        suffixes = {p.suffix for p in files}
        assert suffixes == {".py", ".yaml", ".md"}

    def test_sorted_order(self, tmp_path: Path) -> None:
        _write(tmp_path / "z_agent.py", "")
        _write(tmp_path / "a_agent.py", "")
        files = discover_agent_files(tmp_path)
        assert files[0].name == "a_agent.py"
        assert files[1].name == "z_agent.py"

    def test_empty_directory(self, tmp_path: Path) -> None:
        files = discover_agent_files(tmp_path)
        assert files == []

    def test_not_a_directory(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "file.txt", "hi")
        with pytest.raises(AgentLoadError, match="Not a directory"):
            discover_agent_files(p)

    def test_skips_subdirectories(self, tmp_path: Path) -> None:
        sub = tmp_path / "subdir"
        sub.mkdir()
        _write(sub / "agent.py", "")
        _write(tmp_path / "top.py", "")
        files = discover_agent_files(tmp_path)
        assert len(files) == 1
        assert files[0].name == "top.py"


# ---------------------------------------------------------------------------
# validate_agent
# ---------------------------------------------------------------------------


class TestValidateAgent:
    def test_valid_agent(self) -> None:
        agent = Agent(name="valid")
        validate_agent("valid", agent)  # should not raise

    def test_missing_name(self) -> None:
        bare = object()
        with pytest.raises(AgentLoadError, match="missing 'name'"):
            validate_agent("x", bare)

    def test_missing_run(self) -> None:
        obj = type("NoRun", (), {"name": "nr"})()
        with pytest.raises(AgentLoadError, match="missing 'run'"):
            validate_agent("nr", obj)


# ---------------------------------------------------------------------------
# scan_directory
# ---------------------------------------------------------------------------


class TestScanDirectory:
    def test_loads_markdown(self, tmp_path: Path) -> None:
        _write(tmp_path / "agent.md", "---\nname: md_agent\n---\nHello.")
        result = scan_directory(tmp_path)
        assert "md_agent" in result

    def test_loads_python(self, tmp_path: Path) -> None:
        code = 'from exo.agent import Agent\ndef create_agent():\n    return Agent(name="py_ag")\n'
        _write(tmp_path / "agent.py", code)
        result = scan_directory(tmp_path)
        assert "py_ag" in result

    def test_loads_yaml(self, tmp_path: Path) -> None:
        yaml_content = 'agents:\n  yml_agent:\n    instructions: "Hi"\n'
        _write(tmp_path / "config.yaml", yaml_content)
        result = scan_directory(tmp_path)
        assert "yml_agent" in result

    def test_empty_directory(self, tmp_path: Path) -> None:
        result = scan_directory(tmp_path)
        assert result == {}

    def test_duplicate_name_raises(self, tmp_path: Path) -> None:
        _write(tmp_path / "agent1.md", "---\nname: same_name\n---\nHi.")
        _write(tmp_path / "agent2.md", "---\nname: same_name\n---\nBye.")
        with pytest.raises(AgentLoadError, match="Duplicate agent name"):
            scan_directory(tmp_path)

    def test_validation_disabled(self, tmp_path: Path) -> None:
        code = 'def create_agent():\n    return {"no_methods": object()}\n'
        _write(tmp_path / "fake.py", code)
        # With validation disabled, it loads without error
        result = scan_directory(tmp_path, validate=False)
        assert "no_methods" in result

    def test_validation_enabled_catches_bad_agent(self, tmp_path: Path) -> None:
        code = 'def create_agent():\n    return {"bad": object()}\n'
        _write(tmp_path / "fake.py", code)
        with pytest.raises(AgentLoadError, match="missing 'name'"):
            scan_directory(tmp_path)

    def test_mixed_types(self, tmp_path: Path) -> None:
        _write(tmp_path / "a.md", "---\nname: md_agent\n---\nBody.")
        yaml_content = 'agents:\n  yml_agent:\n    instructions: "Hi"\n'
        _write(tmp_path / "b.yaml", yaml_content)
        result = scan_directory(tmp_path)
        assert "md_agent" in result
        assert "yml_agent" in result


# ---------------------------------------------------------------------------
# sys.modules cleanup (Finding #6)
# ---------------------------------------------------------------------------


class TestSysModulesCleanup:
    def test_python_agent_not_left_in_sys_modules(self, tmp_path: Path) -> None:
        """After loading a Python agent, the transient module is removed from sys.modules."""
        import sys

        code = (
            "from exo.agent import Agent\n"
            "def create_agent():\n"
            "    return Agent(name='cleanup_test', model='test:m')\n"
        )
        p = tmp_path / "cleanup_test.py"
        _write(p, code)
        module_name = "_exo_agent_cleanup_test"

        # Ensure it's not already there
        sys.modules.pop(module_name, None)
        load_python_agent(p)
        assert module_name not in sys.modules

    def test_same_stem_files_load_independently(self, tmp_path: Path) -> None:
        """Two files with the same stem in different directories load without collision."""
        import sys

        dir1 = tmp_path / "d1"
        dir2 = tmp_path / "d2"
        dir1.mkdir()
        dir2.mkdir()

        code1 = (
            "from exo.agent import Agent\n"
            "def create_agent():\n"
            "    return Agent(name='agent_one', model='test:m')\n"
        )
        code2 = (
            "from exo.agent import Agent\n"
            "def create_agent():\n"
            "    return Agent(name='agent_two', model='test:m')\n"
        )
        _write(dir1 / "myagent.py", code1)
        _write(dir2 / "myagent.py", code2)

        r1 = load_python_agent(dir1 / "myagent.py")
        r2 = load_python_agent(dir2 / "myagent.py")

        # Names come from the Agent, not the file stem
        assert "agent_one" in r1
        assert "agent_two" in r2
        # Neither should pollute sys.modules
        assert "_exo_agent_myagent" not in sys.modules
