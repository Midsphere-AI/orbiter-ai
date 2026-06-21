"""Tests for exo.config — configuration types."""

import pytest
from pydantic import ValidationError

from exo.config import (
    ModelConfig,
    RunConfig,
    TaskConfig,
    parse_model_string,
)

# --- parse_model_string ---


class TestParseModelString:
    def test_provider_and_model(self) -> None:
        assert parse_model_string("openai:gpt-4o") == ("openai", "gpt-4o")

    def test_anthropic_provider(self) -> None:
        assert parse_model_string("anthropic:claude-sonnet-4-20250514") == (
            "anthropic",
            "claude-sonnet-4-20250514",
        )

    def test_no_prefix_defaults_to_openai(self) -> None:
        assert parse_model_string("gpt-4o") == ("openai", "gpt-4o")

    def test_empty_string(self) -> None:
        assert parse_model_string("") == ("openai", "")

    def test_multiple_colons(self) -> None:
        assert parse_model_string("custom:my:model") == ("custom", "my:model")

    def test_colon_only(self) -> None:
        assert parse_model_string(":model") == ("", "model")


# --- ModelConfig ---


class TestModelConfig:
    def test_defaults(self) -> None:
        mc = ModelConfig()
        assert mc.provider == "openai"
        assert mc.model_name == "gpt-4o"
        assert mc.api_key is None
        assert mc.base_url is None
        assert mc.max_retries == 3
        assert mc.timeout == 30.0
        assert mc.context_window_tokens is None

    def test_create(self) -> None:
        mc = ModelConfig(
            provider="anthropic",
            model_name="claude-sonnet-4-20250514",
            api_key="sk-test",
            base_url="https://api.example.com",
            max_retries=5,
            timeout=60.0,
        )
        assert mc.provider == "anthropic"
        assert mc.api_key == "sk-test"

    def test_frozen(self) -> None:
        mc = ModelConfig()
        with pytest.raises(ValidationError):
            mc.provider = "changed"  # type: ignore[misc]

    def test_max_retries_ge_zero(self) -> None:
        ModelConfig(max_retries=0)  # should not raise
        with pytest.raises(ValidationError):
            ModelConfig(max_retries=-1)

    def test_timeout_gt_zero(self) -> None:
        ModelConfig(timeout=0.1)  # should not raise
        with pytest.raises(ValidationError):
            ModelConfig(timeout=0)
        with pytest.raises(ValidationError):
            ModelConfig(timeout=-1.0)

    def test_context_window_tokens_field(self) -> None:
        mc = ModelConfig(context_window_tokens=128000)
        assert mc.context_window_tokens == 128000

    def test_context_window_tokens_none(self) -> None:
        mc = ModelConfig(context_window_tokens=None)
        assert mc.context_window_tokens is None

    def test_roundtrip(self) -> None:
        mc = ModelConfig(provider="anthropic", model_name="claude", max_retries=1)
        data = mc.model_dump()
        restored = ModelConfig.model_validate(data)
        assert restored == mc

    def test_roundtrip_with_context_window(self) -> None:
        mc = ModelConfig(provider="openai", model_name="gpt-4o", context_window_tokens=128000)
        data = mc.model_dump()
        restored = ModelConfig.model_validate(data)
        assert restored == mc
        assert restored.context_window_tokens == 128000


# --- TaskConfig ---


class TestTaskConfig:
    def test_defaults(self) -> None:
        tc = TaskConfig(name="my_task")
        assert tc.name == "my_task"
        assert tc.description == ""

    def test_create(self) -> None:
        tc = TaskConfig(name="analyze", description="Analyze the data.")
        assert tc.description == "Analyze the data."

    def test_missing_name(self) -> None:
        with pytest.raises(ValidationError):
            TaskConfig()  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        tc = TaskConfig(name="t")
        with pytest.raises(ValidationError):
            tc.name = "changed"  # type: ignore[misc]

    def test_roundtrip(self) -> None:
        tc = TaskConfig(name="t", description="desc")
        data = tc.model_dump()
        restored = TaskConfig.model_validate(data)
        assert restored == tc


# --- RunConfig ---


class TestRunConfig:
    def test_defaults(self) -> None:
        rc = RunConfig()
        assert rc.max_steps == 10
        assert rc.timeout is None
        assert rc.stream is False
        assert rc.verbose is False

    def test_create(self) -> None:
        rc = RunConfig(max_steps=20, timeout=120.0, stream=True, verbose=True)
        assert rc.max_steps == 20
        assert rc.timeout == 120.0
        assert rc.stream is True
        assert rc.verbose is True

    def test_frozen(self) -> None:
        rc = RunConfig()
        with pytest.raises(ValidationError):
            rc.stream = True  # type: ignore[misc]

    def test_max_steps_ge_one(self) -> None:
        RunConfig(max_steps=1)  # should not raise
        with pytest.raises(ValidationError):
            RunConfig(max_steps=0)

    def test_roundtrip(self) -> None:
        rc = RunConfig(max_steps=5, stream=True)
        data = rc.model_dump()
        restored = RunConfig.model_validate(data)
        assert restored == rc
