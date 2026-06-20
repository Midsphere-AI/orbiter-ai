"""Tests for exo.types — core message and I/O types."""

import pytest
from pydantic import ValidationError

from exo.types import (
    ActionModel,
    AgentInput,
    AgentOutput,
    AssistantMessage,
    ExoError,
    Message,
    RunResult,
    StreamEvent,
    SystemMessage,
    TextEvent,
    ToolCall,
    ToolCallEvent,
    ToolResult,
    Usage,
    UserMessage,
)

# --- Construction & defaults ---


class TestUserMessage:
    def test_create(self) -> None:
        msg = UserMessage(content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_role_is_literal(self) -> None:
        msg = UserMessage(content="hi")
        assert msg.role == "user"


class TestSystemMessage:
    def test_create(self) -> None:
        msg = SystemMessage(content="You are helpful.")
        assert msg.role == "system"
        assert msg.content == "You are helpful."


class TestToolCall:
    def test_create(self) -> None:
        tc = ToolCall(id="tc_1", name="search", arguments='{"q": "hello"}')
        assert tc.id == "tc_1"
        assert tc.name == "search"
        assert tc.arguments == '{"q": "hello"}'

    def test_default_arguments(self) -> None:
        tc = ToolCall(id="tc_2", name="noop")
        assert tc.arguments == ""


class TestAssistantMessage:
    def test_text_only(self) -> None:
        msg = AssistantMessage(content="Sure!")
        assert msg.role == "assistant"
        assert msg.content == "Sure!"
        assert msg.tool_calls == []

    def test_with_tool_calls(self) -> None:
        tc = ToolCall(id="tc_1", name="search", arguments='{"q": "x"}')
        msg = AssistantMessage(tool_calls=[tc])
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "search"
        assert msg.content == ""

    def test_tool_calls_default_is_empty_list(self) -> None:
        msg = AssistantMessage()
        assert msg.tool_calls == []
        assert isinstance(msg.tool_calls, list)

    def test_tool_calls_default_not_shared(self) -> None:
        msg1 = AssistantMessage()
        msg2 = AssistantMessage()
        assert msg1.tool_calls is not msg2.tool_calls


class TestToolResult:
    def test_success(self) -> None:
        tr = ToolResult(tool_call_id="tc_1", tool_name="search", content="found it")
        assert tr.role == "tool"
        assert tr.tool_call_id == "tc_1"
        assert tr.tool_name == "search"
        assert tr.content == "found it"
        assert tr.error is None

    def test_error(self) -> None:
        tr = ToolResult(tool_call_id="tc_1", tool_name="search", error="not found")
        assert tr.content == ""
        assert tr.error == "not found"


# --- Immutability ---


class TestImmutability:
    def test_user_message_frozen(self) -> None:
        msg = UserMessage(content="hi")
        with pytest.raises(ValidationError):
            msg.content = "changed"  # type: ignore[misc]

    def test_system_message_frozen(self) -> None:
        msg = SystemMessage(content="sys")
        with pytest.raises(ValidationError):
            msg.content = "changed"  # type: ignore[misc]

    def test_assistant_message_frozen(self) -> None:
        msg = AssistantMessage(content="ok")
        with pytest.raises(ValidationError):
            msg.content = "changed"  # type: ignore[misc]

    def test_tool_call_frozen(self) -> None:
        tc = ToolCall(id="tc_1", name="x")
        with pytest.raises(ValidationError):
            tc.name = "y"  # type: ignore[misc]

    def test_tool_result_frozen(self) -> None:
        tr = ToolResult(tool_call_id="tc_1", tool_name="x")
        with pytest.raises(ValidationError):
            tr.content = "changed"  # type: ignore[misc]


# --- Serialization roundtrip ---


class TestSerialization:
    def test_user_message_roundtrip(self) -> None:
        msg = UserMessage(content="hello")
        data = msg.model_dump()
        assert data == {"role": "user", "content": "hello"}
        restored = UserMessage.model_validate(data)
        assert restored == msg

    def test_assistant_with_tool_calls_roundtrip(self) -> None:
        tc = ToolCall(id="tc_1", name="search", arguments='{"q": "x"}')
        msg = AssistantMessage(content="Let me search.", tool_calls=[tc])
        data = msg.model_dump()
        restored = AssistantMessage.model_validate(data)
        assert restored == msg
        assert restored.tool_calls[0].name == "search"

    def test_tool_result_roundtrip(self) -> None:
        tr = ToolResult(tool_call_id="tc_1", tool_name="search", content="result", error=None)
        data = tr.model_dump()
        restored = ToolResult.model_validate(data)
        assert restored == tr


# --- Type narrowing (Message union) ---


class TestMessageUnion:
    def test_isinstance_checks(self) -> None:
        msgs: list[Message] = [
            UserMessage(content="hi"),
            AssistantMessage(content="hello"),
            SystemMessage(content="be helpful"),
            ToolResult(tool_call_id="tc_1", tool_name="x", content="done"),
        ]
        assert isinstance(msgs[0], UserMessage)
        assert isinstance(msgs[1], AssistantMessage)
        assert isinstance(msgs[2], SystemMessage)
        assert isinstance(msgs[3], ToolResult)

    def test_match_statement(self) -> None:
        msg: Message = UserMessage(content="test")
        match msg:
            case UserMessage(content=c):
                assert c == "test"
            case _:
                pytest.fail("Should have matched UserMessage")


# --- Validation ---


class TestValidation:
    def test_user_message_missing_content(self) -> None:
        with pytest.raises(ValidationError):
            UserMessage()  # type: ignore[call-arg]

    def test_tool_call_missing_required(self) -> None:
        with pytest.raises(ValidationError):
            ToolCall()  # type: ignore[call-arg]

    def test_tool_result_missing_required(self) -> None:
        with pytest.raises(ValidationError):
            ToolResult()  # type: ignore[call-arg]


# --- ExoError ---


class TestExoError:
    def test_is_exception(self) -> None:
        assert issubclass(ExoError, Exception)

    def test_raise_and_catch(self) -> None:
        with pytest.raises(ExoError, match="something went wrong"):
            raise ExoError("something went wrong")


# --- Usage ---


class TestUsage:
    def test_defaults(self) -> None:
        u = Usage()
        assert u.input_tokens == 0
        assert u.output_tokens == 0
        assert u.total_tokens == 0

    def test_create(self) -> None:
        u = Usage(input_tokens=10, output_tokens=20, total_tokens=30)
        assert u.input_tokens == 10
        assert u.output_tokens == 20
        assert u.total_tokens == 30

    def test_frozen(self) -> None:
        u = Usage(input_tokens=5)
        with pytest.raises(ValidationError):
            u.input_tokens = 10  # type: ignore[misc]

    def test_roundtrip(self) -> None:
        u = Usage(input_tokens=100, output_tokens=50, total_tokens=150)
        data = u.model_dump()
        assert data == {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        assert Usage.model_validate(data) == u


# --- AgentInput ---


class TestAgentInput:
    def test_create(self) -> None:
        inp = AgentInput(query="hello")
        assert inp.query == "hello"
        assert inp.messages == []

    def test_with_messages(self) -> None:
        msgs: list[Message] = [UserMessage(content="hi")]
        inp = AgentInput(query="hello", messages=msgs)
        assert len(inp.messages) == 1

    def test_missing_query(self) -> None:
        with pytest.raises(ValidationError):
            AgentInput()  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        inp = AgentInput(query="hi")
        with pytest.raises(ValidationError):
            inp.query = "changed"  # type: ignore[misc]

    def test_messages_default_not_shared(self) -> None:
        a = AgentInput(query="a")
        b = AgentInput(query="b")
        assert a.messages is not b.messages

    def test_roundtrip(self) -> None:
        inp = AgentInput(query="q", messages=[UserMessage(content="hi")])
        data = inp.model_dump()
        restored = AgentInput.model_validate(data)
        assert restored == inp


# --- AgentOutput ---


class TestAgentOutput:
    def test_defaults(self) -> None:
        out = AgentOutput()
        assert out.text == ""
        assert out.tool_calls == []
        assert out.usage == Usage()

    def test_with_tool_calls(self) -> None:
        tc = ToolCall(id="tc_1", name="search", arguments='{"q": "x"}')
        out = AgentOutput(text="result", tool_calls=[tc])
        assert len(out.tool_calls) == 1
        assert out.tool_calls[0].name == "search"

    def test_frozen(self) -> None:
        out = AgentOutput(text="hi")
        with pytest.raises(ValidationError):
            out.text = "changed"  # type: ignore[misc]

    def test_tool_calls_default_not_shared(self) -> None:
        a = AgentOutput()
        b = AgentOutput()
        assert a.tool_calls is not b.tool_calls

    def test_roundtrip(self) -> None:
        out = AgentOutput(
            text="done",
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        data = out.model_dump()
        restored = AgentOutput.model_validate(data)
        assert restored == out


# --- ActionModel ---


class TestActionModel:
    def test_create(self) -> None:
        a = ActionModel(tool_call_id="tc_1", tool_name="search", arguments={"q": "x"})
        assert a.tool_call_id == "tc_1"
        assert a.tool_name == "search"
        assert a.arguments == {"q": "x"}

    def test_default_arguments(self) -> None:
        a = ActionModel(tool_call_id="tc_1", tool_name="noop")
        assert a.arguments == {}

    def test_frozen(self) -> None:
        a = ActionModel(tool_call_id="tc_1", tool_name="x")
        with pytest.raises(ValidationError):
            a.tool_name = "y"  # type: ignore[misc]

    def test_arguments_default_not_shared(self) -> None:
        a = ActionModel(tool_call_id="1", tool_name="x")
        b = ActionModel(tool_call_id="2", tool_name="y")
        assert a.arguments is not b.arguments

    def test_roundtrip(self) -> None:
        a = ActionModel(tool_call_id="tc_1", tool_name="search", arguments={"q": "hello"})
        data = a.model_dump()
        restored = ActionModel.model_validate(data)
        assert restored == a


# --- RunResult ---


class TestRunResult:
    def test_defaults(self) -> None:
        r = RunResult()
        assert r.output == ""
        assert r.messages == []
        assert r.usage == Usage()
        assert r.steps == 0

    def test_create(self) -> None:
        r = RunResult(
            output="done",
            messages=[UserMessage(content="hi"), AssistantMessage(content="hello")],
            usage=Usage(input_tokens=50, output_tokens=20, total_tokens=70),
            steps=3,
        )
        assert r.output == "done"
        assert len(r.messages) == 2
        assert r.steps == 3

    def test_steps_ge_zero(self) -> None:
        with pytest.raises(ValidationError):
            RunResult(steps=-1)

    def test_frozen(self) -> None:
        r = RunResult(output="hi")
        with pytest.raises(ValidationError):
            r.output = "changed"  # type: ignore[misc]

    def test_messages_default_not_shared(self) -> None:
        a = RunResult()
        b = RunResult()
        assert a.messages is not b.messages

    def test_roundtrip(self) -> None:
        r = RunResult(output="result", steps=2)
        data = r.model_dump()
        restored = RunResult.model_validate(data)
        assert restored == r


# --- TextEvent ---


class TestTextEvent:
    def test_create(self) -> None:
        e = TextEvent(text="hello")
        assert e.type == "text"
        assert e.text == "hello"
        assert e.agent_name == ""

    def test_with_agent_name(self) -> None:
        e = TextEvent(text="hi", agent_name="assistant")
        assert e.agent_name == "assistant"

    def test_frozen(self) -> None:
        e = TextEvent(text="hi")
        with pytest.raises(ValidationError):
            e.text = "changed"  # type: ignore[misc]

    def test_roundtrip(self) -> None:
        e = TextEvent(text="chunk", agent_name="bot")
        data = e.model_dump()
        assert data == {"type": "text", "text": "chunk", "agent_name": "bot"}
        restored = TextEvent.model_validate(data)
        assert restored == e


# --- ToolCallEvent ---


class TestToolCallEvent:
    def test_create(self) -> None:
        e = ToolCallEvent(tool_name="search", tool_call_id="tc_1")
        assert e.type == "tool_call"
        assert e.tool_name == "search"
        assert e.tool_call_id == "tc_1"
        assert e.agent_name == ""

    def test_with_agent_name(self) -> None:
        e = ToolCallEvent(tool_name="search", tool_call_id="tc_1", agent_name="bot")
        assert e.agent_name == "bot"

    def test_frozen(self) -> None:
        e = ToolCallEvent(tool_name="search", tool_call_id="tc_1")
        with pytest.raises(ValidationError):
            e.tool_name = "other"  # type: ignore[misc]

    def test_roundtrip(self) -> None:
        e = ToolCallEvent(tool_name="search", tool_call_id="tc_1", agent_name="bot")
        data = e.model_dump()
        assert data == {
            "type": "tool_call",
            "tool_name": "search",
            "tool_call_id": "tc_1",
            "arguments": "",
            "agent_name": "bot",
        }
        restored = ToolCallEvent.model_validate(data)
        assert restored == e


# --- StreamEvent union ---


class TestStreamEvent:
    def test_text_event_is_stream_event(self) -> None:
        e: StreamEvent = TextEvent(text="hi")
        assert isinstance(e, TextEvent)

    def test_tool_call_event_is_stream_event(self) -> None:
        e: StreamEvent = ToolCallEvent(tool_name="x", tool_call_id="tc_1")
        assert isinstance(e, ToolCallEvent)

    def test_discriminate_by_type_field(self) -> None:
        events: list[StreamEvent] = [
            TextEvent(text="hi"),
            ToolCallEvent(tool_name="search", tool_call_id="tc_1"),
        ]
        assert events[0].type == "text"
        assert events[1].type == "tool_call"
