import pytest

from agent_incident.model import FinalAnswer, ModelError, NoToolCall, ToolCall
from agent_incident.ollama_model import parse_response
from agent_incident.tools import TOOL_SCHEMAS


def reply(content="", tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"message": message}


def call(name="search_logs", arguments=None):
    return {"function": {"name": name, "arguments": arguments}}


def test_tool_call_becomes_a_tool_call_decision():
    body = reply(tool_calls=[call(arguments={"query": "pool"})])

    assert parse_response(body) == ToolCall("search_logs", {"query": "pool"})


def test_plain_text_becomes_a_no_tool_call_decision_not_an_answer():
    assert parse_response(reply("  Let's try again.  ")) == NoToolCall("Let's try again.")


def test_arguments_sent_as_a_json_string_are_decoded():
    body = reply(tool_calls=[call(arguments='{"query": "pool"}')])

    assert parse_response(body) == ToolCall("search_logs", {"query": "pool"})


def test_arguments_that_are_not_valid_json_are_a_model_error():
    body = reply(tool_calls=[call(arguments="{query: pool")])

    with pytest.raises(ModelError):
        parse_response(body)


def test_empty_reply_is_a_model_error():
    with pytest.raises(ModelError):
        parse_response(reply(""))


def test_tool_call_without_a_name_is_a_model_error():
    body = reply(tool_calls=[call(name="", arguments={"query": "pool"})])

    with pytest.raises(ModelError):
        parse_response(body)


def test_known_limitation_only_the_first_of_several_tool_calls_is_used():
    body = reply(
        tool_calls=[
            call(arguments={"query": "pool"}),
            call(arguments={"query": "timeout"}),
        ]
    )

    assert parse_response(body) == ToolCall("search_logs", {"query": "pool"})


def test_submit_root_cause_becomes_a_final_answer():
    body = reply(
        tool_calls=[call("submit_root_cause", {"summary": "  Pool shrank 50 to 5.  "})]
    )

    assert parse_response(body) == FinalAnswer("Pool shrank 50 to 5.")


def test_submit_root_cause_without_a_summary_is_a_model_error():
    with pytest.raises(ModelError):
        parse_response(reply(tool_calls=[call("submit_root_cause", {})]))


def test_submit_root_cause_with_a_blank_summary_is_a_model_error():
    body = reply(tool_calls=[call("submit_root_cause", {"summary": "   "})])

    with pytest.raises(ModelError):
        parse_response(body)


def test_model_is_offered_both_the_search_and_the_submit_tool():
    names = [schema["function"]["name"] for schema in TOOL_SCHEMAS]

    assert names == ["search_logs", "submit_root_cause"]
