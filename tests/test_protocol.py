import pytest

from agent_incident.agent import run_investigation
from agent_incident.model import FinalAnswer, ModelError, NoToolCall, ScriptedModel, ToolCall


class FailingModel:
    name = "failing"
    is_real_llm = False

    def decide(self, messages):
        raise ModelError("Ollama unreachable")


def test_prose_reply_stops_the_run_as_no_tool_call_and_keeps_the_text():
    model = ScriptedModel(
        [
            ToolCall("search_logs", {"query": "latency"}),
            NoToolCall("Let's try a broader query."),
        ]
    )

    trace = run_investigation(model, "controlled")

    assert trace.stop_reason == "no_tool_call"
    assert trace.stop_detail == "Let's try a broader query."
    assert trace.final_answer is None
    assert len(trace.steps) == 1


def test_model_error_keeps_its_reason():
    trace = run_investigation(FailingModel(), "unit_test")

    assert trace.stop_reason == "model_error"
    assert trace.stop_detail == "Ollama unreachable"


def test_trace_records_the_protocol_version_and_the_system_prompt():
    trace = run_investigation(ScriptedModel([FinalAnswer("done")]), "unit_test")

    assert trace.settings["protocol"] == "submit_tool_v2"
    assert "submit_root_cause" in trace.settings["system_prompt"]


def test_default_prompt_is_named_v2():
    trace = run_investigation(ScriptedModel([FinalAnswer("done")]), "unit_test")

    assert trace.settings["prompt"] == "v2"


def test_selected_prompt_variant_is_recorded_by_name_and_text():
    trace = run_investigation(
        ScriptedModel([FinalAnswer("done")]), "unit_test", prompt="v2_no_human"
    )

    assert trace.settings["prompt"] == "v2_no_human"
    assert "no human" in trace.settings["system_prompt"].lower()


def test_unknown_prompt_name_is_rejected():
    with pytest.raises(ValueError):
        run_investigation(ScriptedModel([FinalAnswer("done")]), "unit_test", prompt="v9")
