import json

import pytest

from agent_incident.trace import Trace


def make_trace():
    return Trace(scenario_type="unit_test", model="none", settings={"temperature": 0})


def test_steps_are_numbered_and_record_ok_and_error_outcomes():
    trace = make_trace()
    trace.add_step("search_logs", {"query": "pool"}, observation={"match_count": 4})
    trace.add_step("search_logs", {"query": ""}, error="'query' must not be empty.")

    assert [step.index for step in trace.steps] == [1, 2]
    assert [step.outcome for step in trace.steps] == ["ok", "error"]


def test_finished_trace_exports_stop_reason_and_step_count():
    trace = make_trace()
    trace.add_step("search_logs", {"query": "pool"}, observation={"match_count": 4})
    trace.finish("final_answer", final_answer="Pool shrank from 50 to 5.")

    data = trace.to_dict()

    assert data["scenario_type"] == "unit_test"
    assert data["stop_reason"] == "final_answer"
    assert data["step_count"] == 1
    assert data["final_answer"] == "Pool shrank from 50 to 5."


def test_unknown_scenario_type_is_rejected():
    with pytest.raises(ValueError):
        Trace(scenario_type="spontaneous_llm_behavior", model="none")


def test_unknown_stop_reason_is_rejected():
    with pytest.raises(ValueError):
        make_trace().finish("gave_up_because_bored")


def test_cannot_export_a_trace_that_has_not_finished():
    with pytest.raises(RuntimeError):
        make_trace().to_dict()


def test_cannot_finish_a_trace_twice():
    trace = make_trace()
    trace.finish("final_answer")

    with pytest.raises(RuntimeError):
        trace.finish("model_error")


def test_saved_trace_is_valid_json_named_after_its_scenario_type(tmp_path):
    trace = make_trace()
    trace.add_step("search_logs", {"query": "pool"}, observation={"match_count": 4})
    trace.finish("final_answer", final_answer="done")

    path = trace.save(tmp_path)

    assert path.name.endswith("_unit_test.json")
    assert json.loads(path.read_text(encoding="utf-8"))["step_count"] == 1



def test_no_tool_call_is_a_valid_stop_reason_and_keeps_the_text_as_detail():
    trace = make_trace()
    trace.finish("no_tool_call", detail="Let's try a broader query.")

    data = trace.to_dict()

    assert data["stop_reason"] == "no_tool_call"
    assert data["stop_detail"] == "Let's try a broader query."
    assert data["final_answer"] is None


def test_stop_detail_is_none_when_none_is_given():
    trace = make_trace()
    trace.finish("final_answer", final_answer="done")

    assert trace.to_dict()["stop_detail"] is None


def test_model_error_keeps_its_message_as_detail():
    trace = make_trace()
    trace.finish("model_error", detail="Ollama unreachable")

    assert trace.to_dict()["stop_detail"] == "Ollama unreachable"
