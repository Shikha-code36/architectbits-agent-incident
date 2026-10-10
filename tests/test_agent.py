import pytest

from agent_incident.agent import run_investigation
from agent_incident.model import FinalAnswer, ScriptedModel, ToolCall

SEARCH_POOL = ToolCall("search_logs", {"query": "pool"})
SEARCH_USELESS = ToolCall("search_logs", {"query": "checkout-api p99 latency 4s"})
ANSWER = FinalAnswer("Deploy v2.14.0 shrank DB_POOL_SIZE from 50 to 5.")


class RecordingModel:
    """Test double that records the conversation roles it is shown."""

    name = "recording"
    is_real_llm = False

    def __init__(self, decisions):
        self._decisions = list(decisions)
        self.roles_seen = []

    def decide(self, messages):
        self.roles_seen.append([message["role"] for message in messages])
        return self._decisions.pop(0)


def test_normal_investigation_searches_then_answers():
    trace = run_investigation(ScriptedModel([SEARCH_POOL, ANSWER]), "unit_test")

    assert trace.stop_reason == "final_answer"
    assert trace.final_answer == ANSWER.text
    assert len(trace.steps) == 1
    assert trace.steps[0].observation["match_count"] == 4


def test_invalid_tool_call_is_recorded_as_an_error_and_the_loop_continues():
    bad_call = ToolCall("search_logs", {"query": ""})

    trace = run_investigation(ScriptedModel([bad_call, ANSWER]), "unit_test")

    assert trace.steps[0].outcome == "error"
    assert trace.stop_reason == "final_answer"


def test_model_failure_is_recorded_as_a_stop_reason():
    trace = run_investigation(ScriptedModel([]), "unit_test")

    assert trace.stop_reason == "model_error"
    assert trace.steps == []


def test_without_guards_a_repeating_model_burns_every_step():
    # CONTROLLED scenario: scripted repetition, not real LLM behavior.
    model = ScriptedModel([SEARCH_USELESS] * 6)

    trace = run_investigation(
        model,
        "controlled",
        max_steps=None,
        deadline_seconds=None,
        detect_repeats=False,
    )


    assert trace.scenario_type == "controlled"
    assert len(trace.steps) == 6
    assert all(step.observation["match_count"] == 0 for step in trace.steps)
    assert trace.stop_reason == "model_error"  # only the script ending stopped it


def test_model_is_shown_each_observation_before_its_next_decision():
    model = RecordingModel([SEARCH_POOL, ANSWER])

    run_investigation(model, "unit_test")

    assert model.roles_seen[0] == ["system", "user"]
    assert model.roles_seen[1] == ["system", "user", "assistant", "tool"]


def test_scripted_model_cannot_be_labelled_as_real_llm():
    with pytest.raises(ValueError):
        run_investigation(ScriptedModel([ANSWER]), "real_llm")
