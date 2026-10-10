import itertools

from agent_incident.agent import run_investigation
from agent_incident.model import FinalAnswer, ScriptedModel, ToolCall

ANSWER = FinalAnswer("Deploy v2.14.0 shrank DB_POOL_SIZE from 50 to 5.")


def search(query):
    return ToolCall("search_logs", {"query": query})


def test_step_budget_stops_an_endless_model():
    model = ScriptedModel([search("pool")], repeat_last=True)

    trace = run_investigation(
        model, "controlled", max_steps=3, deadline_seconds=None, detect_repeats=False
    )

    assert len(trace.steps) == 3
    assert trace.stop_reason == "step_budget_exhausted"


def test_repeated_call_is_not_executed_and_stops_the_run():
    model = ScriptedModel([search("pool"), search("pool"), ANSWER])

    trace = run_investigation(model, "controlled")

    assert len(trace.steps) == 1
    assert trace.stop_reason == "repeated_action"


def test_repeat_detection_ignores_letter_case():
    model = ScriptedModel([search("pool"), search("POOL"), ANSWER])

    trace = run_investigation(model, "controlled")

    assert trace.stop_reason == "repeated_action"


def test_known_limitation_reworded_queries_are_not_detected_as_repeats():
    # Documents what repeat detection does NOT catch: same intent, new words.
    model = ScriptedModel(
        [search("pool"), search("pool size"), search("pool waiters"), ANSWER]
    )

    trace = run_investigation(model, "controlled")

    assert len(trace.steps) == 3
    assert trace.stop_reason == "final_answer"


def test_deadline_stops_the_run_between_steps():
    ticks = itertools.count()  # fake clock: 0, 1, 2, 3, ...
    model = ScriptedModel([search("pool")], repeat_last=True)

    trace = run_investigation(
        model,
        "controlled",
        max_steps=None,
        deadline_seconds=2.5,
        detect_repeats=False,
        clock=lambda: next(ticks),
    )

    assert trace.stop_reason == "deadline_exceeded"
    assert len(trace.steps) == 2


def test_trace_records_the_guard_configuration():
    trace = run_investigation(
        ScriptedModel([ANSWER]),
        "unit_test",
        max_steps=3,
        deadline_seconds=10,
        detect_repeats=False,
    )

    assert trace.settings["max_steps"] == 3
    assert trace.settings["deadline_seconds"] == 10
    assert trace.settings["detect_repeats"] is False
