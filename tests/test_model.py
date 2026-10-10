import pytest

from agent_incident.model import FinalAnswer, ModelError, ScriptedModel, ToolCall

SEARCH = ToolCall("search_logs", {"query": "pool"})
ANSWER = FinalAnswer("Pool shrank from 50 to 5.")


def test_scripted_model_replays_decisions_in_order():
    model = ScriptedModel([SEARCH, ANSWER])

    assert model.decide([]) == SEARCH
    assert model.decide([]) == ANSWER


def test_scripted_model_raises_when_it_runs_out_of_decisions():
    model = ScriptedModel([SEARCH])
    model.decide([])

    with pytest.raises(ModelError):
        model.decide([])


def test_repeat_last_keeps_returning_the_final_decision():
    model = ScriptedModel([SEARCH], repeat_last=True)

    assert [model.decide([]) for _ in range(4)] == [SEARCH] * 4
