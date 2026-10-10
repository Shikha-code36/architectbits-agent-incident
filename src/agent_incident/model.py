"""What a model decision looks like, plus a scripted model for tests.

The agent loop only depends on this interface: decide(messages) -> Decision.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict


@dataclass(frozen=True)
class FinalAnswer:
    text: str

@dataclass(frozen=True)
class NoToolCall:
    """The model replied in plain text and requested no tool."""

    text: str
    

Decision = ToolCall | FinalAnswer | NoToolCall



class ModelError(Exception):
    """The model could not produce a usable decision."""


class ScriptedModel:
    """Replays a fixed list of decisions.

    For unit tests and clearly labelled controlled scenarios only. A trace
    produced with this model must never be presented as real LLM behavior.
    """

    name = "scripted"
    is_real_llm = False

    def __init__(self, decisions, repeat_last: bool = False):
        self._decisions = list(decisions)
        self._repeat_last = repeat_last
        self._position = 0

    def decide(self, messages: list[dict]) -> Decision:
        if self._position < len(self._decisions):
            decision = self._decisions[self._position]
            self._position += 1
            return decision
        if self._repeat_last and self._decisions:
            return self._decisions[-1]
        raise ModelError("Scripted model has no more decisions.")
