"""The agent loop: model decision -> tool execution -> observation -> repeat.

Protocol v2: the agent finishes only when the model calls submit_root_cause.
Safeguards are ON by default. Turning one off is an explicit opt-out.
"""

import json
import time

from agent_incident.incident import INCIDENT_ALERT
from agent_incident.model import FinalAnswer, ModelError, NoToolCall
from agent_incident.tools import ToolError, execute_tool
from agent_incident.trace import Trace

PROTOCOL_VERSION = "submit_tool_v2"

_PROMPT_V2 = (
    "You are an on-call engineer investigating a production incident. "
    "Use the search_logs tool to gather evidence. When you have found evidence "
    "for the root cause, call the submit_root_cause tool with a short summary. "
    "Do not answer in plain text: always respond by calling a tool."
)

# Named prompts are experimental conditions. Never edit a prompt after it has
# produced evidence: add a new name instead, so traces stay truthful.
PROMPTS = {
    "v2": _PROMPT_V2,
    "v2_no_human": (
        _PROMPT_V2
        + " There is no human available to answer questions or run tools for you. "
        "Never ask for information; act on your own by calling tools."
    ),
}

DEFAULT_PROMPT = "v2"


def _call_key(decision) -> tuple:
    """Identity of a tool call, ignoring letter case."""
    arguments = json.dumps(decision.arguments, sort_keys=True, default=str)
    return (decision.name, arguments.lower())


def run_investigation(
    model,
    scenario_type: str,
    settings: dict | None = None,
    *,
    max_steps: int | None = 10,
    deadline_seconds: float | None = 60,
    detect_repeats: bool = True,
    prompt: str = DEFAULT_PROMPT,
    clock=time.monotonic,
) -> Trace:
    # A scripted model must never be labelled as real LLM behavior.
    if scenario_type == "real_llm" and not model.is_real_llm:
        raise ValueError("A scripted model cannot be labelled scenario_type='real_llm'.")
    if prompt not in PROMPTS:
        raise ValueError(f"prompt must be one of {sorted(PROMPTS)}, got {prompt!r}")

    system_prompt = PROMPTS[prompt]

    # Evidence files state the rules and the prompt the run was under.
    run_settings = {
        **(settings or {}),
        "protocol": PROTOCOL_VERSION,
        "prompt": prompt,
        "system_prompt": system_prompt,
        "max_steps": max_steps,
        "deadline_seconds": deadline_seconds,
        "detect_repeats": detect_repeats,
    }
    trace = Trace(scenario_type, model.name, run_settings)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": INCIDENT_ALERT},
    ]

    started_at = clock()
    seen_calls = set()

    while True:
        # Guards run BEFORE the model is asked, so we never pay for a decision
        # we have already decided not to use.
        if deadline_seconds is not None and clock() - started_at > deadline_seconds:
            trace.finish("deadline_exceeded")
            return trace
        if max_steps is not None and len(trace.steps) >= max_steps:
            trace.finish("step_budget_exhausted")
            return trace

        try:
            decision = model.decide(messages)
        except ModelError as exc:
            trace.finish("model_error", detail=str(exc))
            return trace

        if isinstance(decision, FinalAnswer):
            trace.finish("final_answer", final_answer=decision.text)
            return trace

        # Prose with no tool call is NOT an answer. Keep the text as evidence.
        if isinstance(decision, NoToolCall):
            trace.finish("no_tool_call", detail=decision.text)
            return trace

        # Checked BEFORE execution: a repeated call is never run a second time.
        if detect_repeats:
            key = _call_key(decision)
            if key in seen_calls:
                trace.finish("repeated_action")
                return trace
            seen_calls.add(key)

        started = time.perf_counter()
        try:
            observation = execute_tool(decision.name, decision.arguments)
            error = None
        except ToolError as exc:
            observation = None
            error = str(exc)
        duration_ms = (time.perf_counter() - started) * 1000

        trace.add_step(
            decision.name,
            decision.arguments,
            observation=observation,
            error=error,
            duration_ms=duration_ms,
        )

        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": decision.name, "arguments": decision.arguments}}
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_name": decision.name,
                "content": json.dumps(observation if error is None else {"error": error}),
            }
        )
