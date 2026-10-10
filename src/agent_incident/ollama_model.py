"""Real LLM adapter: asks a local Ollama model to choose the next action."""

import json
import urllib.error
import urllib.request

from agent_incident.model import Decision, FinalAnswer, ModelError, NoToolCall, ToolCall
from agent_incident.tools import SUBMIT_TOOL_NAME, TOOL_SCHEMAS

OLLAMA_URL = "http://localhost:11434/api/chat"


def _parse_submission(arguments) -> FinalAnswer:
    summary = arguments.get("summary") if isinstance(arguments, dict) else None
    if not isinstance(summary, str) or not summary.strip():
        raise ModelError("submit_root_cause requires a non-empty 'summary' string.")
    return FinalAnswer(summary.strip())


def parse_response(body: dict) -> Decision:
    """Turn Ollama's reply into a Decision. The reply is untrusted input."""
    message = body.get("message") or {}
    tool_calls = message.get("tool_calls") or []

    if tool_calls:
        function = tool_calls[0].get("function") or {}
        name = function.get("name")
        arguments = function.get("arguments", {})

        if not isinstance(name, str) or not name:
            raise ModelError("Tool call had no function name.")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                raise ModelError("Tool arguments were not valid JSON.")

        if name == SUBMIT_TOOL_NAME:
            return _parse_submission(arguments)
        return ToolCall(name, arguments)

    content = (message.get("content") or "").strip()
    if not content:
        raise ModelError("Model returned neither a tool call nor text.")
    return NoToolCall(content)


class OllamaModel:
    is_real_llm = True

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        temperature: float | None = None,
        timeout_seconds: float = 120,
    ):
        self.name = model
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        # Recorded in every trace so a run can be reproduced and compared.
        self.settings = {
            "temperature": temperature if temperature is not None else "ollama_default"
        }

    def decide(self, messages: list[dict]) -> Decision:
        payload = {
            "model": self.name,
            "messages": messages,
            "tools": TOOL_SCHEMAS,
            "stream": False,
        }
        if self.temperature is not None:
            payload["options"] = {"temperature": self.temperature}

        request = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ModelError(f"Could not get a usable reply from Ollama: {exc}") from exc

        return parse_response(body)
