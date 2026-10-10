"""Structured execution trace: what the agent did, what it saw, and why it stopped."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

# Every trace must say what kind of run produced it, so a scripted failure can
# never be mistaken for spontaneous LLM behavior.
SCENARIO_TYPES = ("real_llm", "controlled", "unit_test")

STOP_REASONS = (
    "final_answer",
    "no_tool_call",
    "step_budget_exhausted",
    "deadline_exceeded",
    "repeated_action",
    "model_error",
)


@dataclass
class Step:
    index: int
    tool: str
    arguments: dict
    outcome: str  # "ok" or "error"
    observation: dict | None
    error: str | None
    duration_ms: float


class Trace:
    def __init__(self, scenario_type: str, model: str, settings: dict | None = None):
        if scenario_type not in SCENARIO_TYPES:
            raise ValueError(
                f"scenario_type must be one of {SCENARIO_TYPES}, got {scenario_type!r}"
            )
        self.scenario_type = scenario_type
        self.model = model
        self.settings = settings or {}
        self.started_at = datetime.now(timezone.utc)
        self.finished_at = None
        self.steps: list[Step] = []
        self.stop_reason = None
        self.stop_detail = None
        self.final_answer = None

    def add_step(
        self,
        tool: str,
        arguments: dict,
        observation: dict | None = None,
        error: str | None = None,
        duration_ms: float = 0.0,
    ) -> None:
        self.steps.append(
            Step(
                index=len(self.steps) + 1,
                tool=tool,
                arguments=arguments,
                outcome="error" if error is not None else "ok",
                observation=observation,
                error=error,
                duration_ms=duration_ms,
            )
        )

    def finish(
        self,
        stop_reason: str,
        final_answer: str | None = None,
        detail: str | None = None,
    ) -> None:
        if self.stop_reason is not None:
            raise RuntimeError("Trace is already finished.")
        if stop_reason not in STOP_REASONS:
            raise ValueError(
                f"stop_reason must be one of {STOP_REASONS}, got {stop_reason!r}"
            )
        self.stop_reason = stop_reason
        self.final_answer = final_answer
        self.stop_detail = detail
        self.finished_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        if self.stop_reason is None:
            raise RuntimeError("Cannot export a trace that has not finished.")
        return {
            "scenario_type": self.scenario_type,
            "model": self.model,
            "settings": self.settings,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "duration_seconds": (self.finished_at - self.started_at).total_seconds(),
            "step_count": len(self.steps),
            "stop_reason": self.stop_reason,
            "stop_detail": self.stop_detail,
            "final_answer": self.final_answer,
            "steps": [asdict(step) for step in self.steps],
        }

    def save(self, directory) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = self.started_at.strftime("%Y%m%dT%H%M%S_%f")
        path = directory / f"{stamp}_{self.scenario_type}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path
