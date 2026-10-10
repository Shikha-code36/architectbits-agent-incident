"""Run CONTROLLED scenarios: scripted models that demonstrate each safeguard.

Every trace written here has scenario_type='controlled' and model 'scripted'.
These are NOT observations of real LLM behavior. Scenario 1 and 2 replay the
exact queries of a real qwen2.5:7b run, but the decisions are scripted.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent_incident.agent import run_investigation  
from agent_incident.model import ScriptedModel, ToolCall  

RESULTS_DIR = PROJECT_ROOT / "results" / "runs" / "controlled"

# Queries from a real run: results/runs/.../20261010T213635_265839_real_llm.json
REAL_QUERIES = [
    "2023-04-15T14:02Z",
    "error 2023-04-15T14:02Z",
    "2023-04-15T14:02Z database",
    "2023-04-15T14:02Z timeout",
    "2023-04-15T14:02Z connection refused",
    "2023-04-15T14:02Z database connection refused",
    "2023-04-15T14:02Z database timeout",
    "2023-04-15T14:02Z database error",
    "2023-04-15T14:02Z external service error",
    "2023-04-15T14:02Z external service timeout",
]
REPLAY = "replay of the first 10 queries of real run 20261010T213635_265839"


def search(query):
    return ToolCall("search_logs", {"query": query})


class SlowScriptedModel(ScriptedModel):
    """A scripted model that takes real wall-clock time to 'think'."""

    def __init__(self, decisions, delay_seconds, **kwargs):
        super().__init__(decisions, **kwargs)
        self._delay_seconds = delay_seconds

    def decide(self, messages):
        time.sleep(self._delay_seconds)
        return super().decide(messages)


SCENARIOS = [
    {
        "name": "1_no_guards",
        "note": f"{REPLAY}; all guards OFF, only the end of the script stops it",
        "model": lambda: ScriptedModel([search(q) for q in REAL_QUERIES]),
        "guards": dict(max_steps=None, deadline_seconds=None, detect_repeats=False),
    },
    {
        "name": "2_step_budget",
        "note": f"{REPLAY}; step budget of 4",
        "model": lambda: ScriptedModel([search(q) for q in REAL_QUERIES]),
        "guards": dict(max_steps=4, deadline_seconds=None, detect_repeats=False),
    },
    {
        "name": "3_repeat_detection",
        "note": "scripted exact repeat of the same query",
        "model": lambda: ScriptedModel(
            [search("2023-04-15T14:02Z"), search("2023-04-15T14:02Z"), search("pool")]
        ),
        "guards": dict(max_steps=None, deadline_seconds=None, detect_repeats=True),
    },
    {
        "name": "4_deadline",
        "note": "endless scripted loop, 0.5 s per decision, deadline 1.75 s",
        "model": lambda: SlowScriptedModel([search("pool")], 0.5, repeat_last=True),
        "guards": dict(max_steps=None, deadline_seconds=1.75, detect_repeats=False),
    },
]


def main():
    for scenario in SCENARIOS:
        trace = run_investigation(
            scenario["model"](),
            "controlled",
            settings={"scenario": scenario["name"], "note": scenario["note"]},
            **scenario["guards"],
        )
        path = trace.save(RESULTS_DIR)
        seconds = (trace.finished_at - trace.started_at).total_seconds()
        print(f"{scenario['name']}")
        print(f"  note:        {scenario['note']}")
        print(f"  steps:       {len(trace.steps)}")
        print(f"  stop reason: {trace.stop_reason}")
        if trace.stop_detail:
            print(f"  stop detail: {trace.stop_detail}")
        print(f"  duration:    {seconds:.2f} s")
        print(f"  saved:       {path}")
        print()


if __name__ == "__main__":
    main()
