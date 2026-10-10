"""Summarize the raw traces in results/runs. Read-only: never modifies a trace."""

import json
import re
from collections import Counter
from pathlib import Path

RUNS_DIR = Path(__file__).parent / "results" / "runs"
BASELINE = "v1_prose_reply (no protocol field)"
POOL_WORD = re.compile(r"\bpool\b", re.IGNORECASE)  # matches 'pool', not 'threadpool'


def query_of(step):
    arguments = step["arguments"]
    return arguments.get("query") if isinstance(arguments, dict) else None


def protocol_of(trace):
    return trace["settings"].get("protocol", BASELINE)


def prompt_of(trace):
    # Traces made before prompts were named all used the "v2" prompt.
    if "prompt" in trace["settings"]:
        return trace["settings"]["prompt"]
    return "v2" if "protocol" in trace["settings"] else "v1"


def condition_of(trace):
    """Everything that defines an experimental condition."""
    settings = trace["settings"]
    return (
        trace["scenario_type"],
        trace["model"],
        protocol_of(trace),
        prompt_of(trace),
        settings.get("max_steps"),
        settings.get("deadline_seconds"),
        settings.get("detect_repeats"),
    )


def saw_evidence(trace):
    """True if any search in this run returned at least one log line."""
    return any(
        step["outcome"] == "ok" and step["observation"]["match_count"] > 0
        for step in trace["steps"]
    )


def main():
    paths = sorted(RUNS_DIR.rglob("*.json"))
    traces = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    print(f"traces found: {len(traces)}")

    for condition in sorted({condition_of(t) for t in traces}, key=str):
        scenario, model, protocol, prompt, max_steps, deadline, repeats = condition
        deadline_text = "none" if deadline is None else f"{deadline}s"
        max_steps_text = "none" if max_steps is None else max_steps
        group = [t for t in traces if condition_of(t) == condition]

        print(f"\n== {scenario} | model: {model} | {protocol} | prompt: {prompt}"
              f" | max_steps: {max_steps_text} | deadline: {deadline_text}"
              f" | repeat_guard: {repeats}  ({len(group)} runs) ==")

        print("stop reasons:   ", dict(Counter(t["stop_reason"] for t in group)))
        print("steps per run:  ", dict(sorted(Counter(t["step_count"] for t in group).items())))

        print("first queries:")
        first = Counter(query_of(t["steps"][0]) for t in group if t["steps"])
        for query, count in first.most_common():
            print(f"  {count} x {query!r}")

        searched_pool = sum(
            any(POOL_WORD.search(str(query_of(s))) for s in t["steps"]) for t in group
        )
        final_runs = [t for t in group if t["stop_reason"] == "final_answer"]
        final_mentions_pool = sum(
            bool(POOL_WORD.search(t["final_answer"] or "")) for t in group
        )
        saw = sum(saw_evidence(t) for t in group)
        ungrounded = sum(not saw_evidence(t) for t in final_runs)
        mean_seconds = sum(t["duration_seconds"] for t in group) / len(group)

        print(f"runs with a query containing the word 'pool':  {searched_pool} of {len(group)}")
        print(f"runs that saw at least one matching log line:  {saw} of {len(group)}")
        print(f"runs with stop_reason 'final_answer':          {len(final_runs)} of {len(group)}")
        print(f"final answers with NO matching evidence seen:  {ungrounded} of {len(final_runs)}")
        print(f"final answers containing the word 'pool':      {final_mentions_pool} of {len(group)}")
        print(f"mean run duration: {mean_seconds:.1f} s")


if __name__ == "__main__":
    main()
