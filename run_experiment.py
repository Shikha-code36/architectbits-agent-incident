"""Run real-LLM investigations and save each trace to results/runs."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent_incident.agent import DEFAULT_PROMPT, PROMPTS, run_investigation  # noqa: E402
from agent_incident.ollama_model import OllamaModel  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results" / "runs"


def describe(trace, path):
    print(
        f"  model: {trace.model}   protocol: {trace.settings.get('protocol')}"
        f"   prompt: {trace.settings.get('prompt')}"
        f"   max_steps: {trace.settings.get('max_steps')}"
        f"   deadline: {trace.settings.get('deadline_seconds')}s"
    )
    for step in trace.steps:
        if step.outcome == "ok":
            detail = f"match_count={step.observation['match_count']}"
        else:
            detail = f"ERROR {step.error}"
        print(
            f"  step {step.index}: {step.tool}({step.arguments}) -> {detail}"
            f"  [{step.duration_ms:.1f} ms]"
        )
    print(f"  stop reason: {trace.stop_reason}")
    if trace.stop_detail:
        print(f"  stop detail: {trace.stop_detail}")
    if trace.final_answer:
        print(f"  final answer: {trace.final_answer}")
    print(f"  duration: {(trace.finished_at - trace.started_at).total_seconds():.1f} s")
    print(f"  saved: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, choices=sorted(PROMPTS))
    parser.add_argument("--deadline", type=float, default=60,
                        help="wall-clock seconds before a run is stopped (default 60)")
    args = parser.parse_args()

    for number in range(1, args.runs + 1):
        print(f"=== run {number} of {args.runs} ===")
        model = OllamaModel(model=args.model, temperature=args.temperature)
        trace = run_investigation(
            model,
            "real_llm",
            settings=model.settings,
            prompt=args.prompt,
            deadline_seconds=args.deadline,
        )
        path = trace.save(RESULTS_DIR)
        describe(trace, path)
        print()


if __name__ == "__main__":
    main()
