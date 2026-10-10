# architectbits-agent-incident

What does an LLM agent actually do when its tool calls come back empty?

A controlled experiment for the ArchitectBits video
*"AI Agents Will Replace Engineers. So Why Are They Breaking Production?"*

An agent investigates a simulated production latency incident using one tool
(`search_logs`) over 20 fixed log lines. The true cause is a config change that
shrank a database connection pool from 50 to 5. Everything runs locally against
Ollama models. No real systems are touched.

## What is in here

| Path | Contents |
|---|---|
| `src/agent_incident/` | The agent loop, tool layer, model adapters, trace, guards |
| `tests/` | 57 deterministic unit tests |
| `run_experiment.py` | Runs real-LLM investigations and saves traces |
| `run_controlled.py` | Scripted scenarios that demonstrate each safeguard |
| `summarize_runs.py` | Counts outcomes from the raw traces (read-only) |
| `results/runs/` | 56 raw JSON traces, grouped by condition |
| `results/evidence/` | Saved test and summary output |
| `results/SUMMARY.md` | Findings and limitations |

## Headline results

See [results/SUMMARY.md](results/SUMMARY.md) for the full table and caveats.

- The expected failure (an agent repeating itself) was not the main behavior
  of the 3B model. It searched, got nothing, and stopped, or replied in prose.
- Under the original protocol, 11 of 11 prose replies were recorded as
  `final_answer` with no evidence behind them. The label was wrong, not the model.
- A 7B model persisted, wasted its budget on zero-match searches, and in 2 of 10
  runs submitted a root cause (one without any evidence).
- Each safeguard (step budget, deadline, repeat detection) stops a loop in a
  controlled, scripted scenario. Guards limit waste; they do not make answers correct.

## Honesty notes

- Real-LLM traces are labelled `real_llm`. Scripted ones are labelled
  `controlled` and are never presented as LLM behavior.
- 10 or 11 runs per condition is a sample, not a rate. One task, one tool, one
  dataset, one machine.

## Run it

Requires Python 3.12 and [Ollama](https://ollama.com) with `qwen2.5:3b` (and
optionally `qwen2.5:7b`). On Windows PowerShell:

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install pytest
    .\.venv\Scripts\python.exe -m pytest -v
    python run_experiment.py --runs 10 --model qwen2.5:3b
    python run_controlled.py
    python summarize_runs.py

macOS/Linux: the same commands work with `source .venv/bin/activate` and
`python3`.
