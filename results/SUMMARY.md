# Experiment summary

Simulated production incident: after a deploy, `checkout-api` p99 latency rises
from ~200 ms to ~4 s. The true cause is a config change that shrank
`DB_POOL_SIZE` from 50 to 5. An LLM agent investigates using one tool,
`search_logs` (literal, case-insensitive substring match over 20 fixed log lines).

All runs are raw traces in `results/runs/`. Counts below are produced by
`summarize_runs.py` (see `results/evidence/summarize_runs_output.txt`).

## Setup

- Hardware: i5-12450H, 15.7 GB RAM, GTX 1650 (4 GB VRAM). Ollama, local models.
- Guards (default): 10-step budget, 60 s deadline, exact-repeat detection.
- Protocols: v1 = plain-text reply ends the run. v2 = the agent finishes only by
  calling `submit_root_cause`; plain text ends the run as `no_tool_call`.
- Every trace records the model, protocol, prompt text and guard settings.

## Real-LLM results

| Condition | Runs | Outcome | Saw a matching log line |
|---|---|---|---|
| qwen2.5:3b, v1 (prose = answer) | 11 | `final_answer` 11 (all prose, none grounded) | 0 |
| qwen2.5:3b, v2 | 10 | `no_tool_call` 10 | 0 |
| qwen2.5:3b, v2 + "no human available" | 11 | `no_tool_call` 10, `repeated_action` 1 | 0 |
| qwen2.5:7b, v2, 60 s deadline | 10 | `deadline_exceeded` 8, `final_answer` 2 | 2 |
| qwen2.5:7b, v2, 180 s deadline | 10 | `step_budget_exhausted` 5, `repeated_action` 3, `final_answer` 2 | 4 |

## Findings

1. The failure first expected (an agent repeating a tool call) was not the
   dominant behavior of the 3B model. In all 32 3B runs it made between 1 and
   9 searches, got zero matches every time, and stopped, or (v1) wrote prose.
2. Under v1, 11 of 11 runs ended as `final_answer` with no matching evidence
   seen. The label said "answered"; nothing had been answered. Making the
   finish explicit (v2) turned these into `no_tool_call`.
3. An explicit "no human is available" instruction did not remove the
   behavior (by reading the stop text, 8 of 11 still addressed a human).
4. The 7B model persisted: 16 of 20 runs made 5 to 10 searches. In 8 of 10
   runs at 60 s it was stopped by the clock; with 180 s it was stopped by the
   step budget (5), the repeat guard (3) or it submitted (2). More time did not
   increase the number of runs that submitted (2 of 10 in both batches).
5. Of the 7B 180 s runs, 4 saw the key log lines. One submitted a grounded
   answer. One submitted an answer with no matching evidence ("database
   timeout", all 8 searches returned zero).
6. Every 7B run (20 of 20) began with an invented date (`2023-04-15...`) that
   appears nowhere in the alert or logs, and several used regex or boolean
   syntax the tool does not support.

## Safeguards (controlled scenarios, scripted model, NOT LLM behavior)

| Scenario | Steps | Stop reason |
|---|---|---|
| all guards off (replay of 10 real queries) | 10 | `model_error` (script ended) |
| step budget = 4 | 4 | `step_budget_exhausted` |
| exact repeat | 1 | `repeated_action` |
| deadline 1.75 s | 4 | `deadline_exceeded` |

These traces are labelled `controlled` with model `scripted`. They show the
guards working; they are not observations of a real model.

Unit tests: 57 passed (`results/evidence/pytest_output.txt`).

## What this does not show

- Only one task, one tool, one dataset, one hardware setup. 10 runs per
  condition is a sample, not a rate.
- It does not show that agents "cannot" do this. A 7B model submitted a
  grounded answer in 3 of 20 runs total.
- Guards stop waste; they do not make answers correct. Repeat detection is
  exact-match, so reworded queries pass; guards can also stop a promising run
  (a 7B run found the key lines at step 2, then `repeated_action` stopped it
  after 2 searches).
- "Saw evidence" and the word-match for `pool` are crude checks; answers were
  read by hand.
- The alert says `14:02Z` but log lines carry seconds (`14:02:10Z`), so the
  literal alert time never matches any line. This is a property of the dataset.
- The 3B/7B comparison confounds model size with speed: the 7B model runs
  partly on CPU, so the 60 s deadline cuts it earlier.
- The tool returns only a match count for zero results, with no hint about
  how matching works. Better tool feedback was not tested.

## Data notes

- Traces from the v1 and v2 batches were temporarily removed from the results
  folder and later restored; the restored files are the originals.
- One extra `v2_no_human` trace (stamped before the batch) is included, so
  that condition has 11 runs, not 10.
- `incident.py` was once accidentally emptied and restored from the original
  source. The 20-line / 3 `latency` / 4 `pool` fingerprint and all tests were
  rechecked and match.

## Reproduce

    pytest -v
    python run_experiment.py --runs 10 --model qwen2.5:3b
    python run_experiment.py --runs 10 --model qwen2.5:3b --prompt v2_no_human
    python run_experiment.py --runs 10 --model qwen2.5:7b
    python run_experiment.py --runs 10 --model qwen2.5:7b --deadline 180
    python run_controlled.py
    python summarize_runs.py
