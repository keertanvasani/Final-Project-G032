# The agent loop (step 2)

The controller used to be one 280-line function. It is now split into tools, a
policy, a verifier and a loop, so that step 3 can swap the hand-written rules
for a learned policy without touching anything else. **With the rule policy the
output is identical to before** — see "Equivalence" below.

```
            ┌──────────── policy (FallbackChain: rules → plain) ───────────┐
            │  D1 enhance?            D3 zoom which boxes?   D4 vote?      │
            └──────┬──────────────────────────┬──────────────────┬─────────┘
Observe            │ Act                      │ Act              │ Act
quality ──► enhance ──► detect+fuse ──► ✔ ──► zoom ──► classify ──► vote/plain ──► ✔ ──► result
                                       │                                         │
                          Verify: text filter, refuse gate      Verify: calibration, Known/Unknown
```

| Part | File | What it holds |
|---|---|---|
| Tools | `agentic/tools.py` | `enhance_image`, `zoom_boxes`, `classify_boxes`, `resolve_detection` (vote), `score_plain` |
| Policy | `agentic/policy.py` | `RulePolicy` (the app's rules), `PlainPolicy` (never act), `FallbackChain` |
| Verifier | `agentic/verifier.py` | text filter, refuse gate, calibration, Known/Unknown |
| Loop | `agentic/loop.py` | `run_loop` and the `Guard` (time budget, call limit) |
| Entry points | `agent_controller.py` | `run_agentic_pipeline` (app) and `run_perception` (evaluation) |

## Guardrails

* **Single pass.** Each discretionary tool runs at most `AGENT_MAX_CALLS_PER_TOOL`
  (= 1) times per image, and the vote at most once per box. There is no retry loop.
* **Time budget.** Once an image has used `AGENT_TIME_BUDGET_MS` (10 s), the
  remaining enhance / zoom / vote steps are skipped and the image finishes on the
  plain models. The trace records it as a `Guardrail:` line.
* **Fallback chain.** `AGENT_POLICY_CHAIN = ("learned", "rules", "plain")` since
  2026-11-18 (step 3, confirmed on S_test). If a policy raises or returns an invalid
  decision, the next one answers and the trace says so. The learned policy is read from
  `models/phase_b_machine/bandit_policy.json`, or the identical committed copy in
  `experiments/results/phase_b_machine/step3/`.
* **Note for evaluation scripts:** `run_perception` follows the deployed chain too.
  Results recorded before 2026-11-18 used the rules; `verify_agent_loop.py` pins the
  rule chain on purpose.
* **Safety steps are not choices.** The policy cannot switch off the text filter,
  the refuse gate, calibration or the Known/Unknown flag; the loop calls them
  directly.

The API response is unchanged. The internal summary gains an `agent` entry with
the policy chain, how often each tool was chosen and any guardrail events.

## Equivalence

`experiments/verify_agent_loop.py` loads the pre-refactor controller from git
(commit `8c3a1b0`) and runs it next to the new one. Result file:
`experiments/results/phase_b_machine/verify_agent_loop.json`.

| Check | Images | Result |
|---|---|---|
| App path: detections, trace, summary, annotated image | S_policy (deep-sea) + demo images (both profiles) | identical |
| Evaluation path (`run_perception`) | same | identical |
| Broken first policy falls back to the rules | same | identical to rules |
| Budget 0 ms = plain models (ablation: text filter + gate + Known/Unknown) | same | identical, no tool ran |
| Call limit refuses a repeat call | unit check | passes |

Run of 2026-11-18: 844 S_policy images (1,459 detections), 13 demo images on the
deep-sea profile (45) and on the aquarium profile (137) — 0 mismatches in every
check, 391 s.

## Image agent: a second pass (week 1)

Week 0 (`experiments/results/phase_b_machine/WEEK0_RESULTS.md`) found that enhancement
helps when it is chosen **after** looking at what the first pass found, not before. The
loop therefore has two more decisions, made by the image agent
(`ImageAgentPolicy`, `agentic/policy.py`):

| | When | Choice | State |
|---|---|---|---|
| D5 | after the first pass is finished and verified | re-run the whole pass on an enhanced copy (`agentic/probes.ARMS`), or stop | s1 = image quality + probes of pass 1 |
| D6 | after the second pass | keep the first or the second pass | s2 = s1 + probes of pass 2 |

* **Probes** (`agentic/probes.py`) are cheap extra reads of the frozen classifier on each
  box: mirror and ±10% shift agreement, margin, entropy, energy, distance to the class
  centroids (from S_fit training crops), geometry, plus colour statistics. They change
  nothing.
* **Policy:** two-step fitted Q-iteration on full-information logs
  (`experiments/log_image_agent.py`, `experiments/fit_image_agent.py`). Q2 predicts
  reward(second) − reward(first); Q1 predicts the gain of re-running when D6 then
  chooses, minus λ × the re-run's cost.
* **Guardrails:** at most two passes. Pass 2 has its own call limits (one zoom, one vote
  per box) and shares the image's time budget. Every safety step runs again on pass 2;
  D6 only chooses between two fully verified results. The rules and plain models never
  re-run.
* **Not deployed.** `AGENT_POLICY_CHAIN` does not contain `"image"`, so the app is
  unchanged until the agent is evaluated on S_policy under a registered protocol.
  `experiments/check_image_agent_live.py` checks that the live loop reproduces the
  replay of the logs.
* Perception mode (`run_perception`) does not run the second pass.

## Risk-controlled verifier (weeks 3–6)

`agentic/verifier.taxon_label`, switched on with `config.AGENT_VERIFIER`
(`{"mode": "hierarchical" | "flat", "alpha": 0.1 | 0.2}`; off by default). For every
final box it reports the deepest level of the tree in `agentic/taxonomy.py` whose
combiner probability clears the Learn-then-Test threshold, or abstains (needs review).
Thresholds were calibrated on S_policy and confirmed on S_test (PROTOCOL §16):
realised dive-level risk 0.094 / 0.183 at α 0.10 / 0.20. It is a safety step: no policy
can switch it off or change its threshold. Summary of all results:
`experiments/results/phase_b_machine/AGENT_CONTROLLER_SUMMARY.md`.
