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
* **Fallback chain.** `AGENT_POLICY_CHAIN = ("rules", "plain")`. If a policy raises
  or returns an invalid decision, the next one answers and the trace says so.
  Step 3 puts the learned policy first: `("learned", "rules", "plain")`.
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
