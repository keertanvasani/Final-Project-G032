# Week 2 — a cheaper image agent (S_cal)

Goal: cut the cost of consulting the image agent. In Week 1 the per-box probes cost
~16 ms on every image. **S_cal only**; nothing here reads S_policy or S_test for a
result. Not deployed.

## Which parts of the state matter (offline, same logs, `experiments/week2_cheap_state.py`)

Same two-step fitted Q-iteration and nested dive-grouped cross-fitting as Week 1, with
the state cut to subsets of the logged columns (`agentic.probes.state_columns`).
Arm CLAHE 3.0; reward gain per image over the first pass (+0.403).

| State | Extra crops per box | λ = 0.002 gain | 95% CI | vs full |
|---|---|---|---|---|
| **none**: quality, colour, box count, confidences, disagreement, Unknown share | 0 | **+0.115** | [+0.066, +0.160] | −0.002 [−0.036, +0.027] |
| free: + margin, entropy, energy, centroid distances, geometry | 0 if the pass's forward were reused | +0.116 | [+0.067, +0.162] | −0.001 [−0.031, +0.022] |
| free + mirror | 1 | +0.102 | [+0.054, +0.147] | −0.015 [−0.043, +0.006] |
| full (Week 1) | 5 | +0.117 | [+0.078, +0.155] | — |
| gated (free first, full only near the threshold) | 0–5 | +0.119 | [+0.079, +0.158] | +0.002 [−0.005, +0.009] |

**The probe crops add nothing measurable.** What carries the signal is the first pass's
own output plus colour: how many boxes, how confident, whether detector and classifier
disagree, how many look unknown.

This corrects the reading of Week 0. Its "probes" feature set mixed these first-pass
outputs with the crop probes. The credit belongs to **looking at the first pass before
deciding**, not to the mirror, shift or embedding probes.

## The no-probe agent (`fit_image_agent.py --probes none --lam 0.002`)

| Policy | Gain | 95% CI | Re-runs | Extra ms (logs) |
|---|---|---|---|---|
| one-step | +0.073 | [+0.021, +0.125] | 44% | 21 |
| always re-run + D6 | +0.109 | [+0.059, +0.157] | 100% | 42 |
| two-step, λ = 0 | +0.112 | [+0.063, +0.159] | 65% | 29 |
| **two-step, λ = 0.002** | **+0.115** | [+0.066, +0.160] | 49% | 23 |
| two-step, λ = 0.005 | +0.111 | [+0.065, +0.154] | 34% | 17 |
| oracle | +0.279 | [+0.235, +0.321] | 20% | 10 |

* Two-step − one-step: **+0.039 [+0.008, +0.070]**, unchanged from Week 1.
* λ = 0.005 keeps almost all of the gain with a third of the images re-run: the best
  speed/gain point.
* State: s1 = 8 quality + 11 first-pass features (19); s2 = 30. No centroid file or
  probe crops are needed at run time.

## Checks

* Live loop vs replay, 200 S_cal images: 0 decision and 0 output mismatches
  (92 re-runs, 63 kept), `week2/live_check.json`.
* Live wall time on the laptop GPU, 300 S_cal images (`experiments/week2_time_agent.py`,
  `week2/timing.json`):

  | Chain | Mean | Median | p90 | Re-runs |
  |---|---|---|---|---|
  | deployed (learned → rules → plain) | 38.2 ms | 35.8 ms | 58.3 ms | 0% |
  | image agent in front, λ = 0.002 | 69.0 ms | 56.9 ms | 121.7 ms | 49% |

  +31 ms mean per image for +0.115 net reward per image (about +29%).

## Decisions taken on S_cal

* Deploy candidate: no-probe state, CLAHE 3.0, λ = 0.002, fitted on S_cal
  (`week2/image_agent_policy.json`; `config.AGENT_IMAGE_POLICY` points to it).
  λ = 0.005 is the registered alternative for a speed-constrained setting.
* The probe code stays in `agentic/probes.py` (the logger uses it) but is not called
  by the deploy candidate.

## Next

Register in PROTOCOL.md before S_policy is read:
* the frozen policy file (hash);
* H1 at matched compute vs the rules and the Step 3 bandit;
* two-step > one-step;
* the +ms bound.

Then log S_policy and evaluate once.
