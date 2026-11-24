# Week 1 — two-step image agent (fitted Q-iteration), S_cal

Built from the Week 0 finding (`WEEK0_RESULTS.md`): enhancement helps when it is
chosen **after** the first pass. **S_cal only** (889 images, 315 dives); S_policy and
S_test were not read for any result here. The agent is **not deployed**.

## What was built

| Part | File |
|---|---|
| Probes (cheap reads of the frozen classifier; ~13 ms per 10 crops after moving resize to cv2 + GPU, from 21 ms) | `agentic/probes.py` |
| Loop: the pass is now a function; D5 (re-run on an enhanced image?) and D6 (keep which pass?) after the first pass; pass 2 has its own call limits and re-runs every safety step | `agentic/loop.py` |
| `ImageAgentPolicy`; D5/D6 on every policy (rules and plain never re-run); the chain skips a policy that does not make a decision | `agentic/policy.py` |
| Full-information logs written by the loop itself (forced re-run, both passes, exact states) | `experiments/log_image_agent.py` |
| Two-step fitted Q-iteration with nested dive-grouped cross-fitting; λ sweep | `experiments/fit_image_agent.py` |
| Live loop vs replay check | `experiments/check_image_agent_live.py` |

**Checks:**
* Refactored loop vs the pre-refactor controller (`verify_agent_loop.py`, rule chain): identical (see below).
* Logged passes reproduce the Week 0 logs.
* Live loop with the fitted agent vs replay of the logs on 200 S_cal images: 0 decision mismatches and 0 output mismatches (115 re-runs, 81 kept).

## Method

Reward per image: +1 TP, −1 reported error, 0 Unknown. Δ = R(pass 2) − R(pass 1).

* **Q2(s2) ≈ Δ**: D6 keeps pass 2 iff Q2 > 0.
* **Q1(s1) ≈ Δ · 1[Q2 > 0]**: the realised gain of D6, out of fold. D5 re-runs iff Q1 − λ·c > 0.
* Ridge, strength 10.
* States:
  * s1 = image quality + pass-1 probes (64 features);
  * s2 = s1 + pass-2 probes (120 features).
* Nested cross-fitting. For each outer dive fold, Q1's targets come from out-of-fold Q2 within the training folds, so no image is scored by a model that saw it.

## Results (out of fold, reward gain per image over the first pass, 95% dive bootstrap)

First pass: **+0.403** reward per image.

| Policy | Gain | 95% CI | Re-runs | Extra ms/img |
|---|---|---|---|---|
| always CLAHE 3.0, keep pass 2 | −0.024 | [−0.086, +0.035] | 100% | 57 |
| always CLAHE 3.0 + learned D6 | +0.105 | [+0.062, +0.146] | 100% | 57 |
| one-step CLAHE 3.0 (a single decision from s1) | +0.069 | [+0.015, +0.120] | 38% | 24 |
| **two-step CLAHE 3.0, λ = 0** | **+0.108** | [+0.066, +0.148] | 57% | 34 |
| **two-step CLAHE 3.0, λ = 0.002** | **+0.114** | [+0.075, +0.150] | 36% | 24 |
| two-step, both arms | +0.103 | [+0.061, +0.146] | 63% | 38 |
| oracle | +0.279 | [+0.235, +0.321] | 20% | 13 |

* **Two-step − one-step: +0.039 [+0.008, +0.075].** Deciding again after seeing the second pass beats committing on the first look. This is the evidence that the sequential (RL) formulation matters, not just a bandit.
* **Two-step vs always re-run + D6:** same gain (+0.003 [−0.008, +0.014]), but D5 skips 43–64% of re-runs. D5's value is compute, D6's value is reward.
* **λ curve:** at λ = 0.002 the agent keeps all of the gain (+0.114, about **+28%** on the first pass) with 36% re-runs and +24 ms per image. Larger λ trades gain for speed smoothly: +0.096 at 14 ms, +0.053 at 6 ms.
* CLAHE 3.0 beats CLAHE 2.0 here (+0.108 vs +0.085). Offering both arms does not help.
* The policy captures about 40% of the oracle's gain.

## Cost, honestly

* Consulting the agent at all costs the pass-1 probes, **16 ms per image** (mean), on every image.
* At λ = 0.002 the total added time is about 16 + 24 ≈ 40 ms per image, on top of a ~43 ms pass in this logging run (~27 ms in Step 3's live measurement).
* This is affordable on the laptop GPU. It is the obvious target for Week 2: a cheaper D5 state (fewer crops, or probes only when the first pass is uncertain).

## Not done / next

* λ for an evaluation must be registered before S_policy is read. The candidate is λ = 0.002, chosen on S_cal here.
* Register H1 (matched compute), H-two-step (two-step > one-step) and the cost bound in PROTOCOL.md, then log and evaluate S_policy.
* The verifier side (learned combiner → taxonomy levels → Learn-then-Test) is unchanged and comes in Week 3.
