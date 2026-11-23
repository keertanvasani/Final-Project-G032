# Week 4 — budget control, matched compute, live verifier (S_cal)

**S_cal only.** Neither the image agent nor the verifier is deployed.

## 1. Budget control and the accuracy–compute frontier (`experiments/week4_frontier.py`)

Every policy is replayed exactly from full-information logs of the same 889 S_cal images.
Cost = discretionary ms per image over the plain models. Agent = the deployed bandit plus
the two-step image agent (no-probe state), out of fold. For a budget B, λ is found by
bisection (dual ascent on the compute constraint).

| Policy | Reward/img | F1 | Precision | Recall | AURC | ms/img |
|---|---|---|---|---|---|---|
| plain models | +0.396 | 0.635 | 0.627 | 0.642 | 0.127 | 0.0 |
| rule agent | +0.363 | 0.618 | 0.620 | 0.617 | 0.142 | 11.0 |
| deployed bandit (Step 3) | +0.403 | 0.637 | 0.629 | 0.645 | 0.126 | 1.9 |
| agent, budget 6.9 ms (λ 0.0117) | +0.456 | 0.645 | | | 0.131 | 6.8 |
| **agent at the rules' 11.0 ms** (λ 0.0089) | **+0.475** | **0.646** | 0.666 | 0.627 | 0.134 | 11.0 |
| agent, budget 21.9 ms (λ 0.0034) | +0.514 | 0.645 | | | 0.143 | 21.9 |

**Matched compute (original H1), agent − rules, 95% dive-bootstrap CI:**

| Metric | 95% CI |
|---|---|
| reward/img | **[+0.073, +0.154]** |
| F1 | **[+0.016, +0.041]** |
| precision | **[+0.031, +0.062]** |
| recall | [−0.002, +0.025] |
| AURC | [−0.025, +0.010] |
| ms/img | [−1.8, +1.6] (matched) |

* At the same compute, the image agent beats the rule agent clearly on reward, F1 and
  precision. It also beats the deployed bandit on reward ([+0.032, +0.107]), but uses
  about 9 ms more.
* **AURC does not improve, and worsens as the agent re-runs more** (0.126 → 0.145). The
  enhanced pass's confidences sit on a different scale from the deployed calibration
  map: at displayed confidence ~0.56, enhanced-pass boxes are right 68% of the time,
  first-pass boxes 59%.
* A second-pass isotonic recalibration (out of fold) made AURC unstable (0.112–0.183),
  because tied isotonic blocks dominate the GT-anchored curve. Rejected, and kept in
  the script so the attempt is visible.
* **Consequence for H1:** the image agent is a reward/F1 improvement at matched compute,
  not an AURC improvement. The S_policy registration draft makes reward/F1 primary
  (declared deviation) and AURC a non-inferiority check.

## 2. Live risk-controlled verifier

* `agentic/taxonomy.level_features` holds the combiner features in one function, shared
  by the offline analysis and the live verifier. Week 3's results are reproduced exactly.
* `experiments/week4_fit_verifier.py` fits the per-level combiner on all eligible S_cal
  boxes and writes `verifier.json`.
* The thresholds in that file are **provisional**: Learn-then-Test on out-of-fold S_cal
  probabilities, which carry no guarantee for this combiner.
  * hierarchical: 0.79 / 0.60 / 0.43 / 0.34 at α = 0.10 / 0.15 / 0.20 / 0.25;
  * flat: 0.70 / 0.51 / 0.34 / 0.22.
* `classifier.classify_crops` now also returns the full probability vector.
* `agentic/verifier.taxon_label` attaches a `taxon` answer to every final box: the
  deepest safe level, the node and P(correct), or abstain. The loop calls it when
  `config.AGENT_VERIFIER` is set; it is **off by default**.
* **Checks:**
  * live vs offline, 200 S_cal images: 0 mismatches in 312 boxes, for hierarchical
    α 0.2 (250 answered), flat α 0.2 (222) and hierarchical α 0.1 (188);
  * with the verifier off, the loop is identical to the pre-refactor controller.

## 3. S_policy registration — drafted, not registered

`experiments/PROTOCOL_15_DRAFT.md`:
* frozen files with hashes;
* operating points λ = 0.002 and λ_m = 0.0089;
* Part A: H1-R primary, H1-A non-inferiority, H7 two-step > one-step, H8 vs deployed;
* Part B: a calibration / evaluation dive split of S_policy, H-risk, H4, H5 on
  jellyfish / squid / tunicate, H6.

It waits for approval. A one-step comparator file must be frozen before it is
registered.
