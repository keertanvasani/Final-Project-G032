# Week 6 — confirmatory S_test run (PROTOCOL §16), app support

Order of events:
* the verifier's thresholds were fixed by Learn-then-Test on all of S_policy (frozen
  combiner);
* §16 was registered with every hash and threshold (`d96c82f`);
* the test script was committed (`483c998`) after a dry run on S_policy;
* S_test was logged for these components for the first time;
* the test was run **once**.

This is the **second use of S_test** (the first was §12, Step 3). S_test is now closed
for these components. Results: `week6/S_test_confirmatory.json`.

S_test: 1,170 images, 473 dives.

## Part A — image agent

| Policy | Reward/img | F1 | Precision | Recall | AURC | ms/img | Re-runs |
|---|---|---|---|---|---|---|---|
| rule agent | +0.555 | 0.641 | 0.656 | 0.627 | 0.137 | 11.2 | — |
| deployed bandit | +0.609 | 0.660 | 0.668 | 0.652 | 0.111 | 1.7 | — |
| **agent, λ_m** | **+0.665** | **0.663** | 0.698 | 0.631 | 0.114 | **7.5** | 15% |
| agent, λ = 0.002 | +0.687 | 0.656 | 0.723 | 0.600 | 0.115 | 19.8 | 47% |
| one-step, λ = 0.002 | +0.622 | 0.643 | 0.694 | 0.599 | 0.121 | 14.7 | 33% |

| Hypothesis | S_test | (S_policy, §15) | Verdict |
|---|---|---|---|
| **H1-R** vs rules: reward and F1 | reward **+0.110** [+0.075, +0.145]; F1 **+0.022** [+0.013, +0.031], at 7.5 vs 11.2 ms | +0.136 / +0.029 | **confirmed** |
| H1-A AURC non-inferior | −0.023 (upper 95% bound −0.013): **better** | −0.008 | **confirmed** |
| **H7** two-step vs one-step | **+0.065** [+0.034, +0.095] | +0.081 | **confirmed** |
| **H8** vs deployed bandit | **+0.078** [+0.038, +0.117] at +18 ms | +0.102 | **confirmed** |

## Part B — verifier, S_policy thresholds applied unchanged to all S_test dives

Today's agent: dive-level risk **0.278**.

| α | Verifier | Threshold | Risk | ≤ α | Bootstrap resamples ≤ α | Correct/img | Wrong among reported |
|---|---|---|---|---|---|---|---|
| 0.10 | hierarchical | 0.72 | 0.094 | ✓ | 73% | 1.107 | 13.8% |
| 0.10 | flat | 0.64 | 0.095 | ✓ | 68% | 0.985 | 15.0% |
| 0.20 | hierarchical | 0.38 | 0.183 | ✓ | 92% | 1.262 | 24.0% |
| 0.20 | flat | 0.27 | 0.187 | ✓ | 85% | 1.199 | 26.3% |

| Hypothesis (α = 0.20) | S_test | (S_policy) | Verdict |
|---|---|---|---|
| **H-risk** | risk ≤ α in all four settings | ✓ | **confirmed**, with **less margin** than on S_policy (0.094 vs 0.055 at α = 0.10) |
| **H4** hier > flat, correct/img | **+0.062** [+0.042, +0.084] | +0.108 | **confirmed** |
| **H5** coarse labels on unseen OOD_test animals | 12 of 42 [5, 20]: 11 "Animal", 1 group-level | 13 / 41 | **confirmed**, almost all "Animal" |
| H6 species non-inferior (margin 0.05) | −0.053, lower bound −0.066 | −0.036 | **not supported**, again |

The tighter margin on S_test is expected. The thresholds come from a different split,
and the guarantee holds with probability ≥ 1 − δ over calibration sets, not on every
test set. The realised risk still stayed below α everywhere.

## App support (off by default)

* `config.AGENT_VERIFIER = {"mode": "hierarchical", "alpha": 0.2}` (or 0.1) turns the
  verifier on. It uses `week6/verifier_deploy.json`: the frozen combiner plus the
  S_policy thresholds confirmed here.
* Box labels then read as follows:
  * `starfish 91%` for a species answer;
  * `Echinodermata 84% (~starfish)` or `Animal 89% (~fish)` for a coarse answer;
  * `needs review (~machine)` when the verifier abstains.
* The decision trace explains every coarse answer and abstention. When the image agent
  keeps its second pass, the trace says which lines belong to which pass.
* `"image"` in front of `config.AGENT_POLICY_CHAIN` turns the image agent on.
* **Checks:**
  * live vs offline verifier with the deploy thresholds: 0 mismatches (150 S_cal images,
    3 settings);
  * everything off: identical to the pre-refactor controller on 844 S_policy and 26 demo
    runs.
