# Week 5 — registered S_policy test of the image agent and the verifier (PROTOCOL §15)

Order of events:
* the one-step comparator was frozen on S_cal (`5a77c48e…`);
* §15 was registered (`d418667`);
* the test script was committed (`835c1b2`) after a dry run on S_cal;
* S_policy was logged for these components for the first time;
* the test was run **once**.

Hashes of all frozen files were checked by the script.
Results: `week5/S_policy_test.json`.

S_policy: 844 images, 315 dives. Paired dive bootstrap, 2,000 resamples, seed 0; Holm at
0.05 within each part.

## Part A — image agent (exact replay)

| Policy | Reward/img | F1 | Precision | Recall | AURC | Discretionary ms/img | Re-runs |
|---|---|---|---|---|---|---|---|
| rule agent | +0.325 | 0.607 | 0.603 | 0.611 | 0.138 | 11.3 | — |
| deployed bandit (app today) | +0.380 | 0.626 | 0.621 | 0.631 | 0.124 | 2.1 | — |
| **agent, λ_m (matched to rules on S_cal)** | **+0.461** | **0.636** | 0.660 | 0.615 | 0.130 | **10.1** | 17% |
| agent, λ = 0.002 | +0.482 | 0.628 | 0.683 | 0.582 | 0.125 | 24.0 | 49% |
| one-step, λ = 0.002 | +0.402 | 0.611 | 0.643 | 0.583 | 0.127 | 17.3 | 34% |

| Hypothesis | Result | Verdict |
|---|---|---|
| **H1-R** agent at λ_m vs rules: reward and F1 higher | reward **+0.136** [+0.069, +0.195]; F1 **+0.029** [+0.011, +0.045]; p = 0.001 | **supported** (Holm), with slightly *less* compute (10.1 vs 11.3 ms) |
| H1-A AURC non-inferior (margin 0.01) | −0.008, upper 95% bound +0.003 | **supported** |
| **H7** two-step vs one-step, reward | **+0.081** [+0.039, +0.124], p = 0.0005 | **supported**: the second decision matters |
| **H8** agent (λ = 0.002) vs deployed bandit, reward | **+0.102** [+0.047, +0.161], p < 0.001 | **supported** |

Notes, stated plainly:
* The gains are reward and precision. At λ = 0.002 recall falls (0.631 → 0.582), and F1
  against the deployed bandit is unchanged (0.628 vs 0.626). The agent mostly keeps the
  enhanced pass when it reports fewer, surer boxes. At λ_m, F1 rises over the rules.
* AURC did not worsen on S_policy, unlike the trend seen on S_cal.

## Part B — risk-controlled verifier (calibration on 157 dives, evaluation on 158)

Today's agent on the evaluation half: dive-level risk **0.273**.

| α | Verifier | Threshold | Risk (eval) | ≤ α | Correct/img | Species correct/img | Wrong among reported |
|---|---|---|---|---|---|---|---|
| 0.10 | hierarchical | 0.85 | 0.055 | ✓ (99.9% of resamples) | 0.765 | 0.535 | 8.4% |
| 0.10 | flat | 0.77 | 0.057 | ✓ (99.8%) | 0.657 | 0.657 | 11.3% |
| 0.20 | hierarchical | 0.57 | 0.113 | ✓ (100%) | 0.930 | 0.786 | 18.3% |
| 0.20 | flat | 0.40 | 0.145 | ✓ (99.8%) | 0.822 | 0.822 | 24.3% |

| Hypothesis (α = 0.20) | Result | Verdict |
|---|---|---|
| **H-risk** | realised risk ≤ α in all four settings | **supported** |
| **H4** hierarchical > flat, correct answers/img | **+0.108** [+0.070, +0.142] | **supported** |
| **H5** correct coarse labels on unseen OOD_val animals | 13 of 41 [4, 23] | **supported**, but **all 13 are "Animal"** (12 jellyfish, 1 squid); none reached Cnidaria or Chordata |
| **H6** species answers non-inferior (margin 0.05) | −0.036, lower 95% bound −0.054 | **not supported** (narrowly), as flagged as possible in §15 |

## What this establishes

1. **The two-step image agent is a confirmed improvement over the rule agent at no extra
   compute**, and over the deployed bandit at +22 ms. Two-step beats one-step
   (H7 confirmed), so the sequential (RL) formulation is necessary, not decorative.
2. **The verifier delivers a per-dive error guarantee on unseen dives.** The risk
   target held at every α tested, against today's 0.273. At α = 0.20 the hierarchical
   verifier gives 0.93 correct answers per image with 18% wrong among reported.
3. **The taxonomy adds correct answers but not informative ones** on this classifier.
   They come through "Animal", and species answers fall slightly (H6 fails). The
   group-structure limitation found in Week 3 is confirmed out of sample.

## Status

S_policy is now **used** for these components and must not be re-run or tuned on. A
confirmatory S_test run would be the second use of S_test and needs its own registration.
Neither component is deployed (`AGENT_POLICY_CHAIN` and `AGENT_VERIFIER` unchanged).
