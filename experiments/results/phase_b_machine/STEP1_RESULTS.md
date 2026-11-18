# Step 1 — cheap score fixes on the deep-sea stack (2026-11-17)

Next-semester plan, step 1. No retraining. Every setting is fitted on **S_cal**; every number
below is measured on **S_policy** (844 images, 315 dives, 1,308 known-class animals). S_test is
not touched. Bracketed ranges are paired 95% dive-cluster bootstrap intervals.

Scripts: `experiments/step1_deepsea_uncertainty.py`, `experiments/fit_deepsea_calibration.py`,
`experiments/tune_wbf_cutoff.py`, `experiments/ablation_seed_wbf.py`, `experiments/ablation_deepsea.py`.

## Outcome

| Fix | Result | Decision |
|---|---|---|
| Re-measure AURC / ECE / AUROC | AURC 0.188, ECE 0.269, correctness AUROC 0.832 | Baseline for the paper |
| Isotonic calibration (global) | ECE 0.269 → **0.033** [−0.251, −0.197] | **Deployed** |
| Isotonic calibration (per class) | ECE 0.043, but AURC worse [+0.008, +0.025] | Rejected |
| Flip TTA on the classifier | accuracy 81.9% → 82.3%, +0.4 pt [−0.3, +1.2] | Rejected (2× classifier cost) |
| Energy + Mahalanobis novelty | AUROC 0.769 vs energy 0.784 [−0.045, +0.016] | Rejected |
| Mahalanobis alone | AUROC 0.721 [−0.107, −0.015] vs energy | Rejected |
| Novelty threshold refit on detector boxes | −3.14 → **−3.65** | **Deployed** |
| Second seed + tuned cut-off | F1 +0.012 [−0.004, +0.028] vs tuned single seed, +40% latency | Rejected |

## 1. Deployed confidence, re-measured

Full agent (enhancement, text filter, refuse gate, zoom, vote), Known/Unknown not dropped:
1,459 detections — 822 TP, 251 LOC_ERR, 152 OOD_ERR, 145 BG_FP, 89 CLS_ERR (56.3% correct).

| | AURC ↓ | ECE ↓ | Brier ↓ | AUROC (correct) ↑ |
|---|---|---|---|---|
| Raw combined confidence | 0.188 | 0.269 | 0.253 | 0.832 |
| Temperature (T = 5.93) | 0.188 | 0.142 | 0.200 | 0.832 |
| **Isotonic, global** | **0.187** | **0.033** | **0.166** | 0.831 |
| Isotonic, per predicted class | 0.203 | 0.043 | 0.165 | 0.834 |

The raw score ranks well but is over-confident. Global isotonic makes the shown percentage an
honest P(correct) without changing the ranking. Per-class maps re-order detections across classes
and lose ranking quality (AURC +0.008 to +0.025), so they are not used.

Implementation note: combined confidence saturates at exactly 1.0 for hundreds of detections
(agreement boost cap). The isotonic fit must map tied inputs to one value (their pooled accuracy,
0.90), otherwise interpolation returns the last block's 1.0. Fixed in both scripts.

## 2. Flip test-time augmentation

911 crops whose ground truth is a known class. Single crop 81.9% vs mean of crop + mirror logits
82.3% (+0.4 pt [−0.3, +1.2]); 23 predictions changed. Max-probability AUROC for correctness
0.761 → 0.781. Not worth doubling classifier cost.

## 3. Known/Unknown score

Matched detections: 911 known-class vs 152 unseen-taxon (11 taxa: comb-jelly, jellyfish, octopus,
ragworm, ray, sea-spider, shark, squat-lobster, squid, tunicate, urchin). Threshold keeps 95% of
S_cal known detections.

| Score | AUROC | FPR@95TPR | Unseen flagged | Known flagged |
|---|---|---|---|---|
| **Energy (deployed)** | **0.784** | 0.711 | 14.5% | 4.9% |
| Energy, flip-averaged | 0.788 [−0.009, +0.018] | 0.617 | 21.1% | 5.4% |
| Mahalanobis (unit features) | 0.721 [−0.107, −0.015] | 0.791 | 32.2% | 13.5% |
| Energy + Mahalanobis (z-sum) | 0.769 [−0.045, +0.016] | 0.649 | 34.2% | 10.8% |

The old threshold (−3.14) was fitted on hand-drawn ground-truth crops and flagged only 8.6% of
unseen and 2.6% of known detections. Refitted on the agent's own S_cal detections it is −3.65,
which hits the intended 5% false-alarm budget. Even so, energy misses most unseen animals — the
motivation for the foundation-model fallback in step 4.

Effect on the full pipeline (`ablation_deepsea.py`, row F):

| Threshold | Precision | Recall | OOD_ERR | BG_FP | Detections |
|---|---|---|---|---|---|
| −3.14 (old) | 0.588 [0.553, 0.623] | 0.620 [0.581, 0.661] | 140 | 129 | 1,379 |
| **−3.65 (new)** | **0.603** [0.568, 0.640] | 0.611 [0.571, 0.651] | **131** | **114** | 1,324 |

## 4. Second detector seed

Equal-weight WBF of seed 0 with seed 1 or seed 2 (`ablation_seed_wbf.py`), full agent:
precision −0.075 / −0.056, recall +0.023 / +0.031 (all p < 0.01), F1 0.604 → 0.571 / 0.586,
~1.5× latency.

With a minimum detection confidence tuned on S_cal for each system (`tune_wbf_cutoff.py`,
grid 0.30–0.60, maximise F1; seed 0 best 0.45, fused best 0.35):

| System (S_policy) | Precision | Recall | F1 | ms/image |
|---|---|---|---|---|
| Seed 0, no cut-off (deployed) | 0.603 | 0.611 | 0.607 | 51 |
| Seed 0, cut-off 0.45 | 0.695 | 0.557 | 0.618 | 51 |
| Seed 0 + seed 1, no cut-off | 0.527 | 0.634 | 0.575 | 82 |
| Seed 0 + seed 1, cut-off 0.35 | 0.703 | 0.570 | 0.630 | 72 |

Tuned fused − tuned single: precision [−0.011, +0.028], recall [−0.005, +0.032],
F1 [−0.004, +0.028] (p = 0.15) — break-even at +40% latency, so not adopted. Most of the
fused system's gain over the deployed one comes from the cut-off itself, and for a single
seed that cut-off trades recall for precision (−5 / +9 pts, F1 p = 0.09); the app's confidence
slider already lets a user make that trade.

## Deployed changes

* `models/phase_b_machine/calibration.json` — isotonic map (36 knots); the agent's
  `combined_confidence` is now calibrated and the raw value is kept as `raw_combined_confidence`.
* `models/phase_b_machine/novelty.json` — threshold −3.65 (old value kept as `threshold_gt_crops`;
  backup `novelty_gt_crops_backup.json`). Copies of both are in `step1/`.
* The HF Space replays recordings made before this change; re-record to show calibrated values.
