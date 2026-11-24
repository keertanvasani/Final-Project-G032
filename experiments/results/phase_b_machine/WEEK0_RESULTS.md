# Week 0 — probe signal and headroom on S_cal (PROTOCOL §13)

Exploratory go/no-go check for the multi-step agent controller. **S_cal only**
(889 images, 315 dives, 1,527 base boxes). S_policy and S_test were not read. Criteria
were registered before the run (commit `8b3b8cb`).

Scripts: `experiments/week0_log_probes.py` (full-information logs, 281 s on the RTX 5050
laptop), `experiments/week0_analyse.py` (analysis). Results: `week0/week0_results.json`.
The base pipeline (deployed learned policy) reproduces the Step 3 logs exactly.

## Result: GO — but only for the image agent

Reward per image, +1 TP / −1 reported error / 0 Unknown. Base pipeline: 0.980 TP,
0.577 errors, **net +0.403 per image**. Gains are out-of-fold (5-fold, dive-grouped),
policy = act iff predicted Δ > 0, 95% dive-bootstrap intervals.

| Action | Always act | Oracle headroom | Step 3 features | + probes | 95% CI | probes − features (CI) | Pass |
|---|---|---|---|---|---|---|---|
| CLAHE 1.5 | +0.022 | 0.200 | +0.018 | +0.049 | [+0.005, +0.090] | [+0.000, +0.062] | ✓ |
| **CLAHE 2.0** | +0.004 | 0.226 | +0.004 | **+0.080** | [+0.034, +0.125] | [+0.037, +0.114] ✓ | ✓ |
| **CLAHE 3.0** | −0.024 | 0.240 | +0.002 | **+0.076** | [+0.028, +0.124] | [+0.031, +0.116] ✓ | ✓ |
| gamma | +0.019 | 0.132 | −0.015 | +0.047 | [+0.012, +0.081] | [+0.030, +0.094] ✓ | ✓ |
| grey-world | −0.008 | 0.190 | −0.030 | +0.057 | [+0.015, +0.098] | [+0.049, +0.125] ✓ | ✓ |
| re-detect 960 px | −0.100 | 0.209 | −0.022 | +0.029 | [−0.011, +0.068] | | – |
| 2×2 tiling | **−0.326** | 0.044 | 0.000 | −0.008 | [−0.014, −0.002] | | – |
| zoom (per box) | −0.012 | 0.038 | −0.009 | +0.001 | [−0.009, +0.012] | | – |
| vote (per box) | +0.004 | 0.027 | +0.009 | +0.009 | [−0.010, +0.027] | | – |
| flip TTA (per box) | +0.006 | 0.042 | −0.006 | −0.007 | [−0.021, +0.007] | | – |

Descriptive, not registered: one policy choosing the best of the five enhancements
gains +0.070 [+0.022, +0.119] per image (acts on 63% of images). That is no better than
CLAHE 2.0 alone, so picking among noisy arm estimates costs what the extra arms add.

**Learned combiner: ADOPT.** OOF AUROC for "box is correct if reported":
calibrated confidence 0.811, logistic on the existing fields 0.832, + probes 0.833.
The gain, +0.021 [+0.009, +0.034], comes from learning the combination, not from the
probes.

## What it means

1. **Enhancement helps when chosen after looking at the first detection pass.**
   With quality-only features, which is how Step 3 decided *before* detection, no
   enhancement helps (≈ 0). With the first pass's evidence (box confidences, mirror and
   shift agreement, centroid distance, colour cast), re-detecting on a CLAHE-enhanced
   image gains +0.08 net reward per image, about **+20% on the base +0.403**, for
   ~12 ms per image. This is the observe → act → re-observe loop, and it explains why
   Step 3 found nothing.
2. **The policy still captures only about a third of the headroom.** The oracle for
   CLAHE 2.0 is 0.226 (the best of the five combined is 0.404).
3. **Box-level actions have almost no headroom on this data** (oracle ≤ 0.04 per
   image), and no state predicts them. Tiling and 960 px hurt on 480×360 frames. The
   per-box detection and identification agents have nothing worth deciding here; they
   are dropped from the controller for this dataset (they may matter on HD footage).

## Caveats

* Exploratory: ten actions tested without multiplicity correction. The passing lower
  bounds are small (+0.005 to +0.034). Confirmation must come from S_policy with
  registered hypotheses.
* Fixed ridge strength 10, threshold 0; no tuning. The image-level probes are 54
  features from 889 images, so overfitting is possible but out-of-fold.
* Enhancement after a first pass costs a second detection: about +12 ms per image when
  acting, vs ~27 ms per image today.

## Revised plan

* **Controller:** a two-step image agent. Pass 1 is the base pipeline; probes are
  computed; the agent decides whether to re-run on an enhanced image (CLAHE 2.0 first,
  the menu as an ablation); the improvement-only commit rule keeps the better of the
  two. Offline RL (fitted Q-iteration) over these two steps, with λ for compute.
* **Verifier:** learned combiner → taxonomy levels → Learn-then-Test risk control
  (idea 1), unchanged.
* Box-level zoom, vote and flip stay as today's rules/bandit; tiling and 960 px are
  dropped.

## Correction (Week 2)

Week 2 split the "probes" state into its parts (`WEEK2_RESULTS.md`). The gain comes
from the first pass's own outputs (box count, confidences, disagreement, Unknown share)
and colour statistics. The mirror, shift and embedding crops add nothing measurable.
"Probes credited" above should be read as "first-pass evidence credited".
