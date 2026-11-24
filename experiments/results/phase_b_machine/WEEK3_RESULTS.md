# Week 3 — taxonomy-aware, risk-controlled verifier (S_cal development check)

Design registered first (PROTOCOL §14, commit `617ba78`). **S_cal only**, descriptive:
889 images, 1,527 boxes of the deployed agent, 275 dives with boxes. Following §14.3,
84 boxes matched to other splits' held-out taxa (jellyfish, squid, tunicate, ragworm,
sea-spider, squat-lobster) were excluded from every fit and calibration. 78 boxes of
S_cal's own role taxa (comb-jelly, octopus, ray, shark, urchin) remain.

Code:
* `agentic/taxonomy.py`: tree, levels, correctness, credit.
* `experiments/log_verifier_inputs.py`: every final box with the full classifier
  distribution.
* `experiments/week3_verifier.py`: the analysis.

Results are in `week3/week3_results.json`.

Protocol: three dive folds, rotated. Per-level logistic combiner on A,
Learn-then-Test on B (Hoeffding–Bentkus, δ = 0.10, fixed-sequence over t = 0.99 … 0),
measured on C. Each box is scored once, from the rotation where it was in C.

## 1. Risk control works

Today's agent reports every non-Unknown box. Its dive-level risk (the fraction of a
dive's boxes given a wrong answer) is **0.276**, and 37.1% of the answers it reports
are wrong, with 0.979 correct answers per image.

| α | Flat verifier: risk on C | ≤ α in all 3 rotations | Correct / img | Wrong among reported |
|---|---|---|---|---|
| 0.10 | 0.042 | yes | 0.651 | 11.5% |
| 0.15 | 0.073 | yes | 0.809 | 15.9% |
| 0.20 | 0.112 | yes | 0.900 | 22.0% |
| 0.25 | 0.165 | yes | 0.948 | 27.6% |

The realised risk is always below α. It is conservative, about half of α at the
strict end, because Hoeffding–Bentkus on ~90 calibration dives has little power. At
α = 0.25 the verifier keeps 97% of today's correct answers and cuts the wrong share of
reported answers from 37% to 28%, **with a stated per-dive guarantee** that today's
agent does not have.

## 2. The hierarchy adds answers, but they are almost all "Animal"

Registered single-threshold hierarchy vs flat, same α (dive bootstrap, 95% CI):

| α | Correct / img, hier − flat | Credit / img, hier − flat | Unseen-animal boxes answered correctly | Species answers (hier vs flat) |
|---|---|---|---|---|
| 0.10 | **+0.146** [+0.121, +0.174] | −0.026 [−0.044, −0.007] | 27 / 78 (flat 0) | 474 vs 579 |
| 0.15 | **+0.083** [+0.059, +0.108] | −0.037 [−0.054, −0.021] | 20 / 78 | 636 vs 719 |
| 0.20 | **+0.096** [+0.069, +0.122] | −0.009 [−0.023, +0.004] | 24 / 78 | 752 vs 800 |
| 0.25 | **+0.092** [+0.068, +0.118] | +0.001 [−0.014, +0.014] | 21 / 78 | 812 vs 843 |

* **More correct answers at the same guaranteed risk**, and 20–35% of unseen-animal
  boxes get a correct coarse label instead of a wrong species or nothing. This supports
  the H4/H5 direction.
* **But the information value is not higher.** Specificity credit is equal at α ≥ 0.20
  and lower at α ≤ 0.15, and species answers fall (H6 fails at strict α). Coarse answers
  use risk budget, which raises the shared threshold.
* **Development variants (not registered, reported so the search is visible):**
  * *Two thresholds on independent calibration halves:* species first, then coarse. It
    is worse everywhere, because the halved calibration makes the species threshold
    stricter. Rejected.
  * *Hierarchy without the "Animal" level:* the gain vanishes (correct answers −0.004
    to +0.007 per image; unseen animals 0–2 / 78). Almost all of the hierarchy's value
    comes from the **"Animal" fallback**. Group and phylum answers rarely clear the
    threshold, because the classifier's summed probabilities carry little group
    structure. That is the risk `docs/idea.md` named.

## Verdict

* **Strong and ready to register:** the learned combiner plus Learn-then-Test verifier
  with the dive as the unit. It is a guaranteed per-dive error rate for an open-set
  detector, held in every rotation, at small cost in correct answers.
* **Real but modest:** taxonomy-aware abstention. It gives more correct answers and
  correct coarse labels for unseen animals, but through "Animal", not through
  informative groups. It is not a strong claim on this classifier.
* **What would make the taxonomy strong:** a classifier whose probabilities carry group
  structure, i.e. one retrained with a hierarchical loss (Step 5). It can then be
  re-tested with this exact harness.

## Next

Register the S_policy test (frozen combiner + t per α from S_cal, or Learn-then-Test
calibrated on a dive half of S_policy, to be decided in the registration), with:
* H-risk: realised risk ≤ α on held-out dives;
* H4 as correct answers per image;
* H5 on S_policy's role taxa (jellyfish, squid, tunicate);
* H6 species non-inferiority.
