# Step 5 — taxonomy-trained classifier for the verifier (PROTOCOL §17, S_cal only)

**Outcome: NO-GO under the registered rule.** No S_policy test is registered and nothing
is deployed.

Order of events:
* §17 was registered (`db9fa7a`) and amended before any training (`c779d21`: soft-label
  β 1 / 2 instead of 4 / 10, which on this tree were plain cross-entropy);
* four candidates were trained, seed 0, with the deployed recipe;
* `experiments/step5_dev.py` was run once on S_cal. Results: `step5/step5_dev.json`.

The verifier reads each candidate's probabilities on the same crops of the same agent
boxes (the week-3 S_cal log, 889 images, 1,527 boxes). The species candidate stays the
agent's own final class. The deployed classifier's recomputed probabilities match the log
up to float noise (max |diff| 0.012, no top-class change) and reproduce Week 3 exactly.

## Results (hierarchical verifier, α = 0.20, three dive-fold rotations)

| Classifier | S_modelsel top-1 | Risk | Correct/img | **Credit/img** | Species correct/img | Group answers correct | Unseen OOD_dev correct (node) | Wrong species, right group |
|---|---|---|---|---|---|---|---|---|
| deployed (baseline) | 0.853 | 0.120 | 0.996 | **0.891** | 0.846 | 20 | 24 (all Animal) | 19 / 88 |
| CE re-run (noise reference) | 0.855 | 0.125 | 0.985 | 0.870 | 0.816 | 21 | 23 (21 Animal, 2 Chordata) | 22 / 88 |
| soft labels, β = 1 | 0.868 | 0.116 | 0.981 | 0.872 | 0.817 | **44** | 20 (19 Animal, 1 Echinodermata) | **45 / 88** |
| soft labels, β = 2 | 0.856 | 0.114 | 0.980 | 0.883 | 0.841 | 21 | 24 (23 Animal, 1 Echinodermata) | 38 / 88 |
| marginal loss | 0.854 | 0.118 | 0.991 | **0.896** | 0.853 | 19 | 21 (20 Animal, 1 Echinodermata) | 32 / 88 |

Credit/img minus deployed (dive bootstrap, 2,000 resamples, seed 0):

| Candidate | Difference | 95% CI |
|---|---|---|
| CE re-run | −0.021 | [−0.032, −0.010] |
| soft β = 1 | −0.019 | [−0.032, −0.007] |
| soft β = 2 | −0.008 | [−0.017, +0.002] |
| marginal | +0.005 | [−0.007, +0.017] |

**Selection (§17.4):** marginal (highest credit among the eligible losses).
* Credit CI above 0: **no** (−0.007 lower bound).
* Gain over the CE re-run's gain: yes (+0.025).
* Risk ≤ α in every rotation: yes.

So the registered go rule fails.

## What we learned

1. **The tree loss does give the classifier group structure.** When the species answer
   is wrong, the right group is ranked first in 32–45 of 88 cases (19 for the deployed
   classifier). Soft β = 1 doubles the correct group-level answers (44 vs 20).
2. **That structure does not reach unseen animals.** Shark, ray and urchin already get
   the right group ranked first in 19 / 25 cases with the deployed classifier, and 18–20
   with the new ones. The unseen animals the verifier labels correctly are still almost
   all "Animal". The bottleneck is not ranking; it is that the combiner's confidence at
   group level rarely clears the threshold for unseen taxa.
3. **A second classifier costs species answers.** Even a plain retrain (CE re-run) loses
   0.03 species answers/img, because the verifier then judges the agent's class with a
   different model's probabilities. Measured against the CE re-run, the marginal loss
   gains +0.025 credit/img, but the comparison the protocol registered is against the
   deployed classifier.
4. Training with the tree in the loss does not cost species accuracy on S_modelsel (all
   four ≥ 0.854).

## Not tried (would need its own registration)
* Using the taxonomy-trained classifier as the **agent's** classifier too (no mismatch;
  needs calibration, novelty threshold and policies re-fitted and re-frozen).
* More known classes per group (e.g. adding urchin or jellyfish as training classes),
  which is a data change, not a loss change.

Single seed; S_cal is also where the candidate was selected, so all comparisons above
are optimistic for the selected candidate.
