# Option D: image agent + verifier together (PROTOCOL §21): results

**Descriptive only.** This was a single run on 2026-11-27, on S_policy: 844 images,
315 dives. It used the verifier's frozen combiner and the S_policy thresholds. S_test was
not used. S_policy is where those thresholds were chosen, so the absolute risks are
in-sample and optimistic. The figure that matters is the paired difference between the
two systems.

| System | Final boxes |
|---|---|
| Bandit + verifier (the comparison, `week3/S_policy_verifier.jsonl`) | 1,463 |
| **Option D**: image agent at λ_m + verifier (`S_policy_verifier.jsonl`, sha256 db49d58d…, local only, gitignored) | 1,371 |

## Hierarchical verifier

| α | Risk: bandit | Risk: option D | Difference (95% CI) | Correct per image: bandit | Correct per image: option D | Difference (95% CI) |
|---|---|---|---|---|---|---|
| 0.10 | 0.090 | 0.098 | +0.008 [+0.001, +0.016] | 0.947 | 0.938 | −0.009 [−0.022, +0.006] |
| **0.20** | **0.184** | **0.187** | **+0.003 [−0.008, +0.015]** | 1.031 | 1.002 | **−0.028 [−0.050, −0.008]** |

The flat verifier behaves the same way. Its risk is 0.099 and 0.191 under option D,
against 0.093 and 0.186 for the bandit.

## Reading, by the rule registered in §21.4

* **At α 0.20, the guarantee carries over.** Option D's risk is 0.187, which is ≤ 0.20,
  and the confidence interval for the risk difference includes 0. The app keeps
  option D.
* **The cost is a few correct answers.** Option D gives 0.028 fewer correct answers per
  image (about 3%). The image agent keeps the enhanced pass when that pass reports fewer,
  surer boxes. This raises detection precision and reward (H1-R / H8 on S_test), but it
  leaves the verifier fewer boxes to name (1,371 against 1,463).
* **At α 0.10, the margin is thin.** Risk rises by +0.008, with a confidence interval
  just above 0, to 0.098, which is very close to 0.10 even in-sample. Do not deploy
  option D at α 0.10 without re-calibrating the thresholds on option-D boxes.

## App checks

* **Pinned experiment scripts.** Every app-mode experiment script now calls
  `config.pin_pre_d()`, so it reproduces the committed logs. Both live checks still gave
  0 mismatches after the change: the verifier check on 40 S_cal images, and the image
  agent check on 40 images, with 21 re-runs.
* **Smoke test with option D on.** `run_agentic_pipeline` on 20 S_cal images:
  * every box carries a verifier label (22 species, 1 group, 4 "needs review");
  * 3 of 20 images were re-run;
  * the median is 56 ms per image, for the whole app call.

## Process note

The progress lines of the logging run were lost, because the `tee` target folder did
not exist yet. `log_run.txt` holds only the final lines, which were copied from the
console.
