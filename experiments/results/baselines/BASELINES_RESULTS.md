# Literature baselines for the verifier (PROTOCOL §20, §20.5): results

**Descriptive only.** These comparisons use the same boxes as earlier sections, and the
same calibration data and evaluation data. Every method with one parameter has that
parameter set by the same Learn-then-Test procedure (Hoeffding–Bentkus, δ = 0.10, dive as
the unit). Conformal prediction uses its own calibration rule.

| Run | Output |
|---|---|
| Registered run (§20) | `baselines.json` |
| Post-hoc strict-grid extension for softmax response and DARTS (§20.5) | `baselines_extended_grid.json` |

Settings:

| Setting | Calibration | Evaluation |
|---|---|---|
| **JODD** | S_policy | S_test: 1,170 images, 473 dives, 2,306 boxes, 155 of them unseen animals |
| **FathomNet** | F_cal | F_eval: 1,114 images, 425 dives, 1,660 boxes, 392 of them unseen animals |

## α = 0.20 (the level of H4 and E3)

Correct answers per image (risk in brackets; ✗ means the risk is above α):

| Method | JODD | FathomNet |
|---|---|---|
| **Ours, hierarchical** | **1.262** (0.183) | 0.646 (0.143) |
| Ours, flat | 1.199 (0.187) | 0.405 (0.144) |
| Softmax response (Geifman & El-Yaniv 2017) | 1.138 (0.187) | 0 † / 0.403 ‡ (0.099) |
| Energy score (Liu et al. 2020) | 1.109 (0.174) | 0.496 (0.128) |
| Calibrated confidence, the deployed gate | 1.150 (0.180) | 0.388 (0.139) |
| **DARTS** (Deng et al. 2012), calibrated by our Learn-then-Test | 1.215 (0.179) | **0.741** (0.150) |
| Split conformal + LCA (Sadinle et al. 2019) | 1.127 (0.180) | 0.941 (**0.323 ✗**) |

† The registered grid found no admissible threshold, so the method abstained on every
box. ‡ The §20.5 extended grid.

Ours (hierarchical) minus each baseline, in correct answers per image (95% paired dive
bootstrap):

| Baseline | JODD | FathomNet |
|---|---|---|
| DARTS | **+0.047** [+0.020, +0.074] | **−0.094** [−0.149, −0.037] |
| Best flat method | +0.062 [+0.042, +0.084] (ours, flat) | +0.150 [+0.105, +0.197] (energy) |

## α = 0.10

Rows marked ‡ use the §20.5 extended grid; on the registered grid, softmax response and
DARTS abstain on every box.

| Method | JODD | FathomNet |
|---|---|---|
| **Ours, hierarchical** | **1.107** (0.094) | **0.504** (0.065) |
| Ours, flat | 0.985 (0.095) | 0.256 (0.057) |
| Softmax response ‡ | 0.821 (0.071) | 0.289 (0.056) |
| Energy score | 0.815 (0.072) | 0.307 (0.056) |
| Calibrated confidence | 0.869 (0.077) | 0.237 (0.077) |
| DARTS ‡ | 0.942 (0.079) | 0.491 (0.056) |
| Split conformal + LCA | 1.249 (**0.315 ✗**) | 0.877 (**0.141 ✗**) |

Ours minus DARTS ‡: **+0.165** [+0.122, +0.205] on JODD, and **+0.013** [−0.035, +0.064]
on FathomNet, which is a tie.

## Unseen animals at α = 0.20

| Method | JODD: correct / wrong (of 155) | FathomNet: correct / wrong (of 392) |
|---|---|---|
| Ours, hierarchical | 37 / 90 | 63 / **143** |
| DARTS | 25 / 60 | 41 / 201 |
| Flat methods | 0 / 66–109 | 0 / 131–185 |
| Conformal + LCA | 0 / 75 | 48 / 289 |

## What the results say

1. **On JODD, ours is best at both α levels**, against every method that meets the risk
   target. DARTS comes closest: +0.047 for ours at α 0.20 and +0.165 at α 0.10.
2. **On FathomNet, DARTS ties with ours at α 0.10 and beats it at α 0.20** (−0.094).
   * DARTS gives more species-level answers (0.49 against 0.37 per image).
   * It also gives 58 more unseen animals a wrong answer (201 against 143).

   Our combiner learns from the detector's confidence, the detector–classifier agreement
   and energy, all fitted on JODD. These signals seem to shift on FathomNet footage. DARTS
   uses only the classifier's softmax, so it has less to go wrong out of domain.
3. **Conformal + LCA breaks the error target in 3 of 4 cases**: risk 0.315 and 0.141 at
   α 0.10, and 0.323 at α 0.20 on FathomNet. Its guarantee covers known classes only,
   so unseen animals and false detections break it. §20.4 predicted this.
4. **Every flat method gives fewer correct answers than hierarchical answers** (DARTS or
   ours). This repeats H4 and E3 against published methods.
5. **Softmax response needs a threshold above 0.999** before it can meet α. The classifier
   is badly over-confident, which is why the registered grid failed.

## Implications for the paper

* The combination "hierarchical answers + Learn-then-Test per dive" carries the result.
  DARTS calibrated by our procedure is a strong, simple baseline.
* Our combiner adds value within the training domain (JODD), and does not consistently
  add value outside it (FathomNet). State this as it is.
* Natural follow-up: re-fit the combiner on F_cal (its features may shift out of domain),
  or combine DARTS's node posteriors with the combiner. This would be a new
  registration, with no further use of F_eval for confirmatory claims.

## Process notes

* The plumbing check before the run reproduced all 8 frozen thresholds for ours, on both
  settings.
* §20.5 was registered after the §20 run, which is why it is labelled post-hoc. It
  changes only the strict end of two grids.
* The first `--extended` run wrote to `baselines.json` by mistake: an edit to the output
  path had not applied. Its output was moved to `baselines_extended_grid.json`, and
  `baselines.json` was restored byte for byte from commit 9b69d32, ignoring CRLF line
  endings. The computation was not repeated.
