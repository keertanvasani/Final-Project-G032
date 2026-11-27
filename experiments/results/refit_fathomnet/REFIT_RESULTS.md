# Re-fitting the combiner on FathomNet (PROTOCOL §23): results

**Descriptive only.** This was a single run on 2026-11-27, and the fourth use of F_eval.
F_cal was split by dive:

| Part | Boxes | Dives with boxes | Used for |
|---|---|---|---|
| F_fit | 833 | 170 | fitting the combiner |
| F_thr | 821 | 171 | Learn-then-Test |

Evaluation used all of F_eval: 1,660 boxes and 1,114 images. The plumbing check
reproduced the §18 thresholds on F_cal. Raw output: `refit.json` and `run.txt`.

## Results

| α | Method | Threshold / λ | Risk | Correct per image | Species-correct per image | Credit per image * | Unseen wrong (of 392) |
|---|---|---|---|---|---|---|---|
| 0.20 | M1 frozen JODD combiner | 0.81 | 0.143 | 0.646 | 0.370 | 0.463 | 143 |
| 0.20 | **M2 re-fit, all features** | 0.67 | 0.163 | **1.251** | 0.174 | **0.612** | 106 |
| 0.20 | M3 re-fit, classifier only | 0.64 | 0.149 | 1.245 | 0.045 | 0.552 | **98** |
| 0.20 | M4 DARTS | λ 1,738 | 0.123 | 0.692 | **0.442** | 0.515 | 184 |
| 0.10 | M1 frozen | 0.89 | 0.074 | 0.526 | 0.235 | 0.333 | 86 |
| 0.10 | **M2 re-fit, all features** | 0.79 | 0.064 | **1.329** | 0.020 | **0.515** | 37 |
| 0.10 | M3 re-fit, classifier only | 0.79 | 0.067 | 1.318 | 0.000 | 0.498 | 35 |
| 0.10 | M4 DARTS | λ 27,542 | 0.055 | 0.486 | 0.276 | 0.340 | 115 |

\* Credit per image is the specificity-weighted score: 1 for a species answer, less for
coarser nodes. It is produced by the §20 `metrics` function but was **not one of §23.3's
listed measures**. It is shown because "correct per image" alone rewards coarse answers.

Paired differences in correct answers per image (95% dive bootstrap):

| Comparison | α 0.10 | α 0.20 |
|---|---|---|
| M2 − M1 | +0.803 [+0.696, +0.918] | +0.605 [+0.518, +0.699] |
| M3 − M1 | +0.792 [+0.686, +0.902] | +0.599 [+0.513, +0.689] |
| M2 − DARTS | +0.844 [+0.742, +0.952] | +0.559 [+0.486, +0.641] |
| M3 − DARTS | +0.832 [+0.732, +0.936] | +0.553 [+0.481, +0.629] |

## What it says

1. **The FathomNet gap to DARTS came from the JODD-fitted combiner.**
   * Re-fitted on about 170 FathomNet dives, our verifier gives about twice the correct
     answers of DARTS and of the frozen combiner, at both α levels.
   * Its error rate stays under α at both levels.
   * It also gives fewer unseen animals a wrong answer: 106 against DARTS's 184.
2. **The gain comes from answering almost every box at a coarser level.**
   * The re-fit answers 1.4 of the 1.49 boxes per image, mostly at group level
     (Echinodermata, Decapoda, Porifera and others) or as "Animal".
   * Species answers fall from 0.37 to 0.17 per image. The re-fit has learned that the
     JODD classifier's species call is unreliable on FathomNet (top-1 0.74), while its
     group call usually holds.
   * On credit, which weights answers by specificity, the re-fit still leads:
     0.61 against 0.52 for DARTS and 0.46 for frozen at α 0.20.
3. **The detector features are not what shifted.** M3 (classifier features only) is
   about equal to M2 (all features). This goes against the §23.5 expectation that the
   detector-side features shift. The gain comes from re-learning how much to trust each
   *level*, not from dropping detector signals.
4. **The re-fit's gain is overstated by one important caveat.** Part A scores only boxes
   matched to an annotation (§18.5), so there are no false detections. "Animal" is then
   almost always correct, and a combiner fitted on these boxes learns to say it freely.
   In live use, false boxes exist, and a combiner trained this way would call them
   "Animal". The frozen JODD combiner was fitted with false boxes included. **So the
   re-fit's advantage is an upper bound** until it is tested on data with unmatched
   boxes, such as Part B, which is fully verified.

## For the paper

* State it plainly: our combiner needs a small fit set from the new domain.
  With one, it beats DARTS out of domain by a wide margin. Without one, DARTS is the
  better choice.
* Report the trade-off: the re-fit gets far more correct answers and fewer unseen
  errors, but far fewer species answers. Include the matched-boxes caveat.
* **Next step, if wanted:** a registered test on FathomNet Part B, which includes false
  detections. It would re-fit and calibrate on Part A F_cal and check whether the gain
  survives. It needs no new data.
