# The §23 re-fit on FathomNet Part B, with false detections (PROTOCOL §24): results

**Descriptive only.** This was a single run on 2026-11-27. The data was the Part B
(all-VERIFIED) images whose dive is not an F_cal dive: 602 images, 197 dives and 1,137
final boxes. **426 of those boxes are false detections** (unmatched). Fits and thresholds
are those of §23; the plumbing check reproduced every §23 parameter. M5 got thresholds
0.67 and 0.55. Raw output: `partb.json` and `partb_run.txt`.

## Results

| α | Method | Risk | ≤ α | Correct per image | Species-correct per image | Credit per image | False boxes labelled (of 426) | Unseen wrong |
|---|---|---|---|---|---|---|---|---|
| 0.20 | **M1 frozen JODD combiner** | **0.169** | ✓ | 0.551 | 0.326 | 0.405 | **87** | 75 |
| 0.20 | M2 re-fit, all features | 0.410 | ✗ | 0.983 | 0.221 | 0.524 | 375 | 58 |
| 0.20 | M3 re-fit, classifier only | 0.360 | ✗ | 1.017 | 0.090 | 0.485 | 382 | 29 |
| 0.20 | M4 DARTS | 0.225 | ✗ | 0.561 | 0.397 | 0.443 | 201 | 66 |
| 0.20 | M5 re-fit with unmatched boxes | 0.059 | ✓ | 0.610 | 0.003 | 0.246 | 65 | 4 |
| 0.10 | **M1 frozen** | **0.093** | ✓ | 0.502 | 0.228 | 0.324 | **67** | 31 |
| 0.10 | M2 re-fit, all features | 0.315 | ✗ | 1.058 | 0.042 | 0.436 | 360 | 9 |
| 0.10 | M3 re-fit, classifier only | 0.311 | ✗ | 1.045 | 0.000 | 0.416 | 366 | 11 |
| 0.10 | M4 DARTS | 0.136 | ✗ | 0.395 | 0.282 | 0.319 | 134 | 25 |
| 0.10 | M5 re-fit with unmatched boxes | 0.013 | ✓ | 0.299 | 0.000 | 0.112 | 21 | 1 |

Paired differences are in `partb.json`. M2 − M1 is positive in correct answers and in
credit at both α levels, but M2 does not meet its risk target, so these comparisons are
not like for like.

## What it says

1. **The §23 re-fit advantage was an artefact of matched-only scoring.**
   * Once false detections count, M2 and M3 label almost all of them (375–382 of 426),
     mostly as "Animal".
   * Their risk rises to 0.31–0.41, about twice the target. §24.5 predicted this.
2. **DARTS also breaks the target on Part B**, at 0.136 and 0.225. Its §20 win on
   FathomNet was also measured on matched boxes only. DARTS sees only the classifier's
   softmax, so it cannot tell a false box from a real animal. It labels 201 of the
   false boxes.
3. **Only the frozen JODD combiner, M1, meets the target at both levels**, at 0.093 and
   0.169. It labels the fewest false boxes of any method that still answers usefully
   (67–87).
   * It was fitted on JODD logs that included false detections.
   * Its detector-side features (detector confidence, agreement, box size) are what let
     it hold back on false boxes.
   * The §23 finding that "the classifier-only re-fit matches the full re-fit" is true
     only when no false boxes are scored.
4. **M5 (re-fit with unmatched boxes counted wrong) is safe but over-cautious.**
   * Its risk is 0.013 and 0.059.
   * It gives almost no species answers, and less than half M1's credit at α 0.10.
   * The cause is that Part A's unmatched boxes include real, unannotated animals, which
     teaches it that coarse answers are risky. Part A is not good data for fitting a
     combiner.

## Revised conclusions for the paper (§20, §23, §24 together)

* **With false detections scored, the hierarchical verifier and its JODD-fitted combiner
  are the only method that keeps the guarantee out of domain.** It needs only the
  threshold re-calibrated on the new site; the combiner can stay as it is.
* **DARTS's apparent FathomNet advantage (§20) and the re-fit's apparent advantage (§23)
  both disappear** once unmatched boxes count. Report all three results together, with
  that explanation.
* The detector-side features are what protect against false detections. This supports
  the multi-model design (detector + classifier + agreement) over classifier-only
  uncertainty methods.
* **Caveat.** Part B is descriptive (§18.6). "VERIFIED" does not prove that every animal
  in the frame is boxed, so some of the 426 unmatched boxes may be real animals. This
  affects every method the same way.
