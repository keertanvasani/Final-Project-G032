# External validation on FathomNet (PROTOCOL §18): results

Single registered run of `experiments/test_fathomnet.py` on 2026-11-25. Every input
(the models, the verifier, the policies, the JODD thresholds, the manifest and the
concept-to-tree mapping) was frozen by hash before any FathomNet image was read.
Raw output: `fathomnet_test.json`.

**Data:** 2,897 FathomNet ROV frames (depth ≥ 200 m) from 963 dives, all
sha256-verified and resized to 480 px wide.
- **Part A:** 2,173 images from 850 dives. Split by dive into F_cal (425 dives) and
  F_eval (425 dives: 1,114 images, 1,660 matched boxes, 392 of them unseen animals).
  This is above the §18.7 minimum, so the tests are confirmatory.
- **Part B:** 859 images from 301 dives where every annotation is verified.

## Headline

| Claim | Result |
|---|---|
| **E1**: re-calibrate on F_cal, risk ≤ α on F_eval | **Holds at both levels and for both verifiers**: hierarchical risk 0.065 at α 0.10 and 0.143 at α 0.20 |
| **E2**: JODD thresholds carried over unchanged | **Fails, as predicted**: risk 0.18 at α 0.10 and 0.33 at α 0.20. Calibration does not transfer across datasets. |
| **E3**: hierarchical beats flat on correct answers per image | **Supported**: +0.24 per image (95% CI 0.21–0.27, Holm-significant) |
| **E4**: some unseen animals get a correct coarse answer | **Supported**: 63 of 392 unseen boxes (95% CI 46–82). Flat answers can never do this. |

## E1: re-calibrated thresholds on F_eval

Thresholds learned on F_cal: hierarchical 0.90 (α 0.10) and 0.81 (α 0.20); flat 0.88
and 0.78. These are much stricter than the JODD values of 0.72 / 0.38 and 0.64 / 0.27.

| α | Verifier | Risk | Resamples with risk ≤ α | Answers per image | Correct per image | Species-level correct per image | Unseen animals correct |
|---|---|---|---|---|---|---|---|
| 0.10 | hierarchical | 0.065 | 99.9% | 0.59 | **0.50** | 0.21 | 62 |
| 0.10 | flat | 0.057 | 100% | 0.33 | 0.26 | 0.26 | 0 |
| 0.20 | hierarchical | 0.143 | 100% | 0.81 | **0.65** | 0.37 | 63 |
| 0.20 | flat | 0.144 | 100% | 0.57 | 0.40 | 0.40 | 0 |

The hierarchical verifier gets its extra correct answers by stepping back. At α 0.10 it
gives 236 species-level, 68 group-level and 257 "animal"-level answers. The flat
verifier answers less often, and when it does answer, its error rate is higher (24%
against 14% at α 0.10).

## E2: JODD thresholds unchanged

| α | Hierarchical risk | Flat risk |
|---|---|---|
| 0.10 | 0.183 | 0.210 |
| 0.20 | 0.329 | 0.369 |

Both verifiers break the guarantee by roughly 2×, and no bootstrap resample comes in
under α. This is what §18.5 predicted: the guarantee holds only after re-calibration
on data from the new setting.

## Detector and classifier transfer (descriptive)

- **Detector recall on known classes: 0.26.** It ranges from 0.11 (shrimp) to 0.49
  (machine). FathomNet frames are 1080p reduced to 480 px, so many annotated animals
  shrink to a few pixels. Recall is also understated in Part A, because annotation
  there is not exhaustive.
- **Classifier top-1 on matched known boxes: 0.74.** By class: crab 0.93, machine 0.94,
  fish 0.85, sponge 0.77, coral 0.73, starfish 0.72, shrimp 0.70, crinoid 0.54,
  sea-cucumber 0.41.

## Part B: fully verified images (descriptive, §18.6)

**Verifier with F_cal thresholds:**

| α | Verifier | Risk | Resamples with risk ≤ α | Correct per image |
|---|---|---|---|---|
| 0.10 | hierarchical | 0.086 | 92% | 0.58 |
| 0.10 | flat | 0.076 | 99% | 0.29 |
| 0.20 | hierarchical | 0.173 | 96% | 0.62 |
| 0.20 | flat | 0.174 | 95% | 0.42 |

With the JODD thresholds, Part B risk is 0.25 at α 0.10 (hierarchical and flat alike),
so E2's failure shows up again.

**Agents:**

| Policy | Reward per image | F1 | Precision | Recall | AURC | Extra ms per image |
|---|---|---|---|---|---|---|
| Step 3 rules | −0.489 | 0.278 | 0.360 | 0.226 | 0.435 | 30 |
| Deployed bandit | −0.432 | **0.303** | 0.382 | 0.251 | 0.439 | 6 |
| Two-step agent (λ_m) | −0.347 | 0.294 | 0.395 | 0.234 | 0.429 | 19 |
| Two-step agent (λ) | **−0.250** | 0.289 | **0.416** | 0.222 | **0.422** | 39 |
| One-step agent (λ) | −0.311 | 0.295 | 0.403 | 0.233 | 0.436 | 30 |

**95% bootstrap intervals for the differences:**

| Comparison | Reward | F1 | Extra ms |
|---|---|---|---|
| λ_m agent − rules | +0.06 to +0.22 | −0.005 to +0.037 | −14 to −7 |
| λ agent − deployed bandit | +0.13 to +0.24 | −0.028 to −0.001 | +30 to +37 |
| Two-step − one-step | +0.02 to +0.10 | −0.019 to +0.004 | +7 to +12 |

The agents rank on reward the same way they did on JODD. The extra reward comes from
precision, through more re-runs and refusals, and costs a little F1 compared with the
deployed bandit.

## Deviation

The §18.5 descriptive tables "with the image agent on" are not computed. The image-agent
logs do not store classifier probabilities for second-pass boxes, so the verifier cannot
be replayed on them. E1–E4 are unaffected. This deviation is also recorded in
`fathomnet_test.json`.

## Takeaway for the report

On data from another institute, other vehicles, another ocean and another annotation
scheme:
- the hierarchical verifier's main advantage holds, with about twice as many correct
  answers per image as flat;
- it can still name coarse groups for animals it has never seen;
- the risk guarantee needs a small re-calibration set from the new setting, about 425
  dives here. Without one, the guarantee fails.
- The detector is the weak link out of domain: known-class recall is 0.26, mainly
  because small objects shrink further when frames are resized to 480 px.
