# Consolidated results

Marine organism recognition under domain shift and open-set conditions.
Every figure below is reproducible from the committed code, splits and
manifests. Numbers are labelled by the split they come from; `J_test` has been
read once and `S_test` not at all.

Last updated 2026-09-06.

---

## 0. Status key

| Mark | Meaning |
|---|---|
| **final** | measured out-of-sample on a development split, method frozen |
| *exploratory* | measured, but in-sample or on a small sample — directional only |
| pending | running or not yet started |

Nothing here is a confirmatory test result. The confirmatory family (§11) is
tested once, on the held-out test split, after all design is frozen.

---

## 1. Headline findings

1. **An in-domain-validated system fails badly out of domain.** The aquarium
   pipeline scores mAP@50 0.78 in its own domain and produces a selective risk
   of **0.888** on deep-sea imagery — at its operating point it is wrong
   roughly nine times in ten.

2. **Confidence carries almost no information about correctness under shift.**
   Risk stays near 0.89 across the useful coverage range and the floor at
   minimum coverage is ~0.65. No threshold makes the system reliable.

3. **Confidence carries no usable information about novelty.** Every
   confidence-derived OOD score sits at or below chance (AUROC 0.436–0.533).
   Unseen taxa receive *higher* confidence than correct detections.

4. **On-domain training moves the novelty signal into the logits.** With the
   aquarium classifier only the feature geometry carried signal (Mahalanobis
   0.650 near-OOD, energy 0.553 — useless, because its 7 logits describe
   aquarium species). Retrained on deep-sea taxa, **energy reaches 0.806
   near-OOD and 0.816 far-OOD**, out-of-sample.

5. **A learned state-conditional policy beats the tuned threshold.** At matched
   30% coverage, selective risk falls **0.846 → 0.776** (95% CI on the
   difference **[+0.049, +0.093]**), and the policy operates across the full
   coverage range rather than the fixed gate's 14.7% ceiling.

6. **The policy learned that novelty matters more than confidence.** Its
   standardised weight on novelty distance (**−3.03**) is roughly five times
   its weight on calibrated confidence (+0.62), and classifier confidence is
   ignored entirely (−0.03) — independent corroboration of finding 3.

7. **None of the four discretionary actions improves the system.** Forced-action
   measurement on both platforms: `ENHANCE` is **actively harmful** (−0.057
   TP/image on Phase B, and −0.107 on the images its own trigger selects),
   `ZOOM` is inert with a negative point estimate on the retrained detector,
   `RECLASSIFY` beats the detector but not the classifier, and
   `FULL_IMAGE_REDETECT` is not implemented at all. The hand-written adaptive
   layer is not merely unvalidated — it is a net negative, and removing it
   improves accuracy while saving compute.

8. **The system notices only the degradation it was coded to check.** Under
   controlled in-domain degradation, blur triggers abstention 1.6% → 83.2% and
   risk stays nearly flat. Colour shift and compression raise risk to 0.537 and
   0.610 while abstention barely moves — and colour shift is the degradation
   that dominates underwater imaging.

9. **The abstention rule was never the bottleneck.** Four image-level
   aggregations are statistically indistinguishable as screens (AUC spread
   0.005). A negative result that redirects attention to perception and
   uncertainty.

10. **The supplied dataset splits leak.** 57.3% of the supplied test images
    share a dive with training images. All results use dive-disjoint re-splits,
    which is why they are lower than, and not comparable to, numbers computed on
    the supplied division.

---

## 2. Experimental setup

### 2.1 Data

| | Aquarium Combined | Jedi Organism Detection |
|---|---|---|
| Source | Roboflow `brad-dwyer/aquarium-combined` v2, CC BY 4.0 | supplied |
| Domain | aquarium tanks | deep-sea ROV transects |
| Images | 638 (448 train / 190 val) | 8,151 |
| Dives | — | 3,152 |
| Annotations | — | 15,621 |
| Classes | 7 | 20 |

### 2.2 Splitting

Splits are assigned by **dive group**, never by image. Frames from one dive
share camera, illumination, substrate and often the same individual organism
seconds apart, so an image-level split leaks.

Audit of the supplied Jedi division (`audit_group_leakage.py`) measured **57.3%
test contamination**. It was discarded and all 8,151 images pooled and re-split.

**Phase A** (no training on Jedi) — three splits:

| Split | Images | Role |
|---|---|---|
| `J_cal` | 2,445 | calibration + meta-confidence fitting (hosts `OOD_dev`) |
| `J_policy` | 2,445 | policy learning (hosts `OOD_val`) |
| `J_test` | 3,261 | final evaluation — **read once** (hosts `OOD_test`) |

**Phase B** (on-domain retraining) — five splits, proportioned over dives:

| Split | Dives | Dive % | Images | Annotations | Known | Held-out | Far-OOD |
|---|---|---|---|---|---|---|---|
| `S_fit` | 1,734 | 55.0% | 4,506 | 8,711 | 5,601 | 856 | 2,254 |
| `S_modelsel` | 315 | 10.0% | 742 | 1,367 | 845 | 135 | 387 |
| `S_cal` | 315 | 10.0% | 889 | 1,573 | 966 | 222 | 385 |
| `S_policy` | 315 | 10.0% | 844 | 1,529 | 911 | 221 | 397 |
| `S_test` | 473 | 15.0% | 1,170 | 2,441 | 1,522 | 268 | 651 |

### 2.3 Label spaces

**Phase A** (aquarium model mapped onto Jedi): KNOWN = fish, jellyfish, shark,
starfish (1,538 test annotations, 33.9%); OOD = the other 16 Jedi classes
(3,005, 66.1%). `ray` is treated as OOD rather than as `stingray`.

**Phase B** (Jedi-native): KNOWN (8) = coral/sea-anemone, crab, crinoid, fish,
sea-cucumber, shrimp, sponge, starfish. HELD-OUT UNSEEN (11) = comb-jelly,
jellyfish, octopus, ragworm, ray, sea-spider, shark, squat-lobster, squid,
tunicate, urchin. FAR-OOD (1) = machine.

`jellyfish` and `shark` are KNOWN in Phase A and held out in Phase B, so
per-class figures are **not comparable across phases**.

### 2.4 Correctness and matching

Greedy by descending combined confidence, class-agnostic, one GT per detection,
IoU ≥ 0.50 primary. `correct = (IoU ≥ 0.50) ∧ (predicted_class == gt_class)`.

Error taxonomy: TP; CLS_ERR (matched, wrong known class); OOD_ERR (matched a GT
outside the label space); LOC_ERR (unmatched, IoU ∈ [0.10, 0.50)); BG_FP
(unmatched, IoU < 0.10).

### 2.5 Uncertainty quantification

Cluster bootstrap over dives, ≥1,000 resamples. Dives are resampled with
replacement so frames from one dive are never split across a resample.

### 2.6 Environment

Python 3.12.10 · torch 2.11.0+cu128 · torchvision 0.26.0+cu128 ·
ultralytics 8.4.138 · numpy 2.3.5 · OpenCV 5.0.0 ·
NVIDIA RTX 5050 Laptop (8,151 MiB, sm_120).

---

## 3. Baseline in its own domain — **final**

```
mAP@50                      0.78
Classes                     7
Training images             638
Full-pipeline inference     217.4 ms
Model load                  4,557 ms
```

Measured action costs (30 images, CPU, 8.13 objects/image mean, max 27):

| Stage | mean | median | p95 | max |
|---|---|---|---|---|
| preprocessing | 7.10 ms | 3.05 | 10.26 | 91.00 |
| detection | 88.97 ms | 84.72 | 103.53 | 153.47 |
| WBF | 0.42 ms | 0.32 | — | — |

Accounting residual 0.03 ms, confirming full coverage of the pipeline.

---

## 4. Baseline under domain shift — **final** (`J_test`, single read)

3,261 images · 4,343 detections · 2,003 in-label-space GT objects.

| | images | share |
|---|---|---|
| produced ≥1 detection | 2,115 | 64.9% |
| produced none | 1,146 | 35.1% |

| Outcome | count | share |
|---|---|---|
| TP — correct known class | 465 | 10.7% |
| CLS_ERR — wrong known class | 263 | 6.1% |
| **OOD_ERR — organism outside label space** | **913** | **21.0%** |
| LOC_ERR — poorly localised | 883 | 20.3% |
| BG_FP — nothing there | 1,819 | 41.9% |

Operating point, per-detection acceptance at τ = 0.40:

| Metric | Value | 95% CI |
|---|---|---|
| accepted detections | 3,946 | — |
| **selective risk** | **0.888** | [0.876, 0.900] |
| selective recall (GT-anchored) | 0.221 | [0.206, 0.264] |
| coverage (policy-local) | 0.909 | — |
| **OOD-error rate among accepted** | **0.213** | [0.197, 0.232] |
| AURC vs recall | 0.803 | [0.770, 0.836] |
| AURC vs coverage | 0.834 | — |

Risk floor at minimum coverage ≈ 0.65. The OOD-error panel *increases* toward
low coverage: the most confident deep-sea detections are among the most likely
to be confident misidentifications of organisms never trained on.

---

## 5. Abstention aggregation — **final** (`J_cal`, 2,445 images)

Recoverable images (≥1 known-class TP): 337 (13.8%).

| Aggregation | AUC | retain @ τ=0.4 | recall | yield |
|---|---|---|---|---|
| mean (original gate) | 0.720 | 58.3% | 95.0% | 22.5% |
| median | 0.716 | 58.2% | 95.0% | 22.5% |
| top-3 mean | 0.718 | 58.4% | 95.3% | 22.5% |
| **max (per-box, current)** | **0.721** | 59.7% | 95.8% | 22.1% |

**Interpretation.** Spread is 0.005 AUC — the four are indistinguishable as
screens. The often-quoted 23.7% → 6.3% change is a change in *rejection rate at
a fixed threshold*, not evidence that the original gate discriminated worse.
The abstention rule is not where reliability is lost.

---

## 6. Calibration — **final** (fit `J_cal`, evaluate `J_policy`)

Fit: 3,259 detections, 343 correct (10.5%). Eval: 3,255 detections, 297 correct
(9.1%). Raw mean confidence **0.5582** against a base rate of **0.0912** —
overconfident by **+0.467**.

| Method | ECE | MCE | Brier | NLL | AUROC | mean pred |
|---|---|---|---|---|---|---|
| raw (uncalibrated) | 0.4669 | 0.7716 | 0.3076 | 0.8570 | 0.6414 | 0.5582 |
| temperature | 0.4088 | 0.4088 | 0.2500 | 0.6931 | 0.6414 | 0.5000 |
| Platt (a=0.265, b=−2.258) | 0.0190 | 0.4106 | 0.0817 | 0.2991 | 0.6414 | 0.1048 |
| **isotonic** | **0.0160** | 0.1536 | **0.0810** | **0.2936** | 0.6469 | 0.1057 |
| *trivial constant = base rate* | *0.0140* | — | *0.0831* | *0.3065* | *n/a* | *0.0912* |

**Three findings, two of them cautionary.**

- **Temperature scaling degenerates.** T → ∞ and every score collapses to 0.5
  (mean pred 0.5000, NLL = ln 2). Temperature rescales logits but has no bias
  term, so it cannot correct a level error of +0.47. This is a property of the
  method against this miscalibration, not a fitting failure.

- **The headline ECE reduction is largely regression to the base rate.** A
  constant predictor that ignores the input achieves ECE 0.0140 — *better* than
  isotonic's 0.0160. Brier skill score of isotonic against that constant is
  **+0.0254**. The defensible claim is that calibration makes the score honest
  *while preserving ranking*, which the constant cannot do at all.

- **AUROC is essentially unchanged** (0.6414 → 0.6469). Calibration repairs what
  a score *means*, never how well it *ranks*. Ranking is the agent's problem.

---

## 7. Out-of-distribution detection

### 7.1 Confidence-derived scores — **final**, small sample

| Score | `J_cal` AUROC | FPR@95 | `J_policy` AUROC | FPR@95 |
|---|---|---|---|---|
| MSP (classifier conf) | 0.4693 | 0.9810 | 0.5327 | 0.9663 |
| detector conf | 0.4356 | 0.9684 | 0.6337 | 0.8764 |
| combined conf (deployed) | 0.4383 | 0.9684 | 0.6186 | 0.9213 |
| disagreement-weighted | 0.4918 | 0.9620 | 0.5028 | 0.9663 |

Samples are 74 OOD vs 158 ID (`J_cal`) and 44 vs 178 (`J_policy`) after
§5.1 fit-eligibility filtering. Values flip sign between splits, so treat as
**directional only**: confidence carries no reliable novelty signal, and
FPR@95 ≈ 0.96–0.98 means catching 95% of unseen taxa would discard nearly all
correct detections.

### 7.2 Feature-space, ground-truth crops — *exploratory* (in-sample)

Fitted and scored on the same split, therefore reporting training error.
Retained for the record; §7.4 supersedes it with fit/eval separation.

| | `J_cal` AUROC | FPR@95 | `J_policy` AUROC | FPR@95 |
|---|---|---|---|---|
| Mahalanobis far-OOD | 0.9451 | 0.2717 | 0.9441 | 0.2792 |
| Mahalanobis near-OOD | 0.7741 | 0.7233 | 0.7967 | 0.6975 |
| energy far-OOD | 0.6223 | 0.9250 | 0.6025 | 0.9458 |
| energy near-OOD | 0.5160 | 0.9250 | 0.4706 | 0.9417 |

### 7.3 Feature-space, detector-emitted boxes — **final**

Mahalanobis fitted on 509 in-distribution detections from `J_cal`.

| Split | Regime | AUROC | FPR@95 | n_ID | n_OOD |
|---|---|---|---|---|---|
| `J_cal` *(in-sample)* | far-OOD | *0.9953* | *0.0118* | 509 | 141 |
| `J_cal` *(in-sample)* | near-OOD | *0.9101* | *0.6267* | 509 | 212 |
| **`J_policy`** | **far-OOD** | **0.7948** | 0.5294 | 476 | 161 |
| **`J_policy`** | **near-OOD** | **0.6733** | 0.8088 | 476 | 109 |

**Quote the `J_policy` rows.** The in-sample/out-of-sample gap is large
(near-OOD 0.910 → 0.673) and is the honest measure of generalisation.

**Interpretation.** Feature distance substantially outperforms every
confidence-derived score (0.673–0.795 against ~0.47). Energy score, which reads
the same classification head as MSP, barely helps — so with the *aquarium*
classifier the failure is attributable to the head, not the backbone. FPR@95
remains poor (0.81 near-OOD): at 95% novelty recall, 81% of correct detections
are also discarded, so this is not deployable at high recall.

### 7.4 Does on-domain training improve the feature space? — **final**

Both classifiers scored on identical splits with the label space held fixed, so
the classifier is the only variable. Fitted on `S_cal`, reported on `S_policy`.

| Score | aquarium features | retrained features | Δ |
|---|---|---|---|
| **near-OOD, energy** | 0.5533 | **0.8059** | **+0.253** |
| near-OOD, Mahalanobis | 0.6504 | 0.7127 | +0.062 |
| **far-OOD, energy** | 0.6414 | **0.8161** | **+0.175** |
| far-OOD, Mahalanobis | **0.8163** | 0.7178 | −0.099 |

**Interpretation.** Retraining helps, but through a different mechanism than
expected. The aquarium classifier's logits describe 7 *aquarium* species, so
energy — which reads logit magnitude — is meaningless here, and only the
feature geometry carries signal. The retrained classifier's logits span the 8
relevant deep-sea taxa, so "activates none of these strongly" becomes genuine
evidence of novelty. **On-domain training moves the novelty signal out of the
feature geometry and into the logits.**

Best near-OOD result overall: **0.806**, retrained features with energy score.

### 7.5 A methodological correction worth recording

On the retrained classifier, Mahalanobis initially scored **0.42 far / 0.44
near** out-of-sample — below chance — while scoring 0.87/0.90 in-sample on the
same fit. That was withheld rather than reported, on the grounds that a gap
that size indicates a broken estimator rather than anti-informative features.

The suspected cause (a fixed `1e-3` ridge being negligible against the feature
scale) was **wrong**. The covariance is full rank, 512 of 512, and every
shrinkage variant made the score monotonically worse:

| variant | far AUROC | near AUROC |
|---|---|---|
| euclidean to mean | 0.1890 | 0.2175 |
| diagonal only | 0.1942 | 0.2210 |
| relative ridge, eps=0.50 | 0.3079 | 0.3291 |
| absolute ridge, eps=1e-3 *(original)* | 0.4205 | 0.4391 |
| **L2-normalised + relative ridge 0.10** | **0.7178** | **0.7127** |
| kNN cosine, k=10 | 0.7809 | 0.7494 |

Plain Euclidean distance scoring 0.19 is the diagnostic: the retrained network
assigns **smaller-magnitude features to novel objects**, so any distance using
magnitude reads "novel" as "close to the mean" and inverts. Normalising to the
unit sphere discards the magnitude channel and recovers the signal. Aquarium
features are barely affected by normalisation (0.8196 → 0.8163), consistent
with the mechanism — a weakly-fit model does not concentrate feature norms the
same way.

`ood_features.py` now normalises before fitting and scales the ridge to mean
per-dimension variance. §7.4 reports the corrected figures.

---

## 8. Selective prediction with an uncertainty-aware gate — **final** (`J_policy`)

3,255 detections · 297 correct (9.1%) · 666 OOD_ERR (20.5%).
Mahalanobis threshold m\* = 307.5, chosen on `J_cal` at 90% ID retention.

Areas are integrated over the common coverage range **[0, 0.147]** that all
three gates attain; areas over unequal ranges are not comparable.

| Gate | AURC\* | coverage | risk | OOD-error rate |
|---|---|---|---|---|
| baseline (raw confidence) | 0.1102 | 0.147 | 0.7945 | 0.3732 |
| calibrated | 0.0836 | 0.118 | 0.7807 | 0.3786 |
| **calibrated + OOD-aware** | 0.0890 | 0.147 | **0.7317** | **0.1824** |

Paired dive bootstrap at matched coverage 0.147:

```
risk       0.7945 -> 0.7317
OOD rate   0.3732 -> 0.1824
delta risk = +0.0632   95% CI [+0.0194, +0.1091]
```

**Interpretation.** The CI excludes zero, and the OOD-error rate among accepted
detections is **more than halved**. This is the H3 quantity, measured on a
development split.

**Caveat.** The operating point is 14.7% coverage with residual risk 0.73 — the
gate accepts about one detection in seven and is still wrong most of the time.
The hard mask is what caps coverage; §8.2 removes that ceiling.

### 8.2 Learned state-conditional policy — **final** (`J_policy`)

A logistic model over nine features (calibrated confidence, detector and
classifier confidence, agreement, box area, aspect ratio, novelty distance,
image blur, scene crowding) predicts P(correct); the gate accepts when that
clears τ. The effective bar therefore varies per detection, which a global
constant cannot express.

No exploration data is required: whether accepting was correct is already
recorded for every cached detection, so the counterfactual problem that blocks
ZOOM and RECLASSIFY does not arise for the accept/abstain decision.

| Gate | max coverage | AURC (full) | risk @ 30% | OOD @ 30% |
|---|---|---|---|---|
| baseline (raw confidence) | 1.000 | 0.8575 | 0.8462 | 0.2964 |
| calibrated | 1.000 | 0.8501 | 0.8369 | 0.3017 |
| calibrated + OOD (fixed 2-D) | **0.147** | 0.0890 | 0.7317 | 0.1824 |
| **LEARNED policy** | **1.000** | **0.7874** | **0.7756** | 0.2674 |

Paired dive bootstrap against the baseline at matched coverage:

```
@ 30% coverage:  risk 0.8462 -> 0.7756   delta +0.0708  95% CI [+0.0493, +0.0932]
@ 50% coverage:  risk 0.8801 -> 0.8384   delta +0.0419  95% CI [+0.0288, +0.0555]
```

**Standardised feature weights** (negative pushes toward ABSTAIN):

| feature | weight | | feature | weight |
|---|---|---|---|---|
| **novelty distance** | **−3.027** | | aspect ratio | +0.320 |
| box area | +0.744 | | image blur | +0.298 |
| calibrated confidence | +0.619 | | detector confidence | +0.218 |
| detector/classifier agreement | +0.399 | | classifier confidence | −0.027 |
| scene crowding | −0.359 | | | |

**Interpretation.** Both CIs exclude zero, so the state-conditional policy
accepts fewer wrong detections than the tuned threshold at the same coverage.
The weights are the more interesting result: the policy relies on novelty
distance roughly **five times** more than on calibrated confidence, and ignores
classifier confidence entirely — learning from data what §4 and §7.1 argued
separately.

**Caveats.** Removing the coverage ceiling is a structural consequence of using
a score rather than a hard mask, not evidence that learning caused it. And the
fixed 2-D gate still achieves a lower OOD rate (0.182) than the learned policy
at 30% coverage (0.267) — it is more aggressive on novelty, at a coverage too
low to be useful. The gates trade off rather than one dominating.

---

## 9. Controlled degradation on in-domain data — **final**

190 aquarium validation images, seven degradations at three severities, ground
truth unchanged, running the frozen baseline. Isolates one factor at a time, in
contrast to §4 where quality, taxa, optics and illumination all shift together.

| Degradation | risk clean → severe | abstention clean → severe |
|---|---|---|
| **gaussian blur** | 0.364 → **0.393** | 1.6% → **83.2%** |
| noise | 0.364 → 0.681 | 1.6% → 30.5% |
| motion blur | 0.364 → 0.319 | 1.6% → 26.8% |
| low contrast | 0.364 → 0.533 | 1.6% → 11.1% |
| low light | 0.364 → 0.489 | 1.6% → 10.0% |
| **colour shift** | 0.364 → **0.537** | 1.6% → **2.1%** |
| **compression** | 0.364 → **0.610** | 1.6% → **3.7%** |

**Interpretation.** Gaussian blur is the success case: abstention rises to 83%
and risk stays nearly flat, so the system correctly refuses rather than
guessing. It is also the *only* degradation the pipeline explicitly tests for
(blur variance < 5.0). Under colour shift and compression, risk rises
substantially while abstention stays at its clean-image level — the system does
not notice its evidence has degraded.

**Hand-written quality rules catch only what their author thought to check
for.** Colour shift is the most damaging and is exactly what depth does to
underwater imagery, since red attenuates fastest — giving a mechanistic account
of part of the §4 failure rather than attributing it to domain shift in
aggregate.

**Caveat.** Selective risk is conditioned on the accepted set, so when
abstention rises the survivors are an easier population. Motion blur's apparent
risk *decrease* (0.364 → 0.319 at 26.8% abstention) is that selection effect,
not an improvement. Risk must be read alongside its abstention rate.

---

## 10. Are the discretionary actions worth their cost? — **final**

The decision trace records what the rules *did*, never what would have happened
otherwise, so it cannot answer whether an action earned its compute. Every
logged zoom is one the rule chose to fire; there is no record of a zoom that
would have helped but was not taken. A policy fitted on that data can only
re-derive the existing thresholds.

`log_zoom_outcomes.py` and `log_reclassify_outcomes.py` produce the missing
counterfactual: each action is forced on **every** detection, not only those in
the rule's trigger band, re-detected and re-classified, and scored against
ground truth. Forcing exhaustively rather than sampling ε-greedily costs one
extra pass per detection but removes sampling noise, which matters because the
effects turn out to be small.

Both actions are measured on **both platforms** — the frozen aquarium stack
(base accuracy 10.9%) and the retrained Phase B stack (51.4%) — to test whether
any inertness is a property of the action or merely of a model that was failing
anyway.

### 10.1 ZOOM — no measurable benefit on either platform

| | Phase A (`J_policy`) | Phase B (`S_policy`) |
|---|---|---|
| forced actions | 3,253 | 1,343 |
| candidate box found | 32.2% | 72.7% |
| rule would fire | 19.7% | 24.9% |
| **rule actually changed a box** | **0.3%** | **4.5%** |
| correct without zoom | 0.1085 | 0.5138 |
| correct under the rule | 0.1091 | 0.5101 |
| **net effect** | **+0.00064** | **−0.00376** |
| 95% CI | [−0.00124, +0.00251] | [−0.01007, +0.00221] |
| cost on the split | 18.9 s / 641 firings | 9.6 s / 335 firings |

Both intervals include zero. On Phase A the rule fires 641 times, spends 18.9
seconds and alters the outcome of **9 detections**.

On Phase B the action becomes *more* active and the point estimate turns
**negative**. Of its 29 outcome changes, **11 are worse and 6 better** —
`TP → CLS_ERR` six times, `TP → LOC_ERR` five. A better detector produces
better boxes, which the zoom is then more likely to damage than improve.

`Q(s, ZOOM)` is not fittable: 15 outcome changes on the Phase A fit split is no
signal. `fit_action_value.py` detects this and declines to fit rather than
reporting a model of noise.

### 10.2 RECLASSIFY_LARGER_CROP — beats the detector, not the classifier

Two structural facts narrow the question. On **agreement** the vote already has
a majority of two, so the loose opinion provably cannot change the outcome. And
the action never moves the box, so it can only flip TP ↔ CLS_ERR. The
measurable question is whether, on disagreements, buying a third opinion beats
trusting one of the two already in hand.

Correctness on disagreements:

| policy | Phase A | Phase B | pays extra compute? |
|---|---|---|---|
| **rule** (majority vote) | 0.1066 | 0.2404 | yes |
| classifier only | 0.1000 | 0.2236 | no |
| detector only | 0.0555 | 0.1755 | no |

Paired dive bootstrap, rule minus each free alternative, over all detections:

| | Phase A | Phase B |
|---|---|---|
| vs detector only | **+0.02172** [+0.0137, +0.0311] | **+0.02033** [+0.0024, +0.0384] |
| vs classifier only | +0.00271 [−0.0006, +0.0065] | +0.00517 [−0.0062, +0.0161] |

**The rule clearly beats the detector on both platforms**, so the classifier
stage earns its place. The *expensive* part — the third opinion and the vote —
is not measurably better than simply taking the classifier's class. On Phase A
it buys **+9 correct detections for 7.1 s**; on Phase B, **+7 for 2.4 s**.
Neither gain is distinguishable from zero.

A mechanism worth recording: in 8.2% of Phase A firings the loose crop returns a
*third* distinct class, the vote deadlocks, and the rule defaults to the
**detector** — the worst of the three policies. There the extra compute is not
merely wasted, it steers the answer toward the weakest option.

### 10.3 The image-level actions — ENHANCE is actively harmful

ENHANCE and FULL_IMAGE_REDETECT act on the whole frame, so their counterfactual
compares whole pipeline runs: enhancement changes the pixels, the detector may
return a different number of boxes, and detections cannot be paired across
variants. The outcome is therefore counted **per image** — how many correct
detections an image yields under each variant. Three variants run on every
image, so the rule's trigger never decides what gets measured.

**`FULL_IMAGE_REDETECT` does not exist in the pipeline.** PROTOCOL §6 lists it
and the report's action space names it, but the deployed system implements
*three* discretionary actions, not four. Re-running the detector on identical
pixels is deterministic and would measure nothing, so it is operationalised here
as **full-frame re-detection at 1280 px** against the default 640. That
definition is ours and the result carries it.

| variant | Phase A total TP | Phase B total TP | ms/image |
|---|---|---|---|
| raw (baseline) | 300 | 656 | 30.4 / 35.0 |
| ENHANCE forced | 287 | 608 | 27.5 / 32.0 |
| re-detect @1280 | 189 | 637 | 46.7 / 45.0 |

Paired dive bootstrap on TP per image, action minus baseline:

| | Phase A | Phase B |
|---|---|---|
| ENHANCE, all images | −0.0052 [−0.0117, +0.0017] | **−0.0572** [−0.0909, −0.0291] |
| **ENHANCE, images the rule fires on** | −0.0107 [−0.0241, +0.0025] | **−0.1069** [−0.1697, −0.0572] |
| re-detect @1280, all images | **−0.0453** [−0.0577, −0.0328] | **−0.0227** [−0.0424, −0.0014] |

**ENHANCE reduces correct detections**, significantly so on the retrained
platform — and it does nearly twice the damage on precisely the images its own
trigger selects (−0.107 where the rule fires, against −0.057 overall). The
quality heuristic is not merely failing to help; it is choosing the images where
CLAHE and sharpening hurt most. Phase B loses 48 correct detections to it.

**Re-detection at 1280 px also reduces correct detections** on both platforms,
while returning substantially *more* boxes (4,985 against 3,439 on Phase A) and
costing +16 ms per image — more detections, fewer of them right.

**Caveat that must stay attached to the re-detection result.** The detectors
were trained at 640 px, so a train/test resolution mismatch is a sufficient
explanation on its own. The finding is that *re-detecting at double resolution
with a 640-trained model hurts*, not that full-frame re-detection is useless in
principle. A version using multi-scale augmentation, or a detector trained at
that resolution, is untested.

### 10.4 What this establishes

All four discretionary actions in the protocol's action space have now been
measured, on two platforms, across a fivefold difference in base accuracy:

| action | verdict |
|---|---|
| `ENHANCE` | **actively harmful** — significant on Phase B, worst where its own rule fires |
| `ZOOM` | inert on both platforms; point estimate negative on the retrained detector |
| `RECLASSIFY_LARGER_CROP` | beats the detector, not the classifier |
| `FULL_IMAGE_REDETECT` | not implemented; the operationalised version is harmful |

**None improves the system.** That is a claim about the rule set rather than
about one action, and it is direct evidence for the project's thesis: the
hand-written adaptive layer is not merely unvalidated, it is a net negative.

It also delivers **H2** — reduced computation at matched reliability —
emphatically, though not by the route the report anticipated. Removing the
discretionary layer *improves* accuracy while saving compute. The saving comes
from measuring that the actions do not pay, not from a value model allocating
them: no learned policy is needed to decline to spend on something worthless.

### 10.5 Caveats

- **A naming bug invalidated the first Phase B run.**
  `ensemble.fuse_detections` names detections by indexing `config.CLASS_NAMES`,
  which holds the 7 aquarium classes, so Phase B detections were labelled with
  aquarium names — one came back as `puffin`, a class the model cannot predict.
  Apparent disagreement reached 97.3%. `platform_b.py` now rebinds the label
  space and remaps the classifier's ImageFolder-mangled names; disagreement fell
  to 31.0% and the invalid runs were discarded.
- **The two platforms differ in ensemble composition.** Phase A is
  YOLOv8-n + YOLOv8-s; Phase B is YOLOv8-n seed 0 + seed 1, because YOLOv8-s was
  never trained. Cross-platform differences therefore carry two changes, not
  one. Every comparison that carries a claim above is *within* a platform.
- **`FULL_IMAGE_REDETECT` is operationalised, not reproduced** (§10.3), and its
  result is confounded by the 640-px training resolution.
- **Every measurement is on a development split.** None of §10 has been
  confirmed on a test split.

---

## 11. Pre-registered confirmatory family

Fixed before any test split is read. One-sided, paired cluster bootstrap over
dives, Holm–Bonferroni across these three only; all other comparisons are
exploratory.

**H1** — the learned budget-aware policy achieves lower AURC than the
`IMAGE_V1_BASELINE` rule agent with mean compute matched to ±5%. *(pending —
requires the learned policy)*

**H2** — at matched selective risk, the learned policy's mean per-image
discretionary cost is lower than the rule agent's. *(pending)*

**H3** — at matched GT-anchored coverage, the OOD-aware gate yields a lower
OOD_ERR rate among accepted detections than the baseline confidence gate.
*(supporting evidence on `J_policy`, §8; confirmatory test pending)*

---

## 12. On-domain retraining — **final**

Label space and split design in §2.2–2.3. Training images after excluding those
containing held-out taxa (§11.3 of the protocol): 3,740 (`S_fit`) and 625
(`S_modelsel`), retaining 96.6% of known annotations.

Per-class training annotations:

| Class | n | Class | n |
|---|---|---|---|
| fish | 1,550 | sponge | 389 |
| starfish | 1,010 | sea-cucumber | 367 |
| coral/sea-anemone | 975 | shrimp | 323 |
| crab | 571 | crinoid | 226 |

**Superseded run** — earlier split with `S_fit` at 25.7% of dives, retained
because it motivated the split correction:

| Seed | mAP@50 | mAP@50-95 |
|---|---|---|
| 0 | 0.6699 | 0.4840 |
| 1 | 0.6599 | 0.4672 |
| 2 | 0.6273 | 0.4445 |
| **mean** | **0.6524 ± 0.0223** | 0.4652 ± 0.0199 |

Per-class AP@50 on that run: sea-cucumber 0.914, fish 0.884, crab 0.716,
shrimp 0.679, sponge 0.677, coral 0.665, starfish 0.649, **crinoid 0.036**.
Crinoid failed to learn on 127 training annotations; the corrected split gives
it 226.

### 12.1 Detector, corrected split (YOLOv8n, 3 seeds, 80 epochs)

| Seed | mAP@50 | mAP@50-95 | precision | recall | minutes |
|---|---|---|---|---|---|
| 0 | 0.6827 | 0.4741 | 0.6465 | 0.6545 | 448.2 |
| 1 | 0.6880 | 0.4783 | 0.6888 | 0.6476 | 181.7 |
| 2 | 0.6932 | 0.4772 | 0.7466 | 0.6373 | 171.8 |
| **mean** | **0.6880 ± 0.0053** | **0.4765 ± 0.0022** | | | |

Against the superseded split (0.6524 ± 0.0223): mAP@50 **+3.6 points** and seed
variance **4× tighter**. The earlier spread was inflated by an early-stopping
confound — seed 2 halted at 96 min and its thin classes collapsed — which
`--patience 80` removes. Per-class conclusions are only trustworthy now.

Per-class AP@50 (seed 0):

| Class | AP@50 | vs superseded | train anns |
|---|---|---|---|
| fish | 0.883 | 0.884 | 1,550 |
| crab | 0.835 | 0.716 (**+0.119**) | 571 |
| sea-cucumber | 0.800 | 0.914 (−0.114) | 367 |
| sponge | 0.706 | 0.677 | 389 |
| coral/sea-anemone | 0.696 | 0.665 | 975 |
| starfish | 0.682 | 0.649 | 1,010 |
| shrimp | 0.637 | 0.679 | 323 |
| **crinoid** | **0.223** | 0.036 (**+0.187**) | 226 |

**Crinoid improved 6×** after roughly doubling its training annotations, which
confirms it was data-starved rather than untrainable. It remains the weakest
class by a wide margin: 0.223 is a *weak* class, not a solved one, and
sea-cucumber reaches 0.800 on fewer annotations — so crinoid is both starved
and intrinsically hard. Seed 0's 448 minutes against ~175 for the others is CPU
contention from a concurrently-running job, not a property of the seed.

### 12.2 Classifier (ResNet-18, 3 seeds, deep-sea crops)

| Seed | val accuracy | macro-recall |
|---|---|---|
| 0 | 0.8285 | 0.7623 |
| 1 | 0.8418 | 0.7521 |
| 2 | 0.8418 | 0.7671 |

Train accuracy reached 0.99 against 0.84 validation — substantial overfitting,
which is the same feature-norm concentration diagnosed in §7.5.

**Still pending:** YOLOv8s, RT-DETR, `IMAGE_V2` re-freeze, and re-running the
Phase A analyses on the retrained perception stack.

---

## 13. Limitations

1. **Residual risk stays high.** Even the learned policy is wrong 78% of the
   time at 30% coverage. The system is improved, not reliable. At 9% base
   accuracy there is little headroom for any gate.
2. **Near-OOD is not deployable at high recall.** Best FPR@95 is 0.627
   (retrained energy): catching 95% of novel taxa still discards 63% of correct
   detections.
3. **Calibration barely beats a trivial baseline** on ECE (Brier skill +0.025);
   its value is preserving ranking while being honest.
4. **OOD samples are small.** 44–212 OOD detections per comparison after
   fit-eligibility filtering; `OOD_val` on `S_policy` holds 109 annotations, and
   Phase B `OOD_test` holds 114 dominated by one taxon (`squat-lobster`,
   104 of 114). Intervals are correspondingly wide.
5. **Far-OOD is partly trained.** 25.9% of retained training images contain
   unlabelled `machine` annotations, so the detector learns ROV hardware as
   background. The far-OOD figure is a lower bound on difficulty.
6. **Split proportions deviate from the protocol's stated 70/10/7/7/6**; adopted
   55/10/10/10/15 because 6% of dives yields only 48 usable `OOD_test`
   annotations. Recorded in protocol §11.8.
7. **Only the accept/abstain decision is learned.** ZOOM and RECLASSIFY have now
   been *measured* (§10), but no value model governs them: `Q(s, ZOOM)` was not
   fittable because the action produces too few outcome changes to learn from,
   and ENHANCE and FULL_IMAGE_REDETECT are not measured at all. The budget
   allocator is unbuilt. H2 is satisfied by the measurement showing the actions
   do not pay, not by a policy allocating them.
8. **Crinoid remains weak** at AP@50 0.223 despite a 6× improvement.
9. **The learned gate's advantage is partly structural.** Removing the coverage
   ceiling follows from using a score instead of a hard mask, independent of
   learning; and the fixed 2-D gate still achieves a lower OOD rate at its own
   restrictive operating point.
10. **No external comparison.** Published numbers on this dataset largely use
    the supplied, leaking splits and different label spaces, so they are not
    comparable.

---

## 14. Reproduction

```bash
python experiments/audit_group_leakage.py                 # 57.3% contamination
python experiments/make_splits.py                         # Phase A splits
python experiments/make_splits_phase_b.py                 # Phase B five-way
python experiments/coco_to_yolo.py                        # YOLO label export
python experiments/verify_phase_b.py --strict             # 22 invariant checks

python experiments/baseline/freeze_baseline.py            # IMAGE_V1_BASELINE
python experiments/run_phase_a.py cache --split J_cal
python experiments/run_phase_a.py cache --split J_policy
python experiments/analyze_aggregation.py --split J_cal
python experiments/calibrate.py --fit J_cal --eval J_policy
python experiments/ood_detect.py --fit J_cal --eval J_policy
python experiments/ood_score_detections.py --fit J_cal --splits J_cal J_policy
python experiments/risk_coverage.py --fit J_cal --eval J_policy

python experiments/learned_gate.py --fit J_cal --eval J_policy
python experiments/degrade.py                             # in-domain degradation

python experiments/train_phase_b.py --arch yolov8n --seeds 0 1 2 --epochs 80
python experiments/train_classifier_phase_b.py --seeds 0 1 2

# aquarium vs retrained feature space, classifier as the only variable
python experiments/ood_features.py --manifest experiments/splits/phase_b_splits.json \
    --fit S_cal --eval S_policy --tag aquarium-feats
python experiments/ood_features.py --manifest experiments/splits/phase_b_splits.json \
    --fit S_cal --eval S_policy --tag retrained-feats \
    --weights models/phase_b/classifier_seed0/best_classifier.pth \
    --class-names models/phase_b/classifier_seed0/class_names.json
python experiments/ood_maha_variants.py --fit S_cal --eval S_policy \
    --weights models/phase_b/classifier_seed0/best_classifier.pth \
    --class-names models/phase_b/classifier_seed0/class_names.json

# forced-action measurement, both actions on both platforms
python experiments/log_zoom_outcomes.py --split J_cal
python experiments/log_zoom_outcomes.py --split J_policy
python experiments/log_reclassify_outcomes.py --split J_cal
python experiments/log_reclassify_outcomes.py --split J_policy
python experiments/fit_action_value.py --fit J_cal --eval J_policy
python experiments/analyze_reclassify.py --split J_policy

python experiments/log_zoom_outcomes.py --split S_cal --platform phase_b
python experiments/log_zoom_outcomes.py --split S_policy --platform phase_b
python experiments/log_reclassify_outcomes.py --split S_cal --platform phase_b
python experiments/log_reclassify_outcomes.py --split S_policy --platform phase_b
python experiments/fit_action_value.py --fit S_cal_phaseB --eval S_policy_phaseB
python experiments/analyze_reclassify.py --split S_policy_phaseB

python experiments/make_presentation_figures.py           # figures for the talk
```

`platform_b.py` rebinds `config.CLASS_NAMES` to the Phase B label space, so a
Phase B platform must not be constructed in the same process as a Phase A run.

Run CPU-heavy analyses **sequentially**, never alongside detector training:
concurrent jobs slowed training roughly ninefold (seed 0 took 448 min against
~175 for the others).

`freeze_baseline.py` must be run **without** `--no-measure` to populate
`measured_action_costs`, which protocol §6 sources the entire cost model from.

Cached predictions and trained weights are gitignored and regenerable.
