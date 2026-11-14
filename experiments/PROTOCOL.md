# Evaluation Protocol — Part 1

**Status:** ACTIVE. All six open decisions resolved (§9); three confirmatory
hypotheses pre-registered (§7.4). Binding from this point — amend by dated
addition, never by editing in place.

This document is step 0 of the Part 1 roadmap. It fixes the unit of
evaluation, the definition of correctness, the data splits, the OOD
regime, the cost model, and the statistical procedure. Everything
downstream — calibration, meta-confidence, policy learning, ablations —
inherits these definitions. Changing them later invalidates prior results,
so they are pinned here first.

Baseline under evaluation: `IMAGE_V1_BASELINE`
(`experiments/baseline/IMAGE_V1_BASELINE.json`).

---

## 1. Unit of evaluation

**Primary unit: the detection.** The agent decides per object — accept this
box, investigate it, or abstain on it — so the object is the natural unit.
An image-level label collapses a mixed image (three confident fish, one
ambiguous ray) into a single verdict and hides exactly the behaviour under
study.

**Secondary unit: the image.** Retained because the deployed system
abstains at image level, and because compute budget is allocated per image
(§6).

Mapping between the two: an image is *abstained* iff zero of its emitted
detections are accepted. An image is *correct* iff every accepted detection
on it is a TP and no known-class ground-truth object on it is missed.

---

## 2. Matching and correctness

### 2.1 Matching

For each image, let `G` be ground-truth objects and `D` the detections
emitted by the perception stack.

1. Sort `D` by `combined_confidence`, descending.
2. Greedily match each detection to the **unmatched** ground-truth box with
   the highest IoU, if that IoU ≥ **0.50**.
3. Matching is **class-agnostic**; the class is checked afterwards.
4. One-to-one: each ground-truth object is matched at most once.

Class-agnostic matching is deliberate. Matching within class (COCO style)
would make a correctly-localised box with the wrong label indistinguishable
from a background false positive. Separating them is the point — the two
failures have different causes and different fixes.

### 2.2 Correctness

The predicted label is **`classifier_class`** — the post-vote label the
system actually presents — not `detector_class`.

```
correct(d) = 1  iff  d matched some g at IoU ≥ 0.50
                     AND classifier_class(d) == class(g)
                     AND class(g) ∈ KNOWN_CLASSES
```

### 2.3 Error taxonomy

Every emitted detection falls into exactly one bucket:

| Outcome | Condition | Meaning |
|---|---|---|
| `TP` | matched, class correct, class ∈ KNOWN | correct detection |
| `CLS_ERR` | matched, class wrong, class ∈ KNOWN | localised, misidentified |
| `OOD_ERR` | matched a GT whose class ∉ KNOWN | **real organism, outside the label space** |
| `LOC_ERR` | unmatched, but max IoU with any GT ∈ [0.10, 0.50) | localisation failure |
| `BG_FP` | unmatched, max IoU < 0.10 | background false positive |

`OOD_ERR` is tracked separately because it is the failure mode the
domain-shift study exists to measure: the system detects a genuine organism
it has no label for and asserts a confident species anyway. On the deep-sea
set this is not an edge case — 66% of test annotations are taxa outside the
7-class label space. The desired behaviour is abstention, and the rate at
which the system instead emits a confident wrong species is a headline
result.

Ground-truth objects with `class ∉ KNOWN_CLASSES` are **excluded from the
recall denominator** — the model cannot be penalised for failing to detect
a class it was never trained on. They are used only to identify `OOD_ERR`.

### 2.4 Label-space mapping (Phase A)

The aquarium model's 7 classes map onto Jedi as follows:

| | classes | test annotations |
|---|---|---|
| **KNOWN** | fish, jellyfish, shark, starfish | **1,538 (33.9%)** |
| No Jedi counterpart | penguin, puffin | — |
| **OOD** | the other 16 Jedi classes | **3,005 (66.1%)** |

**`ray` is treated as OOD, not as `stingray`** (decision 9.3). Batoidea
includes skates, electric rays and guitarfish; a deep-sea *ray* is more
likely a skate than the aquarium stingrays the classifier was trained on.
Scoring a `stingray` prediction as correct would credit the model for a
match that cannot be verified from the annotation. The conservative reading
is used, and this is an explicit assumption, not a silent one. At 18
annotations the choice does not move aggregate numbers; it is made for
correctness rather than effect.

Two-thirds of test annotations being OOD is not an artefact of a narrow
label space — it is the condition the study exists to characterise.

---

## 3. Selective-prediction metrics

Let `A ⊆ D` be the accepted detections after agent decisions.

```
selective_risk     = |{d ∈ A : correct(d) = 0}| / |A|
coverage           = |A| / |D|
selective_recall   = |{g ∈ G_known : matched by some d ∈ A}| / |G_known|
```

**Risk and recall must always be reported together.** Risk alone is trivially
gamed: a policy that abstains on everything hard achieves near-zero risk
while missing most organisms. Recall makes that visible.

Report:
- **Risk–coverage curve** and **AURC** (area under it)
- **Risk–recall curve**
- **Coverage at fixed risk** — e.g. recall achievable at ≤ 10% selective risk
- **Abstention rate**, split by cause (unreadable image vs. low confidence)

### 3.1 Coverage denominator — RESOLVED: report both

`|D|` depends on the policy — zoom-and-recheck alters boxes, so two policies
do not share an identical emitted set and raw coverage is not strictly
comparable between them.

Both axes are therefore reported:

- **Policy-local coverage** (`|A| / |D|`) — the classic selective-prediction
  curve. `|D|` is printed alongside every curve so the denominator is never
  implicit.
- **GT-anchored coverage** (`selective_recall`) — comparable across policies
  because both axes are anchored to ground truth.

Cross-policy claims are made on the GT-anchored curve. The policy-local
curve is reported for continuity with the selective-prediction literature.

---

## 4. Datasets and splits

### 4.1 Group leakage in the supplied splits — must be fixed

`JediOrganismDetectionDataset` images are **ROV video frames**, not
independent photographs. The archive spans six vehicles (`2K`, `3K`, `6K`,
`HPD`, `KAIKO`, `KMROV`), and filenames encode vehicle, dive, camera and
frame index:

```
2K0263OUTUM1016_240.jpg     -> dive 2K0263,    frame 240
HPD0664HDDB402_120.jpg      -> dive HPD0664,   frame 120
KAIKO0283PNCSV1016_240.jpg  -> dive KAIKO0283, frame 240
```

Frames from one dive share camera, illumination, substrate, turbidity, and
frequently the same individual organisms seconds apart.

Measured against the supplied splits (`experiments/audit_group_leakage.py`,
2.10 frames per dive in train):

| | value |
|---|---|
| Dives shared train ∩ test | **749** |
| Dives shared train ∩ valid | 302 |
| Dives shared valid ∩ test | 200 |
| **Test images whose dive appears in train** | **1399 / 2440 (57.3%)** |
| Test images whose dive appears in train *or* valid | 1563 / 2440 (64.1%) |
| Valid images whose dive appears in train | 469 / 847 (55.4%) |

Nearly **two-thirds of the test set is contaminated** at dive level. Using
the supplied splits as-is would report performance largely on near-duplicate
material. **The supplied splits must not be used.** All splits are
re-derived **grouped by dive ID**, so no dive appears in more than one split.

Grouping regex — `^((?:[A-Z]+|\d+K)\d{3,4})` — anchors the digit run
immediately after the vehicle prefix. A greedier pattern runs on into the
camera code (`HPD0664HDDB402` instead of `HPD0664`), splitting one dive into
many groups and **understating** leakage. During drafting, two successive
regexes did exactly that and reported 35.6% and 43.5% before the correct
grouping gave 57.3%. Any future change to this pattern must be re-validated
against a sample from all six vehicle families.

### 4.2 Split structure

Splitting is at the level of **dive groups**, never individual images.

| Split | Purpose | Source |
|---|---|---|
| `S_fit` | detector + classifier weight fitting | ~70% of dives |
| `S_modelsel` | early stopping, hyperparameters | ~10% of dives |
| `S_cal` | calibration + meta-confidence fitting | ~7% of dives |
| `S_policy` | exploration rollouts + policy learning | ~7% of dives |
| `S_test` | final evaluation — **single use** | ~6% of dives |

`S_cal` and `S_policy` must be **unseen by weight fitting**. A classifier is
systematically overconfident on its own training data; fitting a
temperature or a meta-confidence model there yields parameters calibrated
to an optimism that does not exist at deployment.

Proportions are over *dives*, not images, so realised image counts will
differ. Report actual counts once split.

### 4.3 Experimental platform — RESOLVED: two phases

**Phase A — frozen aquarium baseline, Jedi as shift + open-set test.**
No retraining. `IMAGE_V1_BASELINE` is evaluated directly on Jedi under
dive-grouped splits. Because the model never saw any Jedi image, weight
leakage is not a concern here and only three splits are required:

| Split | Purpose |
|---|---|
| `J_cal` | calibration + meta-confidence fitting |
| `J_policy` | exploration rollouts + policy learning |
| `J_test` | final evaluation — single use |

Phase A produces the motivating result — how a system trained in aquaria
behaves on real deep-sea imagery it has no labels for — at zero training
cost, and determines whether Phase B is worth running.

**Phase B — retrain on Jedi with deliberately held-out classes.** The main
platform. Training uses a subset of Jedi's 20 classes; the remainder are
withheld entirely so they constitute genuine unseen taxa at test time.
Requires the full five-way split of §4.2.

Phase B is gated on Phase A. Note that training on all 20 Jedi classes would
destroy the OOD test set — the class holdout is what makes the open-set
claim possible, so it is a design requirement, not a convenience.

---

## 5. OOD protocol

The 15 Jedi classes outside the label space provide genuine OOD data. Two
regimes, reported separately:

- **Near-OOD** — unseen marine taxa (shrimp, crab, crinoid, sea-cucumber…).
  Visually and contextually similar; the hard case.
- **Far-OOD** — `machine` (ROV hardware, 1,144 annotations). Non-biological.
  A confident species label on hardware is the most legible failure.

### Class-disjoint three-way split

OOD classes are partitioned **by class**, not by image:

```
OOD_dev   → scorer choice and hyperparameters
OOD_val   → threshold selection
OOD_test  → reported AUROC / FPR@95TPR — single use
```

Partitioning by class rather than by image tests the property that actually
matters: rejecting taxa **never seen during OOD development**. Splitting the
same classes across dev and test would measure only whether the scorer
memorised those specific taxa. Partitions are stratified by annotation
frequency so no split is composed solely of rare classes.

**Both constraints apply simultaneously.** The class partition is enforced
*within* the dive-disjoint image splits: `OOD_dev` draws only from `J_cal`
images, `OOD_val` from `J_policy`, `OOD_test` from `J_test`. Class-disjoint
alone would still let the scorer tune on one taxon and be tested on another
recorded minutes later in the same dive, under identical optics and
illumination.

### 5.1 Fit eligibility (added after the split was built)

Partitioning OOD *roles* by class does not partition the *images* — a
`J_cal` image can and does contain `shrimp`, whose role is `OOD_test`.
Measured on the Phase A split, more OOD annotations sit outside their host
role than inside it:

| Split | Fit-eligible OOD | Present but excluded |
|---|---|---|
| `J_cal` | 867 | 2,131 |
| `J_policy` | 592 | 2,774 |
| `J_test` | 706 | 3,309 |

Two distinct uses of OOD data must therefore be kept apart:

- **Error accounting.** `OOD_ERR` applies to *any* detection matching a GT
  outside `KNOWN_CLASSES`, in any split, regardless of role. A shrimp called
  `fish` is an error wherever it occurs.
- **Fitting.** Calibration, the meta-confidence model and the policy may use
  as OOD signal **only their host split's role classes**. Far-OOD
  (`machine`) is excluded from fitting everywhere and evaluated at test only.

Without this, a meta-confidence model trained on `J_cal` would learn "this
looks unreliable" from shrimp and then be evaluated on shrimp as an unseen
taxon — the novelty claim behind H3 would be false while every split
remained nominally class-disjoint.

The cost is real: `J_cal` loses ~2,131 OOD detections as training signal.
Negative signal remains available from `CLS_ERR`, `LOC_ERR`, `BG_FP` and the
867 dev-role OOD detections, which is sufficient. The alternative — a claim
that does not survive the question "did the meta-model ever see a shrimp?" —
is not.

`make_splits.py` emits `ood_fit_classes` and `ood_excluded_from_fit` per
split so this is enforced by the manifest rather than by memory.

### Near-OOD partition (Phase A, by descending frequency, round-robin)

| Split | Classes | Test annotations |
|---|---|---|
| `OOD_dev` | coral/sea-anemone, sea-cucumber, squat-lobster, squid, ray | 789 |
| `OOD_val` | crab, crinoid, octopus, urchin, tunicate | 585 |
| `OOD_test` | shrimp, sponge, comb-jelly, sea-spider, ragworm | 487 |

### Far-OOD

`machine` (1,144 annotations — ROV hardware) is held out of the near-OOD
partition entirely and reported as its own category, evaluated only at test
time with no tuning. Folding it into `OOD_dev` would make development almost
entirely far-OOD while testing was entirely biological, and its 1,144
annotations would dominate any frequency-stratified split.

A confident species label on a piece of submersible hardware is the most
legible failure this system can produce, and is reported as a standalone
number.

Metrics: AUROC, FPR@95TPR, AUPR-Out; ID-vs-OOD score separation. Near-OOD
and far-OOD are always reported separately — averaging them hides that the
hard case is the one that matters.

---

## 6. Cost accounting and budget

Costs come from measured wall-clock time on the frozen baseline hardware
(`measured_action_costs` in the baseline manifest), never from ordinal
labels.

**Mandatory** (not discretionary, excluded from the budget): quality
assessment, initial dual detection, WBF, first classification pass,
annotation.

**Discretionary** (governed by the budget): `ENHANCE`, `ZOOM`,
`RECLASSIFY_LARGER_CROP`, `FULL_IMAGE_REDETECT`.

Budget is **per image**, not per object. Per-object budgeting makes a
53-object image cost 53× a single-object image and removes the interesting
problem; a per-image budget forces the agent to *allocate* attention across
objects, which is the research question.

Report cost twice: absolute (ms, hardware-specific) and **normalised** as a
multiple of mandatory baseline cost, so results transfer across hardware.

Reference costs (30 val images, CPU, from the baseline manifest):

| Action | mean when fired | fires on |
|---|---|---|
| ZOOM | 282.07 ms | 21/30 (70%) |
| RECLASSIFY (tie-break) | 20.95 ms | 10/30 |
| ENHANCE (within preprocessing) | 7.15 ms | 30/30 |
| agent decision | 0.05 ms | 30/30 |

Deliberation costs ~0.05 ms — four orders of magnitude below `ZOOM`. The
budget therefore constrains **actions only**; the agent may evaluate every
option at negligible cost.

---

## 7. Statistical protocol

### 7.1 Stochastic training → seeds

Components with training randomness (detector, classifier, meta-confidence
model, learned policy) are trained with **3 seeds** — the accepted floor for
reporting mean ± std. Never a single run.

Small policies over tabular state are often seed-sensitive. If the learned
policy's across-seed std exceeds the effect it is claimed to produce, that
is a finding to report, not a reason to add seeds until it disappears.

### 7.2 Deterministic inference ablations → bootstrap

Ablations that only change inference on frozen weights (no WBF, no zoom, no
calibration, no OOD, alternative abstention rules) are **deterministic** —
repeating them yields identical numbers, so seeds add compute without
adding evidence.

Uncertainty for these comes from **cluster bootstrap over test images**
(≥1,000 resamples), resampling **images, not detections**. Detections within
an image are correlated; bootstrapping detections independently
understates the interval.

Where dive grouping persists into the test split, bootstrap over **dives**.

### 7.3 Paired comparisons

Policies evaluated on the same test set are compared with a **paired**
bootstrap over the per-image difference, or McNemar's test for binary
outcomes. Overlapping independent CIs do not establish absence of a
difference.

### 7.4 Multiple comparisons — pre-registered confirmatory family

Three hypotheses are confirmatory. **Holm–Bonferroni is applied across these
three only.** Every other comparison — the ~12 ablation rows, calibration
method selection, abstention-strategy comparison — is **exploratory**,
reported with bootstrap CIs, no correction, and labelled as such in the
caption.

Correcting across all twelve would very likely erase effects that are real;
correcting across none would forfeit any confirmatory claim. Naming a small
family in advance is what keeps both the power and the honesty.

#### Pre-registered hypotheses

Fixed before `S_test` / `J_test` is touched. Each is one-sided, tested by
**paired cluster bootstrap over dives** (≥1,000 resamples).

**H1 — adaptive acquisition improves reliability at matched compute.**
> The learned budget-aware policy achieves lower AURC (GT-anchored,
> per-detection) than the `IMAGE_V1_BASELINE` rule agent, with mean compute
> cost matched to within ±5%.

**H2 — it does so without proportional compute growth.**
> At matched selective risk, the learned policy's mean per-image
> discretionary cost is lower than the rule agent's.

**H3 — uncertainty-aware abstention reduces confident errors on unseen taxa.**
> At matched GT-anchored coverage, the OOD-aware gate yields a lower
> `OOD_ERR` rate among accepted detections than the baseline confidence gate.

H1 and H2 together are the paper's central claim: *better reliability, not
merely more computation.* H1 alone would be satisfiable by spending more; H2
alone by doing less. Both must hold.

If H1 fails but H2 holds, the honest result is a compute reduction at equal
reliability — still publishable, and it must be reported that way rather
than reframed after the fact.

---

## 8. Test-set discipline

- `S_test` and `OOD_test` are evaluated **once**, after all design decisions
  are frozen.
- Every threshold — abstention, OOD, budget — is selected on `S_cal`,
  `S_policy`, or `OOD_val`. Never on test.
- Model selection and early stopping use `S_modelsel`.
- If test is inspected and the design then changes, the result is
  exploratory and must be reported as such.
- Pre-register: metrics, comparisons, and the primary hypothesis are fixed
  in this document before the test run.

### 8.1 Log — J_test baseline read (Phase A)

`J_test` was evaluated once against the frozen `IMAGE_V1_BASELINE` to
establish the baseline reference (`results/phase_a/RESULTS.md`). Measuring a
frozen system cannot tune it, so this is the canonical single use of the
test set, and those numbers are the locked baseline row of the final
comparison — not a design input.

From here, all Phase 1 design and exploration (calibration, uncertainty
features, OOD scorer, abstention aggregation, meta-confidence, policy
learning, ablations) uses **`J_cal` / `J_policy` only**. `J_test` is not read
again until the final agent-vs-baseline evaluation. Re-examining it during
design would convert every confirmatory result in §7.4 into an exploratory
one.

---

## 9. Resolved decisions

All six are settled. Changing any of them invalidates results produced under
the previous setting; amend by adding a dated entry rather than editing in
place.

| # | Decision | Resolution | §
|---|---|---|---|
| 3.1 | Coverage denominator | Report both; cross-policy claims on GT-anchored | 3.1 |
| 4.3 | Experimental platform | Two phases — frozen aquarium first, Jedi retrain second | 4.3 |
| 9.1 | Seed count | 3, for stochastic training only | 7.1 |
| 9.2 | IoU threshold | 0.50 primary; 0.5:0.95 sweep secondary | 2.1 |
| 9.3 | `ray` mapping | Treated as **OOD**, not as `stingray` | 2.4 |
| 9.4 | Multiple comparisons | 3 pre-registered confirmatory (Holm–Bonferroni); rest exploratory | 7.4 |

**9.2 note.** IoU 0.50 as the primary operating point matches COCO
convention and keeps the error taxonomy interpretable; the 0.5:0.95 sweep is
reported alongside so the result is not an artefact of one threshold.

---

## 10. Reportable claim

Nothing in this protocol licenses a claim about deep-sea generalisation
until `S_test` and `OOD_test` have been run under it. Until then the
defensible statement is:

> Evaluated on aquarium-domain imagery, under an abstention gate calibrated
> on held-out data.

---

## 11. Phase B label space (on-domain retraining)

Added 2026-09-04. §2.4 mapped the *aquarium* model's 7 classes onto Jedi.
The retrained detector learns a Jedi-native label space instead, so that
mapping does not apply to it and the §5 OOD partition — built for the
aquarium model's 4 known classes — has to be rebuilt. Phase A results
remain valid under §2.4; this section governs Phase B only.

### 11.1 Known vs held-out taxa

The 20 Jedi classes are severely long-tailed (27 to 4,074 annotations). The
split is made on trainability: a class needs enough annotations to survive
J_fit taking its share.

| | classes | J_fit train annotations |
|---|---|---|
| **KNOWN** (8) | coral/sea-anemone, crab, crinoid, fish, sea-cucumber, shrimp, sponge, starfish | **2,926** |
| **HELD-OUT UNSEEN** (11) | comb-jelly, jellyfish, octopus, ragworm, ray, sea-spider, shark, squat-lobster, squid, tunicate, urchin | — |
| **FAR-OOD** (1) | machine | — |

`jellyfish` and `shark` were KNOWN in Phase A and are held out in Phase B.
This is deliberate — both are too rare in Jedi (451 and 78 annotations) to
train a detector head on — but it means Phase A and Phase B known-class
numbers are **not directly comparable**. Comparisons must be made within a
phase.

### 11.2 OOD role partition — maximin, not round-robin

§5 partitioned OOD classes round-robin by descending total frequency. That
was sound when the host splits were 30/30/40. Phase B carves J_fit out of
J_cal and J_policy while J_test stays frozen, so the host splits are now
roughly 12.5/12.5/40 and a class's *usable* annotations are only those
falling inside its host split. Ranking on global totals starves dev and val.

Greedy assignment on hosted counts fails worse: `jellyfish` alone carries 253
J_test annotations, absorbs the entire OOD_test budget in one step, and
leaves a **single-taxon OOD_test** — which measures whether the scorer
rejects jellyfish, not whether it rejects taxa it has never seen.

The partition is therefore chosen by exhaustive search over all 3^11
assignments, maximising the **minimum** hosted annotation count across the
three roles, subject to every role holding **at least 3 taxa**. Maximin
protects the scarcest role; the class floor protects the novelty claim.

| Role | Hosted by | Classes | Usable annotations |
|---|---|---|---|
| `OOD_dev` | `J_cal` | jellyfish, octopus, shark, squid, urchin | 118 |
| `OOD_val` | `J_policy` | ray, sea-spider, squat-lobster | 117 |
| `OOD_test` | `J_test` | comb-jelly, ragworm, tunicate | 144 |

**Pre-registered limitation.** These budgets are small — Phase A had 487
usable OOD annotations at test, Phase B has 144. AUROC and FPR@95TPR will
carry wide confidence intervals, and they must be reported with those
intervals rather than as point estimates. This is a measured consequence of
choosing an 8-class known space: a 6-class space was quantified at 207
usable test annotations for 9% fewer training annotations, and the 8-class
space was chosen anyway, prioritising detector strength. Recording the
trade-off here rather than discovering it at write-up time.

### 11.3 J_fit excludes images containing held-out taxa

An unlabelled organism in a training image teaches the detector that that
taxon is background. If held-out taxa are present-but-unlabelled in J_fit,
then rejecting them at test is *learned suppression*, not novelty handling —
H3 would be false while every split remained nominally class-disjoint.

18.0% of J_fit images contain at least one held-out taxon. They are dropped
(2,445 → 2,005 images), which retains **97.0%** of known annotations. The
claim is preserved at almost no cost, so this is the default in
`coco_to_yolo.py`.

### 11.4 Far-OOD is partly trained — known limitation

24.0% of the retained J_fit images still contain unlabelled `machine`
annotations. Excluding those too would cost a quarter of the training set,
so they are kept. The consequence is that the detector learns ROV hardware
as background, and far-OOD rejection is therefore **partly trained rather
than purely novel**. Unlike §11.3 this is not cost-free to fix, so it is
declared rather than corrected: the far-OOD number is a lower bound on
difficulty and must not be reported as unseen-novelty performance.

### 11.5 Early stopping uses a J_fit slice, never J_cal

Model selection needs a validation set. Using `J_cal` would burn it — a
calibration set that chose the checkpoint is no longer naive to the model,
and temperature scaling fitted on it would be optimistically biased. `J_fit`
is therefore split by dive into train (1,786 images) and val (219 images,
76 of 757 dives). `J_cal` is untouched during training.

### 11.6 Split provenance

`J_test` is carried across **byte-identical** from the Phase A manifest, so
the single-use discipline of §8 and the §8.1 read already logged against the
aquarium baseline both remain valid. `J_fit` is carved from the Phase A
J_cal ∪ J_policy dives; dives stay indivisible throughout.

| Split | Images | Dives | Provenance |
|---|---|---|---|
| `J_fit` | 2,445 (2,005 after §11.3) | 810 | carved from Phase A cal+policy |
| `J_cal` | 1,223 | 624 | re-carved |
| `J_policy` | 1,222 | 624 | re-carved |
| `J_test` | 3,261 | 1,094 | **frozen from Phase A** |

Built by `make_splits_phase_b.py` → `splits/phase_b_splits.json`.

### 11.7 Detector baselines

YOLOv8n and YOLOv8s (3 seeds each) plus **RT-DETR** (3 seeds), so the
comparison is not confined to one architecture family and a reviewer cannot
dismiss the baseline as a 2023-era benchmark. RT-DETR is transformer-based
and trains under a different recipe; on 8 GB of VRAM it is expected to need
a reduced batch size, which is a compute-budget note, not a protocol change.

### 11.8 Split proportions — deviation from §4.2, and the S_test re-partition

Added 2026-09-05. Supersedes both §4.2's stated proportions for Phase B and
the earlier frozen-`J_test` carve described in §11.6.

**What changed and why.** The first Phase B split held `J_test` frozen from
Phase A and carved `J_fit` out of the remaining cal+policy dives. That
preserved the §8.1 read, but left `S_fit` at 25.7% of dives — roughly a third
of §4.2's ~70%. The long tail starved: `crinoid` reached AP50 **0.036** across
three seeds on 127 training annotations, while `sea-cucumber` reached 0.914 on
170. A detector that cannot see one of its own classes is a poor substrate for
every later phase, so the split was rebuilt.

**Why not §4.2 literally.** §4.2's 70/10/7/7/6 assumes `S_test` only has to
support detection metrics. Here it must *also* host `OOD_test` under the
class-disjoint × dive-disjoint constraint of §5. Measured at 6% of dives,
`OOD_test` holds **48 usable annotations across 4 taxa**, two of them with 4
and 5. An AUROC on 48 samples cannot support H3, and H3 is one of only three
pre-registered confirmatory hypotheses (§7.4).

**Adopted proportions** (over dives, per §4.2's convention):

| Split | §4.2 | Adopted | Realised images |
|---|---|---|---|
| `S_fit` | 70% | **55%** | 4,506 (3,740 after §11.3) |
| `S_modelsel` | 10% | 10% | 742 (625 after §11.3) |
| `S_cal` | 7% | **10%** | 889 |
| `S_policy` | 7% | **10%** | 844 |
| `S_test` | 6% | **15%** | 1,170 |

Measured consequences of the three candidates:

| Split | S_fit known anns | crinoid train anns | OOD_test anns |
|---|---|---|---|
| frozen-`J_test` carve | 3,209 | 127 | 144 |
| §4.2 literal | 6,847 | 292 | **48** |
| **adopted** | 5,411 | **226** | **114** |

`crinoid` at 226 now exceeds `sea-cucumber`'s 170, which trained successfully,
so the class should become learnable; `OOD_test` at 114 stays in the range
Phase A operated in.

**Test-set discipline — `S_test` is a new partition.** It is no longer Phase
A's `J_test`, so the read logged in §8.1 does not cover it. That read was made
against the *aquarium* baseline, a different model, and those images are
excluded from all Phase B weight fitting, so the exposure is to experimenter
knowledge rather than to weights. It is recorded here rather than left
implicit. Phase B's `S_test` remains single-use and untouched until Phase 3.

**Residual limitation.** `OOD_test`'s 114 annotations are dominated by
`squat-lobster` (104); `ragworm` and `sea-spider` contribute 5 each. The role
holds three taxa nominally but is effectively one. No split fixes this — all
11 held-out taxa are rare, and spreading them across three small hosts leaves
every role thin. It must be reported as a limitation on the novelty claim, not
smoothed over.
