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

---

## 12. Confirmatory test of H1 / H2 on S_test (registered 2026-11-18, before the run)

Added before `S_test` is read for the deep-sea (9-class) stack. Everything below is
fixed now; nothing is tuned after the run.

### 12.1 What is compared

* **Rule agent (baseline)** — the deployed deep-sea agent: enhance when the quality
  check flags the image, zoom boxes in 0.40–0.75, vote on every disagreement; text
  filter, refuse gate, calibration and Known/Unknown fixed.
* **Learned policy** — the frozen contextual bandit fitted on `S_cal` and evaluated on
  `S_policy` (`experiments/results/phase_b_machine/STEP3_RESULTS.md`), file
  `bandit_policy.json`, sha256
  `03ff9e7d8f46abfe2bc53ee2a7a01bb469c2046688551e8ab67b5dec0bd37f32`.
  The test script refuses to run if the hash differs.
* Same frozen detector (YOLOv8-n seed 0), classifier, calibration map and novelty
  threshold for both. Reported detections exclude boxes flagged Unknown (ablation
  row F). Outcomes by the §2 harness, IoU 0.50, 9 known classes.
* Evaluation by exact replay of full-information action logs on `S_test`
  (`log_bandit_outcomes.py --split S_test --confirmatory`), the method validated on
  `S_policy` against the live loop. Descriptive rows (never act, oracle) are reported
  but are not tested.

### 12.2 Deviation from §7.4 H1, declared before the run

H1 as registered asks for lower AURC **at compute matched within ±5%**. The learned
policy is not compute-matched: on `S_policy` it uses ~80% less discretionary compute
than the rule agent. Building a separate compute-matched policy now would be a design
change made for the test. H1 is therefore tested in this form:

> **H1′** — the learned policy has lower AURC (GT-anchored: risk vs selective recall,
> calibrated combined confidence) than the rule agent, **while using no more mean
> discretionary compute**.

This is a stronger requirement on the policy than matched compute (it must win on
reliability without extra spend), but it is not the registered wording and is reported
as a deviation.

**H2** — at matched selective risk, the learned policy's mean per-image discretionary
cost is lower than the rule agent's. Operationalised as: discretionary ms/image lower
(one-sided), **and** risk not higher — the learned policy's selective risk at full
coverage (1 − precision) is non-inferior with margin 0.01 (one-sided 95% upper bound of
learned − rules < +0.01).

Discretionary cost = measured ms of enhance, zoom and vote, from the same logs for
both policies.

### 12.3 Statistics

Paired dive-cluster bootstrap over `S_test` dives, 2,000 resamples, seed 0, one-sided.
Holm–Bonferroni over the §7.4 family of three (H3 is tested separately; it is counted
in the family so the correction is not weakened). Secondary, exploratory: precision,
recall, F1, reward per image, live wall time per image.

---

## 13. Week 0 — probe-signal and headroom check on S_cal (registered 2026-11-23, before the run)

Exploratory go/no-go gate for the multi-step agent controller (offline RL over a
larger action menu). **`S_cal` only.** `S_policy` and `S_test` are not read. Nothing
here is a confirmatory claim; it decides whether the controller is worth building.

### 13.1 Base pipeline

The deployed agent with its learned policy (§12): no enhancement, no zoom, vote on
every detector–classifier disagreement. Text filter, refuse gate, calibration and
Known/Unknown are fixed. Same frozen detector and classifier.

### 13.2 Actions logged (every action on every image / box: full information)

* **Image level** (outcome = the whole image re-run through the base pipeline):
  CLAHE clip 1.5, CLAHE clip 2.0, CLAHE clip 3.0 (L channel, 8×8 tiles), gamma to
  mid-grey (γ clipped to [0.5, 2]), grey-world white balance; re-detect at 960 px;
  2×2 tiling with 25% overlap merged with the full-image boxes by WBF (IoU 0.55).
  Applied to every image, not only those the quality check flags.
* **Box level** (outcome = that box, other boxes unchanged): zoom re-check (existing
  tool, vote downstream); vote (existing tool); flip TTA (classifier probabilities
  averaged over the crop and its mirror image, then the base scoring and vote).

### 13.3 Probes (cheap signals, measured before any action)

Per box: label change and own-class probability under a horizontal mirror; label
agreement and own-class probability over four ±10% shifted crops; top-1 − top-2
margin and entropy of the full classifier distribution; energy; the classifier's
probability of the detector's class; cosine distance of the ResNet embedding to the
predicted class centroid and to the nearest other centroid (centroids from `S_fit`
training crops only); box edge contact and overlap with the nearest other box.
Per image: the mean / extreme of the box probes, number of boxes, and colour
statistics (LAB a/b means, saturation mean).

### 13.4 Rewards and analysis

Reward as in Step 3: +1 TP, −1 reported error, 0 Unknown; Δ = reward(action) −
reward(base) per unit (image or box). For each action, ridge regression of Δ on
(a) the Step 3 features and (b) Step 3 features + probes, 5-fold dive-grouped
out-of-fold (OOF) predictions, ridge strength 10 (fixed, no tuning). Policy: act iff
OOF prediction > 0. Policy gain = Σ Δ over units where it acts. Uncertainty: dive-
cluster bootstrap, 2,000 resamples, seed 0, 95% percentile intervals. Also reported:
always-act gain and the oracle gain Σ max(Δ, 0) (headroom).

Correctness signal (for the learned combiner): OOF AUROC for "box would be TP if
reported" over all base boxes, calibrated combined confidence vs an L2 logistic
model on confidences + probes (same folds), paired dive bootstrap of the difference.

### 13.5 Decision rule (fixed now)

* An action **passes** if the probe policy's gain over never-act has a 95% CI lower
  bound > 0.
* **GO** for the multi-step controller if at least one action passes. Probes are
  credited only if (b) beats (a) for a passing action with paired CI lower bound > 0;
  otherwise the controller is built without them.
* **NO-GO** if no action passes: the multi-step controller is not built; work moves
  to the taxonomy-aware, risk-controlled verifier with the Step 3 bandit as the
  controller.
* **Learned combiner** is adopted if its AUROC gain is ≥ 0.01 with CI lower bound > 0;
  otherwise the verifier keeps the calibrated combined confidence.
* Round-2 (action-pair) logs are not part of Week 0.

---

## 14. Week 3 — taxonomy-aware, risk-controlled verifier (registered 2026-11-23, before any fit)

The verifier stops choosing only between "species" and "Unknown". It reports the most
specific node of a fixed tree whose estimated chance of being right clears a threshold.
The threshold is set by Learn-then-Test so that a target error rate holds per new dive.
This section fixes the tree, the scoring, the data rules and the procedure. **Week 3 is
development on S_cal only.** The S_policy / S_test tests of H4–H6 are registered
separately, before those splits are read.

### 14.1 Tree (fixed now)

```
Animal ─┬─ Chordata ........ fish
        ├─ Arthropoda ─ Decapoda ─ crab, shrimp
        ├─ Echinodermata ... starfish, sea-cucumber, crinoid
        ├─ Cnidaria ........ coral/sea-anemone
        └─ Porifera ........ sponge
Man-made ................... machine
```

Levels per known leaf: species → group → phylum → kingdom. Group is Decapoda for crab
and shrimp, and otherwise the phylum. Phylum and kingdom are Man-made for machine.

Held-out taxa, deepest correct node: squat-lobster → Decapoda; sea-spider → Arthropoda;
urchin → Echinodermata; jellyfish → Cnidaria; shark, ray, tunicate → Chordata;
octopus, squid, ragworm, comb-jelly → Animal. (Comb jellies are Ctenophora, not
Cnidaria.)

### 14.2 Scoring

* A reported node is **correct** iff the box matches a GT object (IoU ≥ 0.50, §2
  harness) whose taxon has that node on its path. LOC_ERR and BG_FP boxes are wrong
  at every node. Abstaining ("needs review") is never wrong.
* **Specificity credit** per correct answer: species 1.0, Decapoda 0.75, phylum or
  Man-made 0.5, Animal 0.25. Wrong answers and abstentions get 0.
* Node probability = the sum of the classifier's 5%-crop probabilities over the
  node's leaves. The candidate at each level is the node with the highest summed
  probability. At species level the candidate is the agent's own final class
  (after the vote), so species answers are those of today's agent.

### 14.3 Data rules (§5.1 applied)

* Boxes: the deployed agent's final boxes (learned → rules → plain chain), including
  boxes flagged Unknown.
* Fitting and calibration on S_cal may use, as unseen-animal signal, **only S_cal's role
  taxa (OOD_dev: comb-jelly, octopus, ray, shark, urchin)**. Boxes matched to any other
  held-out taxon (jellyfish, squid, tunicate, ragworm, sea-spider, squat-lobster) are
  excluded from every fit and calibration on S_cal.

### 14.4 Method

* **Combiner:** one L2 logistic model per level for P(candidate at this level is
  correct). Features:
  * the candidate's summed probability, top-1 probability and entropy;
  * energy and the Unknown flag;
  * detector confidence, raw combined confidence and agreement;
  * whether the detector's class lies under the candidate;
  * log box area.
* **Decision:** report the deepest level whose P ≥ t, otherwise abstain. One threshold
  t for all levels.
* **Risk:** per dive, the fraction of its boxes given a wrong answer, a bounded loss in
  [0, 1]. Dives are the exchangeable unit.
* **Learn-then-Test:** t on the grid 0.99, 0.98, …, 0.00, tested in that fixed order
  with Hoeffding–Bentkus p-values for H0: risk(t) > α. Keep the lowest t reached
  before the first non-rejection. δ = 0.10, α ∈ {0.10, 0.15, 0.20, 0.25}.
* **Baseline ("flat"):** the same pipeline restricted to species-or-abstain (the
  species-level model only), with t set by the same Learn-then-Test.

### 14.5 Week-3 development check (S_cal only, descriptive)

S_cal dives are split into three folds A / B / C, rotated three times:
combiner on A, Learn-then-Test on B, measure on C. Reported per α:
* the realised dive-level risk on C and whether it is ≤ α;
* the count of correct answers and the specificity credit per image;
* species answers given;
* correct coarse answers on unseen-animal boxes;
* hierarchical vs flat, with a dive bootstrap.

These numbers decide whether to register the S_policy test. They are not
confirmatory claims.

---

## 15. S_policy test of the image agent and the risk-controlled verifier (registered 2026-11-23, before S_policy is read for these components)

Registered before any S_policy log of these components is written. S_policy was used
in Step 3 to evaluate the Step 3 bandit; this is the first and only evaluation of the image
agent and the verifier on it. The test script refuses to run if a frozen file's hash
differs.

### 15.1 Frozen inputs (fitted on S_cal only)

| Component | File | sha256 |
|---|---|---|
| Image agent (no-probe state, CLAHE 3.0, two-step fitted Q-iteration) | `week2/image_agent_policy.json` | `fe1f5439a27f0a76c6be93455835db079fd0dbbcb8a323028349d804484f61bb` |
| One-step comparator (same state, single decision, ridge on s1) | `week5/one_step_policy.json` | `5a77c48ec0e1d371e23c1854e179ba90aeeb73174f9e47d8c144b1b482383fa2` |
| Verifier combiner (4 per-level logistic models) | `week4/verifier.json` | `fed9559f23128b18c0b270bad5c5029670a8579d26ca6876233f0e040927dc9b` |

Operating points, both chosen on S_cal:
* **λ = 0.002**: the deploy candidate.
* **λ_m = 0.00892128865561972**: the λ whose S_cal discretionary cost equals the rule
  agent's, 11.0 ms/img (`week4/frontier.json`, bisection).
* Policy rules exactly as in the files. Two-step: re-run iff q1(s1) − λ·cost_ms > 0, then
  keep pass 2 iff q2(s2) > 0. One-step: re-run and keep pass 2 iff q(s1) − λ·cost_ms > 0.

### 15.2 Part A — image agent (exact replay of full-information logs)

Logs: `log_image_agent.py --split S_policy` (opened by this section) and the existing
Step 3 S_policy logs (rules, deployed). Reward +1 TP / −1 reported error / 0 Unknown;
F1, AURC (GT-anchored, deployed calibration) and discretionary ms as in Step 3.

| | Claim | Test |
|---|---|---|
| **H1-R** (primary) | At λ_m, the agent has higher reward per image **and** higher F1 than the rule agent | one-sided paired dive bootstrap, both p < α after Holm |
| H1-A | At λ_m, AURC is not worse than the rule agent's by more than 0.01 | one-sided 95% upper bound of agent − rules < +0.01 |
| H7 | At λ = 0.002, two-step reward per image > one-step | one-sided paired bootstrap |
| H8 | At λ = 0.002, the agent's reward per image > the deployed bandit's | one-sided paired bootstrap |

**Declared deviation from §7.4 H1.** H1 was registered on AURC at matched compute. On S_cal
the image agent raises reward, F1 and precision at matched compute but does not improve
AURC: the enhanced pass's confidences sit on a different scale from the deployed
calibration map (`WEEK4` frontier, isotonic recalibration attempt unstable, rejected). The
agent optimises reward, so H1-R is primary and AURC becomes a non-inferiority check.

### 15.3 Part B — risk-controlled verifier on the deployed agent's boxes

* Boxes: final boxes of the deployed agent, logged by `log_verifier_inputs.py --split S_policy`.
* S_policy dives are split once, seed 0, into a **calibration half** and an **evaluation
  half**. Learn-then-Test (Hoeffding–Bentkus, δ = 0.10, fixed sequence t = 0.99 … 0) sets
  t on the calibration half, with the frozen combiner. Everything is measured on the
  evaluation half.
* §5.1: calibration may use as unseen-animal signal only S_policy's role taxa (OOD_val:
  jellyfish, squid, tunicate). Boxes matched to the other eight held-out taxa are
  excluded from calibration. All boxes count in the error accounting on the evaluation
  half.
* α ∈ {0.10, 0.20}. H4–H6 are tested at α = 0.20; α = 0.10 is descriptive.
* The dive split: sorted S_policy dives, `random.Random(0).shuffle`, first half (rounded down) = calibration.

| | Claim | Test |
|---|---|---|
| **H-risk** (primary) | Dive-level risk on the evaluation half ≤ α, for hierarchical and flat | point estimate ≤ α (the guarantee is probabilistic; also reported: share of dive bootstrap resamples ≤ α) |
| H4 | Hierarchical gives more correct answers per image than flat | one-sided paired dive bootstrap |
| H5 | Hierarchical gives a correct coarse label to > 0 OOD_val boxes (flat: 0 by construction) | count with dive-bootstrap 95% CI; the lower bound must be > 0 |
| H6 | Species-level correct answers per image, hierarchical vs flat: non-inferior, margin 0.05 | one-sided 95% lower bound of hier − flat > −0.05 |

Expectations from S_cal, stated so they cannot be reinterpreted afterwards:
* H-risk and H4 are expected to hold.
* H5 is expected to hold mostly through "Animal" answers (squid is "Animal"; jellyfish
  is Cnidaria; tunicate is Chordata).
* H6 may fail at α = 0.10.

### 15.4 Statistics and multiplicity

2,000 dive-cluster bootstrap resamples, seed 0, paired (the same dives for both arms).
One-sided p = share of resamples with difference ≤ 0 (for H6: ≤ −0.05; for H5: count ≤ 0).
H1-R's p is the larger of the reward and F1 p-values. Holm–Bonferroni at family-wise
α = 0.05 within Part A (H1-R, H7, H8) and within Part B (H4, H5, H6). H1-A and H-risk are reported, not corrected.
S_test is not touched. A confirmatory S_test run would be the second use of S_test
and would need its own registration.

---

## 16. Confirmatory S_test run of §15 (registered 2026-11-23, before S_test is read for these components)

**Second use of S_test.** The first was the Step 3 test, §12. This run repeats the §15
tests once on S_test, with everything frozen. Nothing is tuned afterwards, and S_test is
then closed for these components.

### 16.1 Frozen inputs

| Component | File | sha256 |
|---|---|---|
| Image agent | `week2/image_agent_policy.json` | `fe1f5439a27f0a76c6be93455835db079fd0dbbcb8a323028349d804484f61bb` |
| One-step comparator | `week5/one_step_policy.json` | `5a77c48ec0e1d371e23c1854e179ba90aeeb73174f9e47d8c144b1b482383fa2` |
| Verifier combiner | `week4/verifier.json` | `fed9559f23128b18c0b270bad5c5029670a8579d26ca6876233f0e040927dc9b` |
| Verifier thresholds (Learn-then-Test on all of S_policy, δ = 0.10) | `week6/thresholds_S_policy.json` | `d583d9d2386e840fb9b08644dc7cabacf4f9d87d755219deaac040fc4fe2f749` |

Thresholds:
* hierarchical: **0.72** (α = 0.10), **0.38** (α = 0.20);
* flat: **0.64** (α = 0.10), **0.27** (α = 0.20).

Operating points as in §15: λ_m = 0.00892128865561972 and λ = 0.002.

### 16.2 Tests

* **Part A** is §15.2 unchanged on S_test: H1-R (primary), H1-A, H7, H8. Rules and
  deployed are replayed from the Step 3 S_test logs. The agent is replayed from new
  full-information logs (`log_image_agent.py --split S_test --confirmatory`).
* **Part B** is §15.3 on **all** S_test dives, with the fixed thresholds above (no split,
  no calibration on S_test): H-risk, H4, H5, H6 as in §15.3. H5 uses S_test's role taxa
  (OOD_test: ragworm, sea-spider, squat-lobster). Boxes from
  `log_verifier_inputs.py --split S_test --confirmatory`.
* Statistics as §15.4: 2,000 paired dive-bootstrap resamples, seed 0; Holm at 0.05 within
  each part. The script checks every hash above and refuses a second run.

---

## 17. Step 5 — taxonomy-trained classifier for the verifier (registered 2026-11-24, before any training)

§15 and §16 found that the verifier's coarse answers are almost all "Animal": the
classifier's probabilities carry little group structure, so unseen animals rarely reach
Decapoda, Echinodermata, Cnidaria or Chordata (H5), and species answers fall (H6). Step 5
asks whether a classifier **trained with the tree in its loss** fixes this. **Development
on S_cal only.** S_policy and S_test are not read. Nothing is deployed.

### 17.1 Design: a second classifier used only by the verifier

The agent (detector, classifier, vote, calibration, novelty, image agent) is unchanged.
For every final box, the verifier reads its probabilities from a **taxonomy-trained
ResNet-18** on the same 5% crop, instead of from the agent's classifier. The species
candidate is still the agent's own final class (§14.2). This isolates the effect of the
loss and avoids re-freezing the whole stack. Cost: one extra ResNet-18 pass per box.

### 17.2 Candidates (fixed now)

Same recipe as the deployed classifier (`train_classifier_phase_b.py`: ImageNet ResNet-18,
224 px, 25 epochs, freeze stem + layer1–2 for 10 epochs, Adam 1e-3 then 1e-4, batch 32,
seed 0, the same S_fit / S_modelsel crops of the 9 known classes, no held-out taxon).
Only the loss differs.

* **CE-rerun:** plain cross-entropy, retrained (noise reference for run-to-run variation).
* **Soft-1, Soft-2:** soft-label cross-entropy (Bertinetto et al., CVPR 2020). Target for
  true leaf y: q_j ∝ exp(−β·d(y, j)) with β = 1 or 2, where d is the first level of
  §14.1 at which the two leaves share a node (species 0, group 1, phylum 2, kingdom 3,
  none 4). *Amended 2026-11-24, before any training:* the first draft had β = 4 and 10,
  which on this shallow tree put ≥ 98% (β = 4) and ≈ 100% (β = 10) of the target on the
  true leaf, i.e. plain cross-entropy. β = 1 / 2 put about 59% / 87% on a crab's own
  leaf, 22% / 12% on shrimp and the rest across other animals.
* **Marginal:** cross-entropy on the leaf **plus** the negative log of the summed
  probability of the true node at group, phylum and kingdom (weight 1 each).

### 17.3 Eligibility (S_modelsel) and dev metrics (S_cal)

* **Eligible:** best S_modelsel top-1 ≥ 0.833 (deployed classifier 0.853 − 0.02).
* **Dev evaluation on S_cal**, procedure of §14.5 unchanged (three dive-fold rotations:
  fit combiner / Learn-then-Test / measure; §14.3 fit-eligibility), boxes from the
  week-3 S_cal log, probabilities recomputed by each candidate on the same crops. The
  deployed classifier is the baseline (its recomputed probabilities must reproduce the
  logged ones). Reported for each candidate:
  * hierarchical at α = 0.20: realised risk, correct/img, **credit/img**, species
    correct/img, correct answers by level, held-out boxes correct and at which node;
  * threshold-free: on OOD_dev boxes whose deepest node is not "Animal" (shark, ray →
    Chordata; urchin → Echinodermata), the share whose top group candidate is that
    node; on known-class boxes whose species candidate is wrong, the share whose top
    group candidate is right ("better mistakes").

### 17.4 Selection and go / no-go

* **Selection:** among eligible candidates other than CE-rerun, the one with the highest
  hierarchical credit/img at α = 0.20.
* **Go** to a registered S_policy test (separate section, written before S_policy is
  read for this) only if, for the selected candidate vs the deployed classifier on S_cal:
  1. credit/img is higher with a dive-bootstrap 95% CI above 0 (2,000 resamples, seed 0),
  2. the gain exceeds CE-rerun's gain over the deployed classifier, and
  3. realised risk ≤ α in every rotation.
  The S_cal comparison is optimistic (selection happened on it). Otherwise Step 5 stops
  and is reported as negative.

---

## 18. External validation on FathomNet (registered 2026-11-25, before any FathomNet image is downloaded)

Every result in §1–§17 comes from one dataset (J-EDI / JODD: JAMSTEC, waters around
Japan, 480-px frames). §18 tests the frozen system once on a second, independent source:
**FathomNet**, mostly MBARI ROV dives in Monterey Bay (Tiburon, Doc Ricketts, Ventana),
200–3,984 m. Nothing is trained or tuned on FathomNet except, where stated, the
Learn-then-Test threshold. Images are CC BY-NC-ND 4.0 and boxes CC BY-NC 4.0; they stay
local (`data_external/`, git-excluded) and no modified frame is published.

### 18.1 What is known before the run (metadata only)

* Metadata fetched: `experiments/fathomnet_meta.py` (box counts per concept) and
  `experiments/fathomnet_image_meta.py` (per-image records: ROV, depth ≥ 200 m, up to
  1,000 images per mapped taxon root). 19,543 images, 2,248 dive groups.
* **Dive** = the MBARI dive number in the image URL (`<vehicle>/images/<dive>/`), or
  vehicle + day for `stills/<year>/<day>` frames. Dives are the exchangeable unit.
* **Annotation is not exhaustive.** 51% of images have one box and 92% of boxes are
  UNVERIFIED. A detection on an unboxed animal cannot be scored as an error. Hence two
  parts: Part A scores only labelled objects; Part B uses the full pipeline accounting
  only on images whose boxes are all VERIFIED (859 images / 301 dives in the sample),
  and is descriptive.

### 18.2 Label mapping (fixed now)

Each FathomNet concept is placed on the §14.1 tree through its WoRMS ancestry
(FathomNet taxa provider):

| §14.1 leaf | FathomNet sub-tree |
|---|---|
| coral/sea-anemone | Anthozoa |
| crab | Brachyura |
| crinoid | Crinoidea |
| fish | Actinopterygii |
| sea-cucumber | Holothuroidea |
| shrimp | Caridea, Dendrobranchiata |
| sponge | Porifera |
| starfish | Asteroidea |
| machine | `equipment` |

* Every other animal concept is an **unseen animal**. Its deepest correct node is
  Decapoda (any other Decapoda, e.g. Lithodidae, Galatheoidea), else its phylum if the
  phylum is on the tree (Chordata, Arthropoda, Echinodermata, Cnidaria, Porifera), else
  Animal. So brittle stars (Ophiuroidea) → Echinodermata, king crabs → Decapoda,
  hydromedusae and siphonophores → Cnidaria, sharks/rays/tunicates → Chordata,
  molluscs, polychaetes and ctenophores → Animal. A species answer on them is wrong.
* `trash` → deepest correct node Man-made (the species answer "machine" is wrong).
* **Excluded** (no box of this kind is scored in either part, and detections matched to
  them count neither as right nor wrong): concepts that are not organisms or equipment
  (e.g. detrital aggregate, fecal cast, marine snow, bacterial mat), and concepts whose
  WoRMS rank is coarser than the node a correct answer needs to be judged against
  (e.g. "Animalia", "Crustacea", "Invertebrata"), because a deeper answer could not be judged.
* Unmapped concepts (no WoRMS match) are excluded and listed in the results.

### 18.3 Sample and pre-processing (fixed now)

* `experiments/fathomnet_sample.py`, seed 0, from the fetched records: images with a dive
  id. Up to 200 images per known leaf (drawn so that no single dive gives more than 10
  of a leaf's images), up to 600 images whose boxes include an unseen animal, up to 100
  `equipment`/`trash` images, plus every all-VERIFIED image (Part B). Duplicates
  removed. The manifest (image uuid, url, dive, sha256) is frozen before download, and
  downloaded files are checked against FathomNet's sha256.
* **Frames are resized to 480 px wide** (aspect kept, area interpolation), the condition
  the system was built and calibrated for (JODD was released this way). Boxes are scaled
  to match. Everything downstream is the deployed pipeline unchanged.
* Dive split, seed 0: sorted Part A dives, `random.Random(0).shuffle`, first half
  (rounded down) = **F_cal**, rest = **F_eval**. Part B images are reported on their own.

### 18.4 Frozen inputs

Deployed detector, classifier, calibration and novelty as in §16; image agent
`week2/image_agent_policy.json`; verifier combiner `week4/verifier.json`; JODD
thresholds `week6/thresholds_S_policy.json` (hashes as §16.1). The test script checks
every hash and refuses a second run.

### 18.5 Part A — labelled objects (classifier, verifier, recall)

* Run the deployed agent (plain chain, as deployed: image agent and verifier off for the
  boxes; verifier applied after) on every Part A image. Each scorable GT box is matched
  to the agent's final boxes (IoU ≥ 0.50, greedy by confidence, §2 harness). Unmatched
  GT boxes are detector misses. Final boxes not matched to any GT box are **ignored** in
  Part A (the annotation is incomplete).
* Verifier loss per matched GT box: wrong answer = 1, correct or abstain = 0. Dive risk =
  wrong / matched boxes in the dive.

| | Claim | Test |
|---|---|---|
| **E1** (primary) | Learn-then-Test re-run on F_cal (frozen combiner, δ = 0.10) gives dive-level risk ≤ α on F_eval, α ∈ {0.10, 0.20} | point estimate ≤ α; share of 2,000 dive-bootstrap resamples ≤ α also reported |
| E2 | With the JODD thresholds unchanged (no recalibration), risk on F_eval ≤ α | as E1. **Expected to fail**; it measures how far the guarantee drifts across datasets |
| E3 | Hierarchical gives more correct answers per image than flat (both recalibrated on F_cal), α = 0.20 | one-sided paired dive bootstrap |
| E4 | Unseen-animal boxes on F_eval with a correct coarse answer > 0, α = 0.20 | count, dive-bootstrap 95% CI lower bound > 0; reported by node |

Descriptive: detector recall per known leaf and for unseen animals (F_cal ∪ F_eval), vs
the JODD S_test recall; classifier top-1 on matched known-leaf boxes; specificity credit
per image; the same tables with the image agent switched on.

### 18.6 Part B — full pipeline on fully verified images (descriptive)

On the all-VERIFIED images, with §2 accounting (BG_FP and LOC_ERR count as errors; boxes
on excluded concepts are ignored): reward per image, F1, precision and recall for the
rule agent, the deployed bandit and the image agent at λ_m and λ = 0.002, with paired
dive-bootstrap CIs; and the verifier's realised dive risk with the F_cal thresholds and
with the JODD thresholds. No hypothesis is tested in Part B, because "VERIFIED" does not
prove that every animal in the frame is boxed.

### 18.7 Statistics and stopping

2,000 paired dive-cluster bootstrap resamples, seed 0. Holm–Bonferroni at 0.05 over E3
and E4; E1 and E2 are reported, not corrected. If fewer than 30 F_eval dives or fewer
than 300 matched F_eval boxes result, E1–E4 are reported as descriptive only. FathomNet
is used once for these components.

Frozen manifest (�18.3): `data_external/fathomnet/meta/manifest.json`, 2,897 images / 963 dives, sha256 `b07caf6bd2576e256096c0436452456c5599bffcf75bd16c356b930d6786ca0a`.

**§18.2 implementation note (2026-11-25, before any image is read).** The mapping is
`experiments/fathomnet_labels.py` → `experiments/results/external_fathomnet/concept_labels.json`
(sha256 `929db292376677132c330781aa39c9cf4d3e7b86e46111cda5a569d7d77bf19c`). Details
the text above left open, settled from names only:
* Morphospecies and species complexes WoRMS does not know ("Acanthascinae sp. 1",
  "Coryphaenoides … complex") are placed by climbing FathomNet's MBARI hierarchy to the
  nearest name WoRMS knows. One misspelling is corrected (Opiuroidea → Ophiuroidea).
* `machine` = every concept under `equipment` in that hierarchy (instruments, cables,
  the ROV's own tools, traps). Man-made but not machine = litter and materials (trash,
  metal, plastic, bottles, rope, tyres, wreckage).
* Excluded: non-animal organisms (xenophyophores, bacterial mats), remains and traces
  (shells, bones, fecal casts, eggcases, detritus), image artefacts (shadow, black
  frame), and "Decapoda" (coarser than the crab/shrimp leaves).
* Boxes in the manifest by label: coral/sea-anemone 2,699; sponge 2,199;
  sea-cucumber 1,631; fish 973; crab 846; starfish 845; machine 650; shrimp 629;
  crinoid 600; unseen 5,995 (Echinodermata 2,248, Animal 1,219, Decapoda 794,
  Chordata 638, Cnidaria 326, Arthropoda 70); Man-made 144; excluded 1,040.
