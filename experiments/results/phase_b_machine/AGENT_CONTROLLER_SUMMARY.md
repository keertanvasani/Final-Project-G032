# Agent controller, weeks 0–6 — summary for the paper

Two components were added to the Step 3 agent. Both were built on S_cal, tested once on
S_policy (PROTOCOL §15) and confirmed once on S_test (§16). Each test was registered
before its split was read.

## 1. Two-step image agent (constrained offline RL)

* **What it does:** after the first full pass (detect → verify → classify → vote →
  verify), the agent decides whether to re-run the whole pass on a CLAHE-enhanced image
  (D5). If it re-runs, it decides which of the two verified results to keep (D6). Every
  safety step runs again on the second pass. The agent only chooses between two fully
  verified results.
* **State:** image quality, colour statistics and the first pass's own outputs (box count,
  confidences, detector–classifier disagreement, Unknown share). No extra crops.
* **Learning:** two-step fitted Q-iteration on full-information logs (every action run on
  every image, so targets are exact and no propensities are needed). Nested dive-grouped
  cross-fitting; ridge models. λ prices compute, and a budget is met by bisection on λ
  (dual ascent).
* **Key finding (week 0 → 2):** enhancement chosen *before* detection from image quality
  has no signal (Step 3). Chosen *after* looking at the first pass, it pays. Extra probe
  crops add nothing.

| Confirmed on S_test (§16) | Result |
|---|---|
| vs rule agent, at matched compute (7.5 vs 11.2 ms) | reward/img **+0.110** [+0.075, +0.145], F1 **+0.022** [+0.013, +0.031], AURC −0.023 |
| two-step vs one-step (the second decision matters) | reward/img **+0.065** [+0.034, +0.095] |
| vs the deployed Step 3 bandit | reward/img **+0.078** [+0.038, +0.117], at +18 ms; precision up, recall down |

## 2. Risk-controlled, taxonomy-aware verifier

* **What it does:** for every final box, it reports the most specific node of a fixed tree
  (species → group → phylum → kingdom) whose estimated probability of being correct
  clears a threshold, or abstains.
* **Threshold:** set by Learn-then-Test (Hoeffding–Bentkus, δ = 0.10) with the **dive** as
  the exchangeable unit, so that the fraction of a new dive's boxes given a wrong answer
  is ≤ α.
* **Combiner:** per-level logistic models on the classifier distribution, energy, detector
  confidence, agreement and box size. Fitted under the fit-eligibility rule for unseen
  taxa (§5.1).

| Confirmed on S_test (§16) | Result |
|---|---|
| Realised dive-level risk ≤ α | 0.094 at α 0.10, 0.183 at α 0.20 (today's agent: 0.278) |
| Hierarchical vs species-only, correct answers/img | **+0.062** [+0.042, +0.084] |
| Unseen animals (ragworm, sea-spider, squat-lobster) with a correct label | 12 / 42, almost all "Animal" |
| Species answers non-inferior | **not supported** (−0.053) |

## What can be claimed, and what cannot

* **Claim:** a budgeted, safety-constrained agent for open-set underwater detection.
  * A learned two-step controller beats the rule agent at matched compute, and its
    second decision is shown to matter.
  * A verifier that no policy can override guarantees a per-dive error rate on unseen
    dives.
  * All results come from exact replay of full-information logs, pre-registered and
    tested once per split.
* **Do not claim:**
  * informative taxonomic answers: they are mostly "Animal" because the classifier
    carries little group structure (next step: hierarchical-loss retraining);
  * F1 gains over the deployed bandit;
  * multi-agent RL;
  * a recall gain.
* **Limitations:**
  * the guarantee's margin is thinner across splits;
  * unseen-taxon counts are small (≈ 40 boxes per split);
  * frames are low resolution (480×360), so tiling and zoom had no headroom;
  * every split comes from one dataset.
