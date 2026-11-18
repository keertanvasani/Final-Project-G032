# Step 3 — contextual-bandit policy over the agent's actions (2026-11-18)

Fitted on **S_cal** (889 images, 2,235 boxes), evaluated on **S_policy** (844 images,
315 dives). S_test is not touched. Bracketed ranges are paired 95% dive-cluster
bootstrap intervals (1,000 resamples), learned − rules.

Scripts: `experiments/log_bandit_outcomes.py` (logs), `experiments/fit_bandit_policy.py`
(fit + replay evaluation), `experiments/check_bandit_live.py` (live loop check).
Policy file: `models/phase_b_machine/bandit_policy.json`; class `agentic.policy.LearnedPolicy`.

## Set-up

* **Decisions.** D1 enhance? (per image, state = image quality), D3 zoom? (per box,
  state = fused box + image), D4 vote? (per disagreeing box, state = box + classifier view).
  One pass, one decision each; the verifier (text filter, refuse gate, calibration,
  Known/Unknown) is fixed.
* **Full information.** The models are frozen, so every arm was run on every image and
  box. Policies are evaluated by replaying the logs; no exploration or propensity
  weighting is needed. The replay reproduces the deployed rules exactly
  (1,324 boxes, P 0.603, R 0.611 — the ablation's row F), and the live loop reproduces
  the learned policy's replay exactly.
* **Reward.** +1 per correct known-class identification, −1 per reported error,
  0 when the box is flagged Unknown.
* **Model.** Ridge regression of Δreward(act − skip) on the state; act iff the prediction
  clears a threshold. Ridge strength and threshold chosen on out-of-fold predictions of a
  5-fold dive-grouped cross-fit on S_cal, then refitted on all of S_cal. Deterministic,
  so the 3-seed rule (PROTOCOL §7.1) does not apply.

## What each action is worth (S_cal)

| Decision | Rows | Mean Δ per use | Helps / hurts | Out-of-fold corr(pred, Δ) |
|---|---|---|---|---|
| D1 enhance | 539 flagged images | −0.061 per image | 13.7% / 16.5% | −0.11 |
| D3 zoom | 2,235 boxes | −0.009 per box | 1.6% / 2.0% | −0.01 |
| D4 vote | 1,001 disagreements | +0.032 per box | 3.6% / 2.0% | −0.07 |

The state carries **no usable signal** about when an action helps: out-of-fold
predictions are uncorrelated with the realised gain for all three decisions. The
fitted policy therefore collapses to the best fixed choice: **never enhance, never
zoom, vote on (almost) every disagreement** (96.5%).

## S_policy

| Policy | Reported | Precision | Recall | F1 | AURC ↓ | Discretionary ms/img | Wall ms/img |
|---|---|---|---|---|---|---|---|
| Never act | 1,329 | 0.616 | 0.626 | 0.621 | 0.123 | 0.0 | — |
| **Rules (deployed)** | 1,324 | 0.603 | 0.611 | 0.607 | 0.138 | 11.3 | 46.8 |
| Best constant (S_cal) | 1,329 | 0.621 | 0.631 | 0.626 | 0.124 | 2.1 | — |
| **Learned** | 1,329 | **0.622** | **0.631** | **0.626** | **0.124** | **2.1** | **27.4** |
| Oracle (per decision, hindsight) | 1,285 | 0.671 | 0.659 | 0.665 | 0.124 | 0.7 | — |

Learned − rules: precision +0.019 [−0.006, +0.040], recall +0.020 [+0.006, +0.036],
F1 +0.019 [+0.002, +0.035] (p = 0.03), AURC −0.014 [−0.027, −0.003] (p = 0.02),
discretionary cost −9.2 ms/img [−10.7, −8.1] (−81%); measured wall time −41%.

## Reading

* The learned policy is better than the rules on every metric and much cheaper, but it
  is **not contextual**: it equals the context-free policy "skip enhance and zoom, keep
  the vote", which the best-constant row shows. The gain comes from switching off two
  actions that hurt on average, matching the 8-class findings (enhance −0.057 TP/img,
  zoom null, vote +6.6 pts).
* The oracle shows real headroom (reward +0.52 vs +0.38 per image): the actions do help
  on some images and boxes, but nothing in the current state predicts which.
* **H2** (lower cost at matched risk) is supported on S_policy. **H1** as pre-registered
  asks for lower AURC *at compute matched within ±5%*; the learned policy instead has
  lower AURC at ~80% lower discretionary compute. That is a stronger outcome but not the
  registered comparison, and must be reported as such.
* Not yet done: the single confirmatory run on S_test, and deployment (the app's chain
  is still `("rules", "plain")`; deploying = `("learned", "rules", "plain")`).
