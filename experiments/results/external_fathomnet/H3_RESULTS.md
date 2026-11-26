# H3 on FathomNet (PROTOCOL §19): results

This is the single registered run of `experiments/test_h3_fathomnet.py`, done on
2026-11-26. It used FathomNet Part A's evaluation half, F_eval: 1,660 matched boxes in
344 dives, of which 392 boxes are animals the system was never trained on ("unseen").
Raw output: `h3_test.json`.

> **H3:** at matched coverage, the uncertainty-aware gate (the verifier) gives fewer
> confident errors on unseen animals among the boxes it accepts than a plain
> confidence gate does.

## Result: H3 supported, and H3-U supported

All three gates answer exactly 902 of the 1,660 boxes (54%); that is what "matched
coverage" means here. The verifier uses the threshold learned on F_cal at α = 0.20.

| Gate | Unseen animals answered wrongly, as a share of accepted boxes | Unseen animals answered wrongly, of 392 | Known animals answered wrongly, as a share of accepted boxes | All wrong answers, as a share of accepted boxes |
|---|---|---|---|---|
| Confidence gate | 0.228 | 206 | 0.108 | 0.336 |
| Confidence gate + Unknown flag (the deployed system) | 0.231 | 208 | 0.108 | 0.338 |
| **Verifier** | **0.159** | **143** | **0.043** | **0.202** |

| Test | Reduction in the verifier's rate | 95% CI | Holm |
|---|---|---|---|
| **H3** (against the confidence gate) | **0.070** | [0.050, 0.093] | ✅ supported, p < 0.0005 |
| **H3-U** (against the deployed gate with its Unknown flag) | **0.072** | [0.052, 0.095] | ✅ supported, p < 0.0005 |

The 95% CIs are for the reduction, so a positive interval means the verifier makes fewer
errors.

At α = 0.10 (descriptive only), all gates answer 654 boxes (39%). The verifier's rate is
0.122, against 0.216 for both baselines, and it gives 80 unseen animals a wrong answer
against 141.

## What it means

* At the same coverage, the verifier makes about **30% fewer confident errors on unseen
  animals** (143 against 206). It also makes about **60% fewer errors on known animals**
  (0.043 against 0.108).
* The deployed energy-based **Unknown flag does not help here.** It catches only 40 of the
  392 unseen boxes (10%, close to Step 1's 8.6% on JODD), and it also flags 128 known
  boxes. At matched coverage that leaves the deployed gate no better than plain
  confidence, and slightly worse.
* As §19.3 predicted, H3 held. H3-U, the test that matters, held by a similar margin, not
  a smaller one, because the Unknown flag adds almost nothing.

## Caveats

* F_eval had already been read for §18's E1–E4, before this test. Those results reported
  only *correct* answers; the wrong-answer counts and the gate comparison shown here were
  computed for the first time in this run (see the §19 disclosure).
* The plumbing check before the run used the full data but printed only the output keys,
  and its output was deleted unread.
* The verifier still gives a wrong answer to 36% of unseen animals at α = 0.20. Its gain
  comes from stepping back to coarser answers or abstaining, not from recognising new
  animals.
* This test is on FathomNet, not on JODD's S_test. S_test was closed for the verifier
  (§16) before H3 was tested.
