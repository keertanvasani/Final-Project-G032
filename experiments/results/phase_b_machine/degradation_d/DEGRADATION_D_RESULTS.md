# Synthetic degradation with option D (PROTOCOL §22): results

**Descriptive only.** This was a single run on 2026-11-27 on S_policy: 844 images,
315 dives. It used 22 conditions: clean, plus 7 damage types at 3 strengths each. Each
condition went through option D and through the pre-D bandit, on identical pixels.
The verifier used the frozen combiner and the S_policy thresholds, never re-fitted.

Damaged images were built in memory and never saved. The raw images are unchanged: the
files were last modified in Dec 2024. The per-condition logs are box records only, are
gitignored, and stay local in `logs/`. Output: `degradation_d.json` and `run.txt`.

## Hierarchical verifier at α = 0.20

"Holds" means risk ≤ 0.20. "Share ≤ α" is the share of 2,000 dive bootstrap resamples
with risk ≤ 0.20.

| Condition | Option D risk | Share ≤ α | Bandit risk | D correct per image | Bandit correct per image | D − bandit risk (95% CI) | Images refused ("needs review") | D re-run share |
|---|---|---|---|---|---|---|---|---|
| clean | **0.187** ✓ | 79% | 0.184 | 1.002 | 1.031 | [−0.008, +0.015] | 181 | 17% |
| gaussian blur 1 / 2 / 3 | 0.212 / 0.220 / 0.244 ✗ | 24 / 17 / 27% | 0.195 / 0.181 / 0.250 | 0.96 / 0.68 / 0.05 | 0.99 / 0.71 / 0.05 | 1 and 2 above 0 | 172 / 300 / **800** | 16 / 12 / 1% |
| motion blur 1 / 2 / 3 | 0.229 / **0.401** / **0.441** ✗ | 6 / 0 / 0% | 0.210 / 0.366 / 0.385 | 0.86 / 0.63 / 0.43 | 0.92 / 0.67 / 0.46 | all above 0 | 199 / 202 / 274 | 17 / 20 / 28% |
| noise 1 / 2 / 3 | 0.218 / 0.228 / 0.333 ✗ | 17 / 16 / 1% | 0.194 / 0.206 / 0.313 | 0.64 / 0.20 / 0.05 | 0.70 / 0.21 / 0.06 | 1 above 0 | 344 / 635 / **764** | 20 / 9 / 4% |
| low light 1 / 2 / 3 | 0.196 / 0.191 / 0.194 ✓ | 58 / 71 / 63% | 0.194 / 0.183 / 0.184 | 1.03 / 0.96 / 0.88 | 1.04 / 0.97 / 0.87 | all include 0 | 170 / 184 / 209 | 14 / 14 / 12% |
| low contrast 1 / 2 / 3 | 0.196 / 0.195 / 0.200 ✓ | 60 / 63 / 50% | 0.189 / 0.179 / 0.191 | 1.03 / 0.97 / 0.87 | 1.05 / 0.99 / 0.86 | 2 just above 0 | 177 / 181 / 228 | 18 / 18 / 16% |
| colour shift 1 / 2 / 3 | 0.200 ✓ / 0.221 / 0.213 ✗ | 50 / 11 / 24% | 0.181 / 0.201 / 0.208 | 1.00 / 0.95 / 0.92 | 1.04 / 0.99 / 0.95 | 1 and 2 above 0 | 175 / 189 / 205 | 15 / 14 / 13% |
| JPEG compression 1 / 2 / 3 | 0.215 / 0.251 / **0.413** ✗ | 19 / 0 / 0% | 0.198 / 0.201 / 0.348 | 0.86 / 0.64 / 0.29 | 0.91 / 0.71 / 0.32 | all above 0 | 193 / 280 / 416 | 18 / 20 / 18% |

At α = 0.10, option D's risk stays at or below 0.11 for blur, noise, light, contrast and
colour. It breaks for motion blur 2–3 (0.19) and compression 3 (0.23).

## Reading, by §22.5

1. **The guarantee holds under 8 of the 21 damaged conditions**: all of low light, all
   of low contrast, and colour shift 1. None of the 22 conditions is *robust* (at least
   90% of resamples ≤ α), not even clean (79%). Clean risk is already close to α.
2. **Where §22.5's predictions were right:** motion blur, noise and severe damage break
   the guarantee, and low light is absorbed. **Where they were wrong:** low contrast did
   not break the guarantee, but mild blur, colour shift 2–3 and compression did.
3. **There are two kinds of failure:**
   * **Severe blur or noise: the system notices.** The image-level gate refuses most
     images (800 and 764 of 844), so correct answers collapse to about 0.05 per image.
     Few answers are given at all, though among them the risk is still 0.24–0.33.
   * **Motion blur and compression: the system does not notice.** Boxes stay as many as
     on clean images (about 1,290 at motion blur 2), but far fewer of them match a real
     animal (733 against 1,055 on clean). The combiner's confidence features do not drop
     for these kinds of damage, so wrong answers stay confident: risk reaches 0.40–0.44.
4. **Option D is slightly riskier than the bandit under damage.** The paired risk
   difference is above 0 in 13 of the 21 damaged conditions, and below 0 in none. D also
   gives 0.01–0.09 fewer correct answers per image in most conditions. The image agent
   was fitted on clean images. Its re-run rate rises only for motion blur (17% to 28%);
   for blur and noise it falls, because the images are refused before D5 is reached.
5. **No threshold was changed** (§22.5).

## Implications

* **For the paper:** the guarantee is tied to the calibration distribution.
  * It survives changes in light, contrast and mild colour shift.
  * It fails, sometimes without the system noticing, under motion blur and compression.
  * This completes the E2 story with a controlled cause: degraded footage needs its own
    calibration data, or an input-quality check in front of the verifier.
* **For the app:**
  * On clean, dim or low-contrast footage, option D is fine.
  * On footage with motion blur or heavy compression, the bandit + verifier is the safer
    of the two, though neither meets α there.
  * A cheap guard would be a blur and JPEG-quality check that raises the threshold or
    marks the image for review. It would need registration and testing first.
* **Possible follow-up:** calibrate on a mix of clean and damaged S_policy images, then
  test on S_cal damaged with fresh seeds. This needs a new registration.

## Process notes

* A 12-image dry run, saved in the scratchpad and since deleted, checked the script
  before the full run. Its numbers were printed and are not used.
* The clean condition reproduces §21 exactly: 1,371 option-D boxes, risk 0.187.
