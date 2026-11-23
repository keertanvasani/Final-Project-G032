"""Week 2: how cheap can the image agent's state be? Offline, S_cal logs only.

Same two-step fitted Q-iteration and nested dive-grouped cross-fitting as
fit_image_agent.py, with the state restricted to subsets of the logged
columns (no re-logging). Per-box probe keys by what they cost:

    free     p_top, margin, entropy, energy, p_det_class, centroid distances,
             geometry — all from the 5% crop the pass already classifies
             (0 extra crops if the pass's own forward is reused)
    mirror   flip_change, flip_p_own                     (+1 crop per box)
    shift    shift_agree, shift_p_own, shift_p_min       (+4 crops per box)

Variants: none (quality, colour, box count, confidences only), free,
free+mirror, full (current). Plus a gated D5: decide on the free state, and
pay for the full state only when the free Q1 is within a margin of the
threshold. Reported at λ = 0 and λ = 0.002; cost = extra crops per image.

Usage:
    python experiments/week2_cheap_state.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from ablation_deepsea import MAN  # noqa: E402
from agentic.probes import PROBE_KEYS  # noqa: E402
from eval import harness  # noqa: E402
from fit_image_agent import FOLDS, fit_pred, folds_of, inner_oof, reward  # noqa: E402

DIR = ROOT / "experiments" / "results" / "phase_b_machine" / "week1"
OUT = ROOT / "experiments" / "results" / "phase_b_machine" / "week2"
ARM = "clahe3.0"
N_BOOT = 2000
Q = 8                    # image_features length
NP = len(PROBE_KEYS)
FREE = [k for k in PROBE_KEYS if not k.startswith(("flip", "shift"))]
MIRROR = ["flip_change", "flip_p_own"]
SHIFT = ["shift_agree", "shift_p_own", "shift_p_min"]
CROPS = {"none": 0, "free": 0, "free+mirror": 1, "full": 5}      # extra crops per box beyond the pass


def pass_cols(keys):
    """Column indices inside one pass_state block for the chosen probe keys."""
    cols = list(range(0, 5))                                    # colour (4) + log n boxes
    for block in range(3):                                      # mean, min, max
        cols += [5 + block * NP + PROBE_KEYS.index(k) for k in keys]
    cols += list(range(5 + 3 * NP, 5 + 3 * NP + 6))             # conf mean/min, disagree, unknown, has, refused
    return cols


def state_cols(keys, step):
    blk = 5 + 3 * NP + 6
    c = list(range(Q)) + [Q + i for i in pass_cols(keys)]
    if step == 2:
        c += [Q + blk + i for i in pass_cols(keys)]
    return c


def main() -> None:
    man = json.loads(MAN.read_text())
    harness.KNOWN_CLASSES = set(man["label_space"]["known"])
    gt = harness.load_ground_truth()
    recs = [json.loads(line) for line in (DIR / "S_cal_image_agent.jsonl").open(encoding="utf-8")]
    recs = [r for r in recs if not r["arms"][ARM].get("skipped")]
    n = len(recs)
    a = [r["arms"][ARM] for r in recs]
    dives = [r["dive"] for r in recs]
    S1 = np.array([x["s1"] for x in a], float)
    S2 = np.array([x["s2"] for x in a], float)
    D = np.array([reward(x["passes"][1]["finals"], gt.get(r["file"], []))
                  - reward(x["passes"][0]["finals"], gt.get(r["file"], [])) for x, r in zip(a, recs)], float)
    pass2 = np.array([x["pass2_ms"] for x in a])
    probe_full = np.array([x["probe_ms"] for x in a])            # [pass1, pass2] at 6 crops/box
    nbox = np.array([[len(p["finals"]) for p in x["passes"]] for x in a], float)
    ms_per_crop = float(np.median(probe_full[:, 0][nbox[:, 0] > 0] / (6 * nbox[nbox[:, 0] > 0, 0])))

    variants = {"none": [], "free": FREE, "free+mirror": FREE + MIRROR, "full": list(PROBE_KEYS)}
    outer = folds_of(dives, FOLDS)
    preds = {}
    for name, keys in variants.items():
        c1, c2 = state_cols(keys, 1), state_cols(keys, 2)
        X1, X2 = S1[:, c1], S2[:, c2]
        q1, q2 = np.zeros(n), np.zeros(n)
        for k in range(FOLDS):
            tr, te = np.where(outer != k)[0], np.where(outer == k)[0]
            q2[te] = fit_pred(X2, D, tr, te)
            y1 = np.zeros(n)
            y1[tr] = D[tr] * (inner_oof(X2, D, dives, tr) > 0)
            q1[te] = fit_pred(X1, y1, tr, te)
        preds[name] = (q1, q2)

    by = defaultdict(list)
    for i, d in enumerate(dives):
        by[d].append(i)
    ud = sorted(by)
    rng = random.Random(0)
    samples = [np.array([i for _ in ud for i in by[ud[rng.randrange(len(ud))]]]) for _ in range(N_BOOT)]

    def ci(v):
        b = np.sort([v[s].mean() for s in samples])
        return [float(b[int(0.025 * N_BOOT)]), float(b[int(0.975 * N_BOOT) - 1])]

    def crop_ms(name, step):
        return CROPS[name] * nbox[:, step] * ms_per_crop

    def policy(name, lam, d5_name=None, gate=None):
        """Gain and added ms per image. d5_name: state used for D5 (gated: free first)."""
        q1, q2 = preds[name]
        cost_pass = pass2.mean()
        if gate is None:
            act = q1 - lam * cost_pass > 0
            d5_cost = crop_ms(name, 0)
        else:
            qf = preds["free"][0] - lam * cost_pass
            unsure = np.abs(qf) < gate
            act = np.where(unsure, q1 - lam * cost_pass > 0, qf > 0)
            d5_cost = unsure * crop_ms(name, 0)
        keep = q2 > 0
        gain = D * act * keep
        extra = d5_cost + act * (pass2 + crop_ms(name, 1))
        return gain, extra, act

    res = {"images": n, "arm": ARM, "ms_per_extra_crop": ms_per_crop, "rows": []}
    print(f"S_cal {n} images, arm {ARM}; ~{ms_per_crop:.2f} ms per extra crop; pass 2 {pass2.mean():.1f} ms\n")
    print(f"{'state':14s} {'λ':>6s} {'gain/img':>9s} {'95% CI':>18s} {'re-run %':>8s} {'extra ms':>8s}")
    rows = [(nm, lam, None) for nm in variants for lam in (0.0, 0.002)]
    rows += [("full", lam, g) for lam in (0.0, 0.002) for g in (0.02, 0.05, 0.1)]
    full0 = policy("full", 0.002)[0]
    for nm, lam, g in rows:
        gain, extra, act = policy(nm, lam, gate=g)
        label = nm if g is None else f"gated {g}"
        r = {"state": label, "lambda": lam, "gain": float(gain.mean()), "ci95": ci(gain),
             "rerun_rate": float(act.mean()), "extra_ms": float(extra.mean()),
             "minus_full_l0.002": float((gain - full0).mean()), "minus_full_ci95": ci(gain - full0)}
        res["rows"].append(r)
        print(f"{label:14s} {lam:6.3f} {gain.mean():+9.3f} [{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] "
              f"{100 * act.mean():8.1f} {extra.mean():8.1f}   vs full(λ=.002) {r['minus_full_l0.002']:+.3f} "
              f"[{r['minus_full_ci95'][0]:+.3f},{r['minus_full_ci95'][1]:+.3f}]")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cheap_state.json").write_text(json.dumps(res, indent=2) + "\n")
    print(f"\nwrote {(OUT / 'cheap_state.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
