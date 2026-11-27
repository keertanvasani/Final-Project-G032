"""Re-fitting the combiner on FathomNet (PROTOCOL §23). Descriptive, single use.

F_cal is split by dive into F_fit (fit the combiner) and F_thr (Learn-then-Test); every
method is evaluated on all F_eval boxes with the §20 metrics.
  M1 frozen JODD combiner   M2 re-fit, all 10 features
  M3 re-fit, 5 classifier features   M4 DARTS (λ on the §20.5 extended grid)

Usage:
    python experiments/refit_fathomnet.py [--check]
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import test_baselines as tb  # noqa: E402
from week0_analyse import logistic_fit  # noqa: E402
from week3_verifier import box_rows  # noqa: E402

tf = tb.tf
OUT = tf.ROOT / "experiments" / "results" / "refit_fathomnet" / "refit.json"
HIER = tb.m_ours([0, 1, 2, 3])
CLF_COLS = [0, 1, 2, 3, 4]          # node_prob, p_top, entropy, energy, unknown
ALL_COLS = list(range(10))


def split_cal(B, i_cal):
    ud = sorted({B.dives[i] for i in i_cal})
    random.Random(23).shuffle(ud)
    fit = set(ud[: len(ud) // 2])
    return [i for i in i_cal if B.dives[i] in fit], [i for i in i_cal if B.dives[i] not in fit]


def refit(B, spec, i_fit, cols):
    F, Y = [], []
    for b in B.b:
        f, y, _ = box_rows(b, spec["classes"])
        F.append(f)
        Y.append(y)
    F, Y = np.array(F)[:, :, cols], np.array(Y, float)
    P = np.zeros((len(B.b), 4))
    models = {}
    for j in range(4):
        mu, sd, w = logistic_fit(F[i_fit, j], Y[i_fit, j])
        Z = np.c_[np.ones(len(F)), (F[:, j] - mu) / sd]
        P[:, j] = 1 / (1 + np.exp(-Z @ w))
        models[j] = {"mean": mu.tolist(), "std": sd.tolist(), "w": w.tolist()}
    B2 = copy.copy(B)
    B2.P = P
    return B2, models


def main() -> None:
    for f, h in tb.FROZEN.items():
        got = hashlib.sha256(f.read_bytes()).hexdigest()
        if got != h:
            raise SystemExit(f"hash mismatch for {f.name}: {got}")
    spec = json.loads((tb.R / "week4" / "verifier.json").read_text())
    tf.harness.KNOWN_CLASSES = set(json.loads(tb.MAN.read_text())["label_space"]["known"])
    B, i_cal, i_ev, ipd = tb.fathomnet(spec)
    want = json.loads((tf.RES / "fathomnet_test.json").read_text())["part_a"]["thresholds_F_cal"]["hierarchical"]
    for a in tb.ALPHAS:                                   # §23.4 plumbing check, F_cal only
        got = tb.ltt(B, HIER, tb.GRID, i_cal, a)
        assert got == want[str(a)], (a, got, want[str(a)])
    print("plumbing check: §18 thresholds reproduced on F_cal")
    i_fit, i_thr = split_cal(B, i_cal)
    print(f"F_fit {len(i_fit)} boxes / F_thr {len(i_thr)} boxes "
          f"({len({B.dives[i] for i in i_fit})} / {len({B.dives[i] for i in i_thr})} dives)")
    if "--check" in sys.argv:
        return
    if OUT.exists():
        raise SystemExit(f"{OUT} exists: §23 is single-use")

    B2, mods2 = refit(B, spec, i_fit, ALL_COLS)
    B3, mods3 = refit(B, spec, i_fit, CLF_COLS)
    systems = {"M1_frozen": (B, HIER, tb.GRID), "M2_refit_all": (B2, HIER, tb.GRID),
               "M3_refit_classifier": (B3, HIER, tb.GRID), "M4_darts": (B, tb.m_darts, tb.GRID_LAM_EXT)}
    ud = sorted(ipd)
    rng = random.Random(0)
    samples = [[ud[rng.randrange(len(ud))] for _ in ud] for _ in range(tb.N_BOOT)]
    by = {}
    for i in i_ev:
        by.setdefault(B.dives[i], []).append(i)
    out = {"protocol": "PROTOCOL §23 (descriptive)", "F_fit_boxes": len(i_fit), "F_thr_boxes": len(i_thr),
           "F_eval_boxes": len(i_ev), "F_eval_images": sum(ipd.values()), "alphas": {},
           "models": {"M2": mods2, "M3": mods3}}
    for alpha in tb.ALPHAS:
        a, oks = {}, {}
        for name, (Bx, f, grid) in systems.items():
            t = tb.ltt(Bx, f, grid, i_thr, alpha)
            ans = f(Bx, t) if t is not None else [None] * len(Bx.b)
            a[name] = {"parameter": t, **tb.metrics(Bx, ans, i_ev, ipd, alpha)}
            oks[name] = Bx.correct(ans).astype(float)
        diffs = {}
        for x, y in (("M2_refit_all", "M1_frozen"), ("M3_refit_classifier", "M1_frozen"),
                     ("M2_refit_all", "M4_darts"), ("M3_refit_classifier", "M4_darts")):
            d = oks[x] - oks[y]
            boot = np.sort([sum(d[i] for dv in s for i in by.get(dv, [])) / sum(ipd[dv] for dv in s)
                            for s in samples])
            diffs[f"{x}_minus_{y}"] = {"correct_per_image": float(d[i_ev].sum() / sum(ipd.values())),
                                       "ci95": tf.ci(boot)}
        a["paired"] = diffs
        out["alphas"][str(alpha)] = a
        for name in systems:
            m = a[name]
            print(f"α {alpha}  {name:20s} param {m['parameter']}  risk {m['risk']:.3f}  correct/img "
                  f"{m['correct_per_image']:.3f}  species {m['species_correct_per_image']:.3f}  "
                  f"unseen wrong {m['unseen_wrong']}")
        for k, v in diffs.items():
            print(f"α {alpha}  {k:40s} {v['correct_per_image']:+.3f}  {[round(c, 3) for c in v['ci95']]}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    enc = lambda o: bool(o) if isinstance(o, np.bool_) else float(o)   # noqa: E731
    OUT.write_text(json.dumps(out, indent=1, default=enc) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
