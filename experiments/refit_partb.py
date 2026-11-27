"""The §23 re-fit on FathomNet Part B, with false detections (PROTOCOL §24). Descriptive, single use.

Fits and thresholds come from Part A F_cal exactly as §23 (plus M5, fitted and thresholded
on all F_cal final boxes). Scored on Part B images whose dive is not an F_cal dive, with
every final box kept (unmatched = wrong at every level).

Usage:
    python experiments/refit_partb.py [--check]
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import refit_fathomnet as rf  # noqa: E402

tb, tf = rf.tb, rf.tf
OUT = tf.ROOT / "experiments" / "results" / "refit_fathomnet" / "partb.json"
REFIT = tf.ROOT / "experiments" / "results" / "refit_fathomnet" / "refit.json"


def load(spec):
    man = json.loads((tf.EXT / "meta" / "split_manifest.json").read_text())
    files_a, files_b = set(man["parts"]["A"]), set(man["parts"]["B"])
    recs = [json.loads(line) for line in (tf.LOGS / "F_ext_verifier.jsonl").open(encoding="utf-8")]
    ud = sorted({r["dive"] for r in recs if r["file"] in files_a})
    random.Random(0).shuffle(ud)
    cal = set(ud[: len(ud) // 2])                       # §18.5 F_cal
    boxes, dives, tag = [], [], []
    for r in recs:
        if r["file"] in files_a and r["dive"] in cal:
            t = "cal"
        elif r["file"] in files_b and r["dive"] not in cal:
            t = "partb"
        else:
            continue
        for b in r["boxes"]:
            if b["gt_class"] != "excluded":
                boxes.append(b)
                dives.append(r["dive"])
                tag.append(t)
    B = tb.Boxes(boxes, dives, spec)
    i_cal_m = [i for i in range(len(boxes)) if tag[i] == "cal" and boxes[i]["matched"] and tf.SCORABLE(boxes[i]["gt_class"])]
    i_cal_all = [i for i in range(len(boxes)) if tag[i] == "cal"]
    i_b = [i for i in range(len(boxes)) if tag[i] == "partb"]
    ipd = Counter(r["dive"] for r in recs if r["file"] in files_b and r["dive"] not in cal)
    return B, i_cal_m, i_cal_all, i_b, ipd


def main() -> None:
    for f, h in tb.FROZEN.items():
        got = hashlib.sha256(f.read_bytes()).hexdigest()
        if got != h:
            raise SystemExit(f"hash mismatch for {f.name}: {got}")
    spec = json.loads((tb.R / "week4" / "verifier.json").read_text())
    tf.harness.KNOWN_CLASSES = set(json.loads(tb.MAN.read_text())["label_space"]["known"])
    B, i_cal_m, i_cal_all, i_b, ipd = load(spec)
    i_fit, i_thr = rf.split_cal(B, i_cal_m)
    fit_dives = {B.dives[i] for i in i_fit}
    i_fit_all = [i for i in i_cal_all if B.dives[i] in fit_dives]
    i_thr_all = [i for i in i_cal_all if B.dives[i] not in fit_dives]
    B2, _ = rf.refit(B, spec, i_fit, rf.ALL_COLS)
    B3, _ = rf.refit(B, spec, i_fit, rf.CLF_COLS)
    B5, _ = rf.refit(B, spec, i_fit_all, rf.ALL_COLS)
    systems = {"M1_frozen": (B, rf.HIER, tb.GRID, i_thr), "M2_refit_all": (B2, rf.HIER, tb.GRID, i_thr),
               "M3_refit_classifier": (B3, rf.HIER, tb.GRID, i_thr),
               "M4_darts": (B, tb.m_darts, tb.GRID_LAM_EXT, i_thr),
               "M5_refit_with_unmatched": (B5, rf.HIER, tb.GRID, i_thr_all)}
    params = {str(a): {n: tb.ltt(Bx, f, g, it, a) for n, (Bx, f, g, it) in systems.items()} for a in tb.ALPHAS}
    ref = json.loads(REFIT.read_text())["alphas"]
    for a in params:                                    # §24.4 plumbing check
        for n in ("M1_frozen", "M2_refit_all", "M3_refit_classifier", "M4_darts"):
            assert params[a][n] == ref[a][n]["parameter"], (a, n, params[a][n], ref[a][n]["parameter"])
    print(f"plumbing check: §23 parameters reproduced; M5 {params['0.1']['M5_refit_with_unmatched']} / "
          f"{params['0.2']['M5_refit_with_unmatched']}")
    print(f"Part B held out: {sum(ipd.values())} images, {len(ipd)} dives, {len(i_b)} boxes "
          f"({sum(not B.b[i]['matched'] for i in i_b)} unmatched)")
    if "--check" in sys.argv:
        return
    if OUT.exists():
        raise SystemExit(f"{OUT} exists: §24 is single-use")

    ud = sorted(ipd)
    rng = random.Random(0)
    samples = [[ud[rng.randrange(len(ud))] for _ in ud] for _ in range(tb.N_BOOT)]
    by = {}
    for i in i_b:
        by.setdefault(B.dives[i], []).append(i)
    unmatched = np.array([not b["matched"] for b in B.b])
    out = {"protocol": "PROTOCOL §24 (descriptive)", "images": sum(ipd.values()), "dives": len(ipd),
           "boxes": len(i_b), "unmatched_boxes": int(unmatched[i_b].sum()), "alphas": {}}
    for alpha in tb.ALPHAS:
        a, oks, crs = {}, {}, {}
        for name, (Bx, f, _, _) in systems.items():
            t = params[str(alpha)][name]
            ans = f(Bx, t) if t is not None else [None] * len(Bx.b)
            ok = Bx.correct(ans)
            rep = np.array([x is not None for x in ans])
            a[name] = {"parameter": t, **tb.metrics(Bx, ans, i_b, ipd, alpha),
                       "false_detections_labelled": int((rep & unmatched)[i_b].sum()),
                       "false_detections_labelled_by_answer": dict(Counter(
                           "species" if ans[i] in tb.tx.KNOWN else ans[i] for i in i_b if rep[i] and unmatched[i]))}
            oks[name] = ok.astype(float)
            crs[name] = np.array([tb.tx.credit(ans[i]) if ok[i] else 0.0 for i in range(len(ans))])
        diffs = {}
        for x, y in (("M2_refit_all", "M1_frozen"), ("M2_refit_all", "M4_darts"),
                     ("M5_refit_with_unmatched", "M1_frozen"), ("M5_refit_with_unmatched", "M4_darts")):
            row = {}
            for label, v in (("correct", oks), ("credit", crs)):
                d = v[x] - v[y]
                boot = np.sort([sum(d[i] for dv in s for i in by.get(dv, [])) / sum(ipd[dv] for dv in s)
                                for s in samples])
                row[label] = {"per_image": float(d[i_b].sum() / sum(ipd.values())), "ci95": tf.ci(boot)}
            diffs[f"{x}_minus_{y}"] = row
        a["paired"] = diffs
        out["alphas"][str(alpha)] = a
        for name in systems:
            m = a[name]
            print(f"α {alpha}  {name:24s} risk {m['risk']:.3f}  correct {m['correct_per_image']:.3f}  "
                  f"species {m['species_correct_per_image']:.3f}  credit {m['credit_per_image']:.3f}  "
                  f"unseen wrong {m['unseen_wrong']}  false boxes labelled {m['false_detections_labelled']}")
        for k, v in diffs.items():
            print(f"α {alpha}  {k:36s} correct {v['correct']['per_image']:+.3f} {[round(c, 3) for c in v['correct']['ci95']]}"
                  f"  credit {v['credit']['per_image']:+.3f} {[round(c, 3) for c in v['credit']['ci95']]}")
    enc = lambda o: bool(o) if isinstance(o, np.bool_) else float(o)   # noqa: E731
    OUT.write_text(json.dumps(out, indent=1, default=enc) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
