"""Can a tuned confidence cut-off make the second-seed WBF break even?

ablation_seed_wbf.py found that fusing seed 0 with a second YOLOv8-n seed adds
recall but loses more precision: boxes only one detector proposes survive WBF
at a diluted confidence and are mostly false. A fair second look gives both
systems a minimum fused-detection confidence, tuned on S_cal to maximise F1 of
the full deployed agent (configuration F), and compares the tuned systems once
on S_policy with paired dive-bootstrap intervals. S_test is never touched.

The cut-off is applied to every detection pass, including the zoom re-check,
by wrapping ensemble.run_detection.

Usage:
    python experiments/tune_wbf_cutoff.py [--seed 1] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import ensemble  # noqa: E402
from ablation_deepsea import CONFIGS, DIVE_RE, MAN  # noqa: E402
from ablation_seed_wbf import per_dive, pr  # noqa: E402
from eval import harness  # noqa: E402
from pipeline import EnsemblePipeline  # noqa: E402

DS = ROOT / "models" / "phase_b_machine"
OUT = ROOT / "experiments" / "results" / "phase_b_machine" / "tune_wbf_cutoff.json"
GRID = [None, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
_run_detection = ensemble.run_detection
CUT = {"t": None}


def run_detection_cut(image, pipeline, timings=None):
    dets = _run_detection(image, pipeline, timings=timings)
    t = CUT["t"]
    return dets if t is None else [d for d in dets if d["confidence"] >= t]


ensemble.run_detection = run_detection_cut


def f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


def kgt(files, gt_all):
    k: Counter = Counter()
    for fn in files:
        k[DIVE_RE.match(fn).group(1)] += harness.known_gt_count(gt_all.get(fn, []))
    return k


def paired(a0, a1, kd, n_boot=2000, seed=0):
    rng = random.Random(seed)
    dives = list(kd)
    out = {"precision": [], "recall": [], "f1": []}
    for _ in range(n_boot):
        s = [dives[rng.randrange(len(dives))] for _ in dives]
        p0, r0 = pr(a0, kd, s)
        p1, r1 = pr(a1, kd, s)
        out["precision"].append(p1 - p0)
        out["recall"].append(r1 - r0)
        out["f1"].append(f1(p1, r1) - f1(p0, r0))
    res = {}
    for k, v in out.items():
        v.sort()
        p = 2 * min(sum(x <= 0 for x in v), sum(x >= 0 for x in v)) / n_boot
        res[k] = {"ci95": [v[int(0.025 * n_boot)], v[int(0.975 * n_boot) - 1]], "p_two_sided": min(p, 1.0)}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    man = json.loads(MAN.read_text())
    harness.KNOWN_CLASSES = set(man["label_space"]["known"])
    gt_all = harness.load_ground_truth()
    cal = sorted(man["files"]["S_cal"])[: args.limit]
    pol = sorted(man["files"]["S_policy"])[: args.limit]
    kc, kp = kgt(cal, gt_all), kgt(pol, gt_all)
    flags = next(c[2] for c in CONFIGS if c[0] == "F")

    pipe = EnsemblePipeline(profile="deepsea")
    second = YOLO(str(DS / f"yolov8n_seed{args.seed}" / "weights" / "best.pt"))
    systems = {"seed0": (None, None), f"seed0+seed{args.seed}": (second, [1, 1])}

    t0 = time.time()
    out = {"seed": args.seed, "config": "F (full agent)", "grid": GRID, "tuning": {}, "S_policy": {}}
    best = {}
    for name, (mb, w) in systems.items():
        pipe.model_b, pipe.wbf_weights = mb, w
        rows = []
        for t in GRID:
            CUT["t"] = t
            agg, _, _ = per_dive(pipe, flags, cal, gt_all)
            p, r = pr(agg, kc, list(kc))
            rows.append({"cutoff": t, "precision": p, "recall": r, "f1": f1(p, r)})
            print(f"S_cal  {name:12s} cut {str(t):5s} P {p:.3f} R {r:.3f} F1 {f1(p, r):.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        out["tuning"][name] = rows
        best[name] = max(rows, key=lambda q: q["f1"])["cutoff"]

    aggs = {}
    for name, (mb, w) in systems.items():
        pipe.model_b, pipe.wbf_weights = mb, w
        for t in sorted({None, best[name]}, key=lambda v: -1 if v is None else v):
            CUT["t"] = t
            agg, oc, ms = per_dive(pipe, flags, pol, gt_all)
            p, r = pr(agg, kp, list(kp))
            key = f"{name}@{t}"
            aggs[key] = agg
            out["S_policy"][key] = {"cutoff": t, "tuned": t == best[name], "precision": p, "recall": r,
                                    "f1": f1(p, r), "outcomes": oc, "ms_per_image": ms}
            print(f"S_pol  {key:20s} P {p:.3f} R {r:.3f} F1 {f1(p, r):.3f} {ms:5.0f} ms/img "
                  f"({time.time() - t0:.0f}s)", flush=True)

    b0 = f"seed0@{best['seed0']}"
    b1 = f"seed0+seed{args.seed}@{best[f'seed0+seed{args.seed}']}"
    out["best_cutoff_S_cal"] = best
    out["tuned_fused_minus_tuned_seed0"] = paired(aggs[b0], aggs[b1], kp)
    out["tuned_fused_minus_deployed_seed0"] = paired(aggs["seed0@None"], aggs[b1], kp)
    if best["seed0"] is not None:
        out["tuned_seed0_minus_deployed_seed0"] = paired(aggs["seed0@None"], aggs[b0], kp)
    for k in ("tuned_fused_minus_tuned_seed0", "tuned_fused_minus_deployed_seed0",
              "tuned_seed0_minus_deployed_seed0"):
        if k in out:
            d = out[k]
            print(f"{k:34s} " + "  ".join(
                f"{m} [{d[m]['ci95'][0]:+.3f},{d[m]['ci95'][1]:+.3f}] p={d[m]['p_two_sided']:.3f}"
                for m in ("precision", "recall", "f1")))
    ensemble.run_detection = _run_detection
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
