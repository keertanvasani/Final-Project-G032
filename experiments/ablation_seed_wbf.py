"""Does fusing a second detector seed help the deep-sea pipeline?

Compares the deployed deep-sea stack (one YOLOv8-n, seed 0) with the same
stack plus a second YOLOv8-n trained with a different seed, fused by Weighted
Boxes Fusion with equal weights. Everything else (classifier, enhancement,
text filter, gates, zoom, vote, Known/Unknown flag) is unchanged.

Two pipeline configurations from ablation_deepsea.py are measured:

    A  detector + classifier only
    F  the full deployed agent

For each, precision and recall of the single-seed and the fused stack are
scored on S_policy with the harness taxonomy, and the difference
(fused - single) gets a paired 95% dive-cluster bootstrap interval and a
two-sided bootstrap p-value: both systems are rescored on the same resampled
dives, so dive-to-dive difficulty cancels out. S_test is never touched.

Usage:
    python experiments/ablation_seed_wbf.py [--seeds 1 2] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from ablation_deepsea import CONFIGS, DIVE_RE, MAN, SPLIT, image_path, run_config  # noqa: E402
from eval import harness  # noqa: E402
from pipeline import EnsemblePipeline  # noqa: E402

DS = ROOT / "models" / "phase_b_machine"
OUT = ROOT / "experiments" / "results" / "phase_b_machine" / "ablation_seed_wbf_S_policy.json"
WBF_EQUAL = [1, 1]


def per_dive(pipe, flags, files, gt_all):
    """Score every image; return {dive: {"tp","n","rec"}} plus outcome counts and ms/image."""
    agg = defaultdict(lambda: {"tp": 0, "n": 0, "rec": 0})
    oc: Counter = Counter()
    t0, n_img = time.perf_counter(), 0
    for fn in files:
        p = image_path(fn)
        img = cv2.imread(str(p)) if p else None
        if img is None:
            continue
        n_img += 1
        dets, _ = run_config(img, pipe, flags)
        recs = harness.match_and_classify(dets, gt_all.get(fn, []))
        a = agg[DIVE_RE.match(fn).group(1)]
        a["n"] += len(recs)
        a["tp"] += sum(r["outcome"] == harness.TP for r in recs)
        a["rec"] += len({r["matched_gt"] for r in recs if r["outcome"] == harness.TP})
        oc.update(r["outcome"] for r in recs)
    ms = (time.perf_counter() - t0) * 1000.0 / max(n_img, 1)
    return dict(agg), dict(oc), ms


def pr(agg, kgt_by_dive, dives):
    tp = sum(agg.get(d, {}).get("tp", 0) for d in dives)
    n = sum(agg.get(d, {}).get("n", 0) for d in dives)
    rec = sum(agg.get(d, {}).get("rec", 0) for d in dives)
    kgt = sum(kgt_by_dive[d] for d in dives)
    return (tp / n if n else 0.0), (rec / kgt if kgt else 0.0)


def paired_bootstrap(a0, a1, kgt_by_dive, n_boot=2000, seed=0):
    rng = random.Random(seed)
    dives = list(kgt_by_dive)
    dp, dr = [], []
    for _ in range(n_boot):
        s = [dives[rng.randrange(len(dives))] for _ in dives]
        p0, r0 = pr(a0, kgt_by_dive, s)
        p1, r1 = pr(a1, kgt_by_dive, s)
        dp.append(p1 - p0)
        dr.append(r1 - r0)

    def summ(v):
        v = sorted(v)
        lo, hi = v[int(0.025 * n_boot)], v[int(0.975 * n_boot) - 1]
        p = 2 * min(sum(x <= 0 for x in v), sum(x >= 0 for x in v)) / n_boot
        return {"ci95": [lo, hi], "p_two_sided": min(p, 1.0)}

    return summ(dp), summ(dr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2],
                    help="second-detector seeds to try against seed 0")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    man = json.loads(MAN.read_text())
    harness.KNOWN_CLASSES = set(man["label_space"]["known"])
    files = sorted(man["files"][SPLIT])[: args.limit]
    gt_all = harness.load_ground_truth()
    kgt_by_dive: Counter = Counter()
    for fn in files:
        kgt_by_dive[DIVE_RE.match(fn).group(1)] += harness.known_gt_count(gt_all.get(fn, []))

    pipe = EnsemblePipeline(profile="deepsea")
    assert pipe.model_b is None, "deepsea profile is expected to be single-detector"
    configs = [c for c in CONFIGS if c[0] in ("A", "F")]

    out = {"split": SPLIT, "images": len(files), "dives": len(kgt_by_dive),
           "known_gt": sum(kgt_by_dive.values()), "wbf_weights": WBF_EQUAL, "rows": {}}
    t_start = time.time()
    for key, label, flags in configs:
        pipe.model_b, pipe.wbf_weights = None, None
        base, base_oc, base_ms = per_dive(pipe, flags, files, gt_all)
        p0, r0 = pr(base, kgt_by_dive, list(kgt_by_dive))
        row = {"label": label, "seed0": {"precision": p0, "recall": r0, "outcomes": base_oc,
                                         "ms_per_image": base_ms}}
        print(f"{key} {label:24s} seed0      P {p0:.3f} R {r0:.3f}  {base_ms:6.0f} ms/img "
              f"({time.time() - t_start:.0f}s)", flush=True)
        for s in args.seeds:
            w = DS / f"yolov8n_seed{s}" / "weights" / "best.pt"
            pipe.model_b, pipe.wbf_weights = YOLO(str(w)), WBF_EQUAL
            fused, oc, ms = per_dive(pipe, flags, files, gt_all)
            p1, r1 = pr(fused, kgt_by_dive, list(kgt_by_dive))
            bp, br = paired_bootstrap(base, fused, kgt_by_dive)
            row[f"seed0+seed{s}"] = {"precision": p1, "recall": r1, "outcomes": oc, "ms_per_image": ms,
                                     "delta_precision": p1 - p0, "delta_precision_test": bp,
                                     "delta_recall": r1 - r0, "delta_recall_test": br}
            print(f"{key} {label:24s} seed0+{s}    P {p1:.3f} ({p1 - p0:+.3f} "
                  f"[{bp['ci95'][0]:+.3f},{bp['ci95'][1]:+.3f}] p={bp['p_two_sided']:.3f})  "
                  f"R {r1:.3f} ({r1 - r0:+.3f} [{br['ci95'][0]:+.3f},{br['ci95'][1]:+.3f}] "
                  f"p={br['p_two_sided']:.3f})  {ms:6.0f} ms/img ({time.time() - t_start:.0f}s)", flush=True)
        out["rows"][key] = row
        pipe.model_b, pipe.wbf_weights = None, None

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
