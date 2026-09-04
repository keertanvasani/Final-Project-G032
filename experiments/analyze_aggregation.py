"""Compare per-box confidence aggregations for image-level abstention.

Extends the finding that first-version image-level abstention averaged
detector confidence over all boxes and so over-rejected crowded scenes
(measured 23.7% -> 6.3% rejection when switched to a per-box criterion).

This generalises that to a proper comparison. An image-level gate maps a set
of per-box confidences to one accept/abstain decision. The question is which
*aggregation* best separates images that carry a recoverable known-class
detection from those that do not.

Framing. Label each image "recoverable" iff it has >= 1 known-class TP among
its detections (at any confidence). The gate is then a binary screen: score
each image by agg(confidences), accept when score >= tau. Sweeping tau traces
recoverable-image recall against retention; the aggregation with the larger
area retains more good images at any given retention. "max" is exactly the
per-box ">=1 box clears tau" rule the deployed system uses.

Discipline: runs on a DEV split (default J_cal), never J_test (PROTOCOL §8).

Usage:
    python experiments/analyze_aggregation.py --split J_cal
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "experiments" / "results" / "phase_a"


def agg_mean(cs): return statistics.fmean(cs) if cs else 0.0
def agg_max(cs): return max(cs) if cs else 0.0
def agg_median(cs): return statistics.median(cs) if cs else 0.0
def agg_top3(cs): return statistics.fmean(sorted(cs, reverse=True)[:3]) if cs else 0.0

AGGREGATIONS = {
    "mean (old gate)": agg_mean,
    "median": agg_median,
    "top-3 mean": agg_top3,
    "max (per-box / current)": agg_max,
}


def load(split: str):
    det_path = CACHE / f"{split}_detections.jsonl"
    img_path = CACHE / f"{split}_images.jsonl"
    if not det_path.exists():
        raise SystemExit(f"No cache for {split}. Run: run_phase_a.py cache --split {split}")
    dets = [json.loads(l) for l in det_path.read_text().splitlines()]
    imgs = [json.loads(l) for l in img_path.read_text().splitlines()]

    conf_by_img = defaultdict(list)
    tp_by_img = defaultdict(bool)
    for d in dets:
        conf_by_img[d["file"]].append(d["combined_confidence"])
        if d["outcome"] == "TP":
            tp_by_img[d["file"]] = True

    images = []
    for im in imgs:
        f = im["file"]
        images.append({
            "file": f,
            "confs": conf_by_img.get(f, []),
            "recoverable": tp_by_img.get(f, False),
        })
    return images


def curve(images, agg):
    scored = [(agg(im["confs"]), im["recoverable"]) for im in images]
    n = len(scored)
    n_good = sum(1 for _, r in scored if r)
    # sweep tau over observed scores (descending): retain score >= tau
    taus = sorted({round(s, 4) for s, _ in scored} | {0.0}, reverse=True)
    pts = []
    for tau in taus:
        retained = [(s, r) for s, r in scored if s >= tau]
        kept_good = sum(1 for _, r in retained if r)
        pts.append({
            "tau": tau,
            "retention": len(retained) / n if n else 0.0,
            "recoverable_recall": kept_good / n_good if n_good else 0.0,
            "yield": kept_good / len(retained) if retained else 0.0,
        })
    return pts, n, n_good


def auc(pts):
    """Area under recoverable-recall vs retention (higher = better screen)."""
    p = sorted(((q["retention"], q["recoverable_recall"]) for q in pts))
    return sum((x1 - x0) * (y0 + y1) / 2 for (x0, y0), (x1, y1) in zip(p, p[1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="J_cal")
    ap.add_argument("--tau", type=float, default=0.40, help="operating point to tabulate")
    args = ap.parse_args()

    images = load(args.split)
    n = len(images)
    n_good = sum(1 for im in images if im["recoverable"])
    print(f"\nAbstention-aggregation comparison on {args.split}")
    print(f"images {n}   recoverable (>=1 known TP) {n_good} ({n_good/n:.1%})\n")

    print(f"{'aggregation':<26}{'AUC':>7}   at tau={args.tau}: "
          f"{'retain':>7}{'recall':>8}{'yield':>7}")
    print("-" * 64)
    results = {}
    for name, fn in AGGREGATIONS.items():
        pts, _, _ = curve(images, fn)
        a = auc(pts)
        op = min(pts, key=lambda q: abs(q["tau"] - args.tau))
        print(f"{name:<26}{a:>7.3f}   {'':7}{op['retention']:>7.1%}"
              f"{op['recoverable_recall']:>8.1%}{op['yield']:>7.1%}")
        results[name] = {"auc": a, "operating_point": op,
                         "curve": pts}

    best = max(results, key=lambda k: results[k]["auc"])
    worst = min(results, key=lambda k: results[k]["auc"])
    print(f"\nbest screen : {best}  (AUC {results[best]['auc']:.3f})")
    print(f"worst screen: {worst}  (AUC {results[worst]['auc']:.3f})")
    print(f"\nAt tau={args.tau}, switching {worst} -> {best} changes "
          f"recoverable-image recall "
          f"{results[worst]['operating_point']['recoverable_recall']:.1%} -> "
          f"{results[best]['operating_point']['recoverable_recall']:.1%} "
          f"at retention "
          f"{results[worst]['operating_point']['retention']:.1%} -> "
          f"{results[best]['operating_point']['retention']:.1%}.")

    out = CACHE / f"{args.split}_aggregation.json"
    out.write_text(json.dumps(
        {"split": args.split, "images": n, "recoverable": n_good,
         "tau": args.tau,
         "results": {k: {"auc": v["auc"], "operating_point": v["operating_point"]}
                     for k, v in results.items()}},
        indent=2) + "\n")
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
