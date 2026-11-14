"""Do the image-level actions earn their cost? (report Phase 15)

`log_image_actions.py` ran three whole-pipeline variants over every image:
raw (the baseline), ENHANCE forced on, and a full-frame re-detection at double
resolution. This compares them.

The outcome is counted per image rather than per detection, because enhancement
changes the pixels and the detector may return a different number of boxes — so
detections cannot be paired across variants. The quantity compared is the number
of correct detections (TP) an image yields under each variant.

ENHANCE is reported twice: over all images, and restricted to the images where
the deployed rule would actually fire. The rule only enhances when the image is
dark, bright, low-contrast or blurry, so the all-images figure answers "should
we always enhance?" while the restricted figure answers "is the rule's trigger
picking the right images?"

Usage:
    python experiments/analyze_image_actions.py --split J_policy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ACT = ROOT / "experiments" / "results" / "actions"


def boot_diff(dives, a, b, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(dives)
    idx = {d: np.flatnonzero(dives == d) for d in uniq}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx[d] for d in pick])
        if len(ii):
            out.append(a[ii].mean() - b[ii].mean())
    o = np.array(out)
    return float(o.mean()), (float(np.percentile(o, 2.5)),
                             float(np.percentile(o, 97.5)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="J_policy")
    args = ap.parse_args()

    p = ACT / f"{args.split}_image_actions.jsonl"
    if not p.exists():
        raise SystemExit(f"Missing {p.name}. Run log_image_actions.py first.")
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    dive = np.array([r["dive"] for r in rows])
    fires = np.array([r["rule_would_enhance"] for r in rows])

    tp = {v: np.array([r[v]["TP"] for r in rows], dtype=float)
          for v in ("raw", "enhanced", "hires")}
    dets = {v: np.array([r[f"n_det_{'enh' if v=='enhanced' else v}"] for r in rows],
                        dtype=float) for v in ("raw", "enhanced", "hires")}
    ms = {v: np.array([r[f"ms_{v}"] for r in rows], dtype=float)
          for v in ("raw", "enhanced", "hires")}

    print(f"\n=== Image-level actions on {args.split} ===\n")
    print(f"  images {len(rows)}   ground-truth objects "
          f"{sum(r['n_gt'] for r in rows)}")
    print(f"  rule would enhance: {int(fires.sum())} ({fires.mean():.1%})\n")

    print(f"  {'variant':<12}{'total TP':>10}{'TP/image':>10}"
          f"{'detections':>12}{'ms/image':>10}")
    print("  " + "-" * 54)
    for v in ("raw", "enhanced", "hires"):
        print(f"  {v:<12}{int(tp[v].sum()):>10}{tp[v].mean():>10.4f}"
              f"{int(dets[v].sum()):>12}{ms[v].mean():>10.1f}")

    print("\n  paired dive bootstrap on TP per image, action minus baseline")
    res = {}
    for name, key, mask, label in (
        ("ENHANCE", "enhanced", None, "all images"),
        ("ENHANCE", "enhanced", fires, "images the rule fires on"),
        ("FULL_IMAGE_REDETECT", "hires", None, "all images"),
    ):
        a, b, dv = tp[key], tp["raw"], dive
        if mask is not None:
            if mask.sum() == 0:
                continue
            a, b, dv = a[mask], b[mask], dive[mask]
        d, ci = boot_diff(dv, a, b)
        star = "" if ci[0] <= 0 <= ci[1] else "   *"
        print(f"    {name:<22}{label:<26}{d:+.4f}  "
              f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]{star}")
        res[f"{name}|{label}"] = {"delta_tp_per_image": d, "ci95": ci,
                                  "n": int(len(a))}
    print("    * = CI excludes zero")

    print("\n  cost of each action, over the baseline")
    for v, name in (("enhanced", "ENHANCE"), ("hires", "FULL_IMAGE_REDETECT")):
        extra = (ms[v] - ms["raw"]).mean()
        tot = (ms[v] - ms["raw"]).sum() / 1000
        gain = tp[v].sum() - tp["raw"].sum()
        print(f"    {name:<22}{extra:>+8.1f} ms/image   {tot:>+7.1f} s total"
              f"   {gain:+.0f} TP")
        if gain > 0:
            print(f"      -> {tot/gain:.2f} s per additional correct detection")

    dest = ACT / f"{args.split}_image_actions_analysis.json"
    dest.write_text(json.dumps({
        "split": args.split, "n_images": len(rows),
        "rule_fire_rate": float(fires.mean()),
        "total_tp": {v: int(tp[v].sum()) for v in tp},
        "mean_ms": {v: float(ms[v].mean()) for v in ms},
        "bootstrap": res,
    }, indent=2) + "\n")
    print(f"\nWrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
