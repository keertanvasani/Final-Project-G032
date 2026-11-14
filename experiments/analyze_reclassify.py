"""Is the third opinion worth buying? (report Phase 15)

`log_reclassify_outcomes.py` forced the looser-crop re-classification on every
detection and scored three competing policies against the same ground truth:

    rule        majority vote over detector, 5% classifier, 15% classifier
    detector    always take the detector's class, never re-classify
    classifier  always take the 5% classifier's class, never re-classify

Only `rule` pays for the extra classification. The comparison is therefore not
"does the rule help versus nothing" but "does it beat the free alternatives",
which is the question a compute budget actually asks.

Reported on disagreements — the only cases the rule fires on, and the only ones
where the vote can change anything — and over all detections for context.

Usage:
    python experiments/analyze_reclassify.py --split J_policy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ACT = ROOT / "experiments" / "results" / "actions"
POLICIES = ["rule", "detector_only", "classifier_only"]


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
    if "test" in args.split.lower():
        raise SystemExit("Refusing to touch a test split (PROTOCOL §8).")

    p = ACT / f"{args.split}_reclassify_outcomes.jsonl"
    if not p.exists():
        raise SystemExit(f"Missing {p.name}. Run log_reclassify_outcomes.py first.")
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    dis = [r for r in rows if r["rule_would_fire"]]
    dive_all = np.array([r["dive"] for r in rows])
    dive_dis = np.array([r["dive"] for r in dis])

    def corr(rs, pol):
        return np.array([r[f"outcome_{pol}"] == "TP" for r in rs], dtype=float)

    print(f"\n=== RECLASSIFY_LARGER_CROP on {args.split} ===\n")
    print(f"  detections                {len(rows)}")
    print(f"  disagreements (rule fires){len(dis):>6} ({len(dis)/len(rows):.1%})")
    n_swing = sum(1 for r in dis if r["loose_changed_vote"])
    print(f"  loose opinion was a third, distinct class: {n_swing} "
          f"({n_swing/max(len(dis),1):.1%} of firings)")
    print("    -> in those cases the vote deadlocks and defaults to the detector,")
    print("       so the extra classification cannot change the answer.")

    print(f"\n  correctness on the {len(dis)} disagreements")
    print(f"  {'policy':<20}{'correct':>9}{'rate':>9}")
    print("  " + "-" * 38)
    for pol in POLICIES:
        c = corr(dis, pol)
        print(f"  {pol:<20}{int(c.sum()):>9}{c.mean():>9.4f}")

    print(f"\n  correctness over all {len(rows)} detections")
    print(f"  {'policy':<20}{'correct':>9}{'rate':>9}")
    print("  " + "-" * 38)
    for pol in POLICIES:
        c = corr(rows, pol)
        print(f"  {pol:<20}{int(c.sum()):>9}{c.mean():>9.4f}")

    print("\n  paired dive bootstrap, rule minus each free alternative")
    print("  (positive favours paying for the third opinion)")
    res = {}
    for base in ("detector_only", "classifier_only"):
        for label, rs, dv in (("disagreements", dis, dive_dis),
                              ("all detections", rows, dive_all)):
            d, ci = boot_diff(dv, corr(rs, "rule"), corr(rs, base))
            flag = "" if ci[0] <= 0 <= ci[1] else "   *"
            print(f"    vs {base:<18}{label:<16}{d:+.5f}  "
                  f"95% CI [{ci[0]:+.5f}, {ci[1]:+.5f}]{flag}")
            res[f"{base}|{label}"] = {"delta": d, "ci95": ci}
    print("    * = CI excludes zero")

    cost = np.array([r["cost_ms"] for r in dis], dtype=float)
    print(f"\n  measured cost {cost.mean():.1f} ms per firing, "
          f"{cost.sum()/1000:.1f} s across {len(dis)} firings on this split")

    best_free = max(("detector_only", "classifier_only"),
                    key=lambda b: corr(rows, b).mean())
    gain = corr(rows, "rule").sum() - corr(rows, best_free).sum()
    print(f"\n  best free alternative: {best_free} "
          f"({corr(rows, best_free).mean():.4f})")
    print(f"  extra correct detections bought by the rule: {gain:+.0f} "
          f"for {cost.sum()/1000:.1f} s")
    if gain > 0:
        print(f"  -> {cost.sum()/1000/gain:.1f} s of compute per additional "
              f"correct detection")

    dest = ACT / f"{args.split}_reclassify_analysis.json"
    dest.write_text(json.dumps({
        "split": args.split, "n": len(rows), "n_disagreements": len(dis),
        "rates_all": {p: float(corr(rows, p).mean()) for p in POLICIES},
        "rates_disagreements": {p: float(corr(dis, p).mean()) for p in POLICIES},
        "bootstrap": res, "mean_cost_ms": float(cost.mean()),
        "total_cost_s": float(cost.sum() / 1000),
    }, indent=2) + "\n")
    print(f"\nWrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
