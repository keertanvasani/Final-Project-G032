"""Out-of-distribution scoring for the frozen baseline (report Phase 6).

Phase A's most consequential finding is that the OOD-error rate *rises* with
confidence: the baseline's most confident deep-sea detections are among its most
likely to be confident mis-identifications of taxa it never saw. A closed-set
classifier with no novelty mechanism behaves exactly this way. This module asks
whether any score available at inference separates those detections from correct
ones, and by how much.

Fit eligibility (PROTOCOL §5.1). OOD roles are partitioned by *class*, and each
role is hosted by exactly one image split. A scorer developed on `J_cal` may use
only the `OOD_dev` taxa, even though `J_cal` images also contain `OOD_val`,
`OOD_test` and far-OOD organisms. Without this a scorer would be tuned on shrimp
and then evaluated on shrimp as an "unseen" taxon, and the novelty claim behind
H3 would be false while every split stayed nominally class-disjoint. Far-OOD
(`machine`) is excluded from fitting everywhere.

  positives (OOD) : detections matching a GT in the host split's role classes
  negatives (ID)  : detections matching a known-class GT (TP or CLS_ERR)
  excluded        : other roles' taxa, far-OOD, and unmatched detections
                    (LOC_ERR / BG_FP match no organism, so they are neither)

Scores. The cache stores confidences, not logits or embeddings, so the energy
score and Mahalanobis distance the report also lists are NOT computable here —
they need the pipeline re-run with feature extraction. What is available:

  msp            classifier confidence — the standard cheap baseline
  detector       detector confidence
  combined       the fused score the deployed gate actually uses
  disagreement   1 - classifier confidence, boosted when the detector and
                 classifier disagree; disagreement is itself evidence that a
                 region does not belong to any known class

Metrics: AUROC, FPR@95TPR, AUPR-Out. Near-OOD and far-OOD are reported
separately — averaging them hides that the hard case is the one that matters.

Usage:
    python experiments/ood_detect.py --fit J_cal --eval J_policy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "experiments" / "results" / "phase_a"
SPLITS = Path(__file__).parent / "splits" / "phase_a_splits.json"


def load_rows(split: str) -> list[dict]:
    p = CACHE / f"{split}_detections.jsonl"
    if not p.exists():
        raise SystemExit(f"No cache for {split}. Run: run_phase_a.py cache --split {split}")
    return [json.loads(l) for l in p.read_text().splitlines()]


def scores(rows: list[dict]) -> dict[str, np.ndarray]:
    det = np.array([r["detector_confidence"] for r in rows], dtype=float)
    cls = np.array([r["classifier_confidence"] for r in rows], dtype=float)
    comb = np.array([r["combined_confidence"] for r in rows], dtype=float)
    agree = np.array([1.0 if r.get("agreement") else 0.0 for r in rows])
    # Higher = more OOD-like, so confidence scores are negated.
    return {
        "msp (classifier conf)": -cls,
        "detector conf": -det,
        "combined conf (deployed)": -comb,
        "disagreement-weighted": -(cls * (0.5 + 0.5 * agree)),
    }


def auroc(s: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    npos, nneg = y.sum(), (1 - y).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def fpr_at_tpr(s: np.ndarray, y: np.ndarray, target: float = 0.95) -> float:
    """FPR among ID detections at the threshold giving >= target TPR on OOD."""
    pos = np.sort(s[y == 1])
    if len(pos) == 0 or (y == 0).sum() == 0:
        return float("nan")
    # threshold that keeps at least `target` of positives (scores >= tau)
    k = int(np.floor((1 - target) * len(pos)))
    k = min(max(k, 0), len(pos) - 1)
    tau = pos[k]
    neg = s[y == 0]
    return float((neg >= tau).mean())


def aupr_out(s: np.ndarray, y: np.ndarray) -> float:
    """Average precision treating OOD as the positive class."""
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    npos = ys.sum()
    return float((prec * ys).sum() / npos) if npos else float("nan")


def build(split: str, role_classes: set[str], known: set[str],
          far: set[str]) -> tuple[list[dict], np.ndarray, dict]:
    rows = load_rows(split)
    keep, y = [], []
    counts = {"ood_role": 0, "id": 0, "excluded_other_role": 0,
              "excluded_far": 0, "unmatched": 0}
    for r in rows:
        gt = r.get("gt_class")
        if not r.get("matched_gt") or gt is None:
            counts["unmatched"] += 1
            continue
        if gt in far:
            counts["excluded_far"] += 1
            continue
        if gt in known:
            keep.append(r); y.append(0.0); counts["id"] += 1
        elif gt in role_classes:
            keep.append(r); y.append(1.0); counts["ood_role"] += 1
        else:
            counts["excluded_other_role"] += 1
    return keep, np.asarray(y), counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="J_cal")
    ap.add_argument("--eval", dest="ev", default="J_policy")
    args = ap.parse_args()
    if "test" in args.fit.lower() or "test" in args.ev.lower():
        raise SystemExit("Refusing to touch a test split (PROTOCOL §8).")

    man = json.loads(SPLITS.read_text())
    ls = man["label_space"]
    known, far = set(ls["known"]), set(ls["far_ood"])
    hosts = ls["split_hosts_ood_role"]
    roles = {k: set(v) for k, v in ls["ood_roles"].items()}

    print("\nOOD scoring — Phase A label space (PROTOCOL §2.4, §5.1)")
    print(f"  KNOWN (in-distribution): {', '.join(sorted(known))}")
    print(f"  far-OOD (never fitted) : {', '.join(sorted(far))}\n")

    out = {}
    for split, tag in ((args.fit, "fit"), (args.ev, "eval")):
        role = hosts[split]
        rc = roles[role]
        rows, y, counts = build(split, rc, known, far)
        s = scores(rows)
        print(f"{split}  hosts {role}: {', '.join(sorted(rc))}")
        print(f"  usable: {counts['ood_role']} OOD vs {counts['id']} ID   "
              f"(excluded: {counts['excluded_other_role']} other-role, "
              f"{counts['excluded_far']} far-OOD, {counts['unmatched']} unmatched)")
        if counts["ood_role"] == 0 or counts["id"] == 0:
            print("  -> not evaluable\n")
            continue
        print(f"  {'score':<28}{'AUROC':>8}{'FPR@95':>9}{'AUPR-Out':>10}")
        print("  " + "-" * 55)
        res = {}
        for name, sc in s.items():
            a, f, p = auroc(sc, y), fpr_at_tpr(sc, y), aupr_out(sc, y)
            res[name] = {"auroc": a, "fpr_at_95tpr": f, "aupr_out": p}
            print(f"  {name:<28}{a:>8.4f}{f:>9.4f}{p:>10.4f}")
        best = max(res, key=lambda k: res[k]["auroc"])
        print(f"  best: {best} (AUROC {res[best]['auroc']:.4f})")
        print(f"  base rate: {y.mean():.1%} of matched detections are OOD\n")
        out[split] = {"role": role, "role_classes": sorted(rc),
                      "counts": counts, "n": len(y),
                      "ood_base_rate": float(y.mean()), "scores": res}

    print("Interpretation")
    print("  AUROC 0.50 = the score is useless for novelty; 1.00 = perfect.")
    print("  FPR@95TPR is the operationally honest number: to catch 95% of")
    print("  unseen taxa, this is the fraction of correct detections you also")
    print("  throw away.\n")
    print("NOT computed: energy score and Mahalanobis distance. Both need the")
    print("classifier's logits / penultimate embeddings, which the Phase A cache")
    print("does not store. Adding them means re-running run_phase_a.py with")
    print("feature extraction — the natural next step if these baselines are weak.")

    dest = CACHE / f"{args.ev}_ood.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
