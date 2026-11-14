"""Selective-prediction comparison: does the uncertainty layer move the curve?

Everything before this measured components. This asks the question the project
exists to answer: at the same coverage, does an uncertainty-aware gate accept
fewer wrong detections than the deployed confidence gate?

Three gates, evaluated on the same detections so the comparison is paired:

  baseline      accept if combined_confidence >= tau   (what ships today)
  calibrated    accept if isotonic(confidence) >= tau  (PROTOCOL calibration)
  ood-aware     accept if isotonic(confidence) >= tau AND mahalanobis <= m*
                where m* is chosen on the FIT split at a fixed ID-retention
                rate, never on the split being reported

Reported per PROTOCOL §3:

  coverage         accepted / all detections
  selective risk   incorrect among accepted        (lower is better)
  OOD-error rate   accepted detections whose GT is outside the label space
                   — the quantity H3 is about
  AURC             area under risk vs coverage     (lower is better)

Calibration cannot change ranking, so the calibrated curve is expected to sit
on top of the baseline. It is included precisely to make that visible: any
separation between them would indicate a bug, and the real comparison is
baseline vs ood-aware.

Usage:
    python experiments/risk_coverage.py --fit J_cal --eval J_policy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).parent
CACHE = ROOT / "experiments" / "results" / "phase_a"
SPLITS = EXP / "splits" / "phase_a_splits.json"
sys.path.insert(0, str(EXP))
from calibrate import apply_isotonic, fit_isotonic  # noqa: E402


def load(split: str):
    rows = [json.loads(l) for l in
            (CACHE / f"{split}_detections.jsonl").read_text().splitlines()]
    p = CACHE / f"{split}_ood_scores.json"
    ood = np.asarray(json.loads(p.read_text())["scores"], dtype=float) if p.exists() else None
    if ood is not None and len(ood) != len(rows):
        raise SystemExit(f"{split}: {len(ood)} OOD scores for {len(rows)} detections")
    conf = np.array([r["combined_confidence"] for r in rows], dtype=float)
    correct = np.array([r["outcome"] == "TP" for r in rows], dtype=float)
    is_ood = np.array([r["outcome"] == "OOD_ERR" for r in rows], dtype=float)
    dive = np.array([r["dive"] for r in rows])
    return rows, conf, correct, is_ood, dive, ood


def curve(score: np.ndarray, correct: np.ndarray, is_ood: np.ndarray,
          mask_ok: np.ndarray | None = None):
    """Sweep a threshold over `score`; optionally AND with a hard mask."""
    n = len(score)
    taus = np.unique(np.round(score, 5))[::-1]
    pts = []
    for t in taus:
        sel = score >= t
        if mask_ok is not None:
            sel = sel & mask_ok
        k = int(sel.sum())
        if k == 0:
            continue
        pts.append({"tau": float(t),
                    "coverage": k / n,
                    "risk": float(1.0 - correct[sel].mean()),
                    "ood_rate": float(is_ood[sel].mean()),
                    "n": k})
    return pts


def aurc(pts):
    p = sorted(((q["coverage"], q["risk"]) for q in pts))
    if len(p) < 2:
        return float("nan")
    return sum((x1 - x0) * (y0 + y1) / 2 for (x0, y0), (x1, y1) in zip(p, p[1:]))


def at_coverage(pts, target):
    cands = [q for q in pts if q["coverage"] <= target]
    return max(cands, key=lambda q: q["coverage"]) if cands else None


def boot_delta(dives, a_ok, b_ok, correct, n_boot=1000, seed=0):
    """Paired cluster bootstrap over dives of (risk_a - risk_b)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(dives)
    idx_by_dive = {d: np.flatnonzero(dives == d) for d in uniq}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_dive[d] for d in pick])
        sa, sb = a_ok[idx], b_ok[idx]
        if sa.sum() == 0 or sb.sum() == 0:
            continue
        out.append((1 - correct[idx][sa].mean()) - (1 - correct[idx][sb].mean()))
    if not out:
        return float("nan"), (float("nan"), float("nan"))
    o = np.array(out)
    return float(o.mean()), (float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="J_cal")
    ap.add_argument("--eval", dest="ev", default="J_policy")
    ap.add_argument("--id-retention", type=float, default=0.90,
                    help="ID-retention rate used to pick the OOD threshold on --fit")
    args = ap.parse_args()
    if "test" in args.fit.lower() or "test" in args.ev.lower():
        raise SystemExit("Refusing to touch a test split (PROTOCOL §8).")

    man = json.loads(SPLITS.read_text())
    known = set(man["label_space"]["known"])

    rf, cf, yf, of, _, mf = load(args.fit)
    re_, ce, ye, oe, dive_e, me = load(args.ev)
    if mf is None or me is None:
        raise SystemExit("Missing OOD scores. Run ood_score_detections.py first.")

    # calibrator fitted on the fit split
    ix, iy = fit_isotonic(np.clip(cf, 1e-6, 1 - 1e-6), yf)
    cal_e = apply_isotonic(ix, iy, np.clip(ce, 1e-6, 1 - 1e-6))

    # OOD threshold chosen on the FIT split at a fixed ID retention
    id_f = np.array([r.get("gt_class") in known and r.get("matched_gt") is not None
                     for r in rf])
    m_star = float(np.quantile(mf[id_f], args.id_retention)) if id_f.sum() else float("inf")
    ood_ok = me <= m_star

    print(f"\nSelective prediction — calibrator and OOD threshold from {args.fit}, "
          f"reported on {args.ev}")
    print(f"  detections {len(ce)}   correct {int(ye.sum())} ({ye.mean():.1%})   "
          f"OOD_ERR {int(oe.sum())} ({oe.mean():.1%})")
    print(f"  OOD threshold m* = {m_star:.1f} "
          f"(retains {args.id_retention:.0%} of fit-split ID detections; "
          f"passes {ood_ok.mean():.1%} of {args.ev})\n")

    gates = {
        "baseline (raw confidence)": (ce, None),
        "calibrated": (cal_e, None),
        "calibrated + OOD-aware": (cal_e, ood_ok),
    }
    curves = {k: curve(s, ye, oe, m) for k, (s, m) in gates.items()}

    # A hard mask caps the coverage a gate can reach, so its risk-coverage area
    # is integrated over a shorter x-range than an unmasked gate's. Comparing
    # those raw areas would credit the OOD gate for the coverage it CANNOT
    # reach. Every AURC below is therefore integrated over the common range
    # [0, c_max] that all three gates attain, and the operating point is the
    # coverage all three can actually deliver.
    c_max = min(max(q["coverage"] for q in pts) for pts in curves.values())
    print(f"Common coverage range for comparison: [0, {c_max:.3f}] "
          f"(the OOD gate's ceiling; areas over unequal ranges are not comparable)\n")

    print(f"{'gate':<28}{'AURC*':>8}{'cover':>8}{'risk':>8}{'OOD rate':>10}")
    print("-" * 64)
    summary = {}
    for name, pts in curves.items():
        clipped = [q for q in pts if q["coverage"] <= c_max]
        a = aurc(clipped)
        op = at_coverage(pts, c_max)
        summary[name] = {"aurc_common_range": a, "c_max": c_max,
                         "operating_point": op, "curve": pts}
        if op:
            print(f"{name:<28}{a:>8.4f}{op['coverage']:>8.3f}"
                  f"{op['risk']:>8.4f}{op['ood_rate']:>10.4f}")
    print(f"\n* AURC over the common range only — lower is better.")

    # paired comparison at matched coverage
    base_pts, ood_pts = curves["baseline (raw confidence)"], curves["calibrated + OOD-aware"]
    tgt = c_max
    b_op, o_op = at_coverage(base_pts, tgt), at_coverage(ood_pts, tgt)
    if b_op and o_op:
        b_sel = ce >= b_op["tau"]
        o_sel = (cal_e >= o_op["tau"]) & ood_ok
        d, ci = boot_delta(dive_e, b_sel, o_sel, ye)
        print(f"\nPaired dive bootstrap at coverage ~{tgt:.2f} "
              f"(baseline {b_op['coverage']:.3f} vs OOD-aware {o_op['coverage']:.3f})")
        print(f"  risk      {b_op['risk']:.4f} -> {o_op['risk']:.4f}")
        print(f"  OOD rate  {b_op['ood_rate']:.4f} -> {o_op['ood_rate']:.4f}")
        print(f"  delta risk (baseline - ood-aware) = {d:+.4f}  "
              f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")
        print("  positive delta and a CI excluding 0 favours the OOD-aware gate.")
        summary["paired"] = {"target_coverage": tgt, "delta_risk": d, "ci95": ci,
                             "baseline": b_op, "ood_aware": o_op}

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), dpi=150)
        for name, pts in curves.items():
            xs = [q["coverage"] for q in pts]
            axes[0].plot(xs, [q["risk"] for q in pts], lw=1.6, label=name)
            axes[1].plot(xs, [q["ood_rate"] for q in pts], lw=1.6, label=name)
        axes[0].set_ylabel("selective risk"); axes[1].set_ylabel("OOD-error rate among accepted")
        for ax in axes:
            ax.set_xlabel("coverage"); ax.grid(alpha=0.25, lw=0.6)
        axes[0].legend(fontsize=8, frameon=False)
        fig.suptitle(f"Selective prediction on {args.ev}", fontsize=11)
        fig.tight_layout()
        fig.savefig(CACHE / f"{args.ev}_risk_coverage_gates.png")
        print(f"\nWrote {(CACHE / f'{args.ev}_risk_coverage_gates.png').relative_to(ROOT)}")
    except Exception as exc:
        print(f"\n(figure skipped: {exc})")

    dest = CACHE / f"{args.ev}_gate_comparison.json"
    dest.write_text(json.dumps({"fit": args.fit, "eval": args.ev,
                                "m_star": m_star, "gates": summary}, indent=2) + "\n")
    print(f"Wrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
