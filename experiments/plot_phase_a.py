"""Render the Phase A motivating figure from a cached report.

Reads experiments/results/phase_a/<split>_report.json (produced by
run_phase_a.py report) and draws the risk-coverage / risk-recall curves with
the frozen baseline's operating point marked. No inference; pure plotting, so
it is cheap to re-run as the figure is refined.

Usage:
    python experiments/plot_phase_a.py --split J_test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "experiments" / "results" / "phase_a"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="J_test")
    args = ap.parse_args()

    rep = json.loads((CACHE / f"{args.split}_report.json").read_text())
    curve = sorted(rep["risk_coverage_curve"], key=lambda p: p["coverage_policy_local"])
    op = rep["baseline_operating_point"]

    cov = [p["coverage_policy_local"] for p in curve]
    rec = [p["selective_recall"] for p in curve]
    risk_c = [p["selective_risk"] for p in curve]
    # risk vs recall needs sorting on recall
    by_rec = sorted(curve, key=lambda p: p["selective_recall"])
    rr_x = [p["selective_recall"] for p in by_rec]
    rr_y = [p["selective_risk"] for p in by_rec]
    ood_x = cov
    ood_y = [p["ood_err_rate"] for p in curve]

    INK, ACCENT, WARN, MUTE = "#12222f", "#0d8b80", "#c0392b", "#8a99a6"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3d0da",
                         "axes.linewidth": 0.8, "figure.dpi": 130})

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1))

    # Panel 1 — risk vs coverage (policy-local)
    ax = axes[0]
    ax.plot(cov, risk_c, color=ACCENT, lw=2)
    ax.scatter([op["coverage_policy_local"]], [op["selective_risk"]],
               color=WARN, zorder=5, s=45)
    ax.annotate(f"baseline gate\nrisk={op['selective_risk']:.2f}",
                (op["coverage_policy_local"], op["selective_risk"]),
                textcoords="offset points", xytext=(-8, -34), fontsize=8.5,
                color=WARN, ha="right")
    ax.set_xlabel("coverage (policy-local)")
    ax.set_ylabel("selective risk")
    ax.set_title("Risk vs coverage", fontsize=11, color=INK)

    # Panel 2 — risk vs selective recall (GT-anchored, comparability axis)
    ax = axes[1]
    ax.plot(rr_x, rr_y, color=ACCENT, lw=2)
    ax.scatter([op["selective_recall"]], [op["selective_risk"]],
               color=WARN, zorder=5, s=45)
    ax.set_xlabel("selective recall (GT-anchored)")
    ax.set_ylabel("selective risk")
    ax.set_title(f"Risk vs recall   (AURC={rep['aurc']['vs_recall']:.2f})",
                 fontsize=11, color=INK)

    # Panel 3 — OOD-error rate among accepted, vs coverage
    ax = axes[2]
    ax.plot(ood_x, ood_y, color=INK, lw=2)
    ax.scatter([op["coverage_policy_local"]], [op["ood_err_rate"]],
               color=WARN, zorder=5, s=45)
    ax.annotate(f"{op['ood_err_rate']:.0%} of accepted\nare unknown taxa",
                (op["coverage_policy_local"], op["ood_err_rate"]),
                textcoords="offset points", xytext=(-8, 12), fontsize=8.5,
                color=WARN, ha="right")
    ax.set_xlabel("coverage (policy-local)")
    ax.set_ylabel("OOD-error rate among accepted")
    ax.set_title("Confident labels on unknown taxa", fontsize=11, color=INK)

    for ax in axes:
        ax.grid(True, color="#eef2f5", lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Aquarium-trained baseline on deep-sea imagery ({args.split}: "
        f"{rep['images']} images, {rep['dives']} dives, {rep['detections']} detections)",
        fontsize=11.5, color=INK, y=1.02)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        out = CACHE / f"{args.split}_risk_coverage.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
