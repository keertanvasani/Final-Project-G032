"""Confidence calibration for the frozen baseline (PROTOCOL.md §; report Phase 5).

Phase A established that `combined_confidence` carries almost no signal about
correctness under domain shift: selective risk sits near 0.89 across the useful
range. Before an agent can act on uncertainty, the score it acts on has to mean
something. This fits the mapping

    raw confidence  ->  estimated P(detection is correct)

and reports how far off the raw score was.

Three calibrators, per the report's Change 8:

  temperature   one parameter, p' = sigmoid(logit(p) / T). Cannot reorder
                detections, so it fixes over/under-confidence without touching
                ranking — AUROC is invariant under it.
  platt         two parameters, p' = sigmoid(a.logit(p) + b). Adds a shift, so
                it can correct a systematic bias temperature cannot.
  isotonic      non-parametric monotone fit by pool-adjacent-violators. Most
                flexible, most prone to overfitting on small dev sets, which is
                exactly why it is fitted and evaluated on *different* splits.

Discipline. Parameters are fitted on `--fit` and every number reported comes
from `--eval`, a split the fit never saw (PROTOCOL §4.2: S_cal and S_policy are
unseen by weight fitting, and a calibrator fitted and scored on one split
reports its own training error). Never J_test.

Implemented in numpy alone — scipy and sklearn are not project dependencies,
and adding them for three short routines would cost reproducibility for
nothing.

Usage:
    python experiments/calibrate.py --fit J_cal --eval J_policy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "experiments" / "results" / "phase_a"
EPS = 1e-6


# ---------------------------------------------------------------- data
def load(split: str) -> tuple[np.ndarray, np.ndarray]:
    p = CACHE / f"{split}_detections.jsonl"
    if not p.exists():
        raise SystemExit(f"No cache for {split}. Run: run_phase_a.py cache --split {split}")
    conf, correct = [], []
    for line in p.read_text().splitlines():
        d = json.loads(line)
        conf.append(d["combined_confidence"])
        correct.append(1.0 if d["outcome"] == "TP" else 0.0)
    return (np.clip(np.asarray(conf, dtype=float), EPS, 1 - EPS),
            np.asarray(correct, dtype=float))


def logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1.0 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


# ---------------------------------------------------------------- calibrators
def fit_platt(p: np.ndarray, y: np.ndarray, *, fix_a: bool = False,
              iters: int = 2000, lr: float = 0.05) -> tuple[float, float]:
    """Fit p' = sigmoid(a*logit(p) + b) by gradient descent on NLL.

    With fix_a the slope is held at 1 and only the shift is learned; with b
    fixed at 0 this reduces to temperature scaling (T = 1/a), which is how
    `fit_temperature` calls it.
    """
    z = logit(p)
    a, b = 1.0, 0.0
    n = len(z)
    for _ in range(iters):
        q = sigmoid(a * z + b)
        err = q - y
        ga = float(np.dot(err, z) / n)
        gb = float(err.mean())
        if not fix_a:
            a -= lr * ga
        b -= lr * gb
    return a, b


def fit_temperature(p: np.ndarray, y: np.ndarray,
                    iters: int = 2000, lr: float = 0.05) -> float:
    """Fit p' = sigmoid(logit(p)/T). Returns T."""
    z = logit(p)
    a = 1.0
    n = len(z)
    for _ in range(iters):
        q = sigmoid(a * z)
        a -= lr * float(np.dot(q - y, z) / n)
    a = max(a, EPS)
    return 1.0 / a


def fit_isotonic(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool-adjacent-violators. Returns (sorted x, fitted monotone y)."""
    order = np.argsort(p, kind="mergesort")
    x, v = p[order], y[order].astype(float)
    w = np.ones_like(v)

    # Each block holds (sum of values, sum of weights); merge while decreasing.
    vals, wts = [], []
    for vi, wi in zip(v, w):
        vals.append(vi * wi)
        wts.append(wi)
        while len(vals) > 1 and vals[-2] / wts[-2] > vals[-1] / wts[-1]:
            vals[-2] += vals[-1]
            wts[-2] += wts[-1]
            vals.pop()
            wts.pop()
    out = []
    for sv, sw in zip(vals, wts):
        out.extend([sv / sw] * int(sw))
    return x, np.asarray(out, dtype=float)


def apply_isotonic(xs: np.ndarray, ys: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.interp(p, xs, ys, left=ys[0], right=ys[-1])


# ---------------------------------------------------------------- metrics
def ece(p: np.ndarray, y: np.ndarray, bins: int = 15) -> tuple[float, float]:
    """Expected and maximum calibration error over equal-width bins."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
    e = m = 0.0
    n = len(p)
    for b in range(bins):
        sel = idx == b
        if not sel.any():
            continue
        gap = abs(y[sel].mean() - p[sel].mean())
        e += sel.sum() / n * gap
        m = max(m, gap)
    return e, m


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def nll(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auroc(p: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUROC; ties get average rank."""
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    npos, nneg = y.sum(), (1 - y).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def reliability(p, y, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
    rows = []
    for b in range(bins):
        sel = idx == b
        if not sel.any():
            continue
        rows.append({"bin_mid": float((edges[b] + edges[b + 1]) / 2),
                     "mean_conf": float(p[sel].mean()),
                     "empirical_acc": float(y[sel].mean()),
                     "n": int(sel.sum())})
    return rows


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="J_cal")
    ap.add_argument("--eval", dest="ev", default="J_policy")
    ap.add_argument("--bins", type=int, default=15)
    args = ap.parse_args()

    if "test" in args.fit.lower() or "test" in args.ev.lower():
        raise SystemExit("Refusing to touch a test split (PROTOCOL §8).")

    pf, yf = load(args.fit)
    pe, ye = load(args.ev)
    print(f"\nCalibration — fit on {args.fit}, evaluated on {args.ev}")
    print(f"  fit : {len(pf)} detections, {int(yf.sum())} correct ({yf.mean():.1%})")
    print(f"  eval: {len(pe)} detections, {int(ye.sum())} correct ({ye.mean():.1%})\n")

    T = fit_temperature(pf, yf)
    a, b = fit_platt(pf, yf)
    ix, iy = fit_isotonic(pf, yf)

    # Temperature has no bias term, so it can only change the *sharpness* of the
    # scores, never their level. Here the raw score averages ~0.56 while true
    # accuracy is ~0.10, so the NLL optimum drives T to infinity: every logit is
    # squashed to 0 and the calibrator degenerates to a constant 0.5. That is a
    # property of the method against this miscalibration, not a fitting failure,
    # and it is reported rather than dressed up as a tuned parameter.
    T_degenerate = T > 1e3
    t_label = ("temperature (DEGENERATE)" if T_degenerate
               else f"temperature (T={T:.3f})")

    methods = {
        "raw (uncalibrated)": pe,
        t_label: sigmoid(logit(pe) / T),
        f"platt (a={a:.3f}, b={b:.3f})": sigmoid(a * logit(pe) + b),
        "isotonic": apply_isotonic(ix, iy, pe),
    }

    print(f"{'method':<30}{'ECE':>8}{'MCE':>8}{'Brier':>9}{'NLL':>8}{'AUROC':>8}")
    print("-" * 71)
    out = {}
    for name, q in methods.items():
        e, m = ece(q, ye, args.bins)
        row = {"ece": e, "mce": m, "brier": brier(q, ye),
               "nll": nll(q, ye), "auroc": auroc(q, ye),
               "mean_predicted": float(q.mean()),
               "reliability": reliability(q, ye, args.bins)}
        out[name] = row
        print(f"{name:<30}{e:>8.4f}{m:>8.4f}{row['brier']:>9.4f}"
              f"{row['nll']:>8.4f}{row['auroc']:>8.4f}")

    base = out["raw (uncalibrated)"]
    best = min((k for k in out if not k.startswith("raw")),
               key=lambda k: out[k]["ece"])
    print(f"\nbase rate (actual accuracy) : {ye.mean():.4f}")
    print(f"raw mean confidence          : {base['mean_predicted']:.4f}"
          f"   -> overconfident by {base['mean_predicted'] - ye.mean():+.4f}")
    print(f"best calibrator by ECE       : {best}")
    print(f"  ECE   {base['ece']:.4f} -> {out[best]['ece']:.4f}"
          f"   ({100*(base['ece']-out[best]['ece'])/max(base['ece'],EPS):.1f}% reduction)")
    print(f"  Brier {base['brier']:.4f} -> {out[best]['brier']:.4f}")
    if T_degenerate:
        print("\nNOTE: temperature scaling degenerated (T -> inf, every score "
              "squashed to 0.5).\n  It rescales logits but cannot shift their "
              "level, and the raw score is\n  overconfident by "
              f"{base['mean_predicted'] - ye.mean():+.2f} in the mean. Platt's bias "
              "term is what fixes\n  that, which is why it and isotonic succeed "
              "where temperature cannot.")

    print(f"\nAUROC is ~unchanged by calibration ({base['auroc']:.4f} raw) — "
          f"calibration fixes\nwhat the score *means*, not how well it *ranks*. "
          f"Ranking is the agent's job.")

    if args.fit == args.ev:
        print("\nWARNING: fitted and evaluated on the same split — these are "
              "in-sample numbers.\n  Isotonic will look perfect here because it "
              "interpolates its own training\n  data. Use --fit J_cal --eval "
              "J_policy for figures you can report.")

    # reliability diagram
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.6, 5.2), dpi=150)
        ax.plot([0, 1], [0, 1], "--", color="#8FA5AC", lw=1, label="perfect")
        for name, q in methods.items():
            rows = out[name]["reliability"]
            ax.plot([r["mean_conf"] for r in rows],
                    [r["empirical_acc"] for r in rows],
                    marker="o", ms=3.5, lw=1.4, label=name)
        ax.set_xlabel("mean predicted P(correct)")
        ax.set_ylabel("empirical accuracy")
        ax.set_title(f"Reliability — fit {args.fit}, eval {args.ev}", fontsize=11)
        ax.legend(fontsize=7.5, frameon=False)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(alpha=0.25, lw=0.6)
        fig.tight_layout()
        fig.savefig(CACHE / f"{args.ev}_reliability.png")
        print(f"\nWrote {(CACHE / f'{args.ev}_reliability.png').relative_to(ROOT)}")
    except Exception as exc:  # plotting must never lose the numbers
        print(f"\n(reliability diagram skipped: {exc})")

    dest = CACHE / f"{args.ev}_calibration.json"
    dest.write_text(json.dumps(
        {"fit_split": args.fit, "eval_split": args.ev,
         "n_fit": len(pf), "n_eval": len(pe),
         "base_rate": float(ye.mean()),
         "params": {"temperature": T, "platt_a": a, "platt_b": b},
         "methods": out}, indent=2) + "\n")
    print(f"Wrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
