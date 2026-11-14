"""Learned accept/abstain policy — a state-conditional gate (report Phases 7 & 13).

Every gate so far applies a global constant: accept if confidence clears tau,
optionally AND-ed with a novelty threshold. A constant cannot condition on
anything, so a small low-contrast box where the detectors disagree is judged by
the same bar as a large clean one.

This fits a model instead:

    P(correct | calibrated confidence, novelty distance, detector/classifier
                agreement, box geometry, image quality, scene crowding)

and accepts when that probability clears tau. The effective bar then *varies
per detection*, which is the property a threshold cannot express and the point
the project is arguing.

**No exploration data is required for this decision.** Whether accepting was
right is already recorded for every cached detection, so the counterfactual
problem that blocks the ZOOM/RECLASSIFY actions does not arise here. That is
why the accept/abstain policy is buildable now and the full action-value model
is not.

**It also removes the coverage ceiling.** The fixed gate ANDs a hard mask
(`mahalanobis <= m*`), which caps coverage at whatever fraction passes that
mask — 14.7% in the previous run — so its risk-coverage curve simply stops.
A learned score has no such cap: sweeping tau traces the full curve, and areas
become comparable over the whole range rather than a truncated one.

Fitted on `--fit`, reported on `--eval`. Never touches a test split.
Logistic regression in numpy; scipy and sklearn are not project dependencies.

Usage:
    python experiments/learned_gate.py --fit J_cal --eval J_policy
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

EPS = 1e-6
FEATURES = [
    "logit_calibrated", "detector_conf", "classifier_conf", "agreement",
    "log_box_area", "aspect_ratio", "log_ood", "log_blur", "log_n_dets",
]


def load(split: str):
    det_p = CACHE / f"{split}_detections.jsonl"
    img_p = CACHE / f"{split}_images.jsonl"
    ood_p = CACHE / f"{split}_ood_scores.json"
    for p in (det_p, img_p, ood_p):
        if not p.exists():
            raise SystemExit(f"Missing {p.name}. Run the caching / OOD steps first.")

    rows = [json.loads(l) for l in det_p.read_text().splitlines()]
    imgs = {json.loads(l)["file"]: json.loads(l)
            for l in img_p.read_text().splitlines()}
    ood = np.asarray(json.loads(ood_p.read_text())["scores"], dtype=float)
    if len(ood) != len(rows):
        raise SystemExit(f"{split}: {len(ood)} OOD scores vs {len(rows)} detections")

    conf = np.array([r["combined_confidence"] for r in rows], dtype=float)
    y = np.array([r["outcome"] == "TP" for r in rows], dtype=float)
    is_ood = np.array([r["outcome"] == "OOD_ERR" for r in rows], dtype=float)
    dive = np.array([r["dive"] for r in rows])

    det_c = np.array([r.get("detector_confidence") or 0.0 for r in rows], dtype=float)
    cls_c = np.array([r.get("classifier_confidence") or 0.0 for r in rows], dtype=float)
    agree = np.array([1.0 if r.get("agreement") else 0.0 for r in rows])

    area, aspect = [], []
    for r in rows:
        x0, y0, x1, y1 = r.get("bbox", [0, 0, 1, 1])
        w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
        area.append(w * h)
        aspect.append(w / h)
    area = np.asarray(area, dtype=float)
    aspect = np.asarray(aspect, dtype=float)

    blur = np.array([imgs.get(r["file"], {}).get("blur") or 1.0 for r in rows],
                    dtype=float)
    ndet = np.array([imgs.get(r["file"], {}).get("n_detections") or 1
                     for r in rows], dtype=float)

    return dict(rows=rows, conf=conf, y=y, is_ood=is_ood, dive=dive,
                det_c=det_c, cls_c=cls_c, agree=agree, area=area,
                aspect=aspect, ood=ood, blur=blur, ndet=ndet)


def build_matrix(d: dict, cal_scores: np.ndarray) -> np.ndarray:
    p = np.clip(cal_scores, EPS, 1 - EPS)
    return np.column_stack([
        np.log(p / (1 - p)),
        d["det_c"], d["cls_c"], d["agree"],
        np.log1p(d["area"]), np.clip(d["aspect"], 0, 10),
        np.log1p(np.clip(d["ood"], 0, None)),
        np.log1p(np.clip(d["blur"], 0, None)),
        np.log1p(d["ndet"]),
    ])


def fit_logistic(X, y, l2=1e-3, iters=4000, lr=0.15):
    """Gradient descent on regularised NLL. Features are pre-standardised."""
    n, k = X.shape
    Xb = np.column_stack([np.ones(n), X])
    w = np.zeros(k + 1)
    for _ in range(iters):
        z = Xb @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
        g = Xb.T @ (p - y) / n
        g[1:] += l2 * w[1:]
        w -= lr * g
    return w


def predict(X, w):
    Xb = np.column_stack([np.ones(len(X)), X])
    return 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -60, 60)))


def curve(score, correct, is_ood, mask_ok=None):
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
        pts.append({"tau": float(t), "coverage": k / n,
                    "risk": float(1 - correct[sel].mean()),
                    "ood_rate": float(is_ood[sel].mean()), "n": k})
    return pts


def aurc(pts, cmax=None):
    p = sorted(((q["coverage"], q["risk"]) for q in pts
                if cmax is None or q["coverage"] <= cmax))
    if len(p) < 2:
        return float("nan")
    return sum((x1 - x0) * (y0 + y1) / 2 for (x0, y0), (x1, y1) in zip(p, p[1:]))


def at_cov(pts, target):
    c = [q for q in pts if q["coverage"] <= target]
    return max(c, key=lambda q: q["coverage"]) if c else None


def boot_delta(dives, a_sel, b_sel, correct, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(dives)
    idx = {d: np.flatnonzero(dives == d) for d in uniq}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx[d] for d in pick])
        sa, sb = a_sel[ii], b_sel[ii]
        if sa.sum() == 0 or sb.sum() == 0:
            continue
        out.append((1 - correct[ii][sa].mean()) - (1 - correct[ii][sb].mean()))
    if not out:
        return float("nan"), (float("nan"), float("nan"))
    o = np.array(out)
    return float(o.mean()), (float(np.percentile(o, 2.5)),
                             float(np.percentile(o, 97.5)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="J_cal")
    ap.add_argument("--eval", dest="ev", default="J_policy")
    ap.add_argument("--id-retention", type=float, default=0.90)
    args = ap.parse_args()
    if "test" in args.fit.lower() or "test" in args.ev.lower():
        raise SystemExit("Refusing to touch a test split (PROTOCOL §8).")

    known = set(json.loads(SPLITS.read_text())["label_space"]["known"])
    F, E = load(args.fit), load(args.ev)

    # calibrator, fitted on the fit split
    ix, iy = fit_isotonic(np.clip(F["conf"], EPS, 1 - EPS), F["y"])
    cal_f = apply_isotonic(ix, iy, np.clip(F["conf"], EPS, 1 - EPS))
    cal_e = apply_isotonic(ix, iy, np.clip(E["conf"], EPS, 1 - EPS))

    Xf_raw = build_matrix(F, cal_f)
    Xe_raw = build_matrix(E, cal_e)
    mu, sd = Xf_raw.mean(0), Xf_raw.std(0) + EPS
    Xf, Xe = (Xf_raw - mu) / sd, (Xe_raw - mu) / sd

    w = fit_logistic(Xf, F["y"])
    p_learned = predict(Xe, w)

    # the previous fixed 2-D gate, for comparison
    id_f = np.array([r.get("gt_class") in known and r.get("matched_gt") is not None
                     for r in F["rows"]])
    m_star = float(np.quantile(F["ood"][id_f], args.id_retention)) if id_f.sum() else np.inf
    ood_ok = E["ood"] <= m_star

    print(f"\nLearned accept/abstain gate — fit {args.fit}, reported on {args.ev}")
    print(f"  fit  {len(F['y'])} detections, {int(F['y'].sum())} correct "
          f"({F['y'].mean():.1%})")
    print(f"  eval {len(E['y'])} detections, {int(E['y'].sum())} correct "
          f"({E['y'].mean():.1%})   OOD_ERR {E['is_ood'].mean():.1%}\n")

    print("feature weights (standardised; sign = direction of P(correct))")
    print("-" * 56)
    for name, coef in sorted(zip(FEATURES, w[1:]), key=lambda t: -abs(t[1])):
        bar = "#" * int(min(abs(coef) * 12, 28))
        print(f"  {name:<20}{coef:>+8.3f}  {bar}")

    gates = {
        "baseline (raw confidence)": (E["conf"], None),
        "calibrated": (cal_e, None),
        "calibrated + OOD (fixed 2-D)": (cal_e, ood_ok),
        "LEARNED policy": (p_learned, None),
    }
    curves = {k: curve(s, E["y"], E["is_ood"], m) for k, (s, m) in gates.items()}

    reach = {k: max(q["coverage"] for q in pts) for k, pts in curves.items()}
    print(f"\nmaximum coverage reachable by each gate")
    print("-" * 56)
    for k, v in reach.items():
        note = "  <- capped by the hard mask" if "2-D" in k else ""
        print(f"  {k:<32}{v:>7.3f}{note}")

    cmax = min(reach.values())
    print(f"\nAURC over the common range [0, {cmax:.3f}] (lower is better), "
          f"plus operating points")
    print(f"{'gate':<32}{'AURC*':>8}{'AURC-full':>11}{'risk@.30':>10}{'OOD@.30':>9}")
    print("-" * 72)
    summary = {}
    for k, pts in curves.items():
        a_common = aurc(pts, cmax)
        a_full = aurc(pts)
        op = at_cov(pts, 0.30)
        summary[k] = {"aurc_common": a_common, "aurc_full": a_full,
                      "max_coverage": reach[k], "at_cov_30": op, "curve": pts}
        r = f"{op['risk']:.4f}" if op else "  n/a"
        o = f"{op['ood_rate']:.4f}" if op else "  n/a"
        print(f"{k:<32}{a_common:>8.4f}{a_full:>11.4f}{r:>10}{o:>9}")

    # paired comparison: learned vs baseline at matched coverage
    base_pts = curves["baseline (raw confidence)"]
    lp = curves["LEARNED policy"]
    for target in (0.30, 0.50):
        b, l = at_cov(base_pts, target), at_cov(lp, target)
        if not (b and l):
            continue
        b_sel = E["conf"] >= b["tau"]
        l_sel = p_learned >= l["tau"]
        d, ci = boot_delta(E["dive"], b_sel, l_sel, E["y"])
        print(f"\nPaired dive bootstrap at coverage ~{target:.2f} "
              f"(baseline {b['coverage']:.3f} vs learned {l['coverage']:.3f})")
        print(f"  risk      {b['risk']:.4f} -> {l['risk']:.4f}")
        print(f"  OOD rate  {b['ood_rate']:.4f} -> {l['ood_rate']:.4f}")
        print(f"  delta risk (baseline - learned) = {d:+.4f}  "
              f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")
        summary[f"paired@{target}"] = {"delta_risk": d, "ci95": ci,
                                       "baseline": b, "learned": l}

    print("\nA positive delta with a CI excluding zero means the state-conditional")
    print("policy accepts fewer wrong detections than the fixed threshold at the")
    print("same coverage — the claim a global constant cannot make.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), dpi=150)
        for k, pts in curves.items():
            xs = [q["coverage"] for q in pts]
            axes[0].plot(xs, [q["risk"] for q in pts], lw=1.7, label=k)
            axes[1].plot(xs, [q["ood_rate"] for q in pts], lw=1.7)
        axes[0].set_ylabel("selective risk")
        axes[1].set_ylabel("OOD-error rate among accepted")
        for ax in axes:
            ax.set_xlabel("coverage"); ax.grid(alpha=0.25, lw=0.6)
        axes[0].legend(fontsize=8, frameon=False)
        fig.suptitle(f"Learned vs fixed gates on {args.ev}", fontsize=11)
        fig.tight_layout()
        fig.savefig(CACHE / f"{args.ev}_learned_gate.png")
        print(f"\nWrote {(CACHE / f'{args.ev}_learned_gate.png').relative_to(ROOT)}")
    except Exception as exc:
        print(f"\n(figure skipped: {exc})")

    dest = CACHE / f"{args.ev}_learned_gate.json"
    dest.write_text(json.dumps(
        {"fit": args.fit, "eval": args.ev, "m_star": m_star,
         "features": FEATURES, "weights": list(map(float, w)),
         "gates": summary}, indent=2) + "\n")
    print(f"Wrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
