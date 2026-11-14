"""Diagnose and fix the Mahalanobis collapse on retrained features.

On aquarium features Mahalanobis scored 0.82 far / 0.62 near out of sample. On
the retrained classifier it scored 0.42 / 0.44 — below chance — while scoring
0.87 / 0.90 in-sample on the very same fit. A gap that size is the signature of
a broken estimator, not of anti-informative features, so the number was held
back rather than reported.

The prime suspect is the shrinkage term. `ood_features.py` adds a fixed
`1e-3 * I` to a 512x512 covariance estimated from ~966 samples. That is
absolute, not scaled to the data: if the retrained features have a larger
magnitude than the aquarium ones, 1e-3 is negligible against their variance,
the covariance stays ill-conditioned, and its pseudo-inverse amplifies noise
directions that carry no class information.

Variants tested, all fitted on in-distribution crops only and scored
out-of-sample:

  absolute shrinkage    cov + eps*I                    (the current method)
  relative shrinkage    cov + eps*trace(cov)/d*I       (scale-aware)
  diagonal              variances only, no covariance  (d parameters, not d^2)
  L2-normalised         unit-norm features, relative shrinkage
  euclidean             distance to the mean, no covariance at all
  kNN                   mean distance to the k nearest ID crops (non-parametric)

If a scale-aware variant recovers, the finding is "Mahalanobis needs
regularisation matched to the feature scale", not "retrained features destroy
distance-based novelty detection".

Usage:
    python experiments/ood_maha_variants.py --fit S_cal --eval S_policy \
        --weights models/phase_b/classifier_seed0/best_classifier.pth \
        --class-names models/phase_b/classifier_seed0/class_names.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXP = Path(__file__).parent
CACHE = ROOT / "experiments" / "results" / "phase_a"
sys.path.insert(0, str(EXP))
from ood_features import auroc, embed, fpr_at_tpr  # noqa: E402


def maha_scorer(X_id, eps_abs=None, eps_rel=None, diagonal=False, l2=False):
    """Return a scoring function fitted on in-distribution features only."""
    X = X_id.copy()
    if l2:
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    mu = X.mean(0)
    Xc = X - mu
    d = X.shape[1]

    if diagonal:
        var = Xc.var(0) + 1e-8
        def score(Y):
            if len(Y) == 0:
                return np.zeros(0)
            if l2:
                Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)
            return (((Y - mu) ** 2) / var).sum(1)
        return score

    cov = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    if eps_rel is not None:
        # Scale the ridge to the data: trace(cov)/d is the mean variance per
        # dimension, so eps_rel is a fraction of typical feature variance and
        # means the same thing regardless of how large the features are.
        cov = cov + np.eye(d) * (eps_rel * np.trace(cov) / d)
    else:
        cov = cov + np.eye(d) * eps_abs
    P = np.linalg.pinv(cov)

    def score(Y):
        if len(Y) == 0:
            return np.zeros(0)
        if l2:
            Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)
        D = Y - mu
        return np.einsum("ij,jk,ik->i", D, P, D)
    return score


def euclid_scorer(X_id):
    mu = X_id.mean(0)
    return lambda Y: (((Y - mu) ** 2).sum(1) if len(Y) else np.zeros(0))


def knn_scorer(X_id, k=10):
    Xn = X_id / (np.linalg.norm(X_id, axis=1, keepdims=True) + 1e-9)

    def score(Y):
        if len(Y) == 0:
            return np.zeros(0)
        Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)
        out = np.empty(len(Yn))
        for i in range(0, len(Yn), 256):
            chunk = Yn[i:i + 256]
            d = 1.0 - chunk @ Xn.T          # cosine distance
            part = np.partition(d, k, axis=1)[:, :k]
            out[i:i + 256] = part.mean(1)
        return out
    return score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path,
                    default=EXP / "splits" / "phase_b_splits.json")
    ap.add_argument("--fit", default="S_cal")
    ap.add_argument("--eval", dest="ev", default="S_policy")
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--class-names", type=Path, default=None)
    ap.add_argument("--cap", type=int, default=1200)
    ap.add_argument("--tag", default="retrained")
    args = ap.parse_args()
    if "test" in args.fit.lower() or "test" in args.ev.lower():
        raise SystemExit("Refusing to touch a test split (PROTOCOL §8).")

    import torch.nn as nn
    import config
    from classifier import SpeciesClassifier

    man = json.loads(args.manifest.read_text())
    ls = man["label_space"]
    known, far = set(ls["known"]), set(ls["far_ood"])
    hosts = ls["split_hosts_ood_role"]
    roles = {k: set(v) for k, v in ls["ood_roles"].items()}

    w = args.weights or config.CLASSIFIER_WEIGHTS
    cn = args.class_names or config.CLASSIFIER_CLASS_NAMES
    clf = SpeciesClassifier(weights_path=w, class_names_path=cn, device="cpu")
    model = clf.model
    fc = model.fc
    model.fc = nn.Identity()
    model.eval()

    print(f"\nMahalanobis variants — {args.tag} features")
    print(f"  fit {args.fit} -> eval {args.ev}\n")

    Ff, _ = embed(args.fit, man, {"known": known, "near": roles[hosts[args.fit]],
                                  "far": far}, clf, model, fc, args.cap)
    Fe, _ = embed(args.ev, man, {"known": known, "near": roles[hosts[args.ev]],
                                 "far": far}, clf, model, fc, args.cap)

    X = Ff["known"]
    cov = np.cov(X.T)
    ev = np.linalg.eigvalsh(cov)
    ev = np.clip(ev, 0, None)
    print(f"feature diagnostics (fit split, {X.shape[0]} ID crops, dim {X.shape[1]})")
    print(f"  mean feature norm      {np.linalg.norm(X, axis=1).mean():.2f}")
    print(f"  mean per-dim variance  {np.trace(cov)/X.shape[1]:.4f}")
    print(f"  largest eigenvalue     {ev.max():.4f}")
    print(f"  smallest eigenvalue    {ev.min():.3e}")
    print(f"  effective rank         {int((ev > ev.max()*1e-6).sum())} of {X.shape[1]}")
    print(f"  -> a fixed 1e-3 ridge is {'NEGLIGIBLE' if 1e-3 < np.trace(cov)/X.shape[1]*1e-2 else 'comparable'}"
          f" against a mean variance of {np.trace(cov)/X.shape[1]:.4f}\n")

    variants = {
        "absolute eps=1e-3 (current)": maha_scorer(X, eps_abs=1e-3),
        "absolute eps=1e-1": maha_scorer(X, eps_abs=1e-1),
        "relative eps=0.01": maha_scorer(X, eps_rel=0.01),
        "relative eps=0.10": maha_scorer(X, eps_rel=0.10),
        "relative eps=0.50": maha_scorer(X, eps_rel=0.50),
        "diagonal only": maha_scorer(X, diagonal=True),
        "L2-norm + relative 0.10": maha_scorer(X, eps_rel=0.10, l2=True),
        "euclidean to mean": euclid_scorer(X),
        "kNN cosine (k=10)": knn_scorer(X, k=10),
    }

    print(f"{'variant':<30}{'far AUROC':>11}{'far FPR95':>11}"
          f"{'near AUROC':>12}{'near FPR95':>12}")
    print("-" * 76)
    out = {}
    for name, fn in variants.items():
        sid = fn(Fe["known"])
        row = {}
        cells = []
        for g in ("far", "near"):
            sood = fn(Fe[g])
            if len(sood) == 0 or len(sid) == 0:
                cells += ["  n/a", "  n/a"]
                continue
            s = np.concatenate([sid, sood])
            y = np.concatenate([np.zeros(len(sid)), np.ones(len(sood))])
            a, f = auroc(s, y), fpr_at_tpr(s, y)
            row[g] = {"auroc": a, "fpr_at_95tpr": f, "n_ood": int(len(sood))}
            cells += [f"{a:.4f}", f"{f:.4f}"]
        out[name] = row
        print(f"{name:<30}{cells[0]:>11}{cells[1]:>11}{cells[2]:>12}{cells[3]:>12}")

    best = max((k for k in out if out[k]), key=lambda k: out[k].get("near", {}).get("auroc", 0))
    print(f"\nbest near-OOD variant: {best} "
          f"(AUROC {out[best]['near']['auroc']:.4f})")
    print("All fitted on in-distribution crops only; no OOD data informs any of them.")

    dest = CACHE / f"maha_variants_{args.tag}_{args.ev}.json"
    dest.write_text(json.dumps({"fit": args.fit, "eval": args.ev,
                                "classifier": str(w), "variants": out}, indent=2) + "\n")
    print(f"Wrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
