"""Feature-space OOD scoring on ground-truth crops.

`ood_detect.py` showed that every confidence-derived score sits at or below
chance for novelty (AUROC 0.44-0.53): unseen taxa receive *higher* confidence
than correct detections. This asks whether the representation underneath the
softmax retains what the head destroys.

Scores are computed on *ground-truth* crops, which isolates the representation
by removing the detector's background false positives. That makes this a
feasibility probe rather than a deployment number — `ood_score_detections.py`
is the detector-level version.

**Fit and evaluation use different splits.** An earlier revision fitted the
Gaussians on the same split it scored, which reported training error: the
resulting 0.94/0.79 figures were in-sample, and the agreement between two
splits was two independent in-sample fits rather than evidence of
generalisation. Mahalanobis is now fitted on `--fit` and every reported number
comes from `--eval`. In-sample figures are still printed, labelled as such, so
the gap is visible.

Works against either label space: pass `--manifest` for the Phase A manifest
(4 known taxa, aquarium classifier) or the Phase B one (8 known taxa), and
`--weights` / `--class-names` to score with a retrained classifier. Comparing
the two answers whether on-domain training improves the feature space for
novelty detection.

Usage:
    python experiments/ood_features.py --fit J_cal --eval J_policy
    python experiments/ood_features.py --manifest splits/phase_b_splits.json \
        --fit S_cal --eval S_policy \
        --weights models/phase_b/classifier_seed0/best_classifier.pth \
        --class-names models/phase_b/classifier_seed0/class_names.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATASET = ROOT / "JediOrganismDetectionDataset"
SOURCE_SPLITS = ["coco_train_data", "coco_valid_data", "coco_test_data"]
CACHE = ROOT / "experiments" / "results" / "phase_a"
PAD = 0.05


def gt_boxes(files: set[str]) -> dict[str, list[tuple[str, list[float]]]]:
    out: dict[str, list[tuple[str, list[float]]]] = defaultdict(list)
    for sp in SOURCE_SPLITS:
        coco = json.loads((DATASET / sp / "coco.json").read_text())
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        imgs = {im["id"]: Path(im["file_name"]).name for im in coco["images"]}
        for a in coco["annotations"]:
            fn = imgs.get(a["image_id"])
            if fn in files:
                out[fn].append((cats[a["category_id"]], a["bbox"]))
    return out


def image_path(fname: str) -> Path | None:
    for sp in SOURCE_SPLITS:
        p = DATASET / sp / "images" / fname
        if p.is_file():
            return p
    return None


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
    pos = np.sort(s[y == 1])
    if len(pos) == 0 or (y == 0).sum() == 0:
        return float("nan")
    k = min(max(int(np.floor((1 - target) * len(pos))), 0), len(pos) - 1)
    return float((s[y == 0] >= pos[k]).mean())


def embed(split: str, man: dict, groups: dict[str, set[str]], clf, model,
          fc, cap: int, batch: int = 64):
    """Return {group: (features, logits)} for ground-truth crops of `split`."""
    import cv2
    import torch
    from PIL import Image

    files = set(man["files"][split])
    boxes = gt_boxes(files)
    want = {k: [] for k in groups}
    for fn, anns in boxes.items():
        for cname, bb in anns:
            for g, members in groups.items():
                if cname in members and len(want[g]) < cap:
                    want[g].append((fn, bb))

    feats: dict[str, list] = {k: [] for k in groups}
    logits: dict[str, list] = {k: [] for k in groups}
    for g, items in want.items():
        buf = []
        n = 0
        for fn, bb in items:
            p = image_path(fn)
            if p is None:
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            H, W = img.shape[:2]
            x, y, w, h = bb
            px, py = w * PAD, h * PAD
            x0, y0 = max(int(x - px), 0), max(int(y - py), 0)
            x1, y1 = min(int(x + w + px), W), min(int(y + h + py), H)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            buf.append(clf.transform(
                Image.fromarray(img[y0:y1, x0:x1][:, :, ::-1]).convert("RGB")))
            n += 1
            if len(buf) == batch:
                with torch.no_grad():
                    f = model(torch.stack(buf))
                    feats[g].append(f.numpy()); logits[g].append(fc(f).numpy())
                buf = []
        if buf:
            with torch.no_grad():
                f = model(torch.stack(buf))
                feats[g].append(f.numpy()); logits[g].append(fc(f).numpy())
        print(f"    {split}/{g}: {n} crops")
    F = {g: (np.concatenate(v) if v else np.zeros((0, 512))) for g, v in feats.items()}
    L = {g: (np.concatenate(v) if v else np.zeros((0, 1))) for g, v in logits.items()}
    return F, L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path,
                    default=Path(__file__).parent / "splits" / "phase_a_splits.json")
    ap.add_argument("--fit", default="J_cal")
    ap.add_argument("--eval", dest="ev", default="J_policy")
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--class-names", type=Path, default=None)
    ap.add_argument("--cap", type=int, default=1200)
    ap.add_argument("--tag", default=None)
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
    print(f"\nFeature-space OOD probe (ground-truth crops)")
    print(f"  manifest   {args.manifest.name}")
    print(f"  classifier {Path(w).parent.name}/{Path(w).name}")
    print(f"  fit {args.fit} ({hosts[args.fit]})  ->  eval {args.ev} ({hosts[args.ev]})")
    print(f"  KNOWN ({len(known)}): {', '.join(sorted(known))}\n")

    clf = SpeciesClassifier(weights_path=w, class_names_path=cn, device="cpu")
    model = clf.model
    fc = model.fc
    model.fc = nn.Identity()
    model.eval()

    def groups_for(split):
        return {"known": known, "near": roles[hosts[split]], "far": far}

    Ff, Lf = embed(args.fit, man, groups_for(args.fit), clf, model, fc, args.cap)
    Fe, Le = embed(args.ev, man, groups_for(args.ev), clf, model, fc, args.cap)

    if len(Ff["known"]) < 20:
        raise SystemExit(f"Only {len(Ff['known'])} ID crops on {args.fit}.")

    # Features are L2-normalised before the distance is fitted.
    #
    # Without this, Mahalanobis scored *below chance* on the retrained
    # classifier (0.42 far / 0.44 near out of sample) while scoring 0.87/0.90
    # in-sample. Diagnostics in ood_maha_variants.py ruled out the obvious
    # suspect: the covariance is full-rank, and adding shrinkage made the score
    # monotonically worse. Plain Euclidean distance to the mean scored 0.19 —
    # far enough below chance to identify the mechanism. The retrained network
    # assigns *smaller-magnitude* features to novel objects, so any distance
    # that uses magnitude reads "novel" as "close to the mean" and inverts.
    #
    # Normalising to the unit sphere discards the magnitude channel and keeps
    # direction, which is where the class information lives: 0.44 -> 0.71 near.
    # The ridge is also scaled to the data (a fraction of mean per-dimension
    # variance) rather than a fixed absolute value, so it means the same thing
    # whatever the feature scale.
    def _unit(X):
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    Xn = _unit(Ff["known"])
    mu = Xn.mean(axis=0)
    Xc = Xn - mu
    d_ = Xn.shape[1]
    cov = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    cov = cov + np.eye(d_) * (0.10 * np.trace(cov) / d_)
    P = np.linalg.pinv(cov)

    def maha(X):
        if len(X) == 0:
            return np.zeros(0)
        d = _unit(X) - mu
        return np.einsum("ij,jk,ik->i", d, P, d)

    def energy(X):
        if len(X) == 0:
            return np.zeros(0)
        m = X.max(axis=1, keepdims=True)
        return -(m[:, 0] + np.log(np.exp(X - m).sum(axis=1)))

    print(f"\nMahalanobis fitted on {len(Ff['known'])} ID crops from {args.fit}\n")
    print(f"{'split':<12}{'regime':<14}{'score':<14}{'AUROC':>8}{'FPR@95':>9}{'n_ood':>7}")
    print("-" * 66)
    out = {}
    for tag, F, L, split in (("in-sample", Ff, Lf, args.fit),
                             ("OUT-OF-SAMPLE", Fe, Le, args.ev)):
        idm, idl = maha(F["known"]), energy(L["known"])
        for g, label in (("far", "far-OOD"), ("near", "near-OOD")):
            if len(F[g]) == 0:
                continue
            for sname, sid, sood in (("mahalanobis", idm, maha(F[g])),
                                     ("energy", idl, energy(L[g]))):
                s = np.concatenate([sid, sood])
                y = np.concatenate([np.zeros(len(sid)), np.ones(len(sood))])
                a, f = auroc(s, y), fpr_at_tpr(s, y)
                out[f"{split}:{g}:{sname}"] = {
                    "auroc": a, "fpr_at_95tpr": f, "in_sample": tag == "in-sample",
                    "n_id": int(len(sid)), "n_ood": int(len(F[g]))}
                print(f"{split:<12}{label:<14}{sname:<14}{a:>8.4f}{f:>9.4f}"
                      f"{len(F[g]):>7}")
        print()

    print("Report the OUT-OF-SAMPLE rows. In-sample figures are the fit split's")
    print("own training error and are shown only to expose the generalisation gap.")

    tag = args.tag or args.manifest.stem.replace("_splits", "")
    dest = CACHE / f"{tag}_{args.ev}_ood_features.json"
    dest.write_text(json.dumps(
        {"manifest": str(args.manifest.name), "classifier": str(w),
         "fit_split": args.fit, "eval_split": args.ev,
         "mode": "ground-truth crops", "results": out}, indent=2) + "\n")
    print(f"\nWrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
