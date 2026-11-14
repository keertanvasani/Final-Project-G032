"""Score every cached detection with feature-space OOD distance (report Phase 6).

`ood_features.py` established on ground-truth crops that the ResNet-18's
representation separates novel organisms from known ones (Mahalanobis AUROC
0.94 far-OOD, 0.78-0.80 near-OOD) even though every confidence-derived score
sits at chance. That was a feasibility probe: it isolated the representation by
scoring perfect boxes.

This is the deployment version. It scores the boxes the detector actually
emitted, which means it also faces the 42% of detections that are background
false positives and the 20% that are poorly localised — regions containing no
organism at all, where "is this a known taxon?" has no clean answer. The number
this produces is the one an operational system would live with, and it is
expected to be lower than the probe's.

Output is one Mahalanobis distance per cached detection, written alongside the
cache so `risk_coverage.py` can gate on it.

Fitting (PROTOCOL §5.1). The class-conditional Gaussians are estimated from
in-distribution detections on the fit split only — detections that matched a
known-class GT. No OOD data of any kind informs the estimator, so far-OOD is
never tuned on, and near-OOD taxa outside the fit split's role are excluded.

Usage:
    python experiments/ood_score_detections.py --fit J_cal --splits J_cal J_policy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATASET = ROOT / "JediOrganismDetectionDataset"
SOURCE_SPLITS = ["coco_train_data", "coco_valid_data", "coco_test_data"]
SPLITS = Path(__file__).parent / "splits" / "phase_a_splits.json"
CACHE = ROOT / "experiments" / "results" / "phase_a"
PAD = 0.05


def image_path(fname: str) -> Path | None:
    for sp in SOURCE_SPLITS:
        p = DATASET / sp / "images" / fname
        if p.is_file():
            return p
    return None


def embed_split(split: str, clf, model, batch: int = 64) -> tuple[list[dict], np.ndarray]:
    """Return (rows, features) for every cached detection carrying a bbox."""
    import cv2
    import torch
    from PIL import Image

    p = CACHE / f"{split}_detections.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    if rows and "bbox" not in rows[0]:
        raise SystemExit(
            f"{split} cache has no 'bbox' field. Re-run run_phase_a.py cache "
            f"after the harness change that stores it."
        )

    by_file: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_file.setdefault(r["file"], []).append(i)

    feats = np.zeros((len(rows), 512), dtype=np.float32)
    buf, idx = [], []
    done = 0

    def flush():
        nonlocal buf, idx
        if not buf:
            return
        with torch.no_grad():
            f = model(torch.stack(buf)).numpy()
        feats[idx] = f
        buf, idx = [], []

    for fname, ridx in by_file.items():
        path = image_path(fname)
        if path is None:
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        H, W = img.shape[:2]
        for i in ridx:
            x, y, w, h = rows[i]["bbox"]
            # cached boxes are xyxy from the pipeline
            x0, y0, x1, y1 = x, y, w, h
            if x1 < x0 or y1 < y0:
                continue
            bw, bh = x1 - x0, y1 - y0
            px, py = bw * PAD, bh * PAD
            cx0, cy0 = max(int(x0 - px), 0), max(int(y0 - py), 0)
            cx1, cy1 = min(int(x1 + px), W), min(int(y1 + py), H)
            if cx1 - cx0 < 4 or cy1 - cy0 < 4:
                continue
            pil = Image.fromarray(img[cy0:cy1, cx0:cx1][:, :, ::-1]).convert("RGB")
            buf.append(clf.transform(pil))
            idx.append(i)
            if len(buf) == batch:
                flush()
        done += 1
        if done % 400 == 0:
            print(f"    {done}/{len(by_file)} images")
    flush()
    return rows, feats


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


def fpr_at_tpr(s, y, target=0.95):
    pos = np.sort(s[y == 1])
    if len(pos) == 0 or (y == 0).sum() == 0:
        return float("nan")
    k = min(max(int(np.floor((1 - target) * len(pos))), 0), len(pos) - 1)
    return float((s[y == 0] >= pos[k]).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="J_cal")
    ap.add_argument("--splits", nargs="+", default=["J_cal", "J_policy"])
    args = ap.parse_args()
    if any("test" in s.lower() for s in args.splits + [args.fit]):
        raise SystemExit("Refusing to touch a test split (PROTOCOL §8).")

    import torch.nn as nn
    import config
    from classifier import SpeciesClassifier

    man = json.loads(SPLITS.read_text())
    ls = man["label_space"]
    known, far = set(ls["known"]), set(ls["far_ood"])
    hosts = ls["split_hosts_ood_role"]
    roles = {k: set(v) for k, v in ls["ood_roles"].items()}

    clf = SpeciesClassifier(weights_path=config.CLASSIFIER_WEIGHTS,
                            class_names_path=config.CLASSIFIER_CLASS_NAMES,
                            device="cpu")
    model = clf.model
    model.fc = nn.Identity()
    model.eval()

    print("\nEmbedding cached detections (deployment setting)")
    data = {}
    for sp in args.splits:
        print(f"  {sp}:")
        rows, F = embed_split(sp, clf, model)
        data[sp] = (rows, F)
        print(f"    {len(rows)} detections embedded")

    # ---- fit Mahalanobis on ID detections of the fit split --------------
    rows_f, F_f = data[args.fit]
    id_mask = np.array([r.get("gt_class") in known and r.get("matched_gt") is not None
                        for r in rows_f])
    X = F_f[id_mask]
    if len(X) < 20:
        raise SystemExit(f"Only {len(X)} ID detections on {args.fit}; cannot fit.")
    mu = X.mean(axis=0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / max(len(Xc) - 1, 1) + np.eye(512) * 1e-3
    P = np.linalg.pinv(cov)
    print(f"\nMahalanobis fitted on {len(X)} in-distribution detections from {args.fit}")

    out = {}
    for sp, (rows, F) in data.items():
        d = F - mu
        score = np.einsum("ij,jk,ik->i", d, P, d)
        dest = CACHE / f"{sp}_ood_scores.json"
        dest.write_text(json.dumps(
            {"split": sp, "fit_split": args.fit, "score": "mahalanobis",
             "scores": [float(v) for v in score]}, indent=1))

        role = hosts[sp]
        rc = roles[role]
        idm = np.array([r.get("gt_class") in known and r.get("matched_gt") is not None
                        for r in rows])
        nearm = np.array([r.get("gt_class") in rc for r in rows])
        farm = np.array([r.get("gt_class") in far for r in rows])

        print(f"\n{sp} (hosts {role})")
        print(f"  {'comparison':<26}{'AUROC':>8}{'FPR@95':>9}{'n_id':>7}{'n_ood':>7}")
        print("  " + "-" * 57)
        res = {}
        for tag, m in (("far-OOD (machine)", farm), (f"near-OOD ({role})", nearm)):
            if m.sum() == 0 or idm.sum() == 0:
                continue
            s = np.concatenate([score[idm], score[m]])
            y = np.concatenate([np.zeros(idm.sum()), np.ones(m.sum())])
            a, f = auroc(s, y), fpr_at_tpr(s, y)
            res[tag] = {"auroc": a, "fpr_at_95tpr": f,
                        "n_id": int(idm.sum()), "n_ood": int(m.sum())}
            print(f"  {tag:<26}{a:>8.4f}{f:>9.4f}{idm.sum():>7}{m.sum():>7}")
        out[sp] = res
        print(f"  wrote {dest.name}")

    (CACHE / "ood_detection_level.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\nThese are detector-emitted boxes, so background false positives and")
    print("poorly-localised boxes are present in the pool but belong to neither")
    print("class — they are excluded from the AUROC, which compares matched")
    print("known-class detections against matched OOD ones. Gating them is the")
    print("job of the confidence score, not the novelty score.")


if __name__ == "__main__":
    main()
