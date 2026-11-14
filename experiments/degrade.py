"""Controlled degradation experiment on in-domain data (report §36, condition B).

Phase A measured what happens under a *domain change*, where image quality,
species composition, optics and illumination all shift at once. That result is
strong but confounded: it cannot say which factor the system failed on.

This isolates one factor. The aquarium validation set is the domain the model
was trained for, so at severity 0 the system is at its best. Each degradation
is then applied at increasing strength with the ground truth untouched, giving
a difficulty gradient whose cause is known exactly.

The question is not "does accuracy fall" — of course it does. It is:

    as the evidence gets worse, does the system NOTICE?

A system that knows its limits abstains more as quality drops, holding
selective risk roughly flat. A system blind to its own degradation keeps
answering at the same rate and simply becomes wrong more often. The gap between
those two curves is the thing the uncertainty layer exists to close, and here it
is measurable against a controlled x-axis rather than inferred from a domain
change with a hundred confounds.

Degradations follow the report's condition B: blur, motion blur, noise, low
illumination, low contrast, colour shift, compression.

Runs the frozen IMAGE_V1_BASELINE unchanged. No training, no GPU required.

Usage:
    python experiments/degrade.py                 # full sweep
    python experiments/degrade.py --limit 40      # quick check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "experiments" / "results" / "degradation"
IOU_MATCH = 0.50
TAU = 0.40


# ---------------------------------------------------------------- degradations
def d_clean(img, s):
    return img


def d_blur(img, s):
    import cv2
    k = [3, 7, 13][s]
    return cv2.GaussianBlur(img, (k, k), 0)


def d_motion(img, s):
    import cv2
    n = [5, 11, 21][s]
    k = np.zeros((n, n), dtype=np.float32)
    k[n // 2, :] = 1.0 / n           # horizontal streak, as from a panning ROV
    return cv2.filter2D(img, -1, k)


def d_noise(img, s):
    sigma = [10, 25, 45][s]
    out = img.astype(np.float32) + np.random.normal(0, sigma, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def d_lowlight(img, s):
    f = [0.60, 0.40, 0.25][s]
    return np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8)


def d_lowcontrast(img, s):
    a = [0.60, 0.40, 0.25][s]
    m = img.mean()
    return np.clip((img.astype(np.float32) - m) * a + m, 0, 255).astype(np.uint8)


def d_colorshift(img, s):
    """Blue-green cast: red attenuates fastest in water."""
    f = [(0.80, 1.00, 1.05), (0.60, 1.00, 1.12), (0.40, 1.00, 1.20)][s]
    out = img.astype(np.float32).copy()          # BGR
    out[:, :, 2] *= f[0]                          # red down
    out[:, :, 1] *= f[1]
    out[:, :, 0] *= f[2]                          # blue up
    return np.clip(out, 0, 255).astype(np.uint8)


def d_compress(img, s):
    import cv2
    q = [40, 20, 10][s]
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


DEGRADATIONS = {
    "gaussian_blur": d_blur,
    "motion_blur": d_motion,
    "noise": d_noise,
    "low_light": d_lowlight,
    "low_contrast": d_lowcontrast,
    "colour_shift": d_colorshift,
    "compression": d_compress,
}


# ---------------------------------------------------------------- ground truth
def load_gt(img_dir: Path, lbl_dir: Path, names: list[str]):
    import cv2
    gt = {}
    for p in sorted(img_dir.glob("*.jpg")):
        lp = lbl_dir / (p.stem + ".txt")
        img = cv2.imread(str(p))
        if img is None:
            continue
        H, W = img.shape[:2]
        boxes = []
        if lp.is_file():
            for line in lp.read_text().split("\n"):
                if not line.strip():
                    continue
                c, cx, cy, w, h = line.split()[:5]
                cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
                boxes.append({"cls": names[int(c)],
                              "bbox": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]})
        gt[p] = boxes
    return gt


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(ix1 - ix0, 0), max(iy1 - iy0, 0)
    inter = iw * ih
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def score_image(dets, gts):
    """Greedy IoU matching, class checked after matching (PROTOCOL §2)."""
    order = sorted(range(len(dets)), key=lambda i: -dets[i]["combined_confidence"])
    used, rows = set(), []
    for di in order:
        d = dets[di]
        best, bg = 0.0, -1
        for gi, g in enumerate(gts):
            if gi in used:
                continue
            v = iou(d["bbox"], g["bbox"])
            if v > best:
                best, bg = v, gi
        if best >= IOU_MATCH and bg >= 0:
            used.add(bg)
            correct = d.get("classifier_class") == gts[bg]["cls"]
            rows.append({"conf": d["combined_confidence"],
                         "outcome": "TP" if correct else "CLS_ERR"})
        else:
            any_iou = max((iou(d["bbox"], g["bbox"]) for g in gts), default=0.0)
            rows.append({"conf": d["combined_confidence"],
                         "outcome": "LOC_ERR" if any_iou >= 0.10 else "BG_FP"})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import cv2
    import config
    from agent_controller import run_perception
    from pipeline import EnsemblePipeline

    np.random.seed(args.seed)
    names = ["fish", "jellyfish", "penguin", "puffin", "shark", "starfish", "stingray"]
    img_dir = config.DATA_DIR / "images" / "val"
    lbl_dir = config.DATA_DIR / "labels" / "val"
    gt = load_gt(img_dir, lbl_dir, names)
    files = sorted(gt)[: args.limit] if args.limit else sorted(gt)
    n_gt = sum(len(gt[f]) for f in files)
    print(f"In-domain control set: {len(files)} images, {n_gt} ground-truth objects\n")

    pipe = EnsemblePipeline()
    OUT.mkdir(parents=True, exist_ok=True)

    conditions = [("clean", 0, d_clean)]
    for name, fn in DEGRADATIONS.items():
        for s in range(3):
            conditions.append((name, s + 1, fn))

    results = []
    t0 = time.time()
    for cname, sev, fn in conditions:
        rows, n_unreadable, n_empty = [], 0, 0
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            dimg = fn(img, sev - 1) if sev > 0 else img
            try:
                out = run_perception(dimg, pipe)
            except Exception:
                n_unreadable += 1
                continue
            st = out.get("status")
            if st == "unreadable":
                n_unreadable += 1
                continue
            dets = out.get("detections", []) or []
            if not dets:
                n_empty += 1
                continue
            rows.extend(score_image(dets, gt[f]))

        n_img = len(files)
        n_det = len(rows)
        acc = [r for r in rows if r["conf"] >= TAU]
        tp_acc = sum(1 for r in acc if r["outcome"] == "TP")
        # An image is refused if it was unreadable or no box cleared tau.
        risk = 1 - tp_acc / len(acc) if acc else float("nan")
        rec = {
            "degradation": cname, "severity": sev,
            "images": n_img, "detections": n_det,
            "unreadable_images": n_unreadable, "empty_images": n_empty,
            "abstained_images": n_unreadable + n_empty,
            "abstention_rate": (n_unreadable + n_empty) / n_img,
            "accepted_detections": len(acc),
            "coverage": len(acc) / n_det if n_det else 0.0,
            "selective_risk": risk,
            "mean_confidence": float(np.mean([r["conf"] for r in rows])) if rows else float("nan"),
            "tp": sum(1 for r in rows if r["outcome"] == "TP"),
            "cls_err": sum(1 for r in rows if r["outcome"] == "CLS_ERR"),
            "loc_err": sum(1 for r in rows if r["outcome"] == "LOC_ERR"),
            "bg_fp": sum(1 for r in rows if r["outcome"] == "BG_FP"),
        }
        results.append(rec)
        print(f"{cname:<16} sev {sev}  dets {n_det:>5}  abstain {rec['abstention_rate']:>6.1%}"
              f"  risk {risk:>6.3f}  mean conf {rec['mean_confidence']:>5.3f}")

    print(f"\nelapsed {(time.time()-t0)/60:.1f} min")

    dest = OUT / "degradation_results.json"
    dest.write_text(json.dumps({"tau": TAU, "iou": IOU_MATCH,
                                "images": len(files), "gt_objects": n_gt,
                                "results": results}, indent=2) + "\n")
    print(f"Wrote {dest.relative_to(ROOT)}")

    # ---- summary: does the system notice? ----
    base = results[0]
    print(f"\n{'degradation':<16}{'risk clean -> sev3':>22}{'abstention clean -> sev3':>28}")
    print("-" * 68)
    for name in DEGRADATIONS:
        s3 = [r for r in results if r["degradation"] == name and r["severity"] == 3]
        if not s3:
            continue
        r3 = s3[0]
        rtxt = ("all abstained" if np.isnan(r3["selective_risk"])
                else f"{r3['selective_risk']:.3f}")
        print(f"{name:<16}{base['selective_risk']:>11.3f} -> {rtxt:>8}"
              f"{base['abstention_rate']:>17.1%} -> {r3['abstention_rate']:>7.1%}")

    print("\nReading this table:")
    print("  risk rising while abstention stays flat = the system does NOT notice")
    print("  its evidence has degraded — it keeps answering, and is wrong more often.")
    print("  Both rising together = it does notice.")
    print("\nCaveat: selective risk is conditioned on the accepted set, so when")
    print("abstention rises the surviving detections are a different, easier")
    print("population. Compare risk only alongside its abstention rate, never alone.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=150)
        for name in DEGRADATIONS:
            pts = [r for r in results if r["degradation"] == name]
            pts = [base] + sorted(pts, key=lambda r: r["severity"])
            xs = [r["severity"] for r in pts]
            axes[0].plot(xs, [r["selective_risk"] for r in pts], marker="o", ms=3, lw=1.3, label=name)
            axes[1].plot(xs, [r["abstention_rate"] for r in pts], marker="o", ms=3, lw=1.3)
            axes[2].plot(xs, [r["mean_confidence"] for r in pts], marker="o", ms=3, lw=1.3)
        for ax, t in zip(axes, ["selective risk", "abstention rate", "mean confidence"]):
            ax.set_xlabel("degradation severity"); ax.set_ylabel(t)
            ax.grid(alpha=0.25, lw=0.6); ax.set_xticks([0, 1, 2, 3])
        axes[0].legend(fontsize=7, frameon=False)
        fig.suptitle("In-domain degradation: does the system notice?", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / "degradation_curves.png")
        print(f"Wrote {(OUT / 'degradation_curves.png').relative_to(ROOT)}")
    except Exception as exc:
        print(f"(figure skipped: {exc})")


if __name__ == "__main__":
    main()
