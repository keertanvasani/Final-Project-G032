"""Forced-action measurement for the two image-level actions (report Phase 15).

ZOOM and RECLASSIFY act on a single detection. ENHANCE and FULL_IMAGE_REDETECT
act on the whole frame, so their counterfactual is a different shape: the
comparison is between whole runs of the pipeline over the same image, not
between two versions of one box. Detections cannot be paired one-to-one across
variants — enhancement changes the pixels, so the detector may return a
different number of boxes — and the outcome is therefore counted per image.

Three variants are run on every image, so the rule's trigger condition never
decides what gets measured:

    raw        no enhancement, detection at the default 640 px  (the baseline)
    enhanced   ENHANCE forced on, detection at 640 px
    hires      no enhancement, detection at 1280 px             (REDETECT)

**FULL_IMAGE_REDETECT is operationalised here, not reproduced.** PROTOCOL.md §6
lists it among the discretionary actions and the report's action space names it,
but *it does not exist in the pipeline* — the deployed system implements three
discretionary actions, not four. Re-running the detector unchanged on the same
pixels is deterministic and would measure nothing, so the action is defined as a
**full-frame re-detection at double resolution**, which is genuinely distinct
from ZOOM (which crops) and has a clear hypothesis: small organisms lost to
downsampling at 640 px are recovered at 1280 px. That definition is mine and any
result carries it.

`config.YOLO_IMG_SIZE` is defined but never passed by `ensemble.run_dual_detection`,
so the hires variant calls `predict` directly rather than through the helper.

Discipline: dev splits only. Never a test split.

Usage:
    python experiments/log_image_actions.py --split J_policy
    python experiments/log_image_actions.py --split S_policy --platform phase_b
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "experiments" / "results" / "actions"
IOU_MATCH = 0.50
HIRES = 1280


def score_run(dets, cls_results, gts, known, calculate_iou):
    """Count outcomes for one whole-image run (greedy, one GT per detection)."""
    order = sorted(range(len(dets)),
                   key=lambda i: -dets[i]["confidence"])
    used, counts = set(), {"TP": 0, "CLS_ERR": 0, "OOD_ERR": 0,
                           "LOC_ERR": 0, "BG_FP": 0}
    for di in order:
        box = dets[di]["bbox"]
        pred = cls_results[di]["class_name"]
        best, bi = 0.0, -1
        for gi, g in enumerate(gts):
            if gi in used:
                continue
            v = calculate_iou(box, g["bbox"])
            if v > best:
                best, bi = v, gi
        if best >= IOU_MATCH and bi >= 0:
            used.add(bi)
            gt_cls = gts[bi]["class"]
            if gt_cls not in known:
                counts["OOD_ERR"] += 1
            elif pred == gt_cls:
                counts["TP"] += 1
            else:
                counts["CLS_ERR"] += 1
        else:
            any_iou = max((calculate_iou(box, g["bbox"]) for g in gts), default=0.0)
            counts["LOC_ERR" if any_iou >= 0.10 else "BG_FP"] += 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="J_policy")
    ap.add_argument("--platform", choices=["phase_a", "phase_b"], default="phase_a")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if "test" in args.split.lower():
        raise SystemExit("Refusing to touch a test split (PROTOCOL §8).")

    import cv2
    import config
    import ensemble
    from agent_controller import assess_image_quality, calculate_iou, enhance_image
    from experiments.eval import harness
    from experiments.run_phase_a import dive_of, image_path
    sys.path.insert(0, str(Path(__file__).parent))
    from platform_b import build_platform

    pipe, man, known = build_platform(args.platform)
    files = man["files"][args.split]
    if args.limit:
        files = files[: args.limit]
    gt_all = harness.load_ground_truth()

    OUT.mkdir(parents=True, exist_ok=True)
    tag = "" if args.platform == "phase_a" else "_phaseB"
    dest = OUT / f"{args.split}{tag}_image_actions.jsonl"

    def run_variant(img, imgsz=None):
        t0 = time.perf_counter()
        if imgsz is None:
            ra, rb = ensemble.run_dual_detection(img, pipe.model_a, pipe.model_b)
        else:
            ra = pipe.model_a.predict(source=img, conf=config.YOLO_CONF_THRESHOLD,
                                      iou=config.YOLO_IOU_THRESHOLD,
                                      imgsz=imgsz, verbose=False)
            rb = pipe.model_b.predict(source=img, conf=config.YOLO_CONF_THRESHOLD,
                                      iou=config.YOLO_IOU_THRESHOLD,
                                      imgsz=imgsz, verbose=False)
        dets = ensemble.fuse_detections(ra, rb, img.shape[:2])
        cls = []
        if dets:
            crops = [pipe._extract_crop(img, d["bbox"], padding=0.05) for d in dets]
            cls = pipe.classifier.classify_crops(crops)
        return dets, cls, (time.perf_counter() - t0) * 1000.0

    n = 0
    t_start = time.time()
    with dest.open("w") as fh:
        for i, fname in enumerate(files, 1):
            p = image_path(fname)
            if p is None:
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            q = assess_image_quality(img)
            if q["blur_score"] < config.UNUSABLE_BLUR_THRESHOLD:
                continue
            gts = gt_all.get(fname, [])

            d_raw, c_raw, ms_raw = run_variant(img)
            enh, _ = enhance_image(img, q)
            d_enh, c_enh, ms_enh = run_variant(enh)
            d_hi, c_hi, ms_hi = run_variant(img, imgsz=HIRES)

            row = {
                "file": fname, "dive": dive_of(fname),
                "n_gt": len(gts),
                "rule_would_enhance": bool(q["needs_enhancement"] or q["is_blurry"]),
                "blur": float(q["blur_score"]),
                "brightness": float(q["brightness"]),
                "contrast": float(q["contrast"]),
                "raw": score_run(d_raw, c_raw, gts, known, calculate_iou),
                "enhanced": score_run(d_enh, c_enh, gts, known, calculate_iou),
                "hires": score_run(d_hi, c_hi, gts, known, calculate_iou),
                "n_det_raw": len(d_raw), "n_det_enh": len(d_enh),
                "n_det_hires": len(d_hi),
                "ms_raw": round(ms_raw, 1), "ms_enhanced": round(ms_enh, 1),
                "ms_hires": round(ms_hi, 1),
            }
            fh.write(json.dumps(row) + "\n")
            n += 1
            if i % 300 == 0:
                print(f"  {i}/{len(files)} images · {(time.time()-t_start)/60:.1f} min")

    print(f"\nLogged {n} images in {(time.time()-t_start)/60:.1f} min")
    print(f"Wrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
