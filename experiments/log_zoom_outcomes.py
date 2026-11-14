"""Forced-action logging: what would ZOOM have done? (report Phases 11 & 15)

The decision trace records what the rules *did*. It never records what would
have happened otherwise, so it cannot answer whether an action was worth its
cost — every logged zoom is one the rule chose to fire, and there is no record
of a zoom that would have helped but was not taken. A policy fitted on that
data can only re-derive the existing thresholds.

This produces the missing counterfactual. For **every** detection — not only
those in the rule's 0.40-0.75 ambiguity band — it forces the zoom, re-detects,
re-classifies the resulting crop, and records the outcome against ground truth:

    correctness and confidence BEFORE the action
    correctness and confidence AFTER it
    whether the deployed rule would have fired, and whether it would have kept
    the result
    measured wall-clock cost

Forcing the action on the whole confidence range rather than sampling
epsilon-greedily is exhaustive rather than exploratory: it costs one extra
detection pass per detection but removes sampling noise from the estimate of
Q(s, ZOOM), which matters because the effect is expected to be small.

Faithfulness note: the pipeline's own `run_perception` applies zoom internally,
so its output is already post-zoom for ambiguous boxes. This script therefore
runs the stages individually — quality assessment, enhancement, dual detection,
WBF — to obtain the true pre-zoom state, then applies the action itself.

Discipline: dev splits only. Never J_test.

Usage:
    python experiments/log_zoom_outcomes.py --split J_cal
    python experiments/log_zoom_outcomes.py --split J_cal --limit 100
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
SPLITS = ROOT / "experiments" / "splits" / "phase_a_splits.json"
OUT = ROOT / "experiments" / "results" / "actions"
IOU_MATCH = 0.50
PAD = 0.15          # the rule's zoom padding
RULE_LO, RULE_HI = 0.40, 0.75


def best_gt(box, gts):
    """Return (best_iou, index) against ground-truth boxes."""
    best, bi = 0.0, -1
    for i, g in enumerate(gts):
        from agent_controller import calculate_iou
        v = calculate_iou(box, g["bbox"])
        if v > best:
            best, bi = v, i
    return best, bi


def outcome_of(box, cls_name, gts, known):
    """Correctness under PROTOCOL §2, plus the OOD flag."""
    iou, gi = best_gt(box, gts)
    if iou >= IOU_MATCH and gi >= 0:
        # harness.load_ground_truth keys the label as "class", not "class_name"
        gt_cls = gts[gi]["class"]
        if gt_cls not in known:
            return "OOD_ERR", iou, gt_cls
        return ("TP" if cls_name == gt_cls else "CLS_ERR"), iou, gt_cls
    return ("LOC_ERR" if iou >= 0.10 else "BG_FP"), iou, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="J_cal")
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
    dest = OUT / f"{args.split}{tag}_zoom_outcomes.jsonl"

    n_rows = n_img = n_cand = 0
    t0 = time.time()
    with dest.open("w") as fh:
        for i, fname in enumerate(files, 1):
            p = image_path(fname)
            if p is None:
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue

            q = assess_image_quality(img)
            enh = img
            if q["needs_enhancement"] or q["is_blurry"]:
                enh, _ = enhance_image(img, q)
            if q["blur_score"] < config.UNUSABLE_BLUR_THRESHOLD:
                continue

            res_a, res_b = ensemble.run_dual_detection(enh, pipe.model_a, pipe.model_b)
            dets = ensemble.fuse_detections(res_a, res_b, enh.shape[:2])
            if not dets:
                continue
            n_img += 1
            gts = gt_all.get(fname, [])
            H, W = enh.shape[:2]

            # ---- BEFORE: classify the un-zoomed boxes -------------------
            crops = [pipe._extract_crop(enh, d["bbox"], padding=0.05) for d in dets]
            cls_before = pipe.classifier.classify_crops(crops)

            for di, (det, cb) in enumerate(zip(dets, cls_before)):
                box = det["bbox"]
                conf = float(det["confidence"])
                out_b, iou_b, gt_cls = outcome_of(box, cb["class_name"], gts, known)

                # ---- FORCED ZOOM, regardless of the rule's band ---------
                bw, bh = box[2] - box[0], box[3] - box[1]
                px, py = bw * PAD, bh * PAD
                cx1, cy1 = max(0, int(box[0] - px)), max(0, int(box[1] - py))
                cx2, cy2 = min(W, int(box[2] + px)), min(H, int(box[3] + py))
                crop = enh[cy1:cy2, cx1:cx2]
                if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
                    continue

                t_a = time.perf_counter()
                ra, rb = ensemble.run_dual_detection(crop, pipe.model_a, pipe.model_b)
                fused = ensemble.fuse_detections(ra, rb, crop.shape[:2])

                cand, cand_iou = None, 0.0
                for c in fused:
                    gb = [cx1 + c["bbox"][0], cy1 + c["bbox"][1],
                          cx1 + c["bbox"][2], cy1 + c["bbox"][3]]
                    v = calculate_iou(box, gb)
                    if v > 0.4 and v > cand_iou:
                        cand_iou, cand = v, (gb, float(c["confidence"]), c["class_name"])

                row = {
                    "file": fname, "dive": dive_of(fname), "det_index": di,
                    # ---- state before the action ----
                    "conf_before": conf,
                    "cls_conf_before": float(cb["confidence"]),
                    "cls_name_before": cb["class_name"],
                    "agreement_before": det["class_name"] == cb["class_name"],
                    "box_area": float(bw * bh),
                    "aspect_ratio": float(bw / max(bh, 1.0)),
                    "blur": float(q["blur_score"]),
                    "n_dets": len(dets),
                    "outcome_before": out_b, "iou_before": round(iou_b, 4),
                    "gt_class": gt_cls,
                    # ---- what the deployed rule would do ----
                    "rule_would_fire": RULE_LO <= conf < RULE_HI,
                    # ---- the action's result ----
                    "candidate_found": cand is not None,
                }
                if cand is not None:
                    n_cand += 1
                    gb, ncf, ncls = cand
                    new_crop = pipe._extract_crop(enh, gb, padding=0.05)
                    ca = pipe.classifier.classify_crops([new_crop])[0]
                    out_a, iou_a, _ = outcome_of(gb, ca["class_name"], gts, known)
                    row.update({
                        "conf_after": ncf,
                        "cls_conf_after": float(ca["confidence"]),
                        "cls_name_after": ca["class_name"],
                        "iou_new_vs_old": round(cand_iou, 4),
                        "outcome_after": out_a, "iou_after": round(iou_a, 4),
                        # the rule keeps the new box only if confidence improved
                        "rule_would_keep": ncf > conf,
                    })
                row["cost_ms"] = round((time.perf_counter() - t_a) * 1000.0, 2)
                fh.write(json.dumps(row) + "\n")
                n_rows += 1

            if i % 200 == 0:
                el = time.time() - t0
                print(f"  {i}/{len(files)} images · {n_rows} actions logged "
                      f"· {el/60:.1f} min")

    el = (time.time() - t0) / 60
    print(f"\nLogged {n_rows} forced ZOOM actions over {n_img} images in {el:.1f} min")
    print(f"  candidate box found: {n_cand} ({100*n_cand/max(n_rows,1):.1f}%)")
    print(f"Wrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
