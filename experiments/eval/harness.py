"""Evaluation harness — matching, the five-way error taxonomy, GT loading.

Implements PROTOCOL.md §2. Pure functions over plain dicts so they are
trivially unit-testable and independent of the inference stack.

Coordinate convention: all boxes are ``[x1, y1, x2, y2]`` in absolute
pixels. COCO stores ``[x, y, w, h]``; :func:`load_ground_truth` converts.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "JediOrganismDetectionDataset"
SOURCE_SPLITS = ["coco_train_data", "coco_valid_data", "coco_test_data"]

# PROTOCOL.md §2.4 — aquarium label space mapped onto Jedi. `ray` is OOD
# (decision 9.3), so it is deliberately absent here.
KNOWN_CLASSES = {"fish", "jellyfish", "shark", "starfish"}

# Taxonomy thresholds (§2.1, §2.3).
IOU_MATCH = 0.50        # matched vs unmatched
IOU_LOC = 0.10          # localisation error vs background false positive

# Outcome labels.
TP = "TP"               # matched, class correct, class in KNOWN
CLS_ERR = "CLS_ERR"     # matched, class wrong, class in KNOWN
OOD_ERR = "OOD_ERR"     # matched a GT whose class is not in KNOWN
LOC_ERR = "LOC_ERR"     # unmatched, best IoU in [IOU_LOC, IOU_MATCH)
BG_FP = "BG_FP"         # unmatched, best IoU < IOU_LOC


def iou(a: list[float], b: list[float]) -> float:
    """IoU of two [x1,y1,x2,y2] boxes."""
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    if inter <= 0.0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_ground_truth() -> dict[str, list[dict[str, Any]]]:
    """Map filename -> list of GT objects ``{"bbox":[x1,y1,x2,y2], "class":str}``.

    Pooled across the three supplied coco.json files (the split assignment
    lives in the split manifest, keyed by filename). Image ids repeat across
    files, so the mapping is built per file before keying on filename.
    """
    gt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_files: set[str] = set()

    for split in SOURCE_SPLITS:
        coco = json.loads((DATASET / split / "coco.json").read_text())
        id2name = {c["id"]: c["name"] for c in coco["categories"]}
        imgid2file = {im["id"]: im["file_name"] for im in coco["images"]}
        for im in coco["images"]:
            gt.setdefault(im["file_name"], [])
            seen_files.add(im["file_name"])
        for a in coco["annotations"]:
            fname = imgid2file.get(a["image_id"])
            if fname is None:
                continue
            x, y, w, h = a["bbox"]
            gt[fname].append({"bbox": [x, y, x + w, y + h],
                              "class": id2name[a["category_id"]]})
    return dict(gt)


def match_and_classify(
    detections: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Match detections to GT and assign each an outcome (PROTOCOL.md §2).

    Matching is greedy by descending ``combined_confidence``, class-agnostic,
    IoU >= 0.50, one GT per detection. The class check happens *after*
    matching, so a well-localised box with the wrong label is distinguished
    from a background false positive.

    Returns one record per detection with its outcome and the fields the
    metrics layer needs. GT objects are not returned here; recall is computed
    from ``matched_gt`` indices by the caller.
    """
    order = sorted(range(len(detections)),
                   key=lambda i: detections[i]["combined_confidence"], reverse=True)
    gt_used: set[int] = set()
    records: list[dict[str, Any]] = []

    for di in order:
        det = detections[di]
        dbox = det["bbox"]

        best_iou, best_gt = 0.0, -1
        for gi, g in enumerate(ground_truth):
            if gi in gt_used:
                continue
            v = iou(dbox, g["bbox"])
            if v > best_iou:
                best_iou, best_gt = v, gi

        # best IoU against ANY gt (matched or not) — needed to separate
        # LOC_ERR from BG_FP for an unmatched detection.
        best_any = max((iou(dbox, g["bbox"]) for g in ground_truth), default=0.0)

        rec = {
            "det_index": di,
            # Kept so a downstream analysis can re-crop the region without
            # re-running detection — feature-space OOD scoring needs the pixels,
            # and re-detecting a split to recover boxes costs ~10 minutes.
            "bbox": [round(float(v), 2) for v in dbox],
            "combined_confidence": det["combined_confidence"],
            "predicted_class": det["classifier_class"],
            "detector_confidence": det.get("detector_confidence"),
            "classifier_confidence": det.get("classifier_confidence"),
            "agreement": det.get("agreement"),
            "best_iou": round(best_any, 4),
            "matched_gt": None,
            "gt_class": None,
            "outcome": None,
        }

        if best_iou >= IOU_MATCH and best_gt >= 0:
            gt_used.add(best_gt)
            g = ground_truth[best_gt]
            rec["matched_gt"] = best_gt
            rec["gt_class"] = g["class"]
            if g["class"] not in KNOWN_CLASSES:
                rec["outcome"] = OOD_ERR
            elif det["classifier_class"] == g["class"]:
                rec["outcome"] = TP
            else:
                rec["outcome"] = CLS_ERR
        else:
            rec["outcome"] = LOC_ERR if best_any >= IOU_LOC else BG_FP

        records.append(rec)

    # Restore input order for stable downstream joins.
    records.sort(key=lambda r: r["det_index"])
    return records


def known_gt_count(ground_truth: list[dict[str, Any]]) -> int:
    """GT objects in the label space — the recall denominator (§2.4)."""
    return sum(1 for g in ground_truth if g["class"] in KNOWN_CLASSES)
