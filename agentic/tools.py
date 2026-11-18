"""Tools the agent can call.

Each tool is a plain function over images and detection dicts. They were moved
here unchanged from agent_controller.py so the loop, the experiments and the
app all call the same code. Which tool runs, and on which boxes, is decided
by a policy (policy.py), not inside the tool.

Tool kinds:
    discretionary  enhance, zoom, vote      the policy may skip them
    core           detect, classify          always run (the plain models)
    safety         text filter, refuse gate, Known/Unknown — see verifier.py
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

import config
import ensemble

DISCRETIONARY = ("enhance", "zoom", "vote")

# Zoom re-checks boxes whose fused confidence is in this band.
ZOOM_BAND = (0.4, 0.75)


# ──────────────────────────────────────────────────────────────────────
# Observe: image quality
# ──────────────────────────────────────────────────────────────────────
def assess_image_quality(image: np.ndarray) -> dict[str, Any]:
    """Blur (Laplacian variance), brightness, contrast and resolution."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    is_dark = brightness < 50.0
    is_bright = brightness > 200.0
    is_low_contrast = contrast < 30.0
    is_blurry = blur_score < 100.0

    needs_enhancement = is_dark or is_bright or is_low_contrast

    return {
        "blur_score": blur_score,
        "brightness": brightness,
        "contrast": contrast,
        "resolution": (w, h),
        "is_dark": is_dark,
        "is_bright": is_bright,
        "is_low_contrast": is_low_contrast,
        "is_blurry": is_blurry,
        "needs_enhancement": needs_enhancement,
    }


# ──────────────────────────────────────────────────────────────────────
# Tool: enhance (D1)
# ──────────────────────────────────────────────────────────────────────
def enhance_image(image: np.ndarray, state: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    """CLAHE on the L channel if exposure/contrast is off; sharpen if blurry."""
    enhanced = image.copy()
    trace = []

    if state["needs_enhancement"]:
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        merged = cv2.merge((cl, a_channel, b_channel))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        reason = []
        if state["is_dark"]:
            reason.append(f"low brightness: {state['brightness']:.1f}")
        if state["is_bright"]:
            reason.append(f"high brightness: {state['brightness']:.1f}")
        if state["is_low_contrast"]:
            reason.append(f"low contrast: {state['contrast']:.1f}")
        trace.append(f"Action: Applied CLAHE enhancement due to {', '.join(reason)}")

    if state["is_blurry"]:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        trace.append(f"Action: Applied sharpening filter (blur variance {state['blur_score']:.1f} < 100.0)")

    return enhanced, trace


# ──────────────────────────────────────────────────────────────────────
# Tool: zoom and re-check (D3)
# ──────────────────────────────────────────────────────────────────────
def calculate_iou(box_a: list[float], box_b: list[float]) -> float:
    """IoU of two [x1, y1, x2, y2] boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    intersection = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0.0 else 0.0


def in_zoom_band(det: dict[str, Any]) -> bool:
    return ZOOM_BAND[0] <= det["confidence"] < ZOOM_BAND[1]


def zoom_boxes(
    image: np.ndarray,
    detections: list[dict[str, Any]],
    pipeline: Any,
    indices: set[int],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Re-detect on a 15%-padded crop around each box in *indices*.

    A crop detection that overlaps the box (IoU > 0.4) and is more confident
    replaces it. Boxes not in *indices* pass through untouched.
    """
    updated_detections = []
    trace = []
    h, w = image.shape[:2]

    for i, det in enumerate(detections):
        if i not in indices:
            updated_detections.append(det)
            continue

        box = det["bbox"]
        conf = det["confidence"]
        class_name = det["class_name"]

        box_w = box[2] - box[0]
        box_h = box[3] - box[1]
        pad_x = box_w * 0.15
        pad_y = box_h * 0.15

        cx1 = max(0, int(box[0] - pad_x))
        cy1 = max(0, int(box[1] - pad_y))
        cx2 = min(w, int(box[2] + pad_x))
        cy2 = min(h, int(box[3] + pad_y))

        crop = image[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            updated_detections.append(det)
            continue

        # Looked up on the module so experiments can wrap ensemble.run_detection.
        crop_fused = ensemble.run_detection(crop, pipeline)

        best_crop_det = None
        best_iou = 0.0
        for c_det in crop_fused:
            c_box = c_det["bbox"]
            global_box = [cx1 + c_box[0], cy1 + c_box[1], cx1 + c_box[2], cy1 + c_box[3]]
            iou = calculate_iou(box, global_box)
            if iou > 0.4 and iou > best_iou:
                best_iou = iou
                best_crop_det = (global_box, c_det["confidence"], c_det["class_id"], c_det["class_name"])

        if best_crop_det is not None:
            new_box, new_conf, new_class_id, new_class_name = best_crop_det
            if new_conf > conf:
                det["bbox"] = new_box
                det["confidence"] = new_conf
                det["class_id"] = new_class_id
                det["class_name"] = new_class_name
                trace.append(
                    f"Action: Zoomed-in crop for original class '{class_name}' ({conf:.2f}) "
                    f"improved confidence to {new_conf:.2f} (new class '{new_class_name}')"
                )
            else:
                trace.append(
                    f"Action: Zoomed-in crop for class '{class_name}' ({conf:.2f}) "
                    f"did not yield higher confidence (new: {new_conf:.2f}), keeping original"
                )
        else:
            trace.append(
                f"Action: Zoomed-in crop for class '{class_name}' ({conf:.2f}) "
                f"found no overlapping object in high resolution, keeping original"
            )

        updated_detections.append(det)

    return updated_detections, trace


def rerun_zoomed_detection(
    image: np.ndarray,
    detections: list[dict[str, Any]],
    pipeline: Any,
) -> tuple[list[dict[str, Any]], list[str]]:
    """The rule-based zoom: re-check every box in the 0.40–0.75 band."""
    idx = {i for i, d in enumerate(detections) if in_zoom_band(d)}
    return zoom_boxes(image, detections, pipeline, idx)


# ──────────────────────────────────────────────────────────────────────
# Tool: classify (second opinion) — core
# ──────────────────────────────────────────────────────────────────────
def classify_boxes(image: np.ndarray, detections: list[dict[str, Any]],
                   pipeline: Any) -> list[dict[str, Any]]:
    """Classify the 5%-padded crop of every box."""
    crops = [pipeline._extract_crop(image, d["bbox"], padding=0.05) for d in detections]
    return pipeline.classifier.classify_crops(crops) if crops else []


# ──────────────────────────────────────────────────────────────────────
# Scoring, with and without the vote (D4)
# ──────────────────────────────────────────────────────────────────────
def score_plain(idx: int, det: dict[str, Any], cls_res: dict[str, Any]) -> dict[str, Any]:
    """Combine detector and classifier without a tie-break.

    On agreement this is exactly resolve_detection's result. On disagreement
    it keeps the classifier's label (as in the ablation's no-vote rows).
    """
    det_class, cls_class = det["class_name"], cls_res["class_name"]
    det_conf, cls_conf = det["confidence"], cls_res["confidence"]
    agree = det_class == cls_class
    combined = config.DETECTOR_WEIGHT * det_conf + config.CLASSIFIER_WEIGHT * cls_conf
    if agree:
        combined = min(combined * (1 + config.AGREEMENT_BOOST), 1.0)
    det.update({
        "id": idx + 1,
        "detector_class": det_class,
        "detector_confidence": det_conf,
        "classifier_class": cls_class,
        "classifier_confidence": cls_conf,
        "combined_confidence": combined,
        "agreement": agree,
    })
    return det


def resolve_detection(
    idx: int,
    det: dict[str, Any],
    cls_res: dict[str, Any],
    pipeline: Any,
    enhanced_image: np.ndarray,
) -> tuple[dict[str, Any], list[str], float]:
    """Score one detection; on disagreement, re-classify a looser crop and vote.

    Returns (resolved_det, trace_lines, reclassification_ms).
    """
    trace: list[str] = []
    reclassification_ms = 0.0

    det_class = det["class_name"]
    cls_class = cls_res["class_name"]
    det_conf = det["confidence"]
    cls_conf = cls_res["confidence"]

    if det_class == cls_class:
        return score_plain(idx, det, cls_res), trace, reclassification_ms

    trace.append(
        f"Warning: Disagreement on Object {idx+1}. Detector: '{det_class}' ({det_conf:.2f}) "
        f"vs Classifier (5% crop): '{cls_class}' ({cls_conf:.2f})"
    )

    t_recls = time.perf_counter()
    loose_crop = pipeline._extract_crop(enhanced_image, det["bbox"], padding=0.15)
    cls_loose = pipeline.classifier.classify_crop(loose_crop)
    reclassification_ms += (time.perf_counter() - t_recls) * 1000.0

    loose_class = cls_loose["class_name"]
    loose_conf = cls_loose["confidence"]

    opinions = [det_class, cls_class, loose_class]
    counts: dict[str, int] = {}
    for op in opinions:
        counts[op] = counts.get(op, 0) + 1
    majority_class = max(counts, key=counts.get)
    majority_count = counts[majority_class]

    if majority_count >= 2:
        resolved_class = majority_class
        trace.append(
            f"Action: Disagreement resolved to '{resolved_class}' "
            f"via majority vote: Detector ('{det_class}'), "
            f"Classifier Std ('{cls_class}'), Classifier Loose ('{loose_class}')"
        )
    else:
        resolved_class = det_class
        trace.append(
            f"Action: Disagreement deadlock (no majority). Defaulted to detector class "
            f"'{det_class}'"
        )

    resolved_conf = cls_conf if resolved_class == cls_class else (
        loose_conf if resolved_class == loose_class else det_conf)
    combined = config.DETECTOR_WEIGHT * det_conf + config.CLASSIFIER_WEIGHT * resolved_conf

    det.update({
        "id": idx + 1,
        "detector_class": det_class,
        "detector_confidence": det_conf,
        "classifier_class": resolved_class,
        "classifier_confidence": resolved_conf,
        "combined_confidence": combined,
        "agreement": False,
    })
    return det, trace, reclassification_ms
