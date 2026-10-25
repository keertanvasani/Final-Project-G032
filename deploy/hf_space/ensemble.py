"""
Weighted Boxes Fusion (WBF) ensemble module for the Deep-Sea Creature
Detection & Classification System.

This module runs two YOLO detectors on the same image, extracts their
raw predictions, and fuses them using the WBF algorithm to produce a
single, higher-quality set of bounding-box detections.

Typical usage:
    results_a, results_b = run_dual_detection(image, model_a, model_b)
    fused = fuse_detections(results_a, results_b, image.shape[:2])
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from ensemble_boxes import weighted_boxes_fusion
from ultralytics import YOLO

import config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Dual-model detection
# ──────────────────────────────────────────────────────────────────────

def run_dual_detection(
    image: np.ndarray,
    model_a: YOLO,
    model_b: YOLO,
    timings: dict[str, float] | None = None,
) -> tuple[Any, Any]:
    """Run two YOLO models on the same image and return their raw results.

    Both models are invoked with the confidence and IoU thresholds
    defined in ``config``.

    Parameters
    ----------
    image : np.ndarray
        Input image as a NumPy array (BGR, HWC).
    model_a : YOLO
        First YOLO model instance (e.g. YOLOv8-nano).
    model_b : YOLO
        Second YOLO model instance (e.g. YOLOv8-small).
    timings : dict[str, float], optional
        If supplied, per-model wall-clock cost is *added* into this dict
        under ``detector_nano_ms`` and ``detector_small_ms``. Accumulating
        rather than overwriting keeps the totals correct when detection is
        invoked more than once per image (the zoom-and-recheck action
        re-runs both models on a crop). Timing the two models separately
        is what lets the per-action cost model be measured instead of
        assumed.

    Returns
    -------
    tuple[Any, Any]
        ``(results_a, results_b)`` — raw
        :class:`ultralytics.engine.results.Results` lists returned by
        each model's ``predict`` method.
    """
    logger.debug("Running dual YOLO detection (conf=%.2f, iou=%.2f).",
                 config.YOLO_CONF_THRESHOLD, config.YOLO_IOU_THRESHOLD)

    t0 = time.perf_counter()
    results_a = model_a.predict(
        source=image,
        conf=config.YOLO_CONF_THRESHOLD,
        iou=config.YOLO_IOU_THRESHOLD,
        verbose=False,
    )
    t1 = time.perf_counter()
    results_b = model_b.predict(
        source=image,
        conf=config.YOLO_CONF_THRESHOLD,
        iou=config.YOLO_IOU_THRESHOLD,
        verbose=False,
    )
    t2 = time.perf_counter()

    if timings is not None:
        timings["detector_nano_ms"] = timings.get("detector_nano_ms", 0.0) + (t1 - t0) * 1000.0
        timings["detector_small_ms"] = timings.get("detector_small_ms", 0.0) + (t2 - t1) * 1000.0

    logger.debug(
        "Model-A detections: %d | Model-B detections: %d",
        len(results_a[0].boxes) if results_a else 0,
        len(results_b[0].boxes) if results_b else 0,
    )
    return results_a, results_b


# ──────────────────────────────────────────────────────────────────────
# Prediction extraction
# ──────────────────────────────────────────────────────────────────────

def extract_predictions(
    results: Any,
    img_shape: tuple[int, int],
) -> tuple[list[list[float]], list[float], list[int]]:
    """Extract normalised boxes, scores, and labels from YOLO results.

    Parameters
    ----------
    results : ultralytics Results list
        Raw output from ``model.predict()``.
    img_shape : tuple[int, int]
        ``(height, width)`` of the source image.  Currently used only
        for logging; boxes are already normalised via ``xyxyn``.

    Returns
    -------
    tuple[list[list[float]], list[float], list[int]]
        ``(boxes_list, scores_list, labels_list)`` where:

        * *boxes_list* — each element is ``[x1, y1, x2, y2]`` in 0-1.
        * *scores_list* — confidence scores.
        * *labels_list* — integer class IDs.
    """
    if not results or len(results) == 0:
        logger.debug("No results to extract predictions from.")
        return [], [], []

    boxes_tensor = results[0].boxes.xyxyn  # normalised [0, 1]
    scores_tensor = results[0].boxes.conf
    labels_tensor = results[0].boxes.cls

    boxes_list: list[list[float]] = boxes_tensor.cpu().tolist()
    scores_list: list[float] = scores_tensor.cpu().tolist()
    labels_list: list[int] = [int(c) for c in labels_tensor.cpu().tolist()]

    logger.debug(
        "Extracted %d predictions (img %dx%d).",
        len(scores_list), img_shape[1], img_shape[0],
    )
    return boxes_list, scores_list, labels_list


# ──────────────────────────────────────────────────────────────────────
# Weighted Boxes Fusion
# ──────────────────────────────────────────────────────────────────────

def fuse_detections(
    results_a: Any,
    results_b: Any,
    img_shape: tuple[int, int],
) -> list[dict[str, Any]]:
    """Fuse detections from two YOLO models using Weighted Boxes Fusion.

    The function extracts normalised predictions from each model,
    passes them through the ``weighted_boxes_fusion`` algorithm, and
    converts the fused boxes back to pixel coordinates.

    Parameters
    ----------
    results_a : ultralytics Results list
        Raw predictions from the first YOLO model.
    results_b : ultralytics Results list
        Raw predictions from the second YOLO model.
    img_shape : tuple[int, int]
        ``(height, width)`` of the source image, used for
        de-normalisation.

    Returns
    -------
    list[dict[str, Any]]
        Fused detections. Each dict contains:

        * ``bbox``  — ``[x1, y1, x2, y2]`` in pixel coordinates.
        * ``class_id`` — integer class ID.
        * ``confidence`` — fused confidence score.
        * ``class_name`` — human-readable class name.
    """
    h, w = img_shape[:2]

    boxes_a, scores_a, labels_a = extract_predictions(results_a, img_shape)
    boxes_b, scores_b, labels_b = extract_predictions(results_b, img_shape)

    # Edge case: both models returned zero detections.
    if len(scores_a) == 0 and len(scores_b) == 0:
        logger.info("Both models returned zero detections — returning empty list.")
        return []

    logger.debug(
        "Running WBF fusion (weights=%s, iou_thr=%.2f, skip_box_thr=%.3f).",
        config.WBF_WEIGHTS, config.WBF_IOU_THRESHOLD, config.WBF_SKIP_BOX_THRESHOLD,
    )

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        [boxes_a, boxes_b],
        [scores_a, scores_b],
        [labels_a, labels_b],
        weights=config.WBF_WEIGHTS,
        iou_thr=config.WBF_IOU_THRESHOLD,
        skip_box_thr=config.WBF_SKIP_BOX_THRESHOLD,
        conf_type=config.WBF_CONF_TYPE,
    )

    detections: list[dict[str, Any]] = []
    for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
        x1, y1, x2, y2 = box
        class_id = int(label)
        class_name = (
            config.CLASS_NAMES[class_id]
            if 0 <= class_id < len(config.CLASS_NAMES)
            else f"class_{class_id}"
        )
        detections.append(
            {
                "bbox": [
                    float(x1 * w),
                    float(y1 * h),
                    float(x2 * w),
                    float(y2 * h),
                ],
                "class_id": class_id,
                "confidence": float(score),
                "class_name": class_name,
            }
        )

    logger.info(
        "WBF fusion produced %d detections from %d + %d raw boxes.",
        len(detections), len(scores_a), len(scores_b),
    )
    return detections
