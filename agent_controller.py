"""
Deep-Sea Creature Detection & Classification System — Agentic Decision Controller

This module implements a rule-based agentic pipeline. Instead of executing
a fixed sequential pipeline, it inspects quality metrics of the image
and intermediate model confidence scores, then dynamically decides on
corrective/enhancement actions (such as image enhancement, zoom-and-recheck,
and tie-breaking classification).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

import config
import ensemble
from pipeline import EnsemblePipeline

logger = logging.getLogger("deep_sea_agent")


# ──────────────────────────────────────────────────────────────────────
# Image Quality Assessment
# ──────────────────────────────────────────────────────────────────────
def assess_image_quality(image: np.ndarray) -> dict[str, Any]:
    """Analyze image quality parameters (blur, brightness, contrast, resolution).

    Parameters
    ----------
    image : np.ndarray
        BGR image array.

    Returns
    -------
    dict[str, Any]
        Dictionary of quality parameters and boolean indicators for enhancement.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur detection via Laplacian variance
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Brightness & contrast
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    # Thresholds (can be adjusted)
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


def enhance_image(image: np.ndarray, state: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    """Apply enhancement actions based on image quality state.

    Parameters
    ----------
    image : np.ndarray
        BGR image array.
    state : dict[str, Any]
        Image quality state dict from assess_image_quality.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        The enhanced BGR image and a list of human-readable trace strings.
    """
    enhanced = image.copy()
    trace = []

    # 1. CLAHE Contrast & Brightness Enhancement
    if state["needs_enhancement"]:
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to the L channel
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

    # 2. Sharpening filter if image is blurry
    if state["is_blurry"]:
        # Standard sharpening kernel
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        trace.append(f"Action: Applied sharpening filter (blur variance {state['blur_score']:.1f} < 100.0)")

    return enhanced, trace


# ──────────────────────────────────────────────────────────────────────
# Bounding Box IoU Calculation
# ──────────────────────────────────────────────────────────────────────
def calculate_iou(box_a: list[float], box_b: list[float]) -> float:
    """Calculate Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    box_a : list[float]
        [x1, y1, x2, y2] coords.
    box_b : list[float]
        [x1, y1, x2, y2] coords.

    Returns
    -------
    float
        IoU value in range [0, 1].
    """
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    intersection = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0.0 else 0.0


# ──────────────────────────────────────────────────────────────────────
# Zoom & Recheck Action (Rerun detection on crop)
# ──────────────────────────────────────────────────────────────────────
def rerun_zoomed_detection(
    image: np.ndarray,
    detections: list[dict[str, Any]],
    pipeline: EnsemblePipeline,
) -> tuple[list[dict[str, Any]], list[str]]:
    """For low/ambiguous detections, crop that region and re-run dual YOLO + WBF.

    Parameters
    ----------
    image : np.ndarray
        BGR image array.
    detections : list[dict[str, Any]]
        Current list of detections.
    pipeline : EnsemblePipeline
        Pipeline instance containing loaded YOLO models.

    Returns
    -------
    tuple[list[dict[str, Any]], list[str]]
        Updated detections list and a list of trace logs.
    """
    updated_detections = []
    trace = []
    h, w = image.shape[:2]

    for det in detections:
        box = det["bbox"]
        conf = det["confidence"]
        class_name = det["class_name"]

        # Ambiguous confidence check (0.4 to 0.75) triggers zoom and re-check
        if 0.4 <= conf < 0.75:
            # Crop region with 15% padding
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

            # Run dual detection + WBF on crop
            res_a, res_b = ensemble.run_dual_detection(crop, pipeline.model_a, pipeline.model_b)
            crop_fused = ensemble.fuse_detections(res_a, res_b, crop.shape[:2])

            best_crop_det = None
            best_iou = 0.0

            # Match local predictions back to global coordinate space
            for c_det in crop_fused:
                c_box = c_det["bbox"]
                # Convert back to absolute image coordinates
                global_box = [
                    cx1 + c_box[0],
                    cy1 + c_box[1],
                    cx1 + c_box[2],
                    cy1 + c_box[3]
                ]

                # Measure overlap with original box
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


# ──────────────────────────────────────────────────────────────────────
# Main Agentic Pipeline Orchestrator
# ──────────────────────────────────────────────────────────────────────
def run_agentic_pipeline(
    image: np.ndarray,
    pipeline: EnsemblePipeline,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any], list[str]]:
    """Run the multi-model pipeline with agentic decision-making logic.

    Parameters
    ----------
    image : np.ndarray
        BGR input image.
    pipeline : EnsemblePipeline
        Full pipeline container instance.

    Returns
    -------
    tuple[np.ndarray, list[dict[str, Any]], dict[str, Any], list[str]]
        (annotated_image, detections_list, summary_dict, decision_trace)
    """
    t_start = time.perf_counter()
    trace = []
    status = "complete"

    # Stage 1: Pre-processing Quality Check & Decision
    quality_state = assess_image_quality(image)
    trace.append(
        f"Quality check: blur score = {quality_state['blur_score']:.1f}, "
        f"brightness = {quality_state['brightness']:.1f}, "
        f"contrast = {quality_state['contrast']:.1f}, "
        f"resolution = {quality_state['resolution']}"
    )

    enhanced_image = image
    if quality_state["needs_enhancement"] or quality_state["is_blurry"]:
        enhanced_image, enhance_trace = enhance_image(image, quality_state)
        trace.extend(enhance_trace)
    else:
        trace.append("Action: Image is well-formed, skipping pre-enhancements")

    # Stage 2: Dual Detection + WBF Fusion
    t0 = time.perf_counter()
    res_a, res_b = ensemble.run_dual_detection(
        enhanced_image, pipeline.model_a, pipeline.model_b
    )
    detections = ensemble.fuse_detections(res_a, res_b, enhanced_image.shape[:2])
    detection_ms = (time.perf_counter() - t0) * 1000.0

    # Calculate average detector confidence
    if detections:
        avg_det_conf = float(np.mean([d["confidence"] for d in detections]))
    else:
        avg_det_conf = 0.0

    trace.append(f"Detection results: Found {len(detections)} boxes (avg conf: {avg_det_conf:.2f})")

    # Stage 3: Post-Detection Decision Points
    if not detections or avg_det_conf < 0.4:
        # Action: Flag image as low-confidence/uncertain
        trace.append("Action: Combined detection confidence too low (< 0.4) or no objects found. Flagged for human review")
        status = "needs_review"
    else:
        # Action: Zoom & Recheck low-confidence bounding boxes
        detections, zoom_trace = rerun_zoomed_detection(enhanced_image, detections, pipeline)
        trace.extend(zoom_trace)

    # Stage 4: Species Classification & Disagreement Resolving
    t0 = time.perf_counter()
    resolved_detections = []
    classifier_ms = 0.0
    fusion_ms = 5.0  # Estimated WBF fusion time

    if detections and status != "needs_review":
        # Extract crops with standard padding (5%)
        standard_crops = [pipeline._extract_crop(enhanced_image, det["bbox"], padding=0.05) for det in detections]
        cls_results = pipeline.classifier.classify_crops(standard_crops) if standard_crops else []
        classifier_ms += (time.perf_counter() - t0) * 1000.0

        for idx, (det, cls_res) in enumerate(zip(detections, cls_results)):
            det_class = det["class_name"]
            cls_class = cls_res["class_name"]
            det_conf = det["confidence"]
            cls_conf = cls_res["confidence"]

            agreement = det_class == cls_class

            if agreement:
                # Direct match, compute combined confidence with agreement boost
                combined = config.DETECTOR_WEIGHT * det_conf + config.CLASSIFIER_WEIGHT * cls_conf
                combined = min(combined * (1 + config.AGREEMENT_BOOST), 1.0)

                det.update({
                    "id": idx + 1,
                    "detector_class": det_class,
                    "detector_confidence": det_conf,
                    "classifier_class": cls_class,
                    "classifier_confidence": cls_conf,
                    "combined_confidence": combined,
                    "agreement": True
                })
                resolved_detections.append(det)
            else:
                # Disagreement! Run tie-breaker classifier on slightly larger crop (15%)
                trace.append(
                    f"Warning: Disagreement on Object {idx+1}. Detector: '{det_class}' ({det_conf:.2f}) "
                    f"vs Classifier (5% crop): '{cls_class}' ({cls_conf:.2f})"
                )

                t_t0 = time.perf_counter()
                loose_crop = pipeline._extract_crop(enhanced_image, det["bbox"], padding=0.15)
                cls_loose = pipeline.classifier.classify_crop(loose_crop)
                classifier_ms += (time.perf_counter() - t_t0) * 1000.0

                loose_class = cls_loose["class_name"]
                loose_conf = cls_loose["confidence"]

                # Opinions:
                # 1. Detector: det_class
                # 2. Classifier Std (5%): cls_class
                # 3. Classifier Loose (15%): loose_class
                opinions = [det_class, cls_class, loose_class]

                # Count frequencies
                counts = {}
                for op in opinions:
                    counts[op] = counts.get(op, 0) + 1

                # Find majority vote class
                majority_class = max(counts, key=counts.get)
                majority_count = counts[majority_class]

                if majority_count >= 2:
                    resolved_class = majority_class
                    decision_str = (
                        f"Action: Disagreement resolved to '{resolved_class}' "
                        f"via majority vote: Detector ('{det_class}'), "
                        f"Classifier Std ('{cls_class}'), Classifier Loose ('{loose_class}')"
                    )
                else:
                    # All three disagree, select detector class
                    resolved_class = det_class
                    decision_str = (
                        f"Action: Disagreement deadlock (no majority). Defaulted to detector class "
                        f"'{det_class}'"
                    )

                trace.append(decision_str)

                # Set classifier class to the resolved class and calculate combined confidence
                resolved_conf = cls_conf if resolved_class == cls_class else (loose_conf if resolved_class == loose_class else det_conf)
                combined = config.DETECTOR_WEIGHT * det_conf + config.CLASSIFIER_WEIGHT * resolved_conf

                det.update({
                    "id": idx + 1,
                    "detector_class": det_class,
                    "detector_confidence": det_conf,
                    "classifier_class": resolved_class,
                    "classifier_confidence": resolved_conf,
                    "combined_confidence": combined,
                    "agreement": False
                })
                resolved_detections.append(det)
    else:
        # Fill in empty details for return
        resolved_detections = []

    # Final stage: Annotation drawing
    t0 = time.perf_counter()
    annotated = pipeline.annotate_image(enhanced_image, resolved_detections)
    annotation_ms = (time.perf_counter() - t0) * 1000.0

    total_ms = (time.perf_counter() - t_start) * 1000.0

    # If status is needs_review, add status note to trace
    if status == "needs_review":
        trace.append("Final state: flagged for manual human review due to uncertainty")
    else:
        # Add final labels to trace
        for d in resolved_detections:
            trace.append(f"Final label for Object {d['id']}: {d['classifier_class']} (combined conf: {d['combined_confidence']:.2f})")

    avg_conf = float(np.mean([d["combined_confidence"] for d in resolved_detections])) if resolved_detections else 0.0

    summary = {
        "total_objects": len(resolved_detections),
        "average_confidence": round(avg_conf, 4),
        "inference_time_ms": round(total_ms, 2),
        "inference_breakdown": {
            "detection_ms": round(detection_ms, 2),
            "fusion_ms": round(fusion_ms, 2),
            "classification_ms": round(classifier_ms, 2),
            "annotation_ms": round(annotation_ms, 2),
        },
        "models_used": {
            "detector_1": "YOLOv8n",
            "detector_2": "YOLOv8s",
            "classifier": "ResNet18",
            "fusion": "Weighted Boxes Fusion",
        },
    }

    return annotated, resolved_detections, summary, trace
