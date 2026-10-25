"""
Full ensemble pipeline orchestrator for the Deep-Sea Creature Detection
& Classification System.

:class:`EnsemblePipeline` ties together:

1. **Dual YOLO detection** — two YOLOv8 models run in parallel.
2. **Weighted Boxes Fusion** — raw detections are fused via WBF.
3. **Species classification** — each fused crop is classified by a
   ResNet-18 classifier.
4. **Annotation** — bounding boxes and labels are drawn on the image.

Typical usage::

    pipeline = EnsemblePipeline()
    annotated, detections, summary = pipeline.predict(image)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

import config
import ensemble
import text_overlay
from classifier import SpeciesClassifier, _DummyClassifier, build_classifier

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Pipeline class
# ──────────────────────────────────────────────────────────────────────


class EnsemblePipeline:
    """End-to-end detection + classification pipeline.

    On construction the pipeline loads:

    * Two YOLOv8 detectors (nano and small variants).
    * A ResNet-18 species classifier.

    If any model weights are missing the pipeline falls back to
    pretrained COCO models (YOLO) or a dummy classifier, so it can
    always be instantiated safely.

    Attributes
    ----------
    models_status : dict[str, bool]
        Health flags for each sub-model.
    """

    def __init__(self) -> None:
        logger.info("Initialising EnsemblePipeline …")

        self.models_status: dict[str, bool] = {
            "yolo_n": False,
            "yolo_s": False,
            "classifier": False,
        }

        # ── YOLO-Nano ──────────────────────────────────────────────
        self.model_a: YOLO = self._load_yolo(
            weights_path=config.YOLO_N_WEIGHTS,
            fallback="yolov8n.pt",
            name="YOLOv8-nano",
            status_key="yolo_n",
        )

        # ── YOLO-Small ─────────────────────────────────────────────
        self.model_b: YOLO = self._load_yolo(
            weights_path=config.YOLO_S_WEIGHTS,
            fallback="yolov8s.pt",
            name="YOLOv8-small",
            status_key="yolo_s",
        )

        # ── Species classifier ─────────────────────────────────────
        self.classifier: SpeciesClassifier | _DummyClassifier = build_classifier(
            weights_path=config.CLASSIFIER_WEIGHTS,
            class_names_path=config.CLASSIFIER_CLASS_NAMES,
            device=config.DEVICE,
        )
        self.models_status["classifier"] = not isinstance(
            self.classifier, _DummyClassifier
        )
        if self.models_status["classifier"]:
            logger.info("Species classifier loaded successfully.")
        else:
            logger.warning("Using dummy species classifier (weights not found).")

        logger.info(
            "EnsemblePipeline ready — device=%s | status=%s",
            config.DEVICE, self.models_status,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _load_yolo(
        self,
        weights_path: Path,
        fallback: str,
        name: str,
        status_key: str,
    ) -> YOLO:
        """Load a YOLO model, falling back to a pretrained COCO model.

        Parameters
        ----------
        weights_path : Path
            Expected path to fine-tuned weights.
        fallback : str
            Ultralytics model name used when *weights_path* is missing
            (e.g. ``"yolov8n.pt"``).
        name : str
            Human-readable model name for log messages.
        status_key : str
            Key in :attr:`models_status` to update.

        Returns
        -------
        YOLO
            Loaded model instance.
        """
        if Path(weights_path).exists():
            logger.info("Loading %s from %s.", name, weights_path)
            model = YOLO(str(weights_path))
            self.models_status[status_key] = True
        else:
            logger.warning(
                "%s weights not found at %s — falling back to pretrained %s.",
                name, weights_path, fallback,
            )
            model = YOLO(fallback)
            self.models_status[status_key] = False
        return model

    # ------------------------------------------------------------------ #
    # Main prediction entry-point
    # ------------------------------------------------------------------ #

    def predict(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
        """Run the full detection → fusion → classification → annotation pipeline.

        Parameters
        ----------
        image : np.ndarray
            Input image (BGR, HWC).

        Returns
        -------
        tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]
            ``(annotated_image, detections_list, summary_dict)``

            * **annotated_image** — copy of *image* with drawn boxes.
            * **detections_list** — per-object dicts (see below).
            * **summary_dict** — aggregate statistics.

        Detection dict keys
        --------------------
        id, bbox, detector_class, detector_confidence, classifier_class,
        classifier_confidence, combined_confidence, agreement.
        """
        t_start: float = time.perf_counter()
        h, w = image.shape[:2]
        img_shape: tuple[int, int] = (h, w)

        # 1. Dual detection ──────────────────────────────────────────
        t0 = time.perf_counter()
        results_a, results_b = ensemble.run_dual_detection(
            image, self.model_a, self.model_b
        )
        detection_ms: float = (time.perf_counter() - t0) * 1000.0

        # 2. Weighted Boxes Fusion ───────────────────────────────────
        t0 = time.perf_counter()
        fused: list[dict[str, Any]] = ensemble.fuse_detections(
            results_a, results_b, img_shape
        )
        # Burned-in video overlay — timestamp, depth, watermark — reads to the
        # detector as a cluster of small bright objects. Drop those boxes before
        # anything downstream treats them as organisms. Applied here as well as
        # in the agentic controller so the guarantee does not depend on which
        # entry point a caller uses.
        # Its cost falls inside fusion_ms here; the agentic controller, which is
        # what the web app runs, reports it separately as text_filter_ms.
        fused, text_boxes = text_overlay.filter_text_detections(image, fused)
        if text_boxes:
            logger.info("Suppressed %d detection(s) on overlay text.", len(text_boxes))
        fusion_ms: float = (time.perf_counter() - t0) * 1000.0

        # 3. Crop & classify ─────────────────────────────────────────
        t0 = time.perf_counter()
        crops: list[np.ndarray] = []
        for det in fused:
            crop = self._extract_crop(image, det["bbox"], padding=0.05)
            crops.append(crop)

        cls_results: list[dict[str, Any]] = (
            self.classifier.classify_crops(crops) if crops else []
        )
        classification_ms: float = (time.perf_counter() - t0) * 1000.0

        # 4. Merge & compute combined confidence ─────────────────────
        detections: list[dict[str, Any]] = []
        for idx, (det, cls_res) in enumerate(zip(fused, cls_results)):
            det_conf: float = det["confidence"]
            cls_conf: float = cls_res["confidence"]
            det_class: str = det["class_name"]
            cls_class: str = cls_res["class_name"]

            combined: float = (
                config.DETECTOR_WEIGHT * det_conf
                + config.CLASSIFIER_WEIGHT * cls_conf
            )
            agreement: bool = det_class == cls_class
            if agreement:
                combined = min(combined * (1 + config.AGREEMENT_BOOST), 1.0)

            detections.append(
                {
                    "id": idx + 1,
                    "bbox": det["bbox"],
                    "detector_class": det_class,
                    "detector_confidence": round(det_conf, 4),
                    "classifier_class": cls_class,
                    "classifier_confidence": round(cls_conf, 4),
                    "combined_confidence": round(combined, 4),
                    "agreement": agreement,
                }
            )

        # 5. Annotate ────────────────────────────────────────────────
        t0 = time.perf_counter()
        annotated: np.ndarray = self.annotate_image(image, detections)
        annotation_ms: float = (time.perf_counter() - t0) * 1000.0

        total_ms: float = (time.perf_counter() - t_start) * 1000.0

        # 6. Build summary ───────────────────────────────────────────
        avg_conf: float = (
            float(np.mean([d["combined_confidence"] for d in detections]))
            if detections
            else 0.0
        )
        summary: dict[str, Any] = {
            "total_objects": len(detections),
            "average_confidence": round(avg_conf, 4),
            "inference_time_ms": round(total_ms, 2),
            "inference_breakdown": {
                "detection_ms": round(detection_ms, 2),
                "fusion_ms": round(fusion_ms, 2),
                "classification_ms": round(classification_ms, 2),
                "annotation_ms": round(annotation_ms, 2),
            },
            "models_used": {
                "detector_1": "YOLOv8n",
                "detector_2": "YOLOv8s",
                "classifier": "ResNet18",
                "fusion": "Weighted Boxes Fusion",
            },
        }

        logger.info(
            "Pipeline complete: %d objects in %.1f ms (det=%.1f, fus=%.1f, "
            "cls=%.1f, ann=%.1f).",
            len(detections), total_ms, detection_ms, fusion_ms,
            classification_ms, annotation_ms,
        )
        return annotated, detections, summary

    # ------------------------------------------------------------------ #
    # Crop extraction
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_crop(
        image: np.ndarray,
        bbox: list[float],
        padding: float = 0.05,
    ) -> np.ndarray:
        """Extract a padded crop from an image given a bounding box.

        Parameters
        ----------
        image : np.ndarray
            Source image (BGR, HWC).
        bbox : list[float]
            ``[x1, y1, x2, y2]`` in pixel coordinates.
        padding : float
            Fractional padding to add around the bounding box (0.05 = 5 %).

        Returns
        -------
        np.ndarray
            Cropped region.
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        box_w: float = x2 - x1
        box_h: float = y2 - y1

        pad_x: float = box_w * padding
        pad_y: float = box_h * padding

        cx1: int = max(0, int(x1 - pad_x))
        cy1: int = max(0, int(y1 - pad_y))
        cx2: int = min(w, int(x2 + pad_x))
        cy2: int = min(h, int(y2 + pad_y))

        return image[cy1:cy2, cx1:cx2]

    # ------------------------------------------------------------------ #
    # Annotation
    # ------------------------------------------------------------------ #

    def annotate_image(
        self,
        image: np.ndarray,
        detections: list[dict[str, Any]],
    ) -> np.ndarray:
        """Draw bounding boxes and labels on a copy of the image.

        Parameters
        ----------
        image : np.ndarray
            Original image (BGR, HWC). **Not** modified in-place.
        detections : list[dict[str, Any]]
            Detection dicts produced by :meth:`predict`.

        Returns
        -------
        np.ndarray
            Annotated copy of the image.
        """
        annotated: np.ndarray = image.copy()

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            combined_conf: float = det["combined_confidence"]
            species: str = det.get("classifier_class", det.get("detector_class", "?"))
            color: tuple[int, ...] = config.get_confidence_color(combined_conf)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color,
                          config.ANNOTATION_THICKNESS)

            # Compose label text
            label: str = f"{species} {combined_conf:.0%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), baseline = cv2.getTextSize(
                label, font, config.ANNOTATION_FONT_SCALE,
                config.ANNOTATION_THICKNESS,
            )

            # Semi-transparent background behind text
            overlay: np.ndarray = annotated.copy()
            cv2.rectangle(
                overlay,
                (x1, y1 - th - baseline - 4),
                (x1 + tw + 4, y1),
                color,
                cv2.FILLED,
            )
            cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)

            # Draw text
            cv2.putText(
                annotated,
                label,
                (x1 + 2, y1 - baseline - 2),
                font,
                config.ANNOTATION_FONT_SCALE,
                (255, 255, 255),
                config.ANNOTATION_THICKNESS,
                cv2.LINE_AA,
            )

        return annotated

    # ------------------------------------------------------------------ #
    # Review-flag warning annotation
    # ------------------------------------------------------------------ #

    def annotate_message(
        self,
        image: np.ndarray,
        message: str,
    ) -> np.ndarray:
        """Draw a full-width warning banner with *message* on a copy of the image.

        Used when the agentic controller can't reliably classify anything
        in the image (e.g. it's still too blurry after enhancement) and
        needs to surface that directly on the returned image itself.

        Parameters
        ----------
        image : np.ndarray
            Source image (BGR, HWC). **Not** modified in-place.
        message : str
            Warning text, word-wrapped to fit the image width.

        Returns
        -------
        np.ndarray
            Annotated copy of the image with the warning banner drawn on top.
        """
        annotated: np.ndarray = image.copy()
        h, w = annotated.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        padding = 14

        # Word-wrap the message to fit within the image width.
        words = message.split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            (tw, _), _ = cv2.getTextSize(candidate, font, font_scale, thickness)
            if tw > w - 2 * padding and current:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)

        (_, line_h), baseline = cv2.getTextSize("Ag", font, font_scale, thickness)
        line_height = line_h + baseline + 10
        banner_h = min(h, line_height * len(lines) + 2 * padding)

        overlay: np.ndarray = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 200), cv2.FILLED)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

        y = padding + line_h
        for line in lines:
            (tw, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
            x = max(padding, (w - tw) // 2)
            cv2.putText(
                annotated, line, (x, y), font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA,
            )
            y += line_height

        return annotated

    # ------------------------------------------------------------------ #
    # Health check
    # ------------------------------------------------------------------ #

    def get_models_status(self) -> dict[str, bool]:
        """Return the load status of every sub-model.

        Returns
        -------
        dict[str, bool]
            Keys: ``yolo_n``, ``yolo_s``, ``classifier``.
        """
        return dict(self.models_status)
