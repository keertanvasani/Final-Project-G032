"""
Deep-Sea Creature Detection & Classification System — Agentic Decision Controller

Public entry points for the app and the experiments. The agent itself lives
in the agentic package:

    agentic/tools.py     enhance, zoom, classify, vote (the actions)
    agentic/policy.py    who decides which actions run (rules → plain fallback)
    agentic/verifier.py  text filter, refuse gate, calibration, Known/Unknown
    agentic/loop.py      Observe → Decide → Act → Verify with guardrails

The tool functions are re-exported here so existing imports keep working.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from agentic import verifier
from agentic.loop import run_loop
from agentic.tools import (assess_image_quality, calculate_iou,  # noqa: F401  (re-exported)
                           enhance_image, rerun_zoomed_detection, resolve_detection)
from pipeline import EnsemblePipeline

logger = logging.getLogger("deep_sea_agent")


def run_perception(
    image: np.ndarray,
    pipeline: EnsemblePipeline,
) -> dict[str, Any]:
    """Detection → fusion → zoom → classification → scoring, without the accept gate.

    The perception stage of the evidence-acquisition architecture, separated
    from the acceptance policy. It returns every scored detection with its raw
    combined confidence, so evaluation can sweep a risk–coverage curve; the
    acceptance policy is reconstructed downstream by thresholding
    ``combined_confidence`` (experiments/PROTOCOL.md §3). The one gate it keeps
    is the unreadable-image gate (status ``"unreadable"``).

    Returns dict(detections, status ∈ {"ok", "unreadable", "empty"}, quality, timings).
    """
    run = run_loop(image, pipeline, mode="perception")
    return {"detections": run.detections, "status": run.status, "quality": run.quality,
            "timings": run.det_timings}


def run_agentic_pipeline(
    image: np.ndarray,
    pipeline: EnsemblePipeline,
    enhance: bool = True,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any], list[str]]:
    """Run the agent on one image.

    Returns (annotated_image, detections, summary, decision_trace).

    Every stage is measured rather than estimated: the agent trades reliability
    against compute, so an action's cost has to be an observed quantity.
    """
    t_start = time.perf_counter()
    run = run_loop(image, pipeline, enhance=enhance, mode="app")
    trace = run.trace
    tm = run.timings
    detections = run.detections

    t_annot = time.perf_counter()
    if run.status == "needs_review":
        message = verifier.review_message(run.quality)
        annotated = pipeline.annotate_message(run.work_image, message)
    else:
        annotated = pipeline.annotate_image(run.work_image, detections)
    annotation_ms = (time.perf_counter() - t_annot) * 1000.0

    total_ms = (time.perf_counter() - t_start) * 1000.0

    detector_nano_ms = run.det_timings.get("detector_nano_ms", 0.0)
    detector_small_ms = run.det_timings.get("detector_small_ms", 0.0)
    detection_ms = detector_nano_ms + detector_small_ms

    # Controller overhead (policy, guardrails, gate, vote bookkeeping) is the
    # residual, so the components always reconcile against the measured total.
    measured_ms = (
        tm["preprocessing_ms"] + detection_ms + tm["wbf_ms"] + tm["text_filter_ms"]
        + tm["zoom_ms"] + tm["classification_ms"] + tm["reclassification_ms"] + annotation_ms
    )
    agent_decision_ms = max(0.0, total_ms - measured_ms)

    if run.status == "needs_review":
        trace.append(f"Final state: {message}")
    else:
        for d in detections:
            trace.append(f"Final label for Object {d['id']}: {d['classifier_class']} "
                         f"(combined conf: {d['combined_confidence']:.2f})")

    avg_conf = float(np.mean([d["combined_confidence"] for d in detections])) if detections else 0.0

    summary = {
        "total_objects": len(detections),
        "average_confidence": round(avg_conf, 4),
        "inference_time_ms": round(total_ms, 2),
        # Mutually exclusive components that reconcile to inference_time_ms.
        # detection_ms is the sum of the two detector arms.
        "inference_breakdown": {
            "preprocessing_ms": round(tm["preprocessing_ms"], 2),
            "detector_nano_ms": round(detector_nano_ms, 2),
            "detector_small_ms": round(detector_small_ms, 2),
            "detection_ms": round(detection_ms, 2),
            "wbf_ms": round(tm["wbf_ms"], 2),
            "text_filter_ms": round(tm["text_filter_ms"], 2),
            "zoom_ms": round(tm["zoom_ms"], 2),
            "classification_ms": round(tm["classification_ms"], 2),
            "reclassification_ms": round(tm["reclassification_ms"], 2),
            "annotation_ms": round(annotation_ms, 2),
            "agent_decision_ms": round(agent_decision_ms, 2),
        },
        "models_used": pipeline.models_used() if hasattr(pipeline, "models_used") else {
            "detector_1": "YOLOv8n",
            "detector_2": "YOLOv8s",
            "classifier": "ResNet18",
            "fusion": "Weighted Boxes Fusion",
        },
        "profile": getattr(pipeline, "profile", "aquarium"),
        "agent": {
            "policy": run.policy,
            "actions": _action_counts(run),
            "guardrails": run.guardrails,
        },
    }

    return annotated, detections, summary, trace


def _action_counts(run) -> dict[str, int]:
    """How many times each discretionary tool was chosen on this image."""
    out = {"enhance": 0, "zoom": 0, "vote": 0}
    for d in run.decisions:
        v = d["value"]
        if d["decision"] == "zoom":
            out["zoom"] += len(v)
        elif v:
            out[d["decision"]] += 1
    return out
