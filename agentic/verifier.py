"""Verifier: the checks that always run, whatever the policy decides.

    text filter     drop boxes drawn on burned-in telemetry text
    refuse gate     unreadable image, or no box reaches REVIEW_CONF_THRESHOLD
    calibration     shown % = P(correct) on held-out dives (monotone map)
    Known/Unknown   energy above the fitted threshold is shown as Unknown

None of these is a tool the policy can choose; the loop calls them directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import config
import text_overlay


def text_filter(image: np.ndarray, detections: list[dict[str, Any]]
                ) -> tuple[list[dict[str, Any]], list[str]]:
    """Remove boxes on overlay text. Returns (kept, trace_lines)."""
    detections, text_boxes = text_overlay.filter_text_detections(image, detections)
    trace: list[str] = []
    if text_boxes:
        # Two routes reject a box, and the trace says which. Quoting a glyph
        # percentage for the second would be false: that route fires on a box too
        # small to contain a measurable run of text.
        by_glyphs = [d for d in text_boxes if not d.get("text_reason")]
        by_block = [d for d in text_boxes if d.get("text_reason")]
        parts = []
        if by_glyphs:
            worst = max(d["text_coverage"] for d in by_glyphs)
            parts.append(f"{len(by_glyphs)} on glyphs (up to {worst:.0%} of the box)")
        if by_block:
            parts.append(f"{len(by_block)} character-sized and inside an overlay block")
        trace.append(
            f"Action: Rejected {len(text_boxes)} detection(s) drawn on burned-in "
            f"overlay text — {'; '.join(parts)}. These are video telemetry, "
            f"not organisms"
        )
    return detections, trace


def is_readable(quality: dict[str, Any]) -> bool:
    return quality["blur_score"] >= config.UNUSABLE_BLUR_THRESHOLD


def refuse_gate(quality: dict[str, Any], detections: list[dict[str, Any]]
                ) -> tuple[bool, list[str]]:
    """Decide whether the image is worth analysing. Returns (accepted, trace).

    Two independent reasons to refuse: the image was never readable, or it was
    readable but nothing was detected confidently. The threshold decides whether
    the image is analysed at all, not which boxes survive: weak boxes are kept so
    the classifier still gets its say.
    """
    confident = [d for d in detections if d["confidence"] >= config.REVIEW_CONF_THRESHOLD]
    best = max((d["confidence"] for d in detections), default=0.0)
    if not is_readable(quality):
        return False, [
            f"Action: Image is unreadable (blur variance {quality['blur_score']:.1f} < "
            f"{config.UNUSABLE_BLUR_THRESHOLD:.1f}); detections from it cannot be trusted. "
            f"Flagged for human review"
        ]
    if not confident:
        return False, [
            f"Action: No detection reached the {config.REVIEW_CONF_THRESHOLD:.2f} confidence "
            f"threshold (best was {best:.2f}). Flagged for human review"
        ]
    return True, [
        f"Action: {len(confident)} of {len(detections)} boxes cleared "
        f"{config.REVIEW_CONF_THRESHOLD:.2f}; proceeding with all of them"
    ]


def review_message(quality: dict[str, Any]) -> str:
    if quality["is_blurry"]:
        return "Image too blurry to analyze reliably. Please try another photo."
    return ("Detection confidence too low to reliably identify species. "
            "Please try another photo.")


def finalise(idx: int, resolved: dict[str, Any], cls_res: dict[str, Any],
             pipeline: Any) -> list[str]:
    """Calibrate the confidence and attach the Known/Unknown flag, in place."""
    trace: list[str] = []
    if getattr(pipeline, "calibration", None):
        resolved["raw_combined_confidence"] = resolved["combined_confidence"]
        resolved["combined_confidence"] = pipeline.calibrate(resolved["combined_confidence"])
    nov = pipeline.novelty_for(cls_res.get("energy")) if hasattr(pipeline, "novelty_for") else None
    if nov is not None:
        resolved["novelty"] = nov
        if nov["unknown"]:
            trace.append(
                f"Novelty: Object {idx+1} looks unlike the trained classes "
                f"(energy {nov['score']:.2f} > {nov['threshold']:.2f}); "
                f"marked Unknown — '{resolved['classifier_class']}' is the nearest "
                f"known class, not a confirmed identification")
    return trace


def load_taxon_spec(pipeline: Any) -> dict[str, Any] | None:
    """The fitted taxonomy-aware verifier (week 4), cached on the pipeline; None if off."""
    cfg = getattr(config, "AGENT_VERIFIER", None)
    if not cfg:
        return None
    spec = getattr(pipeline, "_taxon_spec", None)
    if spec is None:
        import json
        from pathlib import Path
        spec = json.loads(Path(config.AGENT_VERIFIER_FILE).read_text())
        pipeline._taxon_spec = spec
    # Fitted on the deep-sea classifier's classes; any other profile gets no taxon answer.
    if list(spec["classes"]) != list(getattr(pipeline.classifier, "class_names", [])):
        return None
    return spec


def taxon_label(resolved: dict[str, Any], cls_res: dict[str, Any], image_shape: tuple[int, ...],
                spec: dict[str, Any], mode: str, alpha: float) -> dict[str, Any]:
    """Most specific safe answer for one finished box (PROTOCOL §14).

    Reports the deepest level whose combiner probability clears the
    Learn-then-Test threshold for *alpha*; otherwise abstains (needs review).
    A safety step: the policy cannot switch it off or change its threshold."""
    import numpy as np

    from agentic import taxonomy as tx
    h, w = image_shape[:2]
    b = resolved["bbox"]
    area = max(b[2] - b[0], 1.0) * max(b[3] - b[1], 1.0) / float(w * h)
    nov = resolved.get("novelty") or {}
    nodes, feats = tx.level_features(
        cls_res["probs"], spec["classes"], resolved["classifier_class"], nov.get("score"),
        bool(nov.get("unknown")), resolved["detector_confidence"],
        resolved.get("raw_combined_confidence", resolved["combined_confidence"]), resolved["agreement"],
        resolved.get("detector_class", resolved.get("class_name")), area)
    t = spec["thresholds"][mode][str(alpha)]
    levels = list(range(len(tx.LEVELS))) if mode == "hierarchical" else [0]
    probs = []
    for j, lv in enumerate(tx.LEVELS):
        m = spec["models"][lv]
        z = (np.asarray(feats[j]) - np.asarray(m["mean"])) / np.asarray(m["std"])
        probs.append(float(1 / (1 + np.exp(-(m["w"][0] + z @ np.asarray(m["w"][1:]))))))
    chosen = next((j for j in levels if t is not None and probs[j] >= t), None)
    return {"mode": mode, "alpha": alpha, "threshold": t,
            "level": tx.LEVELS[chosen] if chosen is not None else None,
            "node": nodes[chosen] if chosen is not None else None,
            "p_correct": probs[chosen] if chosen is not None else None,
            "candidates": [{"level": lv, "node": nd, "p_correct": round(pc, 4)}
                           for lv, nd, pc in zip(tx.LEVELS, nodes, probs)]}
