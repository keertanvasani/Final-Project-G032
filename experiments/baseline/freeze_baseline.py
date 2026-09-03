"""Freeze the current image pipeline as IMAGE_V1_BASELINE.

Captures everything needed to reproduce and attribute later results: model
weight hashes, every threshold the pipeline reads, dataset composition,
software and hardware, and the measured per-action cost table.

Action costs are measured rather than assigned. The agent policy trades
reliability against computation, so a cost model built from guesses would make
that trade-off unfalsifiable.

Usage:
    python experiments/baseline/freeze_baseline.py            # measure on 25 images
    python experiments/baseline/freeze_baseline.py --n 100    # more samples
    python experiments/baseline/freeze_baseline.py --no-measure
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

OUT = Path(__file__).parent / "IMAGE_V1_BASELINE.json"

# Components whose measured times are mutually exclusive and reconcile to total.
EXCLUSIVE = [
    "preprocessing_ms", "detection_ms", "wbf_ms", "zoom_ms",
    "classification_ms", "reclassification_ms", "annotation_ms", "agent_decision_ms",
]


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: Path) -> dict:
    return {
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256(path),
    }


def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except Exception:
            return None
    # -uno: only tracked files. Untracked outputs (this manifest, results/)
    # do not affect whether the *code* at this commit is reproducible, and
    # counting them would flag every run as dirty.
    dirty = run("git", "status", "--porcelain", "-uno")
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "tracked_files_clean": (dirty == "" or dirty is None),
    }


def versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for mod in ("torch", "torchvision", "ultralytics", "cv2", "numpy", "ensemble_boxes", "fastapi"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def dataset_summary() -> dict:
    def count(pattern):
        return len(glob.glob(str(ROOT / pattern)))
    return {
        "name": "Aquarium Combined",
        "source": f"roboflow://{config.ROBOFLOW_WORKSPACE}/{config.ROBOFLOW_PROJECT}/v{config.ROBOFLOW_VERSION}",
        "license": "CC BY 4.0",
        "classes": config.CLASS_NAMES,
        "num_classes": config.NUM_CLASSES,
        "images": {
            "train": count("data/images/train/*.jpg"),
            "val": count("data/images/val/*.jpg"),
        },
        "note": "Aquarium-tank domain. Deep-sea generalisation is NOT established by this baseline.",
    }


def measure_action_costs(n: int) -> dict:
    """Run the frozen pipeline over n validation images and summarise stage cost."""
    import cv2
    from pipeline import EnsemblePipeline
    from agent_controller import run_agentic_pipeline

    paths = sorted(glob.glob(str(ROOT / "data/images/val/*.jpg")))[:n]
    if not paths:
        return {"error": "no validation images found"}

    pipe = EnsemblePipeline()
    rows, objects, residuals, abstained = [], [], [], 0
    t0 = time.perf_counter()

    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        _, _, summary, _ = run_agentic_pipeline(img, pipe)
        b = summary["inference_breakdown"]
        rows.append(b)
        objects.append(summary["total_objects"])
        if summary["total_objects"] == 0:
            abstained += 1
        residuals.append(abs(summary["inference_time_ms"] - sum(b[k] for k in EXCLUSIVE)))

    def stat(key):
        vals = [r[key] for r in rows]
        fired = [v for v in vals if v > 0]
        return {
            "mean_ms": round(statistics.fmean(vals), 3),
            "median_ms": round(statistics.median(vals), 3),
            "p95_ms": round(sorted(vals)[max(0, int(0.95 * len(vals)) - 1)], 3),
            "max_ms": round(max(vals), 3),
            # Conditional cost: what the stage costs when it actually runs.
            # For optional actions this is the number the agent's budget needs.
            "fired_on_images": len(fired),
            "mean_when_fired_ms": round(statistics.fmean(fired), 3) if fired else 0.0,
        }

    keys = EXCLUSIVE + ["detector_nano_ms", "detector_small_ms"]
    return {
        "images_measured": len(rows),
        "wall_clock_s": round(time.perf_counter() - t0, 1),
        "device": config.DEVICE,
        "objects_per_image": {
            "mean": round(statistics.fmean(objects), 2),
            "max": max(objects),
        },
        "abstained_images": abstained,
        "reconciliation": {
            "max_residual_ms": round(max(residuals), 4),
            "note": "Exclusive components minus measured total; ~0 confirms full accounting.",
        },
        "stages": {k: stat(k) for k in keys},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25, help="images to measure action costs on")
    ap.add_argument("--no-measure", action="store_true", help="skip cost measurement")
    args = ap.parse_args()

    manifest = {
        "baseline_id": "IMAGE_V1_BASELINE",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "software": versions(),
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or platform.machine(),
            "device": config.DEVICE,
        },
        "models": {
            "detector_nano": file_record(config.YOLO_N_WEIGHTS),
            "detector_small": file_record(config.YOLO_S_WEIGHTS),
            "classifier": file_record(config.CLASSIFIER_WEIGHTS),
        },
        "dataset": dataset_summary(),
        "parameters": {
            "detection": {
                "conf_threshold": config.YOLO_CONF_THRESHOLD,
                "nms_iou_threshold": config.YOLO_IOU_THRESHOLD,
                "img_size": config.YOLO_IMG_SIZE,
            },
            "fusion_wbf": {
                "iou_threshold": config.WBF_IOU_THRESHOLD,
                "skip_box_threshold": config.WBF_SKIP_BOX_THRESHOLD,
                "weights": config.WBF_WEIGHTS,
                "conf_type": config.WBF_CONF_TYPE,
            },
            "classifier": {
                "backbone": config.CLASSIFIER_BACKBONE,
                "img_size": config.CLASSIFIER_IMG_SIZE,
                "crop_padding_agentic": 0.05,
                "crop_padding_tiebreak": 0.15,
                "note": "The agentic path passes padding=0.05/0.15 explicitly, "
                        f"overriding config.CROP_PADDING={config.CROP_PADDING}.",
            },
            "confidence_fusion": {
                "detector_weight": config.DETECTOR_WEIGHT,
                "classifier_weight": config.CLASSIFIER_WEIGHT,
                "agreement_boost": config.AGREEMENT_BOOST,
            },
            "abstention_gate": {
                "review_conf_threshold": config.REVIEW_CONF_THRESHOLD,
                "unusable_blur_threshold": config.UNUSABLE_BLUR_THRESHOLD,
                "criterion": "per-box (>=1 box must clear review_conf_threshold)",
                "unit": "image-level abstention",
            },
            "quality_triggers": {
                "clahe_if_brightness_below": 50.0,
                "clahe_if_brightness_above": 200.0,
                "clahe_if_contrast_below": 30.0,
                "sharpen_if_blur_below": 100.0,
            },
            "zoom_recheck": {
                "ambiguous_band": [0.40, 0.75],
                "crop_padding": 0.15,
                "accept_if_iou_above": 0.4,
                "accept_only_if_confidence_increases": True,
            },
            "api": {
                "allowed_types": sorted(config.ALLOWED_IMAGE_TYPES),
                "max_image_mb": config.MAX_IMAGE_SIZE_MB,
            },
        },
        "known_limitations": [
            "Trained only on aquarium-tank imagery; deep-sea performance unmeasured.",
            "Abstention operates at image level; per-detection selective risk not yet defined.",
            "Confidence is uncalibrated — scores are not probabilities of correctness.",
            "Closed-set classifier: unseen taxa are forced into one of 7 classes.",
            "Single training run per model; no seed variance or confidence intervals.",
        ],
    }

    if not args.no_measure:
        manifest["measured_action_costs"] = measure_action_costs(args.n)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {OUT.relative_to(ROOT)}")

    mac = manifest.get("measured_action_costs", {})
    if "stages" in mac:
        print(f"\nMeasured on {mac['images_measured']} images ({mac['device']}), "
              f"{mac['wall_clock_s']}s wall clock")
        print(f"max reconciliation residual: {mac['reconciliation']['max_residual_ms']} ms\n")
        print(f"  {'stage':<22}{'mean':>9}{'median':>9}{'p95':>9}{'when fired':>12}{'fired':>7}")
        for k, s in mac["stages"].items():
            print(f"  {k:<22}{s['mean_ms']:>9.2f}{s['median_ms']:>9.2f}"
                  f"{s['p95_ms']:>9.2f}{s['mean_when_fired_ms']:>12.2f}{s['fired_on_images']:>7}")


if __name__ == "__main__":
    main()
