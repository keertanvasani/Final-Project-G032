#!/usr/bin/env python3
"""
train_yolo.py — Fine-tune two YOLOv8 models on the Aquarium Combined dataset.

Model A:  YOLOv8-Nano  (yolov8n.pt) — default augmentations
Model B:  YOLOv8-Small (yolov8s.pt) — enhanced augmentations (mixup + copy-paste)

Both models are trained on the same ``data/data.yaml`` dataset and their
best weights are saved under ``models/``.

Usage
-----
    # Train both models with defaults from config.py
    python train_yolo.py

    # Train only the nano model for 30 epochs
    python train_yolo.py --model n --epochs 30

    # Train the small model on GPU 0
    python train_yolo.py --model s --device 0
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ultralytics import YOLO

import config

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_yolo")


# ── Model registry ──────────────────────────────────────────────────────────
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "n": {
        "pretrained": "yolov8n.pt",
        "project": str(config.MODELS_DIR),
        "name": "yolov8n_aquarium",
        "seed": config.YOLO_N_SEED,
        "augment": config.YOLO_N_AUGMENT,
        "description": "YOLOv8-Nano (Model A)",
    },
    "s": {
        "pretrained": "yolov8s.pt",
        "project": str(config.MODELS_DIR),
        "name": "yolov8s_aquarium",
        "seed": config.YOLO_S_SEED,
        "augment": config.YOLO_S_AUGMENT,
        "description": "YOLOv8-Small (Model B)",
    },
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with sensible defaults from *config.py*.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 models on the Aquarium Combined dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["n", "s", "both"],
        default="both",
        help="Which model(s) to train: 'n' (nano), 's' (small), or 'both'.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.YOLO_TRAIN_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=config.YOLO_TRAIN_BATCH,
        help="Batch size.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=config.YOLO_TRAIN_IMGSZ,
        help="Input image size (pixels).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config.DEVICE,
        help="Device to train on (e.g. 'cpu', '0', 'cuda').",
    )
    return parser.parse_args()


def validate_dataset(data_yaml_path: Path) -> None:
    """Verify that the dataset YAML file exists.

    Parameters
    ----------
    data_yaml_path : Path
        Expected path to the YOLO dataset descriptor.

    Raises
    ------
    FileNotFoundError
        If the YAML file is missing; includes instructions for the user.
    """
    if not data_yaml_path.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found at '{data_yaml_path}'.\n"
            "Please run 'python download_dataset.py' first to fetch the "
            "Aquarium Combined dataset."
        )


def train_single_model(
    variant: str,
    *,
    epochs: int,
    batch: int,
    imgsz: int,
    device: str,
    data_yaml: Path,
) -> Dict[str, Any]:
    """Train a single YOLOv8 model variant.

    Parameters
    ----------
    variant : str
        Model key — ``'n'`` (nano) or ``'s'`` (small).
    epochs : int
        Number of training epochs.
    batch : int
        Batch size.
    imgsz : int
        Training image size in pixels.
    device : str
        Compute device identifier (e.g. ``'cpu'``, ``'0'``).
    data_yaml : Path
        Path to the YOLO dataset YAML.

    Returns
    -------
    dict
        A summary dictionary containing key metrics and the weights path.
    """
    cfg = MODEL_CONFIGS[variant]
    logger.info("=" * 60)
    logger.info("Starting training: %s", cfg["description"])
    logger.info("  Pretrained : %s", cfg["pretrained"])
    logger.info("  Epochs     : %d", epochs)
    logger.info("  Batch      : %d", batch)
    logger.info("  Image size : %d", imgsz)
    logger.info("  Device     : %s", device)
    logger.info("  Seed       : %d", cfg["seed"])
    logger.info("  Augments   : %s", cfg["augment"])
    logger.info("=" * 60)

    start_time = time.time()

    # Load the pre-trained model
    model = YOLO(cfg["pretrained"])

    # Build the training keyword arguments
    train_kwargs: Dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "device": device,
        "patience": config.YOLO_TRAIN_PATIENCE,
        "project": cfg["project"],
        "name": cfg["name"],
        "seed": cfg["seed"],
        "exist_ok": True,
        # Augmentation settings
        "mosaic": cfg["augment"]["mosaic"],
        "mixup": cfg["augment"]["mixup"],
        "copy_paste": cfg["augment"]["copy_paste"],
    }

    results = model.train(**train_kwargs)

    elapsed = time.time() - start_time
    logger.info(
        "Finished training %s in %.1f min.", cfg["description"], elapsed / 60
    )

    # ── Extract metrics ─────────────────────────────────────────────────
    metrics = results.results_dict
    summary: Dict[str, Any] = {
        "model": cfg["description"],
        "variant": variant,
        "pretrained": cfg["pretrained"],
        "precision": metrics.get("metrics/precision(B)", 0.0),
        "recall": metrics.get("metrics/recall(B)", 0.0),
        "mAP50": metrics.get("metrics/mAP50(B)", 0.0),
        "mAP50_95": metrics.get("metrics/mAP50-95(B)", 0.0),
        "elapsed_min": round(elapsed / 60, 2),
        "weights_dir": str(
            Path(cfg["project"]) / cfg["name"] / "weights"
        ),
    }

    return summary


def print_summary(summaries: list[Dict[str, Any]]) -> None:
    """Print a formatted training summary table.

    Parameters
    ----------
    summaries : list[dict]
        List of per-model summary dictionaries produced by
        :func:`train_single_model`.
    """
    print("\n" + "=" * 72)
    print("  TRAINING SUMMARY")
    print("=" * 72)
    header = (
        f"{'Model':<24} {'mAP50':>8} {'mAP50-95':>10} "
        f"{'Precision':>10} {'Recall':>8} {'Time':>8}"
    )
    print(header)
    print("-" * 72)
    for s in summaries:
        row = (
            f"{s['model']:<24} "
            f"{s['mAP50']:>8.4f} "
            f"{s['mAP50_95']:>10.4f} "
            f"{s['precision']:>10.4f} "
            f"{s['recall']:>8.4f} "
            f"{s['elapsed_min']:>6.1f}m"
        )
        print(row)
    print("=" * 72)

    print("\nTrained weights saved to:")
    for s in summaries:
        best_pt = Path(s["weights_dir"]) / "best.pt"
        print(f"  • {s['model']}: {best_pt}")
    print()


def main() -> None:
    """Entry-point: parse CLI args, validate dataset, train models, summarise."""
    args = parse_args()

    # ── Validate dataset ────────────────────────────────────────────────
    try:
        validate_dataset(config.DATA_YAML)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)

    logger.info("Dataset YAML found at: %s", config.DATA_YAML)

    # ── Determine which models to train ─────────────────────────────────
    if args.model == "both":
        variants = ["n", "s"]
    else:
        variants = [args.model]

    logger.info(
        "Will train %d model(s): %s",
        len(variants),
        ", ".join(MODEL_CONFIGS[v]["description"] for v in variants),
    )

    # ── Train ───────────────────────────────────────────────────────────
    summaries: list[Dict[str, Any]] = []
    for variant in variants:
        try:
            summary = train_single_model(
                variant,
                epochs=args.epochs,
                batch=args.batch,
                imgsz=args.imgsz,
                device=args.device,
                data_yaml=config.DATA_YAML,
            )
            summaries.append(summary)
        except Exception:
            logger.exception(
                "Training failed for %s",
                MODEL_CONFIGS[variant]["description"],
            )

    # ── Summary ─────────────────────────────────────────────────────────
    if summaries:
        print_summary(summaries)
    else:
        logger.error("No models were trained successfully.")
        sys.exit(1)


if __name__ == "__main__":
    main()
