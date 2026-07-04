#!/usr/bin/env python3
"""
prepare_classifier_data.py
===========================
Extract cropped bounding-box regions from the YOLO detection dataset and
organise them into a per-class directory tree suitable for training an image
classifier (e.g. ``torchvision.datasets.ImageFolder``).

Usage
-----
    python prepare_classifier_data.py              # use config defaults
    python prepare_classifier_data.py --padding 0.15
    python prepare_classifier_data.py -v            # verbose logging

Output layout::

    classifier_data/
    ├── train/
    │   ├── fish/
    │   │   ├── crop_0001.jpg
    │   │   └── …
    │   ├── jellyfish/
    │   └── …
    └── val/
        ├── fish/
        └── …
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ── project config ──────────────────────────────────────────────────────────
from config import (
    CLASS_NAMES,
    CLASSIFIER_DATA_DIR,
    CROP_PADDING,
    DATA_DIR,
)

# ── logging ─────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# YOLO label parsing
# ---------------------------------------------------------------------------

def parse_yolo_label(
    label_path: Path,
    img_width: int,
    img_height: int,
) -> List[Tuple[int, int, int, int, int]]:
    """Parse a YOLO-format label file into pixel-coordinate bounding boxes.

    Each line in a YOLO label file has the format::

        class_id  x_center  y_center  width  height

    where all values except ``class_id`` are normalised to [0, 1].

    Parameters
    ----------
    label_path : Path
        Path to the ``.txt`` label file.
    img_width : int
        Width of the corresponding image in pixels.
    img_height : int
        Height of the corresponding image in pixels.

    Returns
    -------
    list of tuple[int, int, int, int, int]
        Each tuple is ``(class_id, x1, y1, x2, y2)`` in pixel coordinates.
    """
    boxes: List[Tuple[int, int, int, int, int]] = []

    with open(label_path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 5:
                logger.warning(
                    "%s:%d — expected ≥5 fields, got %d; skipping.",
                    label_path.name,
                    line_no,
                    len(parts),
                )
                continue

            try:
                cls_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                logger.warning(
                    "%s:%d — could not parse numeric values; skipping.",
                    label_path.name,
                    line_no,
                )
                continue

            # Convert normalised coords → pixel coords
            x1 = int((x_center - w / 2) * img_width)
            y1 = int((y_center - h / 2) * img_height)
            x2 = int((x_center + w / 2) * img_width)
            y2 = int((y_center + h / 2) * img_height)

            boxes.append((cls_id, x1, y1, x2, y2))

    return boxes


# ---------------------------------------------------------------------------
# Crop extraction
# ---------------------------------------------------------------------------

def extract_crop(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    padding: float = CROP_PADDING,
) -> np.ndarray:
    """Extract a padded crop from *image*.

    Parameters
    ----------
    image : np.ndarray
        Source image (H×W×C, BGR).
    x1, y1, x2, y2 : int
        Bounding-box corners in pixel coordinates.
    padding : float
        Fractional padding to add around each side (e.g. 0.10 = 10 %).

    Returns
    -------
    np.ndarray
        Cropped region with padding, clamped to image boundaries.
    """
    img_h, img_w = image.shape[:2]

    box_w = x2 - x1
    box_h = y2 - y1

    pad_x = int(box_w * padding)
    pad_y = int(box_h * padding)

    # Apply padding and clamp to image boundaries
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(img_w, x2 + pad_x)
    cy2 = min(img_h, y2 + pad_y)

    return image[cy1:cy2, cx1:cx2]


def process_split(
    split: str,
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    class_names: List[str],
    padding: float,
) -> Counter[str]:
    """Process a single data split (train or val).

    For every image/label pair found, extract bounding-box crops and save
    them into per-class sub-directories.

    Parameters
    ----------
    split : str
        Split name (``"train"`` or ``"val"``).
    images_dir : Path
        Directory containing the split's images.
    labels_dir : Path
        Directory containing the split's YOLO label files.
    output_dir : Path
        Root output directory for this split (e.g. ``classifier_data/train``).
    class_names : list of str
        Ordered list of class names (index = class ID).
    padding : float
        Fractional padding around each crop.

    Returns
    -------
    Counter[str]
        Number of crops saved per class name.
    """
    crop_counts: Counter[str] = Counter()
    # Per-class running index for unique filenames
    class_indices: Counter[str] = Counter()

    # Collect image files
    image_extensions = {".jpg", ".jpeg", ".png"}
    image_files = sorted(
        f for f in images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    )

    if not image_files:
        logger.warning("[%s] No images found in %s", split, images_dir)
        return crop_counts

    logger.info("[%s] Found %d images in %s", split, len(image_files), images_dir)

    skipped_no_label = 0
    skipped_read_error = 0

    for img_path in image_files:
        # Find corresponding label file
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.is_file():
            logger.debug("[%s] No label for image %s — skipping.", split, img_path.name)
            skipped_no_label += 1
            continue

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning("[%s] Failed to read image %s — skipping.", split, img_path.name)
            skipped_read_error += 1
            continue

        img_h, img_w = image.shape[:2]

        # Parse labels
        boxes = parse_yolo_label(label_path, img_w, img_h)
        if not boxes:
            logger.debug("[%s] No valid boxes in %s", split, label_path.name)
            continue

        for cls_id, x1, y1, x2, y2 in boxes:
            # Resolve class name
            if cls_id < 0 or cls_id >= len(class_names):
                logger.warning(
                    "[%s] Unknown class_id %d in %s — skipping.",
                    split,
                    cls_id,
                    label_path.name,
                )
                continue

            cls_name = class_names[cls_id]

            # Extract crop
            crop = extract_crop(image, x1, y1, x2, y2, padding=padding)

            # Skip degenerate crops
            if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
                logger.debug(
                    "[%s] Degenerate crop (class=%s) in %s — skipping.",
                    split,
                    cls_name,
                    img_path.name,
                )
                continue

            # Save crop
            class_output_dir = output_dir / cls_name
            class_output_dir.mkdir(parents=True, exist_ok=True)

            class_indices[cls_name] += 1
            crop_filename = f"crop_{class_indices[cls_name]:05d}.jpg"
            crop_path = class_output_dir / crop_filename

            cv2.imwrite(str(crop_path), crop)
            crop_counts[cls_name] += 1

    if skipped_no_label:
        logger.info("[%s] Skipped %d images with no label file.", split, skipped_no_label)
    if skipped_read_error:
        logger.warning("[%s] Skipped %d images due to read errors.", split, skipped_read_error)

    return crop_counts


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(
    train_counts: Counter[str],
    val_counts: Counter[str],
    class_names: List[str],
) -> None:
    """Print a human-readable summary of extracted crops.

    Parameters
    ----------
    train_counts : Counter[str]
        Per-class crop counts for the training split.
    val_counts : Counter[str]
        Per-class crop counts for the validation split.
    class_names : list of str
        Ordered list of class names.
    """
    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values())

    print("\n" + "=" * 60)
    print("  CLASSIFIER DATA SUMMARY")
    print("=" * 60)
    print(f"  {'Class':<15s}  {'Train':>8s}  {'Val':>8s}  {'Total':>8s}")
    print("  " + "-" * 45)

    for name in class_names:
        t = train_counts.get(name, 0)
        v = val_counts.get(name, 0)
        print(f"  {name:<15s}  {t:>8d}  {v:>8d}  {t + v:>8d}")

    print("  " + "-" * 45)
    print(f"  {'TOTAL':<15s}  {total_train:>8d}  {total_val:>8d}  {total_train + total_val:>8d}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Extract cropped bounding-box regions from YOLO detection labels "
            "and organise them into a classification dataset."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=f"Root detection data directory (default: {DATA_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Output classifier data directory (default: {CLASSIFIER_DATA_DIR}).",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=None,
        help=f"Fractional padding around crops (default: {CROP_PADDING}).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the crop-extraction pipeline."""
    args = parse_args()

    # ── logging ─────────────────────────────────────────────────────────
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── resolve paths and parameters ────────────────────────────────────
    data_dir: Path = Path(args.data_dir).resolve() if args.data_dir else DATA_DIR
    output_dir: Path = Path(args.output_dir).resolve() if args.output_dir else CLASSIFIER_DATA_DIR
    padding: float = args.padding if args.padding is not None else CROP_PADDING
    class_names: List[str] = CLASS_NAMES

    logger.info("Data directory      : %s", data_dir)
    logger.info("Output directory    : %s", output_dir)
    logger.info("Crop padding        : %.0f%%", padding * 100)
    logger.info("Classes (%d)        : %s", len(class_names), class_names)

    # ── validate input directories ──────────────────────────────────────
    for split in ("train", "val"):
        img_dir = data_dir / "images" / split
        lbl_dir = data_dir / "labels" / split
        if not img_dir.is_dir():
            logger.error("Image directory does not exist: %s", img_dir)
            sys.exit(1)
        if not lbl_dir.is_dir():
            logger.error("Label directory does not exist: %s", lbl_dir)
            sys.exit(1)

    # ── process each split ──────────────────────────────────────────────
    train_counts = process_split(
        split="train",
        images_dir=data_dir / "images" / "train",
        labels_dir=data_dir / "labels" / "train",
        output_dir=output_dir / "train",
        class_names=class_names,
        padding=padding,
    )

    val_counts = process_split(
        split="val",
        images_dir=data_dir / "images" / "val",
        labels_dir=data_dir / "labels" / "val",
        output_dir=output_dir / "val",
        class_names=class_names,
        padding=padding,
    )

    # ── summary ─────────────────────────────────────────────────────────
    print_summary(train_counts, val_counts, class_names)

    total = sum(train_counts.values()) + sum(val_counts.values())
    if total == 0:
        logger.warning("No crops were extracted.  Check your data directory and labels.")
        sys.exit(1)

    logger.info("✅  Classifier data ready at %s", output_dir)


if __name__ == "__main__":
    main()
