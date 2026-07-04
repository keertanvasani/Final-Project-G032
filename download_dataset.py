#!/usr/bin/env python3
"""
download_dataset.py
====================
Download the Aquarium Combined dataset from Roboflow and organise it into
the directory layout expected by the YOLO training pipeline.

Usage examples
--------------
    # With an environment variable
    export ROBOFLOW_API_KEY="your_key_here"
    python download_dataset.py

    # With a CLI argument
    python download_dataset.py --api-key your_key_here

    # Print manual-download instructions (no API key required)
    python download_dataset.py --manual

Final directory layout produced::

    data/
    ├── images/
    │   ├── train/   (*.jpg)
    │   └── val/     (*.jpg)
    ├── labels/
    │   ├── train/   (*.txt)
    │   └── val/     (*.txt)
    └── data.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# ── project config ──────────────────────────────────────────────────────────
from config import (
    CLASS_NAMES,
    DATA_DIR,
    PROJECT_ROOT,
    ROBOFLOW_FORMAT,
    ROBOFLOW_PROJECT,
    ROBOFLOW_VERSION,
    ROBOFLOW_WORKSPACE,
)

# ── logging setup ───────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _count_files(directory: Path, extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")) -> int:
    """Return the number of files in *directory* that match *extensions*.

    Parameters
    ----------
    directory : Path
        Directory to scan.
    extensions : tuple[str, ...]
        Acceptable file-name suffixes (case-insensitive).

    Returns
    -------
    int
        File count.
    """
    if not directory.is_dir():
        return 0
    return sum(
        1
        for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    )


def _find_roboflow_download_dir(base_dir: Path) -> Optional[Path]:
    """Locate the directory created by the Roboflow SDK download.

    The SDK typically creates a folder named something like
    ``Aquarium-Combined-2/`` inside the current working directory.

    Parameters
    ----------
    base_dir : Path
        Directory in which to search.

    Returns
    -------
    Path or None
        The located download directory, or ``None`` if not found.
    """
    # Common pattern: <ProjectTitle>-<version>/
    pattern = re.compile(r"aquarium.combined", re.IGNORECASE)
    for child in sorted(base_dir.iterdir()):
        if child.is_dir() and pattern.search(child.name):
            return child
    return None


def _copy_tree(src: Path, dst: Path) -> int:
    """Recursively copy all files from *src* into *dst*.

    Parameters
    ----------
    src : Path
        Source directory.
    dst : Path
        Destination directory (created if it does not exist).

    Returns
    -------
    int
        Number of files copied.
    """
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dst / item.name)
            count += 1
    return count


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def download_dataset(api_key: str) -> Path:
    """Download the dataset from Roboflow and return the download directory.

    Parameters
    ----------
    api_key : str
        Roboflow API key.

    Returns
    -------
    Path
        Path to the downloaded dataset root (e.g. ``Aquarium-Combined-2/``).

    Raises
    ------
    ImportError
        If the ``roboflow`` package is not installed.
    RuntimeError
        If the download fails or the expected directory cannot be found.
    """
    try:
        from roboflow import Roboflow  # type: ignore[import-untyped]
    except ImportError as exc:
        logger.error(
            "The 'roboflow' package is required.  Install it with:\n"
            "    pip install roboflow"
        )
        raise ImportError("roboflow package not installed") from exc

    logger.info("Authenticating with Roboflow …")
    try:
        rf = Roboflow(api_key=api_key)
        workspace = rf.workspace(ROBOFLOW_WORKSPACE)
        project = workspace.project(ROBOFLOW_PROJECT)
        version = project.version(ROBOFLOW_VERSION)
    except Exception as exc:
        logger.error("Failed to connect to Roboflow: %s", exc)
        raise RuntimeError(f"Roboflow authentication/connection error: {exc}") from exc

    logger.info(
        "Downloading dataset  workspace=%s  project=%s  version=%d  format=%s",
        ROBOFLOW_WORKSPACE,
        ROBOFLOW_PROJECT,
        ROBOFLOW_VERSION,
        ROBOFLOW_FORMAT,
    )
    try:
        version.download(ROBOFLOW_FORMAT)
    except Exception as exc:
        logger.error("Dataset download failed: %s", exc)
        raise RuntimeError(f"Download error: {exc}") from exc

    download_dir = _find_roboflow_download_dir(PROJECT_ROOT)
    if download_dir is None:
        raise RuntimeError(
            "Could not locate the Roboflow download directory under "
            f"{PROJECT_ROOT}.  Expected a folder matching 'Aquarium-Combined-*'."
        )

    logger.info("Dataset downloaded to: %s", download_dir)
    return download_dir


def reorganize_dataset(download_dir: Path) -> None:
    """Move downloaded files into the canonical ``data/`` layout.

    The Roboflow download typically contains::

        <download_dir>/
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── valid/
        │   ├── images/
        │   └── labels/
        ├── test/
        │   ├── images/
        │   └── labels/
        └── data.yaml

    This function reorganises them into::

        data/
        ├── images/train/
        ├── images/val/
        ├── labels/train/
        ├── labels/val/
        └── data.yaml

    Parameters
    ----------
    download_dir : Path
        Root of the Roboflow download.
    """
    # Mapping: (roboflow_split_name, subfolder) → (target_split, kind)
    split_map: Dict[str, str] = {
        "train": "train",
        "valid": "val",
        "test": "val",  # merge test into val if present
    }

    for rf_split, target_split in split_map.items():
        for kind in ("images", "labels"):
            src = download_dir / rf_split / kind
            if not src.is_dir():
                logger.debug("Skipping missing source: %s", src)
                continue
            dst = DATA_DIR / kind / target_split
            n = _copy_tree(src, dst)
            logger.info("Copied %4d files  %s → %s", n, src, dst)

    # ── data.yaml ───────────────────────────────────────────────────────
    src_yaml = download_dir / "data.yaml"
    dst_yaml = DATA_DIR / "data.yaml"

    if src_yaml.is_file():
        _rewrite_data_yaml(src_yaml, dst_yaml)
    else:
        logger.warning("No data.yaml found in download – creating a minimal one.")
        _create_minimal_data_yaml(dst_yaml)

    # ── clean up the download directory ─────────────────────────────────
    logger.info("Removing original download directory: %s", download_dir)
    shutil.rmtree(download_dir, ignore_errors=True)


def _rewrite_data_yaml(src: Path, dst: Path) -> None:
    """Copy ``data.yaml`` and update train/val paths to absolute paths.

    Parameters
    ----------
    src : Path
        Source YAML file from the Roboflow download.
    dst : Path
        Destination path inside ``data/``.
    """
    with open(src, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    data["train"] = str(DATA_DIR / "images" / "train")
    data["val"] = str(DATA_DIR / "images" / "val")
    # Remove test key if present to avoid confusion
    data.pop("test", None)

    # Ensure class names are correct
    if "names" not in data:
        data["names"] = CLASS_NAMES
    data["nc"] = len(data["names"])

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False)

    logger.info("data.yaml written to %s with absolute paths.", dst)


def _create_minimal_data_yaml(dst: Path) -> None:
    """Create a minimal ``data.yaml`` when one is not provided.

    Parameters
    ----------
    dst : Path
        Path to write the YAML file.
    """
    data = {
        "train": str(DATA_DIR / "images" / "train"),
        "val": str(DATA_DIR / "images" / "val"),
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False)

    logger.info("Minimal data.yaml created at %s", dst)


# ---------------------------------------------------------------------------
# Validation & summary
# ---------------------------------------------------------------------------

def validate_dataset() -> bool:
    """Validate the reorganised dataset and print a summary.

    Checks that every image has a matching label file and vice-versa.

    Returns
    -------
    bool
        ``True`` if the dataset looks valid, ``False`` otherwise.
    """
    valid = True

    for split in ("train", "val"):
        img_dir = DATA_DIR / "images" / split
        lbl_dir = DATA_DIR / "labels" / split

        if not img_dir.is_dir():
            logger.error("Missing image directory: %s", img_dir)
            valid = False
            continue

        if not lbl_dir.is_dir():
            logger.error("Missing label directory: %s", lbl_dir)
            valid = False
            continue

        images: set[str] = {
            f.stem for f in img_dir.iterdir()
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")
        }
        labels: set[str] = {
            f.stem for f in lbl_dir.iterdir()
            if f.is_file() and f.suffix.lower() == ".txt"
        }

        imgs_without_labels = images - labels
        labels_without_imgs = labels - images

        if imgs_without_labels:
            logger.warning(
                "[%s] %d image(s) have no matching label file.",
                split,
                len(imgs_without_labels),
            )
        if labels_without_imgs:
            logger.warning(
                "[%s] %d label(s) have no matching image file.",
                split,
                len(labels_without_imgs),
            )

        matched = images & labels
        logger.info(
            "[%s]  images=%d  labels=%d  matched=%d",
            split,
            len(images),
            len(labels),
            len(matched),
        )

    # ── class distribution from labels ──────────────────────────────────
    _print_class_distribution()

    # ── data.yaml sanity check ──────────────────────────────────────────
    data_yaml = DATA_DIR / "data.yaml"
    if not data_yaml.is_file():
        logger.error("data.yaml not found at %s", data_yaml)
        valid = False
    else:
        with open(data_yaml, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        class_names: List[str] = cfg.get("names", [])
        logger.info("Classes in data.yaml (%d): %s", len(class_names), class_names)

    return valid


def _print_class_distribution() -> None:
    """Read all label files and print per-class annotation counts."""
    from collections import Counter

    totals: Counter[int] = Counter()

    for split in ("train", "val"):
        split_counter: Counter[int] = Counter()
        lbl_dir = DATA_DIR / "labels" / split
        if not lbl_dir.is_dir():
            continue
        for lbl_file in lbl_dir.iterdir():
            if not lbl_file.suffix == ".txt":
                continue
            with open(lbl_file, "r", encoding="utf-8") as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cls_id = int(parts[0])
                            split_counter[cls_id] += 1
                        except ValueError:
                            continue
        logger.info("[%s] annotation counts: %s", split, dict(split_counter))
        totals.update(split_counter)

    logger.info("Total annotation counts across splits:")
    for cls_id in sorted(totals):
        name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"unknown_{cls_id}"
        logger.info("  %2d  %-15s  %d", cls_id, name, totals[cls_id])


# ---------------------------------------------------------------------------
# Manual-download instructions
# ---------------------------------------------------------------------------

def print_manual_instructions() -> None:
    """Print step-by-step instructions for manually downloading the dataset."""
    instructions = f"""\
╔══════════════════════════════════════════════════════════════════╗
║           Manual Download Instructions                         ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                ║
║  1. Go to https://universe.roboflow.com/                       ║
║     Search for: "{ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT}"      ║
║                                                                ║
║  2. Select  Version {ROBOFLOW_VERSION}                         ║
║                                                                ║
║  3. Click  "Download Dataset" → choose YOLOv8 format           ║
║                                                                ║
║  4. Extract the ZIP into the project root:                     ║
║       {PROJECT_ROOT}                                           ║
║                                                                ║
║  5. Re-run this script without --manual to reorganise:         ║
║       python download_dataset.py --from-dir <extracted_folder> ║
║                                                                ║
║  6. Expected folder layout after extraction:                   ║
║       <folder>/train/images/  <folder>/train/labels/           ║
║       <folder>/valid/images/  <folder>/valid/labels/           ║
║       <folder>/test/images/   <folder>/test/labels/  (opt.)    ║
║       <folder>/data.yaml                                       ║
║                                                                ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(instructions)


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
        description="Download and organise the Aquarium Combined dataset for YOLO training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Roboflow API key.  Falls back to the ROBOFLOW_API_KEY env var.",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Print manual-download instructions and exit.",
    )
    parser.add_argument(
        "--from-dir",
        type=str,
        default=None,
        help=(
            "Path to an already-downloaded Roboflow dataset directory.  "
            "Skips the download step and only reorganises."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate an existing data/ directory; do not download.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the download-and-organise pipeline."""
    args = parse_args()

    # ── logging ─────────────────────────────────────────────────────────
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── manual mode ─────────────────────────────────────────────────────
    if args.manual:
        print_manual_instructions()
        sys.exit(0)

    # ── validate-only mode ──────────────────────────────────────────────
    if args.validate_only:
        ok = validate_dataset()
        sys.exit(0 if ok else 1)

    # ── determine download directory ────────────────────────────────────
    if args.from_dir:
        download_dir = Path(args.from_dir).resolve()
        if not download_dir.is_dir():
            logger.error("Provided --from-dir does not exist: %s", download_dir)
            sys.exit(1)
        logger.info("Using pre-downloaded directory: %s", download_dir)
    else:
        # Resolve API key
        api_key: Optional[str] = args.api_key or os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            logger.error(
                "No API key provided.  Set ROBOFLOW_API_KEY env var, use --api-key, "
                "or run with --manual for instructions."
            )
            sys.exit(1)

        download_dir = download_dataset(api_key)

    # ── reorganise ──────────────────────────────────────────────────────
    logger.info("Reorganising dataset into %s …", DATA_DIR)
    reorganize_dataset(download_dir)

    # ── validate ────────────────────────────────────────────────────────
    logger.info("Validating reorganised dataset …")
    ok = validate_dataset()

    if ok:
        logger.info("✅  Dataset ready at %s", DATA_DIR)
    else:
        logger.warning("⚠️  Dataset has issues – review warnings above.")

    # ── summary ─────────────────────────────────────────────────────────
    train_imgs = _count_files(DATA_DIR / "images" / "train")
    val_imgs = _count_files(DATA_DIR / "images" / "val")
    print("\n" + "=" * 50)
    print("  DOWNLOAD SUMMARY")
    print("=" * 50)
    print(f"  Train images : {train_imgs}")
    print(f"  Val images   : {val_imgs}")
    print(f"  Classes      : {len(CLASS_NAMES)}  {CLASS_NAMES}")
    print(f"  Data dir     : {DATA_DIR}")
    print(f"  data.yaml    : {DATA_DIR / 'data.yaml'}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
