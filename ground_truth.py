"""Ground truth for uploads that are frames from the JEDI dataset.

If an uploaded image is byte-for-byte a JEDI frame, the dataset's own labels say
whether each detection is right. This module finds such a frame and grades the
detections with the evaluation harness taxonomy (TP / CLS_ERR / OOD_ERR /
LOC_ERR / BG_FP), so the interface can colour boxes by correctness.

Matching is by exact bytes: the file size narrows the candidates, then SHA-256
confirms. A resized, re-encoded or cropped copy is not matched, and an image
from anywhere else simply gets no ground truth. Nothing is guessed.

The dataset is optional; without it the module returns None for everything.
"""

from __future__ import annotations

import hashlib
import logging
import sys
import threading
from pathlib import Path
from typing import Any

import config

logger = logging.getLogger(__name__)

DATASET = config.PROJECT_ROOT / "JediOrganismDetectionDataset"
_SPLITS = ("coco_train_data", "coco_valid_data", "coco_test_data")

# The aquarium models can only ever be right about these four JEDI taxa.
AQUARIUM_KNOWN = {"fish", "jellyfish", "shark", "starfish"}

_lock = threading.Lock()
_by_size: dict[int, list[Path]] | None = None
_gt: dict[str, list[dict[str, Any]]] | None = None


def _harness():
    exp = str(config.PROJECT_ROOT / "experiments")
    if exp not in sys.path:
        sys.path.insert(0, exp)
    from eval import harness
    return harness


def _index() -> dict[int, list[Path]]:
    global _by_size, _gt
    if _by_size is None:
        idx: dict[int, list[Path]] = {}
        for sp in _SPLITS:
            d = DATASET / sp / "images"
            if d.is_dir():
                for p in d.iterdir():
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        idx.setdefault(p.stat().st_size, []).append(p)
        _by_size = idx
        _gt = _harness().load_ground_truth() if idx else {}
        logger.info("Ground-truth index: %d dataset frames", sum(len(v) for v in idx.values()))
    return _by_size


def find_frame(contents: bytes) -> str | None:
    """Dataset filename of an upload that is exactly a JEDI frame, else None."""
    if not DATASET.is_dir():
        return None
    with _lock:
        cands = _index().get(len(contents), [])
    if not cands:
        return None
    h = hashlib.sha256(contents).hexdigest()
    for p in cands:
        if hashlib.sha256(p.read_bytes()).hexdigest() == h:
            return p.name
    return None


def grade(contents: bytes, detections: list[dict[str, Any]],
          known: set[str]) -> dict[str, Any] | None:
    """Attach ``gt`` to each detection and return the response's ``ground_truth``.

    ``known`` is the label space of the model that produced the detections.
    Returns None when the upload is not a dataset frame.
    """
    name = find_frame(contents)
    if name is None:
        return None
    h = _harness()
    gt = (_gt or {}).get(name, [])
    with _lock:
        saved = h.KNOWN_CLASSES
        h.KNOWN_CLASSES = set(known)
        try:
            outcomes = h.match_and_classify(detections, gt)
        finally:
            h.KNOWN_CLASSES = saved
    for det, o in zip(detections, outcomes):
        det["gt"] = {"outcome": o["outcome"], "gt_class": o["gt_class"]}
    return {
        "source": name,
        "label_space": sorted(known),
        "boxes": [{"class": g["class"], "bbox": [round(v, 1) for v in g["bbox"]],
                   "in_label_space": g["class"] in known} for g in gt],
    }
