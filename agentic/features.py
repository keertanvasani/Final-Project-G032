"""State features for the learned policy (step 3).

One function per decision point. The same code builds the features when the
policy is fitted (experiments/fit_bandit_policy.py) and when it runs in the
loop, so the two cannot drift apart. Every feature is something the loop has
observed *before* the decision is made.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def image_features(q: dict[str, Any]) -> list[float]:
    """D1 (enhance?): image quality only — decided before detection."""
    return [
        math.log1p(q["blur_score"]),
        q["brightness"] / 255.0,
        q["contrast"] / 128.0,
        float(q["is_dark"]),
        float(q["is_bright"]),
        float(q["is_low_contrast"]),
        float(q["is_blurry"]),
        float(q["needs_enhancement"]),
    ]


def _onehot(name: str, classes: list[str]) -> list[float]:
    return [1.0 if name == c else 0.0 for c in classes]


def _geometry(bbox: list[float], q: dict[str, Any]) -> list[float]:
    w, h = q["resolution"]
    bw, bh = max(bbox[2] - bbox[0], 1.0), max(bbox[3] - bbox[1], 1.0)
    return [math.log(bw * bh / float(w * h)), math.log(bw / bh)]


def zoom_features(det: dict[str, Any], q: dict[str, Any], n_dets: int, enhanced: bool,
                  classes: list[str]) -> list[float]:
    """D3 (zoom this box?): the fused box before any re-check."""
    c = det["confidence"]
    return ([c, c * c, float(0.4 <= c < 0.75)] + _geometry(det["bbox"], q)
            + [math.log1p(n_dets), float(enhanced)] + image_features(q)
            + _onehot(det["class_name"], classes))


def vote_features(det: dict[str, Any], cls_res: dict[str, Any], q: dict[str, Any],
                  n_dets: int, enhanced: bool, classes: list[str]) -> list[float]:
    """D4 (vote on this disagreement?): the box after zoom, plus the classifier's view."""
    return (zoom_features(det, q, n_dets, enhanced, classes)
            + [cls_res["confidence"]] + _onehot(cls_res["class_name"], classes))


def as_matrix(rows: list[list[float]]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float64).reshape(len(rows), -1)
