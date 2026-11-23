"""Cheap probes for the image agent (week 0 → week 1).

A probe is a cheap extra look at what a pass found, used as state for the
image agent's decisions (D5 re-run on an enhanced image? D6 keep which pass?):

    per box    mirror and ±10% shift agreement of the classifier, top-1 − top-2
               margin, entropy, energy, the classifier's probability of the
               detector's class, ResNet-embedding distance to the class centroids
               (centroids from S_fit training crops), edge contact, overlap
    per image  colour statistics and the mean / min / max of the box probes

The same functions build the state when the policy is fitted
(experiments/log_image_agent.py) and when it runs in the loop, so the two
cannot drift apart. Probes only read the frozen models; they change nothing.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

import config
from agentic import tools

PROBE_KEYS = ("p_top", "margin", "entropy", "energy", "p_det_class", "flip_change", "flip_p_own",
              "shift_agree", "shift_p_own", "shift_p_min", "cent_dist_own", "cent_dist_other",
              "cent_gap", "edge", "max_iou_other")
SHIFTS = ((0.1, 0), (-0.1, 0), (0, 0.1), (0, -0.1))


# ── Image transforms the agent may re-run with ───────────────────────
def clahe(img: np.ndarray, clip: float) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_, a, b = cv2.split(lab)
    l_ = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(l_)
    return cv2.cvtColor(cv2.merge((l_, a, b)), cv2.COLOR_LAB2BGR)


ARMS = {"clahe2.0": lambda im: clahe(im, 2.0), "clahe3.0": lambda im: clahe(im, 3.0)}


# ── Classifier with full outputs ─────────────────────────────────────
class FullClassifier:
    """Probabilities, energy and the 512-d embedding for a batch of crops."""

    def __init__(self, clf: Any) -> None:
        self.clf = clf
        self._emb = None
        self._mean = torch.tensor(config.IMAGENET_MEAN, device=clf.device).view(1, 3, 1, 1)
        self._std = torch.tensor(config.IMAGENET_STD, device=clf.device).view(1, 3, 1, 1)
        clf.model.avgpool.register_forward_hook(lambda m, i, o: setattr(self, "_emb", o))

    def __call__(self, crops: list[np.ndarray]):
        if not crops:
            return np.zeros((0, len(self.clf.class_names))), np.zeros(0), np.zeros((0, 512))
        # cv2 resize + normalisation on the device: ~10× cheaper than the PIL
        # transform crop by crop. Values differ slightly from the classifier's own
        # input path, which is fine: probes are only state, and fitting and the
        # live loop both use this function.
        s = config.CLASSIFIER_IMG_SIZE
        batch = np.stack([cv2.resize(c, (s, s), interpolation=cv2.INTER_LINEAR)[:, :, ::-1] for c in crops])
        t = torch.from_numpy(np.ascontiguousarray(batch)).to(self.clf.device).permute(0, 3, 1, 2).float() / 255.0
        t = (t - self._mean) / self._std
        with torch.no_grad():
            logits = self.clf.model(t)
        probs = torch.softmax(logits, 1).cpu().numpy()
        energy = (-torch.logsumexp(logits, 1)).cpu().numpy()
        return probs, energy, self._emb.flatten(1).cpu().numpy()


def prober(pipeline: Any) -> tuple[FullClassifier, np.ndarray]:
    """The pipeline's probing classifier and class centroids, built once."""
    cached = getattr(pipeline, "_agent_prober", None)
    if cached is None:
        path = Path(getattr(config, "AGENT_CENTROIDS", ""))
        if not path.is_file():
            raise FileNotFoundError(f"class centroids not found: {path}")
        cents = np.load(path)
        cents = cents["c"] if hasattr(cents, "files") else cents
        cached = (FullClassifier(pipeline.classifier), np.asarray(cents, dtype=np.float64))
        pipeline._agent_prober = cached
    return cached


def _shifted(bbox, dx, dy):
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return [bbox[0] + dx * w, bbox[1] + dy * h, bbox[2] + dx * w, bbox[3] + dy * h]


def box_probes(img: np.ndarray, dets: list[dict[str, Any]], pipeline: Any) -> tuple[list[dict], float]:
    """Probe dict per box (one batched classifier call) and its cost in ms."""
    from pipeline import EnsemblePipeline
    fc, cents = prober(pipeline)
    names = list(pipeline.classifier.class_names)
    t = time.perf_counter()
    crops = []
    for d in dets:
        c0 = EnsemblePipeline._extract_crop(img, d["bbox"], padding=0.05)
        crops += [c0, c0[:, ::-1]]
        for dx, dy in SHIFTS:
            c = EnsemblePipeline._extract_crop(img, _shifted(d["bbox"], dx, dy), padding=0.05)
            crops.append(c if c.size else c0)       # a shift off the frame falls back to the box
    probs, energy, emb = fc(crops)
    h, w = img.shape[:2]
    out = []
    for k, d in enumerate(dets):
        p, e = probs[6 * k: 6 * k + 6], energy[6 * k: 6 * k + 6]
        top = int(p[0].argmax())
        srt = np.sort(p[0])[::-1]
        e0 = emb[6 * k] / max(np.linalg.norm(emb[6 * k]), 1e-9)
        cos = cents @ e0
        other = np.delete(cos, top)
        b = d["bbox"]
        det_class = d.get("detector_class", d.get("class_name"))
        di = names.index(det_class) if det_class in names else None
        ious = [tools.calculate_iou(b, o["bbox"]) for j, o in enumerate(dets) if j != k]
        out.append({
            "p_top": float(srt[0]), "margin": float(srt[0] - srt[1]),
            "entropy": float(-(p[0] * np.log(np.clip(p[0], 1e-12, 1))).sum()),
            "energy": float(e[0]), "p_det_class": float(p[0][di]) if di is not None else 0.0,
            "flip_change": float(int(p[1].argmax()) != top), "flip_p_own": float(p[1][top]),
            "shift_agree": float(np.mean([int(p[j].argmax()) == top for j in range(2, 6)])),
            "shift_p_own": float(p[2:6, top].mean()), "shift_p_min": float(p[2:6, top].min()),
            "cent_dist_own": float(1 - cos[top]), "cent_dist_other": float(1 - other.max()),
            "cent_gap": float(cos[top] - other.max()),
            "edge": float(b[0] <= 2 or b[1] <= 2 or b[2] >= w - 2 or b[3] >= h - 2),
            "max_iou_other": float(max(ious, default=0.0)),
        })
    return out, (time.perf_counter() - t) * 1000.0


def colour_stats(img: np.ndarray) -> dict[str, float]:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    return {"lab_a": float(lab[:, 1].mean() - 128), "lab_b": float(lab[:, 2].mean() - 128),
            "sat": float(hsv[:, 1].mean() / 255.0), "sat_std": float(hsv[:, 1].std() / 255.0)}


def pass_state(img: np.ndarray, finals: list[dict[str, Any]], pipeline: Any,
               refused: bool) -> tuple[list[float], float]:
    """Image-level state of one finished pass (week-0 layout) and its cost in ms.

    *finals* are the pass's final boxes (after vote, calibration and
    Known/Unknown); *img* is the image the pass ran on."""
    t = time.perf_counter()
    c = colour_stats(img)
    v = [c["lab_a"] / 20, c["lab_b"] / 20, c["sat"], c["sat_std"], math.log1p(len(finals))]
    ms = 0.0
    if finals and not refused:
        probes, ms = box_probes(img, finals, pipeline)
        P = np.array([[pr[k] for k in PROBE_KEYS] for pr in probes])
        conf = [d["combined_confidence"] for d in finals]
        v += list(P.mean(0)) + list(P.min(0)) + list(P.max(0))
        v += [float(np.mean(conf)), float(np.min(conf)),
              float(np.mean([d["detector_class"] != d["classifier_class"] for d in finals])),
              float(np.mean([bool((d.get("novelty") or {}).get("unknown")) for d in finals])), 1.0]
    else:
        v += [0.0] * (3 * len(PROBE_KEYS) + 4) + [0.0]
    v.append(float(refused))
    return v, (time.perf_counter() - t) * 1000.0
