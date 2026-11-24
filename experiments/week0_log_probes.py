"""Week 0 (PROTOCOL §13): full-information logs of a larger action menu plus cheap
probes, on S_cal only.

For every S_cal image:

  base     the deployed pipeline (learned policy): no enhance, no zoom, vote on
           every disagreement; text filter, refuse gate, calibration, Known/Unknown.
  image-level actions, each re-run through the base pipeline:
           clahe1.5, clahe2.0, clahe3.0, gamma, greyworld, det960, tile2x2
  box-level actions on the base boxes (one box changed at a time):
           zoom (vote downstream), novote (plain scoring), fliptta
  probes   per box (mirror, shifts, margin, entropy, energy, centroid distances,
           geometry) and per image (colour statistics), with their cost.

Each final box is stored with the fields the harness needs, so rewards are
computed in week0_analyse.py. S_policy and S_test are refused.

Usage:
    python experiments/week0_log_probes.py [--limit N]
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import config  # noqa: E402
import ensemble  # noqa: E402
from ablation_deepsea import DIVE_RE, MAN, image_path  # noqa: E402
from agentic import tools, verifier  # noqa: E402
from ensemble_boxes import weighted_boxes_fusion  # noqa: E402
from pipeline import EnsemblePipeline  # noqa: E402

OUT = ROOT / "experiments" / "results" / "phase_b_machine" / "week0"
CROPS = ROOT / "experiments" / "classifier_crops_phase_b_machine" / "train"
SPLIT = "S_cal"
IMAGE_ACTIONS = ("clahe1.5", "clahe2.0", "clahe3.0", "gamma", "greyworld", "det960", "tile2x2")
KEEP = ("bbox", "confidence", "class_name")


# ── Classifier with full outputs ─────────────────────────────────────
class FullClassifier:
    """Probabilities, energy and the 512-d embedding for a batch of crops."""

    def __init__(self, clf):
        self.clf = clf
        self._emb = None
        clf.model.avgpool.register_forward_hook(lambda m, i, o: setattr(self, "_emb", o))

    def __call__(self, crops):
        if not crops:
            return np.zeros((0, len(self.clf.class_names))), np.zeros(0), np.zeros((0, 512))
        t = torch.stack([self.clf.transform(Image.fromarray(np.ascontiguousarray(c[:, :, ::-1])).convert("RGB"))
                         for c in crops]).to(self.clf.device)
        with torch.no_grad():
            logits = self.clf.model(t)
        probs = torch.softmax(logits, 1).cpu().numpy()
        energy = (-torch.logsumexp(logits, 1)).cpu().numpy()
        emb = self._emb.flatten(1).cpu().numpy()
        return probs, energy, emb


def centroids(fc, names, per_class=200):
    """Unit-norm mean embedding per class from S_fit training crops (never S_cal)."""
    cache = OUT / "centroids.npz"
    if cache.is_file():
        z = np.load(cache)
        return z["c"]
    man = json.loads(MAN.read_text())
    cal_dives = {DIVE_RE.match(f).group(1) for f in man["files"][SPLIT]}
    rng = random.Random(0)
    cs = []
    for name in names:
        files = sorted((CROPS / name.replace("/", "_")).glob("*.jpg"))
        files = [f for f in files if DIVE_RE.match(f.name).group(1) not in cal_dives]
        rng.shuffle(files)
        embs = []
        for k in range(0, min(per_class, len(files)), 64):
            crops = [cv2.imread(str(f)) for f in files[k:min(k + 64, per_class)]]
            embs.append(fc([c for c in crops if c is not None])[2])
        e = np.concatenate(embs)
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        m = e.mean(0)
        cs.append(m / np.linalg.norm(m))
    c = np.stack(cs)
    np.savez(cache, c=c)
    return c


# ── Image-level transforms ───────────────────────────────────────────
def clahe(img, clip):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_, a, b = cv2.split(lab)
    l_ = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(l_)
    return cv2.cvtColor(cv2.merge((l_, a, b)), cv2.COLOR_LAB2BGR)


def gamma(img):
    m = max(float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()) / 255.0, 1e-3)
    g = float(np.clip(np.log(0.5) / np.log(m), 0.5, 2.0)) if 0 < m < 1 else 1.0
    lut = (np.clip((np.arange(256) / 255.0) ** g, 0, 1) * 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def greyworld(img):
    f = img.astype(np.float32)
    mean = f.reshape(-1, 3).mean(0)
    f *= mean.mean() / np.maximum(mean, 1e-3)
    return np.clip(f, 0, 255).astype(np.uint8)


TRANSFORMS = {"clahe1.5": lambda im: clahe(im, 1.5), "clahe2.0": lambda im: clahe(im, 2.0),
              "clahe3.0": lambda im: clahe(im, 3.0), "gamma": gamma, "greyworld": greyworld}


def to_dets(boxes, scores, labels, shape, names):
    h, w = shape[:2]
    return [{"bbox": [float(b[0] * w), float(b[1] * h), float(b[2] * w), float(b[3] * h)],
             "class_id": int(c), "confidence": float(s),
             "class_name": names[int(c)] if 0 <= int(c) < len(names) else f"class_{int(c)}"}
            for b, s, c in zip(boxes, scores, labels)]


def detect_960(img, pipe):
    res = pipe.model_a.predict(source=img, conf=config.YOLO_CONF_THRESHOLD, iou=config.YOLO_IOU_THRESHOLD,
                               imgsz=960, verbose=False)
    return ensemble.fuse_detections(res, None, img.shape[:2], class_names=pipe.class_names,
                                    weights=getattr(pipe, "wbf_weights", None))


def detect_tiles(img, pipe, overlap=0.25):
    """Full image + 2×2 overlapping tiles, one pooled box list merged by WBF.

    Tile boxes touching an interior tile cut are dropped (they are cut-off parts)."""
    h, w = img.shape[:2]
    tw, th = int(round(w * (1 + overlap) / 2)), int(round(h * (1 + overlap) / 2))
    pred = lambda im: pipe.model_a.predict(source=im, conf=config.YOLO_CONF_THRESHOLD,
                                           iou=config.YOLO_IOU_THRESHOLD, verbose=False)
    B, S, L = ensemble.extract_predictions(pred(img), img.shape[:2])
    B, S, L = list(B), list(S), list(L)
    for x0 in (0, w - tw):
        for y0 in (0, h - th):
            tile = img[y0:y0 + th, x0:x0 + tw]
            tb, ts, tl = ensemble.extract_predictions(pred(tile), tile.shape[:2])
            for b, s, c in zip(tb, ts, tl):
                px = [x0 + b[0] * tw, y0 + b[1] * th, x0 + b[2] * tw, y0 + b[3] * th]
                cut = ((x0 > 0 and px[0] <= x0 + 2) or (x0 + tw < w and px[2] >= x0 + tw - 2)
                       or (y0 > 0 and px[1] <= y0 + 2) or (y0 + th < h and px[3] >= y0 + th - 2))
                if not cut:
                    B.append([px[0] / w, px[1] / h, px[2] / w, px[3] / h])
                    S.append(s)
                    L.append(c)
    if not S:
        return []
    fb, fs, fl = weighted_boxes_fusion([B], [S], [L], weights=[1], iou_thr=config.WBF_IOU_THRESHOLD,
                                       skip_box_thr=config.WBF_SKIP_BOX_THRESHOLD,
                                       conf_type=config.WBF_CONF_TYPE)
    return to_dets(fb, fs, fl, img.shape, pipe.class_names)


# ── Base pipeline ────────────────────────────────────────────────────
def final_fields(d):
    nov = d.get("novelty") or {}
    return {"bbox": [float(v) for v in d["bbox"]], "classifier_class": d["classifier_class"],
            "detector_class": d.get("detector_class", d["class_name"]),
            "combined_confidence": d["combined_confidence"],
            "raw_combined_confidence": d.get("raw_combined_confidence", d["combined_confidence"]),
            "detector_confidence": d["detector_confidence"],
            "classifier_confidence": d["classifier_confidence"],
            "agreement": d["agreement"], "unknown": bool(nov.get("unknown")),
            "energy": nov.get("score")}


def finish_box(i, det, c, work, pipe, vote=True):
    det = dict(det)
    if vote and det["class_name"] != c["class_name"]:
        d, _, _ = tools.resolve_detection(i, det, c, pipe, work)
    else:
        d = tools.score_plain(i, det, c)
    verifier.finalise(i, d, c, pipe)
    return final_fields(d)


def base_run(work, q, pipe, dets=None):
    """Base pipeline from detection to final boxes. Returns (dets, cls, finals, refused)."""
    if dets is None:
        dets = ensemble.run_detection(work, pipe)
    dets, _ = verifier.text_filter(work, dets)
    ok, _ = verifier.refuse_gate(q, dets)
    if not ok:
        return dets, [], [], True
    cls = tools.classify_boxes(work, dets, pipe)
    return dets, cls, [finish_box(i, d, c, work, pipe) for i, (d, c) in enumerate(zip(dets, cls))], False


# ── Probes ───────────────────────────────────────────────────────────
def shifted(bbox, dx, dy):
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return [bbox[0] + dx * w, bbox[1] + dy * h, bbox[2] + dx * w, bbox[3] + dy * h]


def box_probes(img, dets, fc, cents, names):
    """Mirror, shift, distribution and embedding probes for every box, one batch."""
    crops = []
    for d in dets:
        c0 = EnsemblePipeline._extract_crop(img, d["bbox"], padding=0.05)
        crops += [c0, c0[:, ::-1]]
        for dx, dy in ((0.1, 0), (-0.1, 0), (0, 0.1), (0, -0.1)):
            crops.append(EnsemblePipeline._extract_crop(img, shifted(d["bbox"], dx, dy), padding=0.05))
    t = time.perf_counter()
    probs, energy, emb = fc(crops)
    ms = (time.perf_counter() - t) * 1000.0
    h, w = img.shape[:2]
    out = []
    for k, d in enumerate(dets):
        p, e = probs[6 * k: 6 * k + 6], energy[6 * k: 6 * k + 6]
        top = int(p[0].argmax())
        srt = np.sort(p[0])[::-1]
        emb0 = emb[6 * k] / max(np.linalg.norm(emb[6 * k]), 1e-9)
        cos = cents @ emb0
        other = np.delete(cos, top)
        b = d["bbox"]
        ious = [tools.calculate_iou(b, o["bbox"]) for j, o in enumerate(dets) if j != k]
        det_idx = names.index(d["class_name"]) if d["class_name"] in names else None
        out.append({
            "p_top": float(srt[0]), "margin": float(srt[0] - srt[1]),
            "entropy": float(-(p[0] * np.log(np.clip(p[0], 1e-12, 1))).sum()),
            "energy": float(e[0]),
            "p_det_class": float(p[0][det_idx]) if det_idx is not None else 0.0,
            "flip_change": float(int(p[1].argmax()) != top), "flip_p_own": float(p[1][top]),
            "shift_agree": float(np.mean([int(p[j].argmax()) == top for j in range(2, 6)])),
            "shift_p_own": float(p[2:6, top].mean()), "shift_p_min": float(p[2:6, top].min()),
            "cent_dist_own": float(1 - cos[top]), "cent_dist_other": float(1 - other.max()),
            "cent_gap": float(cos[top] - other.max()),
            "edge": float(b[0] <= 2 or b[1] <= 2 or b[2] >= w - 2 or b[3] >= h - 2),
            "max_iou_other": float(max(ious, default=0.0)),
            "flip_probs": ((p[0] + p[1]) / 2).tolist(), "flip_energy": float((e[0] + e[1]) / 2),
        })
    return out, ms


def colour_stats(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    return {"lab_a": float(lab[:, 1].mean() - 128), "lab_b": float(lab[:, 2].mean() - 128),
            "sat": float(hsv[:, 1].mean() / 255.0), "sat_std": float(hsv[:, 1].std() / 255.0)}


# ── Main ─────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", default=SPLIT)
    args = ap.parse_args()
    if args.split != SPLIT:
        raise SystemExit("Week 0 reads S_cal only (PROTOCOL §13)")

    man = json.loads(MAN.read_text())
    files = sorted(man["files"][SPLIT])[: args.limit]
    pipe = EnsemblePipeline(profile="deepsea")
    names = list(pipe.classifier.class_names)
    OUT.mkdir(parents=True, exist_ok=True)
    fc = FullClassifier(pipe.classifier)
    cents = centroids(fc, names)
    dest = OUT / f"{SPLIT}_week0.jsonl"
    t0, n_boxes = time.time(), 0
    with dest.open("w", encoding="utf-8") as fh:
        for k, fn in enumerate(files):
            p = image_path(fn)
            img = cv2.imread(str(p)) if p else None
            if img is None:
                continue
            q = tools.assess_image_quality(img)
            rec = {"file": fn, "dive": DIVE_RE.match(fn).group(1),
                   "quality": {k2: (list(v) if isinstance(v, tuple) else v) for k2, v in q.items()},
                   "colour": colour_stats(img), "actions": {}}

            t = time.perf_counter()
            dets, cls, finals, refused = base_run(img, q, pipe)
            rec["base"] = {"ms": (time.perf_counter() - t) * 1000.0, "refused": refused,
                           "n_dets": len(dets), "finals": finals}

            # Image-level actions.
            for a in IMAGE_ACTIONS:
                t = time.perf_counter()
                if a in TRANSFORMS:
                    work = TRANSFORMS[a](img)
                    _, _, fa, ra = base_run(work, q, pipe)
                elif a == "det960":
                    _, _, fa, ra = base_run(img, q, pipe, detect_960(img, pipe))
                else:
                    _, _, fa, ra = base_run(img, q, pipe, detect_tiles(img, pipe))
                rec["actions"][a] = {"ms": (time.perf_counter() - t) * 1000.0, "refused": ra, "finals": fa}

            # Box-level actions and probes.
            boxes = []
            if not refused:
                probes, probe_ms = box_probes(img, dets, fc, cents, names)
                for i, (d, c) in enumerate(zip(dets, cls)):
                    t = time.perf_counter()
                    zd, _ = tools.zoom_boxes(img, copy.deepcopy(dets), pipe, {i})
                    zc = tools.classify_boxes(img, [zd[i]], pipe)[0]
                    zoom_final = finish_box(i, zd[i], zc, img, pipe)
                    zoom_ms = (time.perf_counter() - t) * 1000.0
                    novote = finish_box(i, d, c, img, pipe, vote=False)
                    pr = probes[i]
                    fp = np.array(pr.pop("flip_probs"))
                    top = int(fp.argmax())
                    fcls = {"class_name": names[top], "confidence": float(fp[top]),
                            "energy": pr.pop("flip_energy")}
                    t = time.perf_counter()
                    flip_final = finish_box(i, d, fcls, img, pipe)
                    flip_ms = (time.perf_counter() - t) * 1000.0 + probe_ms / (3 * max(len(dets), 1))
                    boxes.append({"det": {k2: d[k2] for k2 in KEEP},
                                  "cls": {"class_name": c["class_name"], "confidence": c["confidence"]},
                                  "base": finals[i], "zoom": zoom_final, "zoom_ms": zoom_ms,
                                  "zoom_changed": zd[i]["confidence"] != d["confidence"],
                                  "novote": novote, "fliptta": flip_final, "fliptta_ms": flip_ms,
                                  "disagree": d["class_name"] != c["class_name"], "probes": pr})
                rec["probe_ms"] = probe_ms
            rec["boxes"] = boxes
            n_boxes += len(boxes)
            fh.write(json.dumps(rec) + "\n")
            if (k + 1) % 100 == 0:
                print(f"{k + 1}/{len(files)} images, {n_boxes} boxes ({time.time() - t0:.0f}s)", flush=True)
    print(f"wrote {dest.relative_to(ROOT)}: {len(files)} images, {n_boxes} boxes ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
