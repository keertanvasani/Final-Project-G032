"""Ablation of the deployed deep-sea pipeline, one component at a time.

Runs the 9-class deep-sea stack (the app's `deepsea` profile) over the S_policy
validation split and adds each agent component in turn:

    A  detector + classifier only         (no enhancement, no filter, no gate)
    B  + D1 conditional enhancement       (CLAHE / sharpening when the image needs it)
    C  + telemetry-text filter            (drop boxes drawn on burned-in overlay text)
    D  + D2 refuse gate                   (unreadable image, or no box >= 0.40)
    E  + D3 zoom and D4 three-opinion vote
    F  + Known/Unknown flag               (energy > threshold is shown as Unknown, not accepted)

Every detection the configuration emits is scored against the JEDI ground
truth with the harness taxonomy (TP / CLS_ERR / OOD_ERR / LOC_ERR / BG_FP),
using the 9 trained classes as the label space. S_policy hosts the held-out
taxa jellyfish, squid and tunicate, so OOD_ERR counts unseen animals that were
reported as a known class.

Precision and recall come with 95% dive-cluster bootstrap intervals. The
learned enhancement predictor and colour correction are not part of this
pipeline and are not measured here. S_test is never touched (PROTOCOL §8).

Usage:
    python experiments/ablation_deepsea.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import config  # noqa: E402
import ensemble  # noqa: E402
import text_overlay  # noqa: E402
from agent_controller import (assess_image_quality, enhance_image,  # noqa: E402
                              rerun_zoomed_detection, resolve_detection)
from eval import harness  # noqa: E402
from pipeline import EnsemblePipeline  # noqa: E402

MAN = ROOT / "experiments" / "splits" / "phase_b_machine_splits.json"
OUT = ROOT / "experiments" / "results" / "phase_b_machine" / "ablation_S_policy.json"
SPLIT = "S_policy"
DIVE_RE = re.compile(r"^((?:[A-Z]+|\d+K)\d{3,4})")

CONFIGS = [
    ("A", "Detector + classifier", dict()),
    ("B", "+ enhancement (D1)", dict(enh=True)),
    ("C", "+ text filter", dict(enh=True, text=True)),
    ("D", "+ refuse gate (D2)", dict(enh=True, text=True, gate=True)),
    ("E", "+ zoom and vote (D3, D4)", dict(enh=True, text=True, gate=True, zoom=True, vote=True)),
    ("F", "+ Known/Unknown flag", dict(enh=True, text=True, gate=True, zoom=True, vote=True, novelty=True)),
]


def image_path(fname: str) -> Path | None:
    for sp in harness.SOURCE_SPLITS:
        p = harness.DATASET / sp / "images" / fname
        if p.is_file():
            return p
    return None


def run_config(img, pipe, f):
    """Return (detections, refused) for one image under flags f."""
    q = assess_image_quality(img)
    work = img
    if f.get("enh") and (q["needs_enhancement"] or q["is_blurry"]):
        work, _ = enhance_image(img, q)
    if f.get("gate") and q["blur_score"] < config.UNUSABLE_BLUR_THRESHOLD:
        return [], True
    dets = ensemble.run_detection(work, pipe)
    if f.get("text"):
        dets, _ = text_overlay.filter_text_detections(work, dets)
    if f.get("gate") and not any(d["confidence"] >= config.REVIEW_CONF_THRESHOLD for d in dets):
        return [], True
    if not dets:
        return [], False
    if f.get("zoom"):
        dets, _ = rerun_zoomed_detection(work, dets, pipe)
    crops = [pipe._extract_crop(work, d["bbox"], padding=0.05) for d in dets]
    cls = pipe.classifier.classify_crops(crops)
    out = []
    for i, (d, c) in enumerate(zip(dets, cls)):
        if f.get("vote"):
            r, _, _ = resolve_detection(i, d, c, pipe, work)
        else:
            agree = d["class_name"] == c["class_name"]
            comb = config.DETECTOR_WEIGHT * d["confidence"] + config.CLASSIFIER_WEIGHT * c["confidence"]
            if agree:
                comb = min(comb * (1 + config.AGREEMENT_BOOST), 1.0)
            r = dict(d, id=i + 1, detector_class=d["class_name"], detector_confidence=d["confidence"],
                     classifier_class=c["class_name"], classifier_confidence=c["confidence"],
                     combined_confidence=comb, agreement=agree)
        nov = pipe.novelty_for(c.get("energy"))
        r["unknown"] = bool(nov and nov["unknown"])
        if f.get("novelty") and r["unknown"]:
            continue          # shown as Unknown in the app, not reported as a species
        out.append(r)
    return out, False


def summarise(recs, kgt):
    n = len(recs)
    tp = sum(r["outcome"] == harness.TP for r in recs)
    recalled = {(r["file"], r["matched_gt"]) for r in recs if r["outcome"] == harness.TP}
    return {"precision": tp / n if n else 0.0, "recall": len(recalled) / kgt if kgt else 0.0}


def bootstrap(by_dive, kgt_by_dive, key, n_boot=1000, seed=0):
    rng = random.Random(seed)
    dives = list(by_dive)
    vals = []
    for _ in range(n_boot):
        recs, kgt = [], 0
        for k in range(len(dives)):
            d = dives[rng.randrange(len(dives))]
            recs += [dict(r, file=f"{k}|{r['file']}") for r in by_dive[d]]
            kgt += kgt_by_dive[d]
        vals.append(summarise(recs, kgt)[key])
    vals.sort()
    return [vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot) - 1]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    man = json.loads(MAN.read_text())
    known = set(man["label_space"]["known"])
    harness.KNOWN_CLASSES = known        # score against the 9 trained classes
    files = sorted(man["files"][SPLIT])[: args.limit]
    gt_all = harness.load_ground_truth()

    pipe = EnsemblePipeline(profile="deepsea")
    if pipe.novelty is None:
        raise SystemExit("models/phase_b_machine/novelty.json missing - run fit_novelty.py first")

    kgt_by_dive: Counter = Counter()
    for fn in files:
        kgt_by_dive[DIVE_RE.match(fn).group(1)] += harness.known_gt_count(gt_all.get(fn, []))
    kgt = sum(kgt_by_dive.values())

    results = {}
    t0 = time.time()
    for key, label, flags in CONFIGS:
        by_dive = defaultdict(list)
        refused = 0
        for fn in files:
            p = image_path(fn)
            img = cv2.imread(str(p)) if p else None
            if img is None:
                continue
            dets, ref = run_config(img, pipe, flags)
            refused += ref
            for r in harness.match_and_classify(dets, gt_all.get(fn, [])):
                r["file"] = fn
                by_dive[DIVE_RE.match(fn).group(1)].append(r)
        for d in kgt_by_dive:
            by_dive.setdefault(d, [])
        recs = [r for rs in by_dive.values() for r in rs]
        oc = Counter(r["outcome"] for r in recs)
        s = summarise(recs, kgt)
        results[key] = {
            "label": label, "flags": flags, "images": len(files), "refused_images": refused,
            "detections": len(recs), "outcomes": dict(oc),
            "precision": s["precision"], "precision_ci": bootstrap(by_dive, kgt_by_dive, "precision"),
            "recall": s["recall"], "recall_ci": bootstrap(by_dive, kgt_by_dive, "recall"),
        }
        r = results[key]
        print(f"{key} {label:28s} dets {r['detections']:5d} | P {r['precision']:.3f} "
              f"[{r['precision_ci'][0]:.3f},{r['precision_ci'][1]:.3f}] | R {r['recall']:.3f} "
              f"[{r['recall_ci'][0]:.3f},{r['recall_ci'][1]:.3f}] | OOD_ERR {oc.get('OOD_ERR', 0):3d} "
              f"BG_FP {oc.get('BG_FP', 0):4d} | refused {refused:3d}  ({time.time() - t0:.0f}s)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "split": SPLIT, "profile": "deepsea", "known_classes": sorted(known),
        "known_gt": kgt, "dives": len(kgt_by_dive),
        "novelty_threshold": pipe.novelty["threshold"], "configs": results}, indent=2) + "\n")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
