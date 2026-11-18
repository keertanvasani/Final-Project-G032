"""Full-information action logs for the step-3 contextual bandit.

The models are frozen, so every action can be tried on every image and box:
there is nothing to explore, and the logs give the outcome of every policy
without bias (no propensities needed).

For each image, for each enhancement choice e (both when the quality check
flags the image, else only e = 0 because enhancement would be a no-op):

    detect + fuse → text filter → refuse gate                 (fixed)
    for every box:
        z = 0  the box as fused       z = 1  zoomed re-check of this box alone
        for each z: classify the 5% crop; plain scoring, and — when detector and
        classifier disagree — the three-opinion vote; then calibration and
        Known/Unknown.

Each final box is stored with everything needed to rebuild any policy's
output and score it with the harness. Zoom acts on one box at a time and the
vote on one box, so per-box variants compose exactly into a whole-image run
(up to batch-vs-single float noise in the classifier). Costs are measured.

Dev splits only, except the single registered confirmatory run on S_test
(PROTOCOL §12, --confirmatory).

Usage:
    python experiments/log_bandit_outcomes.py --split S_cal
    python experiments/log_bandit_outcomes.py --split S_policy
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import ensemble  # noqa: E402
from ablation_deepsea import DIVE_RE, MAN, image_path  # noqa: E402
from agentic import tools, verifier  # noqa: E402
from pipeline import EnsemblePipeline  # noqa: E402

OUT = ROOT / "experiments" / "results" / "phase_b_machine" / "step3"
KEEP = ("bbox", "confidence", "class_name")


def final_fields(d: dict) -> dict:
    nov = d.get("novelty") or {}
    return {"bbox": [float(v) for v in d["bbox"]],
            "classifier_class": d["classifier_class"],
            "combined_confidence": d["combined_confidence"],
            "raw_combined_confidence": d.get("raw_combined_confidence", d["combined_confidence"]),
            "detector_confidence": d["detector_confidence"],
            "classifier_confidence": d["classifier_confidence"],
            "agreement": d["agreement"], "unknown": bool(nov.get("unknown"))}


def box_variants(work, dets, pipe):
    """All (zoom, vote) outcomes for every box of one detection pass."""
    n = len(dets)
    zoomed, zoom_ms = [], []
    for i in range(n):
        one = copy.deepcopy(dets)
        t = time.perf_counter()
        out, _ = tools.zoom_boxes(work, one, pipe, {i})
        zoom_ms.append((time.perf_counter() - t) * 1000.0)
        zoomed.append(out[i])
    rows = []
    cls_by_z = {0: tools.classify_boxes(work, dets, pipe),
                1: tools.classify_boxes(work, zoomed, pipe)}
    for i in range(n):
        row = {"det": {k: dets[i][k] for k in KEEP}, "zoomed": {k: zoomed[i][k] for k in KEEP},
               "zoom_changed": zoomed[i]["confidence"] != dets[i]["confidence"],
               "zoom_ms": zoom_ms[i]}
        for z, src in ((0, dets[i]), (1, zoomed[i])):
            c = cls_by_z[z][i]
            plain = tools.score_plain(i, dict(src), c)
            verifier.finalise(i, plain, c, pipe)
            v = {"cls": {"class_name": c["class_name"], "confidence": c["confidence"]},
                 "plain": final_fields(plain), "vote": None, "vote_ms": 0.0}
            if src["class_name"] != c["class_name"]:
                voted, _, ms = tools.resolve_detection(i, dict(src), c, pipe, work)
                verifier.finalise(i, voted, c, pipe)
                v["vote"], v["vote_ms"] = final_fields(voted), ms
            row[f"z{z}"] = v
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["S_cal", "S_policy", "S_test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--confirmatory", action="store_true",
                    help="required for S_test: the single registered run (PROTOCOL §12)")
    args = ap.parse_args()
    if args.split == "S_test" and not (args.confirmatory and args.limit is None):
        raise SystemExit("S_test is single-use: only the full --confirmatory run (PROTOCOL §12)")

    man = json.loads(MAN.read_text())
    files = sorted(man["files"][args.split])[: args.limit]
    pipe = EnsemblePipeline(profile="deepsea")
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{args.split}_actions.jsonl"
    t0 = time.time()
    n_boxes = 0
    with dest.open("w", encoding="utf-8") as fh:
        for k, fn in enumerate(files):
            p = image_path(fn)
            img = cv2.imread(str(p)) if p else None
            if img is None:
                continue
            q = tools.assess_image_quality(img)
            flagged = bool(q["needs_enhancement"] or q["is_blurry"])
            rec = {"file": fn, "dive": DIVE_RE.match(fn).group(1), "flagged": flagged,
                   "quality": {k2: (list(v) if isinstance(v, tuple) else v) for k2, v in q.items()},
                   "variants": {}}
            for e in ((0, 1) if flagged else (0,)):
                t = time.perf_counter()
                work = tools.enhance_image(img, q)[0] if e else img
                enh_ms = (time.perf_counter() - t) * 1000.0 if e else 0.0
                t = time.perf_counter()
                dets = ensemble.run_detection(work, pipe)
                det_ms = (time.perf_counter() - t) * 1000.0
                dets, _ = verifier.text_filter(work, dets)
                ok, _ = verifier.refuse_gate(q, dets)
                var = {"enh_ms": enh_ms, "det_ms": det_ms, "refused": not ok, "n_dets": len(dets),
                       "boxes": box_variants(work, dets, pipe) if ok else []}
                n_boxes += len(var["boxes"])
                rec["variants"][str(e)] = var
            fh.write(json.dumps(rec) + "\n")
            if (k + 1) % 100 == 0:
                print(f"{args.split} {k + 1}/{len(files)} images, {n_boxes} boxes "
                      f"({time.time() - t0:.0f}s)", flush=True)
    print(f"wrote {dest.relative_to(ROOT)}: {len(files)} images, {n_boxes} boxes "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
