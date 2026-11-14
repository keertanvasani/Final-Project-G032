"""Forced-action logging for RECLASSIFY_LARGER_CROP (report Phases 11 & 15).

The deployed rule fires when the detector and the classifier disagree about a
species. It re-classifies the region on a looser 15% crop and takes a majority
vote over three opinions — detector, classifier at 5%, classifier at 15% — with
a three-way deadlock defaulting to the detector.

Two structural facts narrow what has to be measured:

  * On **agreement** the action is provably inert. The vote is over
    [det, cls, loose]; if det == cls that is already a majority of two, so the
    loose opinion cannot change the outcome whatever it says. No measurement is
    needed to establish this.
  * The action never moves the box, so it can only ever flip TP <-> CLS_ERR.
    OOD_ERR and LOC_ERR are fixed by geometry and the ground-truth label.

So the question is narrow and answerable: **on disagreements, does buying a
third opinion beat simply trusting one of the first two?** Three policies are
scored against the same ground truth:

    rule        majority vote over the three opinions (what ships today)
    detector    always take the detector's class, never re-classify
    classifier  always take the 5% classifier's class, never re-classify

Only `rule` pays for the extra classification. If it does not beat the two free
policies, the compute is being spent for nothing — the same question
`log_zoom_outcomes.py` asked of ZOOM.

Discipline: dev splits only. Never J_test.

Usage:
    python experiments/log_reclassify_outcomes.py --split J_cal
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPLITS = ROOT / "experiments" / "splits" / "phase_a_splits.json"
OUT = ROOT / "experiments" / "results" / "actions"
IOU_MATCH = 0.50


def vote(det_class: str, cls_class: str, loose_class: str) -> str:
    """The deployed majority vote, reproduced exactly (agent_controller)."""
    counts: dict[str, int] = {}
    for op in (det_class, cls_class, loose_class):
        counts[op] = counts.get(op, 0) + 1
    top = max(counts, key=counts.get)
    return top if counts[top] >= 2 else det_class


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="J_cal")
    ap.add_argument("--platform", choices=["phase_a", "phase_b"], default="phase_a")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if "test" in args.split.lower():
        raise SystemExit("Refusing to touch a test split (PROTOCOL §8).")

    import cv2
    import config
    import ensemble
    from agent_controller import assess_image_quality, calculate_iou, enhance_image
    from experiments.eval import harness
    from experiments.run_phase_a import dive_of, image_path
    sys.path.insert(0, str(Path(__file__).parent))
    from platform_b import build_platform

    pipe, man, known = build_platform(args.platform)
    files = man["files"][args.split]
    if args.limit:
        files = files[: args.limit]
    gt_all = harness.load_ground_truth()

    OUT.mkdir(parents=True, exist_ok=True)
    tag = "" if args.platform == "phase_a" else "_phaseB"
    dest = OUT / f"{args.split}{tag}_reclassify_outcomes.jsonl"

    n_rows = n_dis = 0
    t0 = time.time()
    with dest.open("w") as fh:
        for i, fname in enumerate(files, 1):
            p = image_path(fname)
            if p is None:
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            q = assess_image_quality(img)
            enh = img
            if q["needs_enhancement"] or q["is_blurry"]:
                enh, _ = enhance_image(img, q)
            if q["blur_score"] < config.UNUSABLE_BLUR_THRESHOLD:
                continue

            ra, rb = ensemble.run_dual_detection(enh, pipe.model_a, pipe.model_b)
            dets = ensemble.fuse_detections(ra, rb, enh.shape[:2])
            if not dets:
                continue
            gts = gt_all.get(fname, [])

            crops = [pipe._extract_crop(enh, d["bbox"], padding=0.05) for d in dets]
            cls5 = pipe.classifier.classify_crops(crops)

            for di, (det, c5) in enumerate(zip(dets, cls5)):
                box = det["bbox"]
                det_class = det["class_name"]
                cls_class = c5["class_name"]
                agree = det_class == cls_class

                # match once — the action cannot move the box
                best, bi = 0.0, -1
                for gi, g in enumerate(gts):
                    v = calculate_iou(box, g["bbox"])
                    if v > best:
                        best, bi = v, gi
                matched = best >= IOU_MATCH and bi >= 0
                gt_cls = gts[bi]["class"] if matched else None
                in_label = matched and gt_cls in known

                # ---- FORCED: the looser crop, regardless of agreement ----
                t_a = time.perf_counter()
                loose = pipe.classifier.classify_crop(
                    pipe._extract_crop(enh, box, padding=0.15))
                cost_ms = (time.perf_counter() - t_a) * 1000.0
                loose_class = loose["class_name"]

                resolved = vote(det_class, cls_class, loose_class)

                def outcome(pred: str) -> str:
                    if not matched:
                        return "LOC_ERR" if best >= 0.10 else "BG_FP"
                    if not in_label:
                        return "OOD_ERR"
                    return "TP" if pred == gt_cls else "CLS_ERR"

                row = {
                    "file": fname, "dive": dive_of(fname), "det_index": di,
                    "det_class": det_class, "det_conf": float(det["confidence"]),
                    "cls_class": cls_class, "cls_conf": float(c5["confidence"]),
                    "loose_class": loose_class,
                    "loose_conf": float(loose["confidence"]),
                    "agreement": agree,
                    "rule_would_fire": not agree,
                    "loose_changed_vote": (not agree) and loose_class not in (det_class, cls_class),
                    "gt_class": gt_cls, "matched": matched, "in_label": in_label,
                    "iou": round(best, 4),
                    # three competing policies, same ground truth
                    "outcome_rule": outcome(resolved),
                    "outcome_detector_only": outcome(det_class),
                    "outcome_classifier_only": outcome(cls_class),
                    "resolved_class": resolved,
                    "cost_ms": round(cost_ms, 2),
                }
                fh.write(json.dumps(row) + "\n")
                n_rows += 1
                n_dis += 0 if agree else 1

            if i % 400 == 0:
                print(f"  {i}/{len(files)} images · {n_rows} logged "
                      f"· {(time.time()-t0)/60:.1f} min")

    print(f"\nLogged {n_rows} forced RECLASSIFY actions in "
          f"{(time.time()-t0)/60:.1f} min")
    print(f"  detector/classifier disagreements: {n_dis} "
          f"({100*n_dis/max(n_rows,1):.1f}%) — the only cases the rule fires on")
    print(f"Wrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
