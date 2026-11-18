"""Does training on `machine` stop hardware being reported as an animal?

The visible failure of the Phase B detector is ROV hardware boxed and named as an
organism. This measures that failure directly, for any number of detectors, on
the dive-disjoint calibration split `S_cal`. The test split is never touched.

For every ground-truth `machine` box, the detector's highest-confidence
prediction overlapping it at IoU >= 0.5 is classified as one of

    machine   the detector says "hardware"                (correct)
    organism  the detector names it as an animal          (the failure)
    missed    no prediction overlaps it

A model that cannot predict `machine` (the 8-taxa Phase B detectors) can only
ever be `organism` or `missed`, which is exactly the comparison of interest.

Also reported: mAP@0.50 over the eight organism classes, so any gain on hardware
can be checked against a loss on animals, and the per-class AP of `machine`.

Usage:
    python experiments/eval_machine_effect.py \
        v1_n0=models/phase_b/yolov8n_seed0/weights/best.pt \
        v2_n0=models/phase_b_machine/yolov8n_seed0/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).parent
sys.path.insert(0, str(EXP))
from coco_to_yolo import load_coco  # noqa: E402

MAN_V2 = EXP / "splits" / "phase_b_machine_splits.json"
YOLO_V2 = EXP / "yolo_phase_b_machine"
YOLO_V1 = EXP / "yolo_phase_b"
OUT = EXP / "results" / "phase_b_machine" / "detector_eval.json"
SPLIT = "S_cal"
HW = "machine"


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between boxes a (n,4) and b (m,4), xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def val_yaml(model_names: dict, cal_list: Path, tmp: Path) -> Path:
    lines = [f"path: {cal_list.parent.as_posix()}", f"train: {cal_list.name}",
             f"val: {cal_list.name}", "names:"]
    lines += [f"  {i}: {n}" for i, n in sorted(model_names.items())]
    tmp.write_text("\n".join(lines) + "\n")
    return tmp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="name=path/to/best.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-match", type=float, default=0.5)
    ap.add_argument("--device", default="0")
    ap.add_argument("--split", default=SPLIT)
    args = ap.parse_args()

    from ultralytics import YOLO

    man = json.loads(MAN_V2.read_text())
    files = man["files"][args.split]
    dims, boxes = load_coco()
    # Image paths from the self-contained v2 tree (same pixels as the originals).
    split_dir = {"S_cal": "cal", "S_policy": "policy", "S_modelsel": "val",
                 "S_test": "test"}[args.split]
    paths = [str(YOLO_V2 / "images" / split_dir / f) for f in files]

    gt_hw = {f: np.array([[x, y, x + w, y + h] for c, x, y, w, h in boxes.get(f, []) if c == HW],
                         dtype=float).reshape(-1, 4) for f in files}
    n_gt = sum(len(v) for v in gt_hw.values())
    print(f"{args.split}: {len(files)} images, {n_gt} ground-truth {HW} boxes\n")

    results: dict = {"split": args.split, "n_images": len(files), "n_machine_gt": n_gt,
                     "conf": args.conf, "iou_match": args.iou_match, "models": {}}

    for spec in args.models:
        name, path = spec.split("=", 1)
        model = YOLO(path)
        names = model.names
        has_hw = HW in names.values()

        outcome: Counter = Counter()
        organism_on_hw = n_org_total = 0
        by_class: Counter = Counter()
        for i in range(0, len(paths), 32):
            chunk = paths[i:i + 32]
            for p, r in zip(chunk, model.predict(chunk, conf=args.conf, iou=0.45, imgsz=640,
                                                 device=args.device, verbose=False, stream=False)):
                f = Path(p).name
                xyxy = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
                conf = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros(0)
                cls = [names[int(c)] for c in r.boxes.cls.cpu().numpy()] if len(r.boxes) else []
                n_org_total += sum(1 for c in cls if c != HW)
                gt = gt_hw[f]
                m = iou_matrix(gt, xyxy)
                for gi in range(len(gt)):
                    hits = np.where(m[gi] >= args.iou_match)[0]
                    if len(hits) == 0:
                        outcome["missed"] += 1
                        continue
                    best = hits[np.argmax(conf[hits])]
                    if cls[best] == HW:
                        outcome["machine"] += 1
                    else:
                        outcome["organism"] += 1
                        by_class[cls[best]] += 1
                if len(gt) and len(xyxy):
                    on_hw = (m >= args.iou_match).any(axis=0)
                    organism_on_hw += sum(1 for k, c in enumerate(cls) if c != HW and on_hw[k])

        # mAP over the organism classes, on the same split.
        yaml = val_yaml(names, (YOLO_V2 if has_hw else YOLO_V1) / f"{split_dir}.txt",
                        ROOT / "models" / f"_val_{name}.yaml")
        v = model.val(data=str(yaml), split="val", imgsz=640, batch=16, workers=0,
                      device=args.device, verbose=False, plots=False, conf=0.001)
        ap50 = {names[int(k)]: float(a) for k, a in zip(v.box.ap_class_index, v.box.ap50)}
        org = {k: a for k, a in ap50.items() if k != HW}
        yaml.unlink(missing_ok=True)

        tot = sum(outcome.values())
        results["models"][name] = {
            "weights": path, "can_predict_machine": has_hw,
            "machine_outcome": {k: outcome[k] for k in ("machine", "organism", "missed")},
            "machine_outcome_rate": {k: round(outcome[k] / max(tot, 1), 4)
                                     for k in ("machine", "organism", "missed")},
            "organism_labels_on_hardware": dict(by_class.most_common()),
            "organism_predictions_total": n_org_total,
            "organism_predictions_on_hardware": organism_on_hw,
            "organism_mAP50": round(float(np.mean(list(org.values()))), 4),
            "machine_AP50": round(ap50.get(HW, float("nan")), 4) if has_hw else None,
            "per_class_AP50": {k: round(a, 4) for k, a in ap50.items()},
        }
        r = results["models"][name]
        print(f"{name}  ({'9-class' if has_hw else '8-class'})")
        print(f"  hardware boxes -> machine {r['machine_outcome_rate']['machine']:.1%} | "
              f"called an ORGANISM {r['machine_outcome_rate']['organism']:.1%} | "
              f"missed {r['machine_outcome_rate']['missed']:.1%}   (n={tot})")
        print(f"  organism predictions sitting on hardware: {organism_on_hw} of {n_org_total} "
              f"({organism_on_hw / max(n_org_total, 1):.1%})")
        print(f"  organism mAP@0.50 = {r['organism_mAP50']:.4f}"
              + (f"   machine AP50 = {r['machine_AP50']:.4f}" if has_hw else ""))
        if by_class:
            print("  hardware named as: " + ", ".join(f"{k} {n}" for k, n in by_class.most_common(4)))
        print()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
