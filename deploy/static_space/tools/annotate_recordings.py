"""Attach ground truth to the recorded deep-sea samples.

The eight deep-sea samples are unmodified frames from the JEDI dataset, so the
dataset's own labels say what is really in each one. This matches every recorded
detection to those labels with the evaluation harness's rules (IoU >= 0.5, class
checked after matching, label space of the aquarium baseline) and stores the
outcome next to the detection, so the demo page can say, for the boxes the
threshold slider is currently showing, how many are actually right.

It reuses `experiments/eval/harness.py` rather than re-implementing the rules, so
the demo's notion of "correct" cannot drift from the one the results use.

The aquarium stock samples have no ground truth and are left untouched.

Each frame is tied to its source by SHA-256 of the file bytes, so a sample that
was ever re-encoded or swapped fails loudly instead of being scored against the
wrong labels.

Usage (after record_demo.py, with the JEDI dataset on disk):
    python tools/annotate_recordings.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]            # deploy/static_space
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from eval.harness import (KNOWN_CLASSES, load_ground_truth,  # noqa: E402
                          match_and_classify)

DATASET = ROOT / "JediOrganismDetectionDataset"
SAMPLES = HERE / "assets" / "samples"
RECORDED = HERE / "recorded"

# Sample -> its file in the JEDI dataset (verified byte-for-byte below).
SOURCE = {
    "ds_01_hardware_as_puffin.jpg": "2K1317CCDDB1023_360.jpg",
    "ds_06_dark_telemetry_filtered.jpg": "3K0380CCDSV1011_480.jpg",
    "ds_07_crowded.jpg": "6K1562C2HDF107_100.jpg",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_source(fname: str) -> Path:
    for split in ("coco_train_data", "coco_valid_data", "coco_test_data"):
        p = DATASET / split / "images" / fname
        if p.is_file():
            return p
    raise SystemExit(f"{fname} not found under {DATASET}")


def main() -> None:
    if not DATASET.is_dir():
        raise SystemExit(f"JEDI dataset not found at {DATASET}")
    manifest = json.loads((RECORDED / "manifest.json").read_text())
    by_name = {v["name"]: k for k, v in manifest["images"].items()}
    gt_all = load_ground_truth()

    print(f"{'sample':<36}{'dets':>5}  outcomes")
    for name, src_name in SOURCE.items():
        sample = SAMPLES / name
        if sha(sample) != sha(find_source(src_name)):
            raise SystemExit(f"{name} is not byte-identical to {src_name}; refusing to score it")
        key = by_name.get(name)
        if key is None:
            raise SystemExit(f"{name} has no recording; run tools/record_demo.py first")

        rec_path = RECORDED / f"{key}.json"
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        gt = gt_all[src_name]
        outcomes = match_and_classify(rec["detections"], gt)

        for det, o in zip(rec["detections"], outcomes):
            det["gt"] = {"outcome": o["outcome"], "gt_class": o["gt_class"]}
        rec["ground_truth"] = {
            "source": src_name,
            "label_space": sorted(KNOWN_CLASSES),
            "boxes": [{"class": g["class"], "bbox": [round(v, 1) for v in g["bbox"]],
                       "in_label_space": g["class"] in KNOWN_CLASSES} for g in gt],
        }
        rec_path.write_text(json.dumps(rec), encoding="utf-8")
        c = Counter(o["outcome"] for o in outcomes)
        print(f"{name:<36}{len(outcomes):>5}  " +
              ", ".join(f"{k} {v}" for k, v in c.most_common()))

    print("\nground truth attached; the aquarium stock samples have none and are unchanged")


if __name__ == "__main__":
    main()
