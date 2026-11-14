"""Convert the Jedi COCO annotations to YOLO format on the Phase B splits.

Reads `splits/phase_b_splits.json` and emits, for every image, a YOLO label
file containing only the KNOWN taxa, remapped to contiguous class ids. Held-out
and far-OOD annotations are never written: they are not in the detector's label
space, and that is the whole point of the open-set design.

Two things this script is careful about.

**J_fit excludes images containing held-out taxa** (`--exclude-heldout-images`,
on by default). An unlabelled organism in a training image teaches the detector
that that taxon is background, so rejecting it at test time would be *learned
suppression* rather than novelty handling — which would falsify H3 while every
split remained nominally class-disjoint. Excluding those images costs ~3% of
known annotations (PROTOCOL.md §11.3).

**The early-stopping split is carved from J_fit, not J_cal.** Ultralytics needs
a validation set for model selection. Using J_cal would burn it: a calibration
set that chose the checkpoint is no longer naive to the model, and temperature
scaling fitted on it would be optimistically biased. So J_fit is split by dive
into `train` / `val`, and J_cal is not touched during training at all.

Images are not copied. Label files are written next to each source `images/`
directory as `labels/`, which is where Ultralytics looks, and each split is
described by a text file of image paths.

Usage:
    python experiments/coco_to_yolo.py
    python experiments/coco_to_yolo.py --val-frac 0.15 --no-exclude-heldout-images
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "JediOrganismDetectionDataset"
SOURCE_SPLITS = ["coco_train_data", "coco_valid_data", "coco_test_data"]
MANIFEST = Path(__file__).parent / "splits" / "phase_b_splits.json"

sys.path.insert(0, str(Path(__file__).parent))
from make_splits import dive_of  # noqa: E402


def load_coco() -> tuple[dict, dict]:
    """Return (filename -> (w, h), filename -> [(class_name, x, y, w, h)])."""
    dims: dict[str, tuple[int, int]] = {}
    boxes: dict[str, list] = defaultdict(list)
    for split in SOURCE_SPLITS:
        p = DATASET / split / "coco.json"
        if not p.is_file():
            raise SystemExit(f"Missing {p}")
        coco = json.loads(p.read_text())
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        imgs = {}
        for im in coco["images"]:
            fn = Path(im["file_name"]).name
            imgs[im["id"]] = fn
            dims[fn] = (im["width"], im["height"])
            boxes.setdefault(fn, [])
        for a in coco["annotations"]:
            fn = imgs.get(a["image_id"])
            if fn is None:
                continue
            x, y, w, h = a["bbox"]
            boxes[fn].append((cats[a["category_id"]], x, y, w, h))
    return dims, boxes


def image_path(fname: str) -> Path:
    for split in SOURCE_SPLITS:
        p = DATASET / split / "images" / fname
        if p.is_file():
            return p
    raise SystemExit(f"Image not found on disk: {fname}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "yolo_phase_b")
    ap.add_argument("--val-frac", type=float, default=0.10,
                    help="dive-disjoint share of J_fit held out for early stopping")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-heldout-images", dest="exclude", action="store_true",
                    default=True)
    ap.add_argument("--no-exclude-heldout-images", dest="exclude",
                    action="store_false")
    args = ap.parse_args()

    man = json.loads(args.manifest.read_text())
    known = sorted(man["label_space"]["known"])
    held = set(man["label_space"]["held_out_unseen"])
    far = set(man["label_space"]["far_ood"])
    cls_id = {c: i for i, c in enumerate(known)}

    dims, boxes = load_coco()

    # ---- drop images containing held-out taxa (PROTOCOL §11.3) ----------
    # Applied to both weight-fitting splits. S_modelsel chooses the checkpoint,
    # so a held-out taxon visible there would steer model selection toward
    # suppressing it just as surely as training on it would.
    def clean(split: str) -> list[str]:
        files = man["files"][split]
        if not args.exclude:
            return sorted(files)
        return sorted(f for f in files
                      if not any(c in held for c, *_ in boxes.get(f, [])))

    fit_train = clean("S_fit")
    fit_val = clean("S_modelsel")

    # S_modelsel is a first-class split from the manifest (PROTOCOL §4.2), not
    # a slice carved out of S_fit here. Early stopping therefore never touches
    # S_cal, which must stay naive for calibration (§11.5).
    splits = {
        "train": fit_train,
        "val": fit_val,
        "cal": man["files"]["S_cal"],
        "policy": man["files"]["S_policy"],
        "test": man["files"]["S_test"],
    }

    # ---- write label files ----------------------------------------------
    written = 0
    skipped_degenerate = 0
    label_counts: Counter = Counter()
    all_files = sorted({f for v in splits.values() for f in v})
    for fname in all_files:
        img = image_path(fname)
        lbl_dir = img.parent.parent / "labels"
        lbl_dir.mkdir(exist_ok=True)
        W, H = dims[fname]
        lines = []
        for cname, x, y, w, h in boxes.get(fname, []):
            if cname not in cls_id:
                continue  # held-out and far-OOD are never labelled
            if w <= 0 or h <= 0:
                skipped_degenerate += 1
                continue
            xc, yc = (x + w / 2) / W, (y + h / 2) / H
            nw, nh = w / W, h / H
            # Clamp: a few COCO boxes run a pixel past the image edge.
            xc, yc = min(max(xc, 0.0), 1.0), min(max(yc, 0.0), 1.0)
            nw, nh = min(nw, 1.0), min(nh, 1.0)
            lines.append(f"{cls_id[cname]} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
            label_counts[cname] += 1
        (lbl_dir / (Path(fname).stem + ".txt")).write_text("\n".join(lines) + "\n")
        written += 1

    # ---- split list files + data.yaml -----------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    for name, files in splits.items():
        (args.out / f"{name}.txt").write_text(
            "\n".join(str(image_path(f)) for f in files) + "\n"
        )

    yaml = [
        "# Phase B — on-domain retraining (generated by coco_to_yolo.py)",
        f"path: {args.out.resolve().as_posix()}",
        "train: train.txt",
        "val: val.txt",
        "",
        "names:",
    ]
    yaml += [f"  {i}: {c}" for i, c in enumerate(known)]
    (args.out / "data.yaml").write_text("\n".join(yaml) + "\n")

    # ---- report ----------------------------------------------------------
    fit = fit_train + fit_val
    fit_all = man["files"]["S_fit"] + man["files"]["S_modelsel"]
    fit_machine = sum(
        1 for f in fit if any(c in far for c, *_ in boxes.get(f, []))
    )
    print(f"KNOWN label space ({len(known)}): {', '.join(known)}\n")
    print(f"{'split':<9}{'images':>8}{'known anns':>12}")
    print("-" * 29)
    for name, files in splits.items():
        n = sum(1 for f in files for c, *_ in boxes.get(f, []) if c in cls_id)
        print(f"{name:<9}{len(files):>8}{n:>12}")
    print("-" * 29)
    print(f"\nS_fit + S_modelsel: {len(fit_all)} images -> {len(fit)} after "
          f"{'excluding' if args.exclude else 'keeping'} held-out-taxa images")
    print(f"  train {len(fit_train)} (S_fit) / val {len(fit_val)} (S_modelsel)")
    if set(dive_of(f) for f in fit_train) & set(dive_of(f) for f in fit_val):
        raise SystemExit("BUG: train/val share a dive")

    print(f"\nper-class training annotations:")
    for c in known:
        n = sum(1 for f in fit_train for cc, *_ in boxes.get(f, []) if cc == c)
        flag = "  <-- thin" if n < 100 else ""
        print(f"  {c:<22}{n:>6}{flag}")

    print(f"\nLabel files written: {written}")
    if skipped_degenerate:
        print(f"Degenerate boxes skipped (w or h <= 0): {skipped_degenerate}")
    print(f"J_fit images still containing far-OOD 'machine': {fit_machine} "
          f"({100*fit_machine/max(len(fit),1):.1f}%)")
    print("  These are unlabelled, so the detector learns ROV hardware as")
    print("  background. Far-OOD rejection is therefore partly trained, not")
    print("  purely novel — see PROTOCOL.md §11.4.")
    print(f"\nWrote {(args.out / 'data.yaml').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
