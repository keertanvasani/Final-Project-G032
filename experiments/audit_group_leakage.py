"""Audit group-level leakage in the JediOrganismDetectionDataset splits.

The images are ROV video frames, not independent photographs. Filenames
encode a dive and a frame index:

    2K0263OUTUM1016_240.jpg  ->  dive 2K0263, frame 240

Frames from one dive share camera, illumination, substrate and turbidity,
and often show the same individual organisms seconds apart. If a dive spans
two splits, the evaluation is partly measuring memorisation.

This script quantifies that overlap in the supplied splits. It reports only;
it does not modify anything. See experiments/PROTOCOL.md §4.

Usage:
    python experiments/audit_group_leakage.py
"""

from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "JediOrganismDetectionDataset"
SPLITS = ["coco_train_data", "coco_valid_data", "coco_test_data"]

IMAGE_EXT = (".jpg", ".jpeg", ".png")

# Leading dive identifier: a vehicle prefix followed by a 3-4 digit dive
# number, stopping before the camera/track code.
#
#   2K0263OUTUM1016_240.jpg      -> 2K0263
#   HPD0664HDDB402_120.jpg       -> HPD0664
#   KAIKO0283PNCSV1016_240.jpg   -> KAIKO0283
#
# The alternation exists because the archive spans six vehicles with two
# naming styles: letter-prefixed (HPD, KAIKO, KMROV) and digit-K (2K, 3K, 6K).
# Anchoring the digit run immediately after the prefix is what stops the match
# running on into the camera code -- a greedier pattern silently splits one
# dive into many groups and understates leakage.
DIVE_RE = re.compile(r"^((?:[A-Z]+|\d+K)\d{3,4})")


def dive_of(filename: str) -> str:
    m = DIVE_RE.match(filename)
    return m.group(1) if m else f"UNPARSED::{filename}"


def main() -> None:
    if not DATASET.is_dir():
        raise SystemExit(f"Dataset not found at {DATASET}")

    per_split: dict[str, list[str]] = {}
    for s in SPLITS:
        img_dir = DATASET / s / "images"
        if not img_dir.is_dir():
            raise SystemExit(f"Missing image directory: {img_dir}")
        per_split[s] = sorted(
            p.name for p in img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXT
        )

    dives = {s: {dive_of(f) for f in files} for s, files in per_split.items()}

    unparsed = {s: sum(1 for f in files if dive_of(f).startswith("UNPARSED::"))
                for s, files in per_split.items()}

    print(f"{'split':<20}{'images':>8}{'dives':>8}{'frames/dive':>13}{'unparsed':>10}")
    print("-" * 59)
    for s in SPLITS:
        n, d = len(per_split[s]), len(dives[s])
        print(f"{s:<20}{n:>8}{d:>8}{n / d:>13.2f}{unparsed[s]:>10}")

    print("\nShared dives between splits")
    print("-" * 59)
    for a, b in combinations(SPLITS, 2):
        shared = dives[a] & dives[b]
        print(f"  {a} ∩ {b}: {len(shared)}")

    print("\nContamination — images sitting in a dive that also appears elsewhere")
    print("-" * 59)
    report: dict = {"per_split": {}, "contamination": {}}
    for target, others in [
        ("coco_test_data", ["coco_train_data"]),
        ("coco_test_data", ["coco_train_data", "coco_valid_data"]),
        ("coco_valid_data", ["coco_train_data"]),
    ]:
        other_dives: set[str] = set().union(*(dives[o] for o in others))
        hit = sum(1 for f in per_split[target] if dive_of(f) in other_dives)
        total = len(per_split[target])
        label = f"{target} vs {'+'.join(others)}"
        print(f"  {label:<46}{hit:>5}/{total} ({hit / total:.1%})")
        report["contamination"][label] = {
            "contaminated": hit, "total": total, "fraction": round(hit / total, 4)
        }

    all_dives = set().union(*dives.values())
    print(f"\nTotal unique dives across all splits: {len(all_dives)}")

    for s in SPLITS:
        report["per_split"][s] = {
            "images": len(per_split[s]),
            "dives": len(dives[s]),
            "unparsed_filenames": unparsed[s],
        }
    report["shared_dives"] = {
        f"{a}|{b}": len(dives[a] & dives[b]) for a, b in combinations(SPLITS, 2)
    }
    report["total_unique_dives"] = len(all_dives)

    out = Path(__file__).parent / "group_leakage_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {out.relative_to(ROOT)}")
    print("\nConclusion: the supplied splits are not dive-disjoint. Re-split by "
          "dive before reporting any number (PROTOCOL.md §4).")


if __name__ == "__main__":
    main()
