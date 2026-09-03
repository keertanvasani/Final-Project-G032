"""Build dive-disjoint Phase A splits for the Jedi dataset.

Phase A evaluates the frozen aquarium baseline on deep-sea imagery. No
training happens on Jedi, so the supplied train/valid/test division carries
no meaning here and is discarded — it is also contaminated (57.3% of its
test images share a dive with training; see audit_group_leakage.py). All
8,151 images are pooled and re-split.

Splits are assigned by *dive*, never by image. Frames from one dive share
camera, illumination, substrate and often the same individual organisms
seconds apart, so an image-level split leaks.

    J_cal     calibration + meta-confidence fitting   (hosts OOD_dev)
    J_policy  exploration rollouts + policy learning  (hosts OOD_val)
    J_test    final evaluation, single use            (hosts OOD_test)

The OOD class partition is applied *within* these image splits, so the OOD
scorer can neither tune on a taxon it is later tested on, nor tune on a dive
it is later tested in.

See experiments/PROTOCOL.md §4 and §5.

Usage:
    python experiments/make_splits.py
    python experiments/make_splits.py --seed 7 --cal 0.35 --policy 0.35 --test 0.30
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "JediOrganismDetectionDataset"
SOURCE_SPLITS = ["coco_train_data", "coco_valid_data", "coco_test_data"]
IMAGE_EXT = (".jpg", ".jpeg", ".png")

# Validated against all six vehicle families (2K, 3K, 6K, HPD, KAIKO, KMROV).
# Anchoring the digit run immediately after the prefix is what stops the match
# running into the camera code; see audit_group_leakage.py.
DIVE_RE = re.compile(r"^((?:[A-Z]+|\d+K)\d{3,4})")

# PROTOCOL.md §2.4 — the aquarium model's label space, mapped onto Jedi.
# `ray` is deliberately absent: decision 9.3 treats it as OOD, since Batoidea
# spans skates and guitarfish and a deep-sea ray is unlikely to be the
# aquarium stingray the classifier learned.
KNOWN_CLASSES = {"fish", "jellyfish", "shark", "starfish"}

# PROTOCOL.md §5 — near-OOD partitioned by class, descending frequency,
# round-robin. Each role is hosted by exactly one image split.
OOD_ROLE_CLASSES = {
    "OOD_dev":  {"coral/sea-anemone", "sea-cucumber", "squat-lobster", "squid", "ray"},
    "OOD_val":  {"crab", "crinoid", "octopus", "urchin", "tunicate"},
    "OOD_test": {"shrimp", "sponge", "comb-jelly", "sea-spider", "ragworm"},
}
FAR_OOD_CLASSES = {"machine"}          # evaluated at test only, never tuned on

SPLIT_HOSTS_OOD = {"J_cal": "OOD_dev", "J_policy": "OOD_val", "J_test": "OOD_test"}


def dive_of(filename: str) -> str:
    m = DIVE_RE.match(filename)
    if not m:
        raise ValueError(f"Filename does not match the dive pattern: {filename}")
    return m.group(1)


def load_dataset() -> tuple[dict[str, list[str]], list[str]]:
    """Return (filename -> class names of its annotations, all image filenames)."""
    ann_by_file: dict[str, list[str]] = defaultdict(list)
    all_files: list[str] = []
    seen: dict[str, str] = {}

    for split in SOURCE_SPLITS:
        coco_path = DATASET / split / "coco.json"
        img_dir = DATASET / split / "images"
        if not coco_path.is_file():
            raise SystemExit(f"Missing {coco_path}")

        coco = json.loads(coco_path.read_text())
        id2name = {c["id"]: c["name"] for c in coco["categories"]}
        # Image ids are only unique within a file, so key on filename.
        imgid2file = {im["id"]: im["file_name"] for im in coco["images"]}

        on_disk = {p.name for p in img_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXT}

        for fname in sorted(on_disk):
            if fname in seen:
                raise SystemExit(
                    f"Duplicate filename across supplied splits: {fname} "
                    f"({seen[fname]} and {split}). Pooling would double-count it."
                )
            seen[fname] = split
            all_files.append(fname)
            ann_by_file.setdefault(fname, [])

        for a in coco["annotations"]:
            fname = imgid2file.get(a["image_id"])
            if fname in on_disk:
                ann_by_file[fname].append(id2name[a["category_id"]])

    return ann_by_file, all_files


def assign_dives(
    dive_sizes: dict[str, int], targets: dict[str, float], seed: int
) -> dict[str, str]:
    """Greedy bin-packing: largest dives first, each to the split most in deficit.

    Dives are indivisible, so exact proportions are unreachable; placing the
    largest groups first keeps the realised sizes close to target, and a
    seeded shuffle breaks size ties deterministically without biasing toward
    any particular dive ordering.
    """
    rng = random.Random(seed)
    dives = list(dive_sizes)
    rng.shuffle(dives)
    dives.sort(key=lambda d: dive_sizes[d], reverse=True)

    current = {s: 0 for s in targets}
    assignment: dict[str, str] = {}
    for dive in dives:
        split = max(targets, key=lambda s: targets[s] - current[s])
        assignment[dive] = split
        current[split] += dive_sizes[dive]
    return assignment


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cal", type=float, default=0.30)
    ap.add_argument("--policy", type=float, default=0.30)
    ap.add_argument("--test", type=float, default=0.40)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "splits" / "phase_a_splits.json")
    args = ap.parse_args()

    props = {"J_cal": args.cal, "J_policy": args.policy, "J_test": args.test}
    if abs(sum(props.values()) - 1.0) > 1e-6:
        raise SystemExit(f"Proportions must sum to 1.0, got {sum(props.values())}")

    ann_by_file, all_files = load_dataset()
    print(f"Pooled {len(all_files)} images, "
          f"{sum(len(v) for v in ann_by_file.values())} annotations "
          f"from {len(SOURCE_SPLITS)} supplied splits\n")

    files_by_dive: dict[str, list[str]] = defaultdict(list)
    for f in all_files:
        files_by_dive[dive_of(f)].append(f)
    dive_sizes = {d: len(fs) for d, fs in files_by_dive.items()}

    targets = {s: p * len(all_files) for s, p in props.items()}
    assignment = assign_dives(dive_sizes, targets, args.seed)

    split_files: dict[str, list[str]] = {s: [] for s in props}
    split_dives: dict[str, list[str]] = {s: [] for s in props}
    for dive, split in assignment.items():
        split_dives[split].append(dive)
        split_files[split].extend(files_by_dive[dive])
    for s in split_files:
        split_files[s].sort()
        split_dives[s].sort()

    # ---- verification -------------------------------------------------
    seen_dives: dict[str, str] = {}
    for s, dives in split_dives.items():
        for d in dives:
            if d in seen_dives:
                raise SystemExit(f"BUG: dive {d} in both {seen_dives[d]} and {s}")
            seen_dives[d] = s
    assigned = sum(len(v) for v in split_files.values())
    if assigned != len(all_files):
        raise SystemExit(f"BUG: assigned {assigned} of {len(all_files)} images")

    # ---- reporting ----------------------------------------------------
    print(f"{'split':<10}{'images':>8}{'target':>9}{'dives':>8}{'anns':>8}"
          f"{'known':>8}{'ood':>8}{'far':>7}")
    print("-" * 66)
    report: dict = {}
    for s in ["J_cal", "J_policy", "J_test"]:
        cls = Counter(c for f in split_files[s] for c in ann_by_file[f])
        n_ann = sum(cls.values())
        known = sum(v for k, v in cls.items() if k in KNOWN_CLASSES)
        far = sum(v for k, v in cls.items() if k in FAR_OOD_CLASSES)
        ood = n_ann - known - far
        print(f"{s:<10}{len(split_files[s]):>8}{targets[s]:>9.0f}"
              f"{len(split_dives[s]):>8}{n_ann:>8}{known:>8}{ood:>8}{far:>7}")
        report[s] = {
            "images": len(split_files[s]),
            "target_images": round(targets[s]),
            "dives": len(split_dives[s]),
            "annotations": n_ann,
            "known": known,
            "ood_non_far": ood,
            "far_ood": far,
            "class_counts": dict(cls.most_common()),
        }

    # Each split must actually contain the OOD taxa it is meant to host.
    print(f"\n{'role':<10}{'hosted by':<11}{'anns':>7}   classes present / total")
    print("-" * 66)
    warnings: list[str] = []
    for split, role in SPLIT_HOSTS_OOD.items():
        role_classes = OOD_ROLE_CLASSES[role]
        cls = Counter(c for f in split_files[split] for c in ann_by_file[f])
        present = {c: cls.get(c, 0) for c in sorted(role_classes)}
        n = sum(present.values())
        n_present = sum(1 for v in present.values() if v > 0)
        print(f"{role:<10}{split:<11}{n:>7}   {n_present}/{len(role_classes)}")
        for c, v in present.items():
            flag = "  <-- sparse" if v < 10 else ""
            print(f"             {c:<26}{v:>6}{flag}")
            if v == 0:
                warnings.append(f"{role}: class '{c}' has NO annotations in {split}")
            elif v < 10:
                warnings.append(f"{role}: class '{c}' has only {v} annotations in {split}")
        report[split]["hosted_ood_role"] = role
        report[split]["hosted_ood_counts"] = present

        # Classes carrying another split's OOD role, or far-OOD, still occur in
        # this split's images. They are legitimate OOD_ERR for error accounting,
        # but must not become fitting signal: a meta-confidence model trained on
        # J_cal that saw shrimp would no longer be naive to shrimp at test time,
        # and the "unseen taxa" claim would be false.
        other_roles = set().union(
            *(v for k, v in OOD_ROLE_CLASSES.items() if k != role)
        )
        excluded = sorted((other_roles | FAR_OOD_CLASSES) & set(cls))
        report[split]["ood_fit_classes"] = sorted(role_classes)
        report[split]["ood_excluded_from_fit"] = {c: cls[c] for c in excluded}
        report[split]["ood_excluded_from_fit_total"] = sum(cls[c] for c in excluded)

    print(f"\n{'split':<10}{'fit-eligible OOD':>18}{'excluded from fit':>19}")
    print("-" * 66)
    for s in ["J_cal", "J_policy", "J_test"]:
        print(f"{s:<10}{sum(report[s]['hosted_ood_counts'].values()):>18}"
              f"{report[s]['ood_excluded_from_fit_total']:>19}")
    print("  Excluded annotations remain valid OOD_ERR for error accounting;")
    print("  they are barred only from fitting calibration, meta-confidence")
    print("  and the policy, so those never see another split's OOD taxa.")

    if warnings:
        print("\nWARNINGS")
        for w in warnings:
            print(f"  - {w}")
        print("\n  Rare taxa are sparse in the source data, so this is usually a")
        print("  dataset limit rather than a splitter fault. If a class is absent")
        print("  from its designated split, either re-seed or fold it into a")
        print("  neighbouring role and record the change in PROTOCOL.md §5.")

    manifest = {
        "phase": "A",
        "description": "Dive-disjoint splits over pooled Jedi imagery for "
                       "evaluating the frozen aquarium baseline.",
        "config": {
            "seed": args.seed,
            "proportions": props,
            "dive_regex": DIVE_RE.pattern,
            "pooled_source_splits": SOURCE_SPLITS,
            "supplied_splits_discarded": True,
            "reason": "Supplied splits share dives (57.3% test contamination); "
                      "Phase A does not train on Jedi so they carry no meaning.",
        },
        "label_space": {
            "known": sorted(KNOWN_CLASSES),
            "far_ood": sorted(FAR_OOD_CLASSES),
            "ood_roles": {k: sorted(v) for k, v in OOD_ROLE_CLASSES.items()},
            "split_hosts_ood_role": SPLIT_HOSTS_OOD,
            "note": "'ray' is OOD, not stingray (PROTOCOL.md decision 9.3).",
        },
        "totals": {
            "images": len(all_files),
            "dives": len(dive_sizes),
            "annotations": sum(len(v) for v in ann_by_file.values()),
        },
        "splits": report,
        "verification": {
            "dive_disjoint": True,
            "every_image_assigned_once": True,
            "duplicate_filenames": 0,
        },
        "files": split_files,
        "dives": split_dives,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote {args.out.relative_to(ROOT)}")
    print("Verified: dive-disjoint, every image assigned exactly once.")


if __name__ == "__main__":
    main()
