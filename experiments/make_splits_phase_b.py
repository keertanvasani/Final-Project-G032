"""Build the five-way dive-disjoint Phase B splits for on-domain retraining.

PROTOCOL.md §4.2 specifies five splits for Phase B, proportioned **over dives**:

    S_fit       detector + classifier weight fitting     ~70%
    S_modelsel  early stopping, hyperparameters          ~10%
    S_cal       calibration + meta-confidence fitting     ~7%   (hosts OOD_dev)
    S_policy    exploration rollouts + policy learning    ~7%   (hosts OOD_val)
    S_test      final evaluation — single use             ~6%   (hosts OOD_test)

An earlier revision of this script carved `J_fit` out of the Phase A cal+policy
dives while holding `J_test` frozen. That preserved the §8.1 test-set read, but
it left S_fit at 25.7% of dives — roughly a third of what §4.2 specifies — and
the resulting detector starved on the long tail (crinoid reached AP50 0.036 on
127 training annotations). The protocol's own proportions win: the detector is
the substrate every later phase is built on.

Consequence, recorded deliberately: `S_test` is a **new** partition, so the
Phase A `J_test` read logged in §8.1 no longer covers it. That read was made
against the aquarium baseline — a different model — and the images are excluded
from Phase B fitting, so the exposure is to experimenter knowledge rather than
to weights. It is logged in §8.1 rather than left implicit.

Proportions are over dives, so realised image counts will differ; both are
reported. Dives stay indivisible throughout.

Usage:
    python experiments/make_splits_phase_b.py
    python experiments/make_splits_phase_b.py --test 0.15 --fit 0.55
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "JediOrganismDetectionDataset"
SOURCE_SPLITS = ["coco_train_data", "coco_valid_data", "coco_test_data"]

sys.path.insert(0, str(Path(__file__).parent))
from make_splits import dive_of, load_dataset  # noqa: E402

# PROTOCOL.md §11.1 — Phase B label space.
KNOWN_CLASSES_B = {
    "fish", "starfish", "coral/sea-anemone", "crab",
    "sea-cucumber", "sponge", "shrimp", "crinoid",
}
HELD_OUT_CLASSES_B = {
    "jellyfish", "squat-lobster", "squid", "octopus", "comb-jelly",
    "urchin", "shark", "sea-spider", "tunicate", "ray", "ragworm",
}
FAR_OOD_CLASSES = {"machine"}

SPLIT_ORDER = ["S_fit", "S_modelsel", "S_cal", "S_policy", "S_test"]
SPLIT_HOSTS_OOD = {"S_cal": "OOD_dev", "S_policy": "OOD_val", "S_test": "OOD_test"}
SPLIT_HOSTS_OOD_INV = {v: k for k, v in SPLIT_HOSTS_OOD.items()}
ROLES = ["OOD_dev", "OOD_val", "OOD_test"]
MIN_CLASSES_PER_ROLE = 3


def assign_dives_by_count(
    dives: list[str], props: dict[str, float], seed: int
) -> dict[str, str]:
    """Allocate whole dives so each split gets its share of the *dive* count.

    §4.2 proportions are over dives, not images, so this allocates by count and
    lets image counts fall where they may (both are reported). A seeded shuffle
    makes the assignment deterministic without ordering bias.
    """
    rng = random.Random(seed)
    shuffled = list(dives)
    rng.shuffle(shuffled)

    targets = {s: p * len(shuffled) for s, p in props.items()}
    current = {s: 0 for s in props}
    out: dict[str, str] = {}
    for d in shuffled:
        split = max(props, key=lambda s: targets[s] - current[s])
        out[d] = split
        current[split] += 1
    return out


def partition_ood_roles(
    class_totals: dict[str, int], hosted: dict[str, dict[str, int]]
) -> dict[str, set[str]]:
    """Maximin exhaustive search over role assignments (PROTOCOL.md §11.2)."""
    classes = sorted(class_totals)
    gains = {
        c: {r: hosted[c].get(SPLIT_HOSTS_OOD_INV[r], 0) for r in ROLES}
        for c in classes
    }
    best_key = None
    best = None
    for combo in itertools.product(ROLES, repeat=len(classes)):
        members: dict[str, list[str]] = {r: [] for r in ROLES}
        for cls, role in zip(classes, combo):
            members[role].append(cls)
        if any(len(members[r]) < MIN_CLASSES_PER_ROLE for r in ROLES):
            continue
        h = {r: sum(gains[c][r] for c in members[r]) for r in ROLES}
        key = (min(h.values()), sum(h.values()), tuple(combo))
        if best_key is None or key > best_key:
            best_key, best = key, {r: set(members[r]) for r in ROLES}
    if best is None:
        raise SystemExit("No partition gives every role the minimum class count.")
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    # §4.2 specifies 70/10/7/7/6. Those proportions assume S_test only has to
    # support detection metrics; here it must *also* host OOD_test under the
    # class-disjoint x dive-disjoint constraint, and 6% of dives cannot do both
    # — it yields 48 usable OOD_test annotations, which cannot support an AUROC.
    # Test is widened to 15% and fit reduced to 55%. See PROTOCOL.md §11.8.
    ap.add_argument("--fit", type=float, default=0.55)
    ap.add_argument("--modelsel", type=float, default=0.10)
    ap.add_argument("--cal", type=float, default=0.10)
    ap.add_argument("--policy", type=float, default=0.10)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "splits" / "phase_b_splits.json")
    args = ap.parse_args()

    props = {
        "S_fit": args.fit, "S_modelsel": args.modelsel, "S_cal": args.cal,
        "S_policy": args.policy, "S_test": args.test,
    }
    if abs(sum(props.values()) - 1.0) > 1e-6:
        raise SystemExit(f"Proportions must sum to 1.0, got {sum(props.values())}")

    ann_by_file, all_files = load_dataset()
    all_classes = Counter(c for v in ann_by_file.values() for c in v)
    declared = KNOWN_CLASSES_B | HELD_OUT_CLASSES_B | FAR_OOD_CLASSES
    if declared != set(all_classes):
        raise SystemExit(
            "Label space does not cover the dataset.\n"
            f"  missing: {sorted(set(all_classes) - declared)}\n"
            f"  extra:   {sorted(declared - set(all_classes))}"
        )

    files_by_dive: dict[str, list[str]] = defaultdict(list)
    for f in all_files:
        files_by_dive[dive_of(f)].append(f)
    dives = sorted(files_by_dive)

    assignment = assign_dives_by_count(dives, props, args.seed)
    split_files: dict[str, list[str]] = {s: [] for s in props}
    split_dives: dict[str, list[str]] = {s: [] for s in props}
    for d, s in assignment.items():
        split_dives[s].append(d)
        split_files[s].extend(files_by_dive[d])
    for s in props:
        split_files[s].sort()
        split_dives[s].sort()

    # ---- verification ----------------------------------------------------
    seen: dict[str, str] = {}
    for s in SPLIT_ORDER:
        for d in split_dives[s]:
            if d in seen:
                raise SystemExit(f"BUG: dive {d} in both {seen[d]} and {s}")
            seen[d] = s
    assigned = sum(len(split_files[s]) for s in SPLIT_ORDER)
    if assigned != len(all_files):
        raise SystemExit(f"BUG: assigned {assigned} of {len(all_files)}")

    cls_by_split = {
        s: Counter(c for f in split_files[s] for c in ann_by_file[f])
        for s in SPLIT_ORDER
    }

    hosted = {c: {s: cls_by_split[s].get(c, 0) for s in SPLIT_ORDER}
              for c in HELD_OUT_CLASSES_B}
    ood_roles = partition_ood_roles(
        {c: all_classes[c] for c in HELD_OUT_CLASSES_B}, hosted
    )

    # ---- S_fit contamination (PROTOCOL §11.3) ----------------------------
    def clean(split: str) -> list[str]:
        return [f for f in split_files[split]
                if not any(c in HELD_OUT_CLASSES_B for c in ann_by_file[f])]

    fit_clean, sel_clean = clean("S_fit"), clean("S_modelsel")
    fit_known = sum(1 for f in split_files["S_fit"]
                    for c in ann_by_file[f] if c in KNOWN_CLASSES_B)
    fit_known_clean = sum(1 for f in fit_clean
                          for c in ann_by_file[f] if c in KNOWN_CLASSES_B)

    # ---- reporting -------------------------------------------------------
    n_d, n_i = len(dives), len(all_files)
    print(f"Pooled {n_i} images across {n_d} dives (proportions are over DIVES)\n")
    print(f"{'split':<12}{'dives':>7}{'dive%':>7}{'target%':>9}{'images':>8}"
          f"{'img%':>7}{'anns':>7}{'known':>7}{'held':>6}{'far':>6}")
    print("-" * 79)
    report: dict = {}
    for s in SPLIT_ORDER:
        cls = cls_by_split[s]
        n_ann = sum(cls.values())
        known = sum(v for k, v in cls.items() if k in KNOWN_CLASSES_B)
        held = sum(v for k, v in cls.items() if k in HELD_OUT_CLASSES_B)
        far = sum(v for k, v in cls.items() if k in FAR_OOD_CLASSES)
        print(f"{s:<12}{len(split_dives[s]):>7}"
              f"{100*len(split_dives[s])/n_d:>6.1f}%{100*props[s]:>8.0f}%"
              f"{len(split_files[s]):>8}{100*len(split_files[s])/n_i:>6.1f}%"
              f"{n_ann:>7}{known:>7}{held:>6}{far:>6}")
        report[s] = {
            "dives": len(split_dives[s]),
            "dive_fraction": round(len(split_dives[s]) / n_d, 4),
            "target_dive_fraction": props[s],
            "images": len(split_files[s]),
            "annotations": n_ann,
            "known": known,
            "held_out": held,
            "far_ood": far,
            "class_counts": dict(cls.most_common()),
        }

    print(f"\n{'role':<10}{'hosted by':<12}{'usable anns':>12}   classes")
    print("-" * 79)
    warnings: list[str] = []
    for role in ROLES:
        host = SPLIT_HOSTS_OOD_INV[role]
        members = sorted(ood_roles[role])
        counts = {c: cls_by_split[host].get(c, 0) for c in members}
        print(f"{role:<10}{host:<12}{sum(counts.values()):>12}   {len(members)} classes")
        for c, v in sorted(counts.items(), key=lambda kv: -kv[1]):
            flag = "  <-- sparse" if v < 10 else ""
            print(f"             {c:<24}{v:>6}{flag}")
            if v < 10:
                warnings.append(f"{role}: '{c}' has only {v} annotations in {host}")
        report[host]["hosted_ood_role"] = role
        report[host]["hosted_ood_counts"] = counts
        report[host]["ood_fit_classes"] = members
        other = set().union(*(ood_roles[r] for r in ROLES if r != role))
        excl = sorted((other | FAR_OOD_CLASSES) & set(cls_by_split[host]))
        report[host]["ood_excluded_from_fit"] = {c: cls_by_split[host][c] for c in excl}
        report[host]["ood_excluded_from_fit_total"] = sum(
            cls_by_split[host][c] for c in excl)

    print("\nheld-out contamination of the training splits (PROTOCOL §11.3)")
    print("-" * 79)
    for label, full, cl in (("S_fit", split_files["S_fit"], fit_clean),
                            ("S_modelsel", split_files["S_modelsel"], sel_clean)):
        print(f"  {label:<11}{len(full):>6} images -> {len(cl):>6} clean "
              f"({100*(len(full)-len(cl))/max(len(full),1):.1f}% dropped)")
    print(f"  KNOWN annotations retained in S_fit: {fit_known_clean} of {fit_known} "
          f"({100*fit_known_clean/max(fit_known,1):.1f}%)")

    print("\nper-class KNOWN annotations available for training (clean S_fit)")
    print("-" * 79)
    for c in sorted(KNOWN_CLASSES_B):
        n = sum(1 for f in fit_clean for cc in ann_by_file[f] if cc == c)
        flag = "  <-- thin" if n < 200 else ""
        print(f"  {c:<24}{n:>6}{flag}")

    if warnings:
        print("\nWARNINGS")
        for w in warnings:
            print(f"  - {w}")

    manifest = {
        "phase": "B",
        "description": "Five-way dive-disjoint splits for on-domain retraining, "
                       "proportioned over dives per PROTOCOL.md §4.2.",
        "config": {
            "seed": args.seed,
            "proportions_over_dives": props,
            "pooled_source_splits": SOURCE_SPLITS,
            "supersedes": "the frozen-J_test carve; S_fit was 25.7% of dives, "
                          "starving the long tail. S_test is a new partition, so "
                          "the Phase A read logged in PROTOCOL §8.1 does not "
                          "cover it — see §8.1 for the entry.",
        },
        "label_space": {
            "known": sorted(KNOWN_CLASSES_B),
            "held_out_unseen": sorted(HELD_OUT_CLASSES_B),
            "far_ood": sorted(FAR_OOD_CLASSES),
            "ood_roles": {r: sorted(ood_roles[r]) for r in ROLES},
            "split_hosts_ood_role": SPLIT_HOSTS_OOD,
        },
        "totals": {"images": n_i, "dives": n_d,
                   "annotations": sum(len(v) for v in ann_by_file.values())},
        "contamination": {
            "s_fit_images_clean": sorted(fit_clean),
            "s_modelsel_images_clean": sorted(sel_clean),
            "s_fit_known_anns_all": fit_known,
            "s_fit_known_anns_clean": fit_known_clean,
        },
        "splits": report,
        "verification": {"dive_disjoint": True, "every_image_assigned_once": True},
        "files": {s: split_files[s] for s in SPLIT_ORDER},
        "dives": {s: split_dives[s] for s in SPLIT_ORDER},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
