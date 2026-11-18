"""Derive the Phase B+M manifest: Phase B with `machine` as a trained class.

Phase B held `machine` (ROV hardware, 4,074 annotations and the largest category
in the dataset) out as the far-OOD class. Because it was never labelled, the
detector learned hardware as background, and the most visible failure of the
system on deep-sea frames is hardware reported as an animal. This manifest
promotes `machine` to a ninth known class so the detector and classifier can say
"hardware, not an organism".

**The dive assignment is copied, not recomputed.** Every split contains exactly
the same dives and images as `phase_b_splits.json`, so the v1 and v2 models are
compared on identical data and every paired bootstrap stays paired. Only the
label space changes:

    known    8 taxa -> 9 (adds `machine`)
    far_ood  {machine} -> {}

Consequence, recorded deliberately: the far-OOD evaluation is retired. `machine`
is now a known class, so a score on it measures recognition on unseen *dives*,
not rejection of an unseen *class*. Near-OOD (11 held-out taxa) is unaffected.

Usage:
    python experiments/make_machine_manifest.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

EXP = Path(__file__).parent
sys.path.insert(0, str(EXP))
from make_splits import load_dataset  # noqa: E402

SRC = EXP / "splits" / "phase_b_splits.json"
DST = EXP / "splits" / "phase_b_machine_splits.json"
ORDER = ["S_fit", "S_modelsel", "S_cal", "S_policy", "S_test"]
NEW_CLASS = "machine"


def main() -> None:
    m = json.loads(SRC.read_text())
    ls = m["label_space"]
    if NEW_CLASS not in ls["far_ood"]:
        raise SystemExit(f"{NEW_CLASS!r} is not the far-OOD class in {SRC.name}")

    ann_by_file, _ = load_dataset()
    known_old = set(ls["known"])
    known_new = known_old | {NEW_CLASS}

    ls["known"] = sorted(known_new)
    ls["far_ood"] = []

    for s in ORDER:
        r = m["splits"][s]
        cls = Counter(c for f in m["files"][s] for c in ann_by_file[f])
        r["known"] = sum(v for k, v in cls.items() if k in known_new)
        r["far_ood"] = 0
        excl = r.get("ood_excluded_from_fit")
        if excl is not None:
            excl.pop(NEW_CLASS, None)
            r["ood_excluded_from_fit_total"] = sum(excl.values())

    clean = set(m["contamination"]["s_fit_images_clean"])
    fit_files = m["files"]["S_fit"]
    m["contamination"]["s_fit_known_anns_all"] = sum(
        1 for f in fit_files for c in ann_by_file[f] if c in known_new)
    m["contamination"]["s_fit_known_anns_clean"] = sum(
        1 for f in fit_files if f in clean for c in ann_by_file[f] if c in known_new)

    m["description"] = (
        "Phase B+M: the Phase B five-way dive-disjoint splits with `machine` "
        "promoted from far-OOD to a trained class. Dive assignment is identical "
        "to phase_b_splits.json.")
    m["config"]["derived_from"] = SRC.name
    m["config"]["label_space_change"] = (
        "known 8 -> 9 (+machine); far_ood {machine} -> {}. Far-OOD evaluation is "
        "retired: machine is now known, so it measures recognition on unseen "
        "dives, not rejection of an unseen class.")

    old = json.loads(SRC.read_text())
    for s in ORDER:
        assert m["files"][s] == old["files"][s], f"{s} changed"
        assert m["dives"][s] == old["dives"][s], f"{s} dives changed"

    DST.write_text(json.dumps(m, indent=2) + "\n")
    print(f"known {len(known_old)} -> {len(known_new)}: {', '.join(ls['known'])}")
    print(f"{'split':<11}{'images':>8}{'known anns':>12}{'machine':>9}")
    for s in ORDER:
        n_m = sum(1 for f in m["files"][s] for c in ann_by_file[f] if c == NEW_CLASS)
        print(f"{s:<11}{len(m['files'][s]):>8}{m['splits'][s]['known']:>12}{n_m:>9}")
    print(f"\nS_fit known annotations (clean): "
          f"{old['contamination']['s_fit_known_anns_clean']} -> "
          f"{m['contamination']['s_fit_known_anns_clean']}")
    print("dive assignment identical to phase_b_splits.json: verified")
    print(f"Wrote {DST.relative_to(EXP.parent)}")


if __name__ == "__main__":
    main()
