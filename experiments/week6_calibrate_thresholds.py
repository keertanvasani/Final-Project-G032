"""Week 6: Learn-then-Test thresholds for the S_test confirmation, calibrated on all of S_policy.

The combiner (week4/verifier.json) is frozen. S_policy is used only as calibration data
(§5.1: as unseen-animal signal only its own role taxa, OOD_val). The thresholds written
here are copied into PROTOCOL §16 before S_test is read.

Usage:
    python experiments/week6_calibrate_thresholds.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from ablation_deepsea import MAN  # noqa: E402
from agentic import taxonomy as tx  # noqa: E402
from week3_verifier import box_rows, ltt  # noqa: E402

R = ROOT / "experiments" / "results" / "phase_b_machine"


def main() -> None:
    man = json.loads(MAN.read_text())
    spec = json.loads((R / "week4" / "verifier.json").read_text())
    excluded = set(man["label_space"]["held_out_unseen"]) - set(man["label_space"]["ood_roles"]["OOD_val"])
    boxes, dives = [], []
    for line in (R / "week3" / "S_policy_verifier.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        for b in r["boxes"]:
            boxes.append(b)
            dives.append(r["dive"])
    F, Y = [], []
    for b in boxes:
        f, y, _ = box_rows(b, spec["classes"])
        F.append(f)
        Y.append(y)
    F, Y = np.array(F), np.array(Y, bool)
    P = np.zeros((len(boxes), 4))
    for j, lv in enumerate(tx.LEVELS):
        m = spec["models"][lv]
        P[:, j] = 1 / (1 + np.exp(-(m["w"][0] + ((F[:, j] - np.array(m["mean"])) / np.array(m["std"]))
                                    @ np.array(m["w"][1:]))))
    idx = np.array([i for i, b in enumerate(boxes) if b["gt_class"] not in excluded])
    out = {"calibrated_on": "S_policy (all dives, eligible boxes)", "delta": 0.10, "boxes": int(len(idx)),
           "dives": len({dives[i] for i in idx}), "thresholds": {}}
    for mode, levels in (("hierarchical", [0, 1, 2, 3]), ("flat", [0])):
        out["thresholds"][mode] = {str(a): ltt(P, Y, dives, idx, a, levels) for a in (0.10, 0.20)}
    (R / "week6" / "thresholds_S_policy.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
