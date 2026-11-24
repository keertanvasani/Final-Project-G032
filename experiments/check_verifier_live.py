"""Week 4: the live taxonomy-aware verifier must match the offline computation.

Runs the deployed loop with config.AGENT_VERIFIER switched on (per mode, α)
over S_cal images and compares each box's answer (level, node) with the one
computed offline from week3/S_cal_verifier.jsonl and the same verifier.json.

Usage:
    python experiments/check_verifier_live.py [--limit 200]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import config  # noqa: E402
from ablation_deepsea import image_path  # noqa: E402
from agentic import loop  # noqa: E402
from agentic import taxonomy as tx  # noqa: E402
from pipeline import EnsemblePipeline  # noqa: E402
from week3_verifier import box_rows  # noqa: E402

W3 = ROOT / "experiments" / "results" / "phase_b_machine" / "week3"
OUT = ROOT / "experiments" / "results" / "phase_b_machine" / "week4"


def offline(b, spec, classes, mode, alpha):
    feats, _, nodes = box_rows(b, classes)
    t = spec["thresholds"][mode][str(alpha)]
    levels = range(4) if mode == "hierarchical" else [0]
    for j in levels:
        m = spec["models"][tx.LEVELS[j]]
        z = (np.asarray(feats[j]) - np.asarray(m["mean"])) / np.asarray(m["std"])
        if t is not None and 1 / (1 + np.exp(-(m["w"][0] + z @ np.asarray(m["w"][1:])))) >= t:
            return tx.LEVELS[j], nodes[j]
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()
    spec = json.loads(Path(config.AGENT_VERIFIER_FILE).read_text())
    recs = [json.loads(line) for line in (W3 / "S_cal_verifier.jsonl").open(encoding="utf-8")]
    classes = recs[0]["classes"]
    pipe = EnsemblePipeline(profile="deepsea")
    stats = {}
    for mode, alpha in (("hierarchical", 0.2), ("flat", 0.2), ("hierarchical", 0.1)):
        config.AGENT_VERIFIER = {"mode": mode, "alpha": alpha}
        boxes = mism = answered = 0
        for rec in recs[: args.limit]:
            run = loop.run_loop(cv2.imread(str(image_path(rec["file"]))), pipe)
            if len(run.detections) != len(rec["boxes"]):
                mism += 1
                continue
            for d, b in zip(run.detections, rec["boxes"]):
                boxes += 1
                live = (d["taxon"]["level"], d["taxon"]["node"])
                answered += live[0] is not None
                mism += live != offline(b, spec, classes, mode, alpha)
        stats[f"{mode} α={alpha}"] = {"boxes": boxes, "answered": answered, "mismatches": mism}
        print(f"{mode:12s} α={alpha}: {boxes} boxes, {answered} answered, {mism} mismatches")
    config.AGENT_VERIFIER = None
    stats["identical"] = all(v["mismatches"] == 0 for v in stats.values() if isinstance(v, dict))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "verifier_live_check.json").write_text(json.dumps(stats, indent=2) + "\n")
    sys.exit(0 if stats["identical"] else 1)


if __name__ == "__main__":
    main()
