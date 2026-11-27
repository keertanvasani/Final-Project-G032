"""Week 1: the live loop with the fitted image agent must match the replay of the logs.

On S_cal images (the fit split; nothing is evaluated here), run
agentic/loop.run_loop with the chain image → deployed chain and check, per
image, that D5 (re-run arm), D6 (kept pass) and the final boxes equal what
replaying week1/S_cal_image_agent.jsonl with the same policy file predicts.

Usage:
    python experiments/check_image_agent_live.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import config  # noqa: E402
from ablation_deepsea import image_path  # noqa: E402
from agentic import loop, policy
from agentic.probes import PROBE_KEYS, state_columns  # noqa: E402
from log_image_agent import harness_fields  # noqa: E402
from pipeline import EnsemblePipeline  # noqa: E402

config.pin_pre_d()   # reproduce the committed logs: the pre-option-D app settings

DIR = ROOT / "experiments" / "results" / "phase_b_machine" / "week1"


def replay(rec, spec):
    keys = spec.get("probe_keys", list(PROBE_KEYS))
    c1, c2 = state_columns(keys, 1), state_columns(keys, 2)
    best, arm = 0.0, None
    for a, m in spec["q1"].items():
        v = policy._ridge(m, [rec["arms"][a]["s1"][i] for i in c1]) - spec["lambda"] * spec["cost_ms"][a]
        if v > best:
            best, arm = v, a
    if arm is None:
        return None, "first", rec["arms"][next(iter(rec["arms"]))]["passes"][0]["finals"]
    r = rec["arms"][arm]
    keep = "second" if policy._ridge(spec["q2"][arm], [r["s2"][i] for i in c2]) > 0 else "first"
    return arm, keep, r["passes"][1 if keep == "second" else 0]["finals"]


def keys_full(spec):
    return list(spec.get("probe_keys", PROBE_KEYS)) == list(PROBE_KEYS)


def key(finals):
    return [(d["classifier_class"], round(d["combined_confidence"], 5), [round(v, 2) for v in d["bbox"]],
             d["unknown"]) for d in finals]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=150)
    args = ap.parse_args()
    spec = json.loads(Path(config.AGENT_IMAGE_POLICY).read_text())
    recs = [json.loads(line) for line in (DIR / "S_cal_image_agent.jsonl").open(encoding="utf-8")][: args.limit]
    pipe = EnsemblePipeline(profile="deepsea")
    chain = policy.build_chain(("image",) + tuple(config.AGENT_POLICY_CHAIN))
    stats = {"images": 0, "decision_mismatch": [], "output_mismatch": [], "reruns": 0, "kept_second": 0}
    for rec in recs:
        img = cv2.imread(str(image_path(rec["file"])))
        run = loop.run_loop(img, pipe, chain=chain)
        d5 = next((d for d in run.decisions if d["decision"] == "rerun"), None)
        d6 = next((d for d in run.decisions if d["decision"] == "commit"), None)
        live = (d5["value"] if d5 else None, d6["value"] if d6 else "first")
        arm, keep, finals = replay(rec, spec)
        stats["images"] += 1
        stats["reruns"] += live[0] is not None
        stats["kept_second"] += live[1] == "second"
        if live != (arm, keep):
            stats["decision_mismatch"].append([rec["file"], live, [arm, keep]])
        elif key([harness_fields(d) for d in run.detections]) != key(finals):
            stats["output_mismatch"].append(rec["file"])
    stats["identical"] = not stats["decision_mismatch"] and not stats["output_mismatch"]
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in stats.items()}))
    out = DIR if keys_full(spec) else DIR.parent / "week2"
    out.mkdir(parents=True, exist_ok=True)
    (out / "live_check.json").write_text(json.dumps(stats, indent=2) + "\n")
    sys.exit(0 if stats["identical"] else 1)


if __name__ == "__main__":
    main()
