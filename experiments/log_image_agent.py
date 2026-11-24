"""Week 1: full-information logs for the two-step image agent, written by the loop itself.

For every image and every arm in agentic.probes.ARMS, the real agent loop
(agentic/loop.run_loop, app mode, deployed chain for D1/D3/D4) runs with a
forced image policy that always re-runs (D5 = arm). At D6 it records both
passes and the exact states s1, s2 the learned image agent would see, built
by the same code (agentic.policy.pass_states). So fitting, replay and the live
policy share one feature path.

Per image and arm: pass-1 and pass-2 final boxes (harness fields), states,
costs. S_cal only for now; S_policy and S_test are refused until the
evaluation is registered in PROTOCOL.md.

Usage:
    python experiments/log_image_agent.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import config  # noqa: E402
from ablation_deepsea import DIVE_RE, MAN, image_path  # noqa: E402
from agentic import loop, policy, probes  # noqa: E402
from pipeline import EnsemblePipeline  # noqa: E402

OUT = ROOT / "experiments" / "results" / "phase_b_machine" / "week1"
ALLOWED = ("S_cal", "S_policy")      # S_policy opened by PROTOCOL §15


def harness_fields(d):
    nov = d.get("novelty") or {}
    return {"bbox": [float(v) for v in d["bbox"]], "classifier_class": d["classifier_class"],
            "detector_class": d.get("detector_class"), "combined_confidence": d["combined_confidence"],
            "raw_combined_confidence": d.get("raw_combined_confidence", d["combined_confidence"]),
            "detector_confidence": d["detector_confidence"], "classifier_confidence": d["classifier_confidence"],
            "agreement": d["agreement"], "unknown": bool(nov.get("unknown")), "energy": nov.get("score")}


class Forced(policy.Policy):
    """Always re-run with *arm*; at D6 record everything, keep the second pass."""
    name = "forced"

    def __init__(self, arm):
        self.arm = arm
        self.rec = {}

    def rerun(self, obs):
        self.rec["t_rerun"] = obs.elapsed_ms
        return self.arm

    def commit(self, obs):
        t = time.perf_counter()
        s1, s2 = policy.pass_states(obs)
        st = obs.extra["state"]
        one, two = obs.extra["passes"]
        self.rec.update({
            "s1": s1, "s2": s2, "t_commit": obs.elapsed_ms,
            "probe_ms": [st["probe_ms_0"], st["probe_ms_1"]],
            "arm_ms": two.get("arm_ms", 0.0), "state_ms": (time.perf_counter() - t) * 1000.0,
            "passes": [{"status": p["status"], "finals": [harness_fields(d) for d in p["detections"]]}
                       for p in (one, two)]})
        return "second"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="S_cal")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--confirmatory", action="store_true",
                    help="required for S_test: the single registered run (PROTOCOL §16)")
    args = ap.parse_args()
    if args.split == "S_test":
        if not (args.confirmatory and args.limit is None):
            raise SystemExit("S_test: only the full --confirmatory run registered in PROTOCOL §16")
    elif args.split not in ALLOWED:
        raise SystemExit(f"{args.split} is not open yet (register the evaluation in PROTOCOL.md first)")

    man = json.loads(MAN.read_text())
    files = sorted(man["files"][args.split])[: args.limit]
    pipe = EnsemblePipeline(profile="deepsea")
    base = [policy.POLICIES[n]() for n in config.AGENT_POLICY_CHAIN]
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{args.split}_image_agent.jsonl"
    t0 = time.time()
    with dest.open("w", encoding="utf-8") as fh:
        for k, fn in enumerate(files):
            p = image_path(fn)
            img = cv2.imread(str(p)) if p else None
            if img is None:
                continue
            rec = {"file": fn, "dive": DIVE_RE.match(fn).group(1), "arms": {}}
            for arm in probes.ARMS:
                f = Forced(arm)
                t = time.perf_counter()
                run = loop.run_loop(img.copy(), pipe, chain=policy.FallbackChain([f] + base))
                total = (time.perf_counter() - t) * 1000.0
                r = f.rec
                if "s1" not in r:          # the loop never reached D6 (guardrail)
                    rec["arms"][arm] = {"skipped": True, "trace": run.guardrails}
                    continue
                # Pass-1 cost = time to D5; pass-2 cost = D5 → D6 (the forced policy
                # computes no probe before D6). Probe costs are logged separately.
                r["pass1_ms"] = r.pop("t_rerun")
                r["pass2_ms"] = r.pop("t_commit") - r["pass1_ms"]
                r["total_ms"] = total
                rec["arms"][arm] = r
            fh.write(json.dumps(rec) + "\n")
            if (k + 1) % 100 == 0:
                print(f"{k + 1}/{len(files)} images ({time.time() - t0:.0f}s)", flush=True)
    print(f"wrote {dest.relative_to(ROOT)}: {len(files)} images ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
