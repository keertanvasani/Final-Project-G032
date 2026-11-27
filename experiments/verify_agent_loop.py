"""Check that the step-2 agent loop reproduces the old controller exactly.

The pre-refactor agent_controller.py is loaded from git (default commit
8c3a1b0) and run side by side with the new one on the same images:

  1. app path (run_agentic_pipeline): detections, decision trace, status
     message, summary (minus timings) and the annotated image must be identical;
  2. perception path (run_perception): detections and status identical;
  3. guardrails:
       - a time budget of 0 ms skips every discretionary tool and must equal
         the ablation's plain-model configuration (text filter + refuse gate +
         Known/Unknown, no enhance/zoom/vote);
       - a policy that raises must fall back to the rules and give the same
         output as the rules alone;
       - the call limit refuses a second call of the same tool.

Images: S_policy (deep-sea profile) and demo_images (both profiles).
S_test is never touched.

Usage:
    python experiments/verify_agent_loop.py [--limit N] [--ref 8c3a1b0]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import config  # noqa: E402

# This script checks the refactor against the old rule controller, so it runs
# the rule chain whatever the app's deployed chain is.
config.pin_pre_d()
config.AGENT_POLICY_CHAIN = ("rules", "plain")

import agent_controller as new  # noqa: E402
from ablation_deepsea import CONFIGS, MAN, image_path, run_config  # noqa: E402
from agentic import loop, policy  # noqa: E402
from pipeline import EnsemblePipeline  # noqa: E402

OUT = ROOT / "experiments" / "results" / "phase_b_machine" / "verify_agent_loop.json"
TIMING_KEYS = {"inference_time_ms", "inference_breakdown", "agent"}


def load_reference(ref: str) -> types.ModuleType:
    src = subprocess.run(["git", "show", f"{ref}:agent_controller.py"], cwd=ROOT,
                         capture_output=True, text=True, encoding="utf-8", check=True).stdout
    mod = types.ModuleType("agent_controller_ref")
    exec(compile(src, f"{ref}:agent_controller.py", "exec"), mod.__dict__)
    return mod


def same(a, b) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    return a == b


class Broken(policy.Policy):
    name = "broken"

    def enhance(self, obs):
        raise RuntimeError("test")

    def zoom(self, obs):
        return {"not-an-index"}

    def vote(self, obs, idx, det, cls_res):
        raise RuntimeError("test")


def check_images(files, pipe, ref, label, plain_flags):
    stats = {"images": 0, "app_mismatch": [], "perception_mismatch": [], "fallback_mismatch": [],
             "budget0_mismatch": [], "detections": 0, "budget0_actions": 0}
    chain_broken = policy.FallbackChain([Broken(), policy.RulePolicy(), policy.PlainPolicy()])
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        stats["images"] += 1
        name = p.name

        a_img, a_det, a_sum, a_tr = ref.run_agentic_pipeline(img.copy(), pipe)
        b_img, b_det, b_sum, b_tr = new.run_agentic_pipeline(img.copy(), pipe)
        stats["detections"] += len(b_det)
        sa = {k: v for k, v in a_sum.items() if k not in TIMING_KEYS}
        sb = {k: v for k, v in b_sum.items() if k not in TIMING_KEYS}
        if not (same(a_det, b_det) and a_tr == b_tr and same(sa, sb) and np.array_equal(a_img, b_img)):
            stats["app_mismatch"].append(name)

        pa = ref.run_perception(img.copy(), pipe)
        pb = new.run_perception(img.copy(), pipe)
        if not (same(pa["detections"], pb["detections"]) and pa["status"] == pb["status"]):
            stats["perception_mismatch"].append(name)

        # A broken first policy must fall back to the rules without changing anything.
        rb = loop.run_loop(img.copy(), pipe, chain=chain_broken)
        rr = loop.run_loop(img.copy(), pipe)
        if not same(rb.detections, rr.detections):
            stats["fallback_mismatch"].append(name)

        # Budget 0: no discretionary tool runs; equals the plain-model ablation row.
        r0 = loop.run_loop(img.copy(), pipe, budget_ms=0)
        if plain_flags is not None:
            exp, _ = run_config(img.copy(), pipe, plain_flags)
            got = [dict(d, combined_confidence=d.get("raw_combined_confidence", d["combined_confidence"]))
                   for d in r0.detections if not (d.get("novelty") or {}).get("unknown")]
            keys = ("bbox", "class_name", "classifier_class", "combined_confidence", "agreement")
            if not same([{k: d[k] for k in keys} for d in exp], [{k: d[k] for k in keys} for d in got]):
                stats["budget0_mismatch"].append(name)
        if any(line.startswith("Action: Applied") or line.startswith("Action: Zoomed")
               or "majority vote" in line or "deadlock" in line for line in r0.trace):
            stats["budget0_actions"] += 1
    print(f"{label:32s} images {stats['images']:4d} dets {stats['detections']:5d} | "
          f"app {len(stats['app_mismatch'])} perception {len(stats['perception_mismatch'])} "
          f"fallback {len(stats['fallback_mismatch'])} budget0 {len(stats['budget0_mismatch'])} "
          f"(discretionary actions under budget 0: {stats['budget0_actions']}) mismatches", flush=True)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ref", default="8c3a1b0")
    args = ap.parse_args()

    # Call limit: a second call of the same tool on the same key is refused.
    g = loop.Guard(None, 1, time.perf_counter())
    assert g.allow("zoom")[0] and not g.allow("zoom")[0]
    assert g.allow("vote", 0)[0] and g.allow("vote", 1)[0] and not g.allow("vote", 0)[0]

    ref = load_reference(args.ref)
    man = json.loads(MAN.read_text())
    pol = [image_path(f) for f in sorted(man["files"]["S_policy"])[: args.limit]]
    demo = sorted(p for p in (ROOT / "demo_images").iterdir() if p.suffix.lower() in (".jpg", ".png"))
    plain = {"gate": True, "text": True, "novelty": True}
    assert not any(c[2] == plain for c in CONFIGS)   # plain-model row is not one of A–F

    t0 = time.time()
    out = {"reference_commit": args.ref, "results": {}}
    ds = EnsemblePipeline(profile="deepsea")
    out["results"]["deepsea S_policy"] = check_images([p for p in pol if p], ds, ref, "deepsea S_policy", plain)
    out["results"]["deepsea demo_images"] = check_images(demo, ds, ref, "deepsea demo_images", plain)
    aq = EnsemblePipeline(profile="aquarium")
    out["results"]["aquarium demo_images"] = check_images(demo, aq, ref, "aquarium demo_images", None)
    out["seconds"] = round(time.time() - t0)
    ok = all(not v[k] for v in out["results"].values()
             for k in ("app_mismatch", "perception_mismatch", "fallback_mismatch", "budget0_mismatch")) \
        and all(v["budget0_actions"] == 0 for v in out["results"].values())
    out["identical"] = ok
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n{'IDENTICAL' if ok else 'MISMATCH'} — wrote {OUT.relative_to(ROOT)} ({out['seconds']} s)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
