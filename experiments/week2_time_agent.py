"""Week 2: live wall time per image, deployed chain vs image agent in front (S_cal, 300 images).

Usage:
    python experiments/week2_time_agent.py [--limit 300]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import config  # noqa: E402
from ablation_deepsea import MAN, image_path  # noqa: E402
from agentic import loop, policy  # noqa: E402
from pipeline import EnsemblePipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args()
    files = sorted(json.loads(MAN.read_text())["files"]["S_cal"])[: args.limit]
    imgs = [cv2.imread(str(image_path(f))) for f in files]
    pipe = EnsemblePipeline(profile="deepsea")
    chains = {"deployed": policy.build_chain(tuple(config.AGENT_POLICY_CHAIN)),
              "image agent": policy.build_chain(("image",) + tuple(config.AGENT_POLICY_CHAIN))}
    for im in imgs[:10]:                                       # warm-up
        for c in chains.values():
            loop.run_loop(im.copy(), pipe, chain=c)
    out = {}
    for name, c in chains.items():
        ms, reruns = [], 0
        for im in imgs:
            t = time.perf_counter()
            run = loop.run_loop(im.copy(), pipe, chain=c)
            ms.append((time.perf_counter() - t) * 1000.0)
            reruns += any(d["decision"] == "rerun" and d["value"] for d in run.decisions)
        ms = np.array(ms)
        out[name] = {"mean_ms": float(ms.mean()), "median_ms": float(np.median(ms)),
                     "p90_ms": float(np.percentile(ms, 90)), "rerun_rate": reruns / len(imgs)}
        print(f"{name:12s} mean {ms.mean():6.1f}  median {np.median(ms):6.1f}  p90 {np.percentile(ms, 90):6.1f} ms"
              f"  re-run {100 * reruns / len(imgs):.0f}%")
    d = ROOT / "experiments" / "results" / "phase_b_machine" / "week2"
    d.mkdir(parents=True, exist_ok=True)
    (d / "timing.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
