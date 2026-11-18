"""Run the live agent loop with the learned policy on S_policy and score it.

Confirms that the replay in fit_bandit_policy.py predicts what the real loop
does with chain ("learned", "rules", "plain"), and measures real wall time
for the rules and learned chains. Dev split only.

Usage:
    python experiments/check_bandit_live.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from ablation_deepsea import MAN, image_path  # noqa: E402
from agentic.loop import run_loop  # noqa: E402
from agentic.policy import build_chain  # noqa: E402
from eval import harness  # noqa: E402
from pipeline import EnsemblePipeline  # noqa: E402

OUT = ROOT / "experiments" / "results" / "phase_b_machine" / "step3" / "live_check.json"


def main() -> None:
    man = json.loads(MAN.read_text())
    harness.KNOWN_CLASSES = set(man["label_space"]["known"])
    gt_all = harness.load_ground_truth()
    files = sorted(man["files"]["S_policy"])
    pipe = EnsemblePipeline(profile="deepsea")
    imgs = [(fn, cv2.imread(str(image_path(fn)))) for fn in files]
    run_loop(imgs[0][1], pipe)                                   # warm-up
    out = {}
    for name, chain in (("rules", ("rules", "plain")), ("learned", ("learned", "rules", "plain"))):
        ch = build_chain(chain)
        recs, kgt, ms, fallbacks = [], 0, 0.0, 0
        for fn, img in imgs:
            t = time.perf_counter()
            run = run_loop(img, pipe, chain=ch)
            ms += (time.perf_counter() - t) * 1000.0
            fallbacks += sum("falling back" in g for g in run.guardrails)
            dets = [d for d in run.detections if not (d.get("novelty") or {}).get("unknown")]
            gts = gt_all.get(fn, [])
            recs += harness.match_and_classify(dets, gts)
            kgt += harness.known_gt_count(gts)
        tp = sum(r["outcome"] == harness.TP for r in recs)
        out[name] = {"reported": len(recs), "precision": tp / len(recs), "recall": tp / kgt,
                     "ms_per_image": ms / len(imgs), "fallbacks": fallbacks,
                     "outcomes": dict(Counter(r["outcome"] for r in recs))}
        o = out[name]
        print(f"{name:8s} rep {o['reported']} P {o['precision']:.3f} R {o['recall']:.3f} "
              f"{o['ms_per_image']:.1f} ms/img  fallbacks {fallbacks}", flush=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
