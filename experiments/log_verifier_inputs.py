"""Week 3: log what the verifier sees for every final box of the deployed agent.

Runs the deployed loop (agentic/loop.run_loop, app mode, AGENT_POLICY_CHAIN)
on every image. For each final box, including boxes flagged Unknown, it stores
the harness fields, the classifier's full 5%-crop probability vector (the
classifier's own input transform), the energy, the detector class, the box
area and the matched GT class. S_cal only until the S_policy test is
registered (PROTOCOL §14).

Usage:
    python experiments/log_verifier_inputs.py [--limit N]
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

from ablation_deepsea import DIVE_RE, MAN, image_path  # noqa: E402
from agentic import loop, taxonomy  # noqa: E402
from eval import harness  # noqa: E402
from log_image_agent import harness_fields  # noqa: E402
from pipeline import EnsemblePipeline  # noqa: E402

OUT = ROOT / "experiments" / "results" / "phase_b_machine" / "week3"
ALLOWED = ("S_cal", "S_policy")      # S_policy opened by PROTOCOL §15


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="S_cal")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.split not in ALLOWED:
        raise SystemExit(f"{args.split} is not open yet (register the test in PROTOCOL.md first)")
    man = json.loads(MAN.read_text())
    harness.KNOWN_CLASSES = set(man["label_space"]["known"])
    gt_all = harness.load_ground_truth()
    files = sorted(man["files"][args.split])[: args.limit]
    pipe = EnsemblePipeline(profile="deepsea")
    classes = list(pipe.classifier.class_names)
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{args.split}_verifier.jsonl"
    t0, nb = time.time(), 0
    with dest.open("w", encoding="utf-8") as fh:
        for k, fn in enumerate(files):
            img = cv2.imread(str(image_path(fn)))
            if img is None:
                continue
            run = loop.run_loop(img, pipe)
            fields = [harness_fields(d) for d in run.detections]
            crops = [EnsemblePipeline._extract_crop(run.work_image, d["bbox"], padding=0.05) for d in run.detections]
            probs = taxonomy.full_probs(crops, pipe.classifier)
            gts = gt_all.get(fn, [])
            matched = harness.match_and_classify([dict(f, unknown=False) for f in fields], gts)
            h, w = img.shape[:2]
            boxes = []
            for f, p, m in zip(fields, probs, matched):
                b = f["bbox"]
                boxes.append(dict(f, probs=[float(x) for x in p],
                                  area=max(b[2] - b[0], 1.0) * max(b[3] - b[1], 1.0) / float(w * h),
                                  outcome=m["outcome"], gt_class=m["gt_class"],
                                  matched=m["matched_gt"] is not None))
            nb += len(boxes)
            fh.write(json.dumps({"file": fn, "dive": DIVE_RE.match(fn).group(1), "status": run.status,
                                 "classes": classes if k == 0 else None, "boxes": boxes}) + "\n")
            if (k + 1) % 200 == 0:
                print(f"{k + 1}/{len(files)} images, {nb} boxes ({time.time() - t0:.0f}s)", flush=True)
    print(f"wrote {dest.relative_to(ROOT)}: {len(files)} images, {nb} boxes ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
