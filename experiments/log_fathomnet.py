"""External validation (FathomNet, PROTOCOL §18): run the frozen system's own loggers.

    prepare  resize every downloaded frame to 480 px wide (§18.3), write the
             ground truth in the harness format and a split manifest ("F_ext")
    verifier experiments/log_verifier_inputs.py   (deployed agent's final boxes)
    agent    experiments/log_image_agent.py       (full-information image-agent logs)
    bandit   experiments/log_bandit_outcomes.py   (Step 3 rules / deployed bandit logs)

The loggers run unchanged; only their image source, dive parser, ground truth and
output folder are redirected to FathomNet. GT classes: a §14.1 leaf, "unseen:<node>",
"Man-made", or "excluded" (§18.2; matched detections on these are ignored later).

Usage:
    python experiments/log_fathomnet.py prepare
    python experiments/log_fathomnet.py verifier|agent|bandit [--limit N]
"""
from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

EXT = ROOT / "data_external" / "fathomnet"
IMG480 = EXT / "images480"
GT_PATH = EXT / "meta" / "gt480.json"
MAN_PATH = EXT / "meta" / "split_manifest.json"
RES = ROOT / "experiments" / "results" / "external_fathomnet"
WIDTH = 480
FN_DIVE = re.compile(r"^(.+?)__")


def fname(m):
    return f"{m['dive'].replace(' ', '-')}__{m['uuid']}.png"


def prepare():
    from ablation_deepsea import MAN
    man = json.loads((EXT / "meta" / "manifest.json").read_text())["images"]
    labels = json.loads((RES / "concept_labels.json").read_text())
    recs = {json.loads(l)["uuid"]: json.loads(l) for l in (EXT / "meta" / "image_records.jsonl").open()}
    IMG480.mkdir(parents=True, exist_ok=True)
    gt, files, part_a, part_b, missing = {}, [], [], [], []
    for m in man:
        src = EXT / "images" / f"{m['uuid']}.png"
        img = cv2.imread(str(src))
        if img is None:
            missing.append(m["uuid"])
            continue
        h, w = img.shape[:2]
        s = WIDTH / w
        small = cv2.resize(img, (WIDTH, round(h * s)), interpolation=cv2.INTER_AREA)
        fn = fname(m)
        cv2.imwrite(str(IMG480 / fn), small)
        gt[fn] = [{"bbox": [b["x"] * s, b["y"] * s, (b["x"] + b["w"]) * s, (b["y"] + b["h"]) * s],
                   "class": labels[b["concept"]]["label"] or "excluded", "concept": b["concept"]}
                  for b in recs[m["uuid"]]["boxes"]]
        files.append(fn)
        if "partB" in m["why"]:
            part_b.append(fn)
        if any(t != "partB" for t in m["why"]):
            part_a.append(fn)
    GT_PATH.write_text(json.dumps(gt))
    jedi = json.loads(MAN.read_text())
    MAN_PATH.write_text(json.dumps({"label_space": jedi["label_space"],
                                    "files": {"F_ext": sorted(files)},
                                    "parts": {"A": sorted(part_a), "B": sorted(part_b)}}, indent=1))
    print(f"prepared {len(files)} images (Part A {len(part_a)}, Part B {len(part_b)}), missing {len(missing)}")


def image_path(fn):
    p = IMG480 / fn
    return p if p.is_file() else None


def load_gt():
    return json.loads(GT_PATH.read_text())


def run(which, extra):
    from eval import harness
    mod = importlib.import_module({"verifier": "log_verifier_inputs", "agent": "log_image_agent",
                                   "bandit": "log_bandit_outcomes"}[which])
    harness.load_ground_truth = load_gt
    mod.MAN, mod.image_path, mod.DIVE_RE = MAN_PATH, image_path, FN_DIVE
    mod.OUT = RES / "logs"
    if hasattr(mod, "ALLOWED"):
        mod.ALLOWED = ("F_ext",)
    sys.argv = [mod.__file__, "--split", "F_ext", *extra]
    mod.main()


if __name__ == "__main__":
    cmd, rest = sys.argv[1], sys.argv[2:]
    prepare() if cmd == "prepare" else run(cmd, rest)
