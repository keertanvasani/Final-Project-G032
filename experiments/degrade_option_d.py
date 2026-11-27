"""Synthetic degradation with option D (PROTOCOL §22). Descriptive.

S_policy images, each of the 7 damage types of experiments/degrade.py at 3 severities,
plus clean (22 conditions). Each condition runs through two systems on identical pixels:
  option_d  image agent at λ_m in front of the deployed chain
  bandit    the pre-D chain (learned, rules, plain)
The verifier (frozen combiner, S_policy thresholds) is applied offline to the logged boxes,
exactly as in experiments/option_d_check.py. Damaged images are never written to disk.

Logs (gitignored): experiments/results/phase_b_machine/degradation_d/logs/<system>__<cond>.jsonl
Result:            experiments/results/phase_b_machine/degradation_d/degradation_d.json

Usage:
    python experiments/degrade_option_d.py [--limit N] [--out DIR]   # resumes finished logs
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import config  # noqa: E402
from ablation_deepsea import DIVE_RE, MAN, image_path  # noqa: E402
from agentic import loop, policy, taxonomy  # noqa: E402
from agentic import taxonomy as tx  # noqa: E402
from degrade import DEGRADATIONS  # noqa: E402
from eval import harness  # noqa: E402
from log_image_agent import harness_fields  # noqa: E402
from option_d_check import load  # noqa: E402
from pipeline import EnsemblePipeline  # noqa: E402
from week3_verifier import decide, dive_risk  # noqa: E402

R = ROOT / "experiments" / "results" / "phase_b_machine"
LAMBDA_M = 0.00892128865561972
CHAINS = {"option_d": ("image", "learned", "rules", "plain"), "bandit": ("learned", "rules", "plain")}
N_BOOT = 2000
CONDITIONS = [("clean", 0)] + [(n, s) for n in DEGRADATIONS for s in (1, 2, 3)]


def cname(name, sev):
    return name if sev == 0 else f"{name}_{sev}"


def damage(img, fn, name, sev):
    if sev == 0:
        return img
    np.random.seed(zlib.crc32(f"{fn}|{name}|{sev}".encode()) & 0xFFFFFFFF)   # same pixels for both systems
    return DEGRADATIONS[name](img, sev - 1)


def log_condition(files, pipe, chain, name, sev, dest, classes, gt_all):
    tmp = dest.with_suffix(".part")
    nb = 0
    with tmp.open("w", encoding="utf-8") as fh:
        for k, fn in enumerate(files):
            img = damage(cv2.imread(str(image_path(fn))), fn, name, sev)
            run = loop.run_loop(img, pipe, chain=chain)
            fields = [harness_fields(d) for d in run.detections]
            crops = [EnsemblePipeline._extract_crop(run.work_image, d["bbox"], padding=0.05) for d in run.detections]
            probs = taxonomy.full_probs(crops, pipe.classifier) if crops else []
            matched = harness.match_and_classify([dict(f, unknown=False) for f in fields], gt_all.get(fn, []))
            h, w = img.shape[:2]
            boxes = []
            for f, p, m in zip(fields, probs, matched):
                b = f["bbox"]
                boxes.append(dict(f, probs=[float(x) for x in p],
                                  area=max(b[2] - b[0], 1.0) * max(b[3] - b[1], 1.0) / float(w * h),
                                  outcome=m["outcome"], gt_class=m["gt_class"], matched=m["matched_gt"] is not None))
            d5 = next((d for d in run.decisions if d["decision"] == "rerun"), None)
            d6 = next((d for d in run.decisions if d["decision"] == "commit"), None)
            nb += len(boxes)
            fh.write(json.dumps({"file": fn, "dive": DIVE_RE.match(fn).group(1), "status": run.status,
                                 "rerun": d5["value"] if d5 else None, "kept": d6["value"] if d6 else "first",
                                 "classes": classes if k == 0 else None, "boxes": boxes}) + "\n")
    tmp.replace(dest)
    return nb


def analyse(path, spec, img_per_dive, samples):
    boxes, dives, P, Y = load(path, spec)
    recs = [json.loads(line) for line in path.open(encoding="utf-8")]
    n_img = sum(img_per_dive.values())
    thr = spec["thresholds"]
    out = {"final_boxes": len(boxes), "matched_boxes": int(sum(b["matched"] for b in boxes)),
           "status": dict(Counter(r["status"] for r in recs)),
           "rerun_share": float(np.mean([r["rerun"] is not None for r in recs])),
           "kept_second_share": float(np.mean([r["kept"] == "second" for r in recs]))}
    chs = {}
    for key, mode, alpha, levels in (("hier_0.2", "hierarchical", 0.2, [0, 1, 2, 3]),
                                     ("hier_0.1", "hierarchical", 0.1, [0, 1, 2, 3]),
                                     ("flat_0.2", "flat", 0.2, [0])):
        ch = decide(P, thr[mode][str(alpha)], levels) if len(boxes) else np.zeros(0, int)
        chs[key] = ch
        ok = np.array([ch[i] >= 0 and Y[i, ch[i]] for i in range(len(ch))], bool)
        by = defaultdict(list)
        for i, d in enumerate(dives):
            by[d].append(float(ch[i] >= 0 and not Y[i, ch[i]]))
        pdr = {d: float(np.mean(v)) for d, v in by.items()}
        share = [np.mean(v) <= alpha for v in ([pdr[d] for d in s if d in pdr] for s in samples) if v]
        out[key] = {"risk": dive_risk(ch, Y, dives, np.arange(len(ch)))[0] if len(ch) else 0.0,
                    "share_resamples_le_alpha": float(np.mean(share)) if share else 1.0,
                    "answers_per_image": float((ch >= 0).sum() / n_img),
                    "correct_per_image": float(ok.sum() / n_img),
                    "species_correct_per_image": float((ok & (ch == 0)).sum() / n_img),
                    "level_mix": {("abstain" if j < 0 else tx.LEVELS[j]): int((ch == j).sum()) for j in range(-1, 4)}}
    return out, (dives, chs["hier_0.2"], Y)


def paired(a, b, img_per_dive, samples):
    """option D (a) − bandit (b): risk and correct per image at hier α 0.2, dive bootstrap."""
    def per(d_ch_y):
        dives, ch, Y = d_ch_y
        wr, ok = defaultdict(list), defaultdict(float)
        for i, d in enumerate(dives):
            wr[d].append(float(ch[i] >= 0 and not Y[i, ch[i]]))
            ok[d] += float(ch[i] >= 0 and Y[i, ch[i]])
        return {d: float(np.mean(v)) for d, v in wr.items()}, ok
    (ra, oa), (rb, ob) = per(a), per(b)
    dr, dc = [], []
    for s in samples:
        ni = sum(img_per_dive[d] for d in s)
        va, vb = [ra[d] for d in s if d in ra], [rb[d] for d in s if d in rb]
        dr.append((np.mean(va) if va else 0.0) - (np.mean(vb) if vb else 0.0))
        dc.append((sum(oa.get(d, 0) for d in s) - sum(ob.get(d, 0) for d in s)) / ni)
    dr, dc = np.sort(dr), np.sort(dc)
    ci = lambda v: [float(v[int(0.025 * N_BOOT)]), float(v[int(0.975 * N_BOOT) - 1])]  # noqa: E731
    return {"risk_ci95": ci(dr), "correct_ci95": ci(dc)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(R / "degradation_d"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    logs = out_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    man = json.loads(MAN.read_text())
    harness.KNOWN_CLASSES = set(man["label_space"]["known"])
    gt_all = harness.load_ground_truth()
    files = sorted(man["files"]["S_policy"])[: args.limit]
    files = [f for f in files if image_path(f) and cv2.imread(str(image_path(f)), cv2.IMREAD_REDUCED_GRAYSCALE_8) is not None]
    pipe = EnsemblePipeline(profile="deepsea")
    classes = list(pipe.classifier.class_names)
    config.AGENT_VERIFIER = None                 # applied offline; it does not change the boxes
    config.AGENT_IMAGE_LAMBDA = LAMBDA_M
    chains = {k: policy.build_chain(v) for k, v in CHAINS.items()}
    t0 = time.time()
    for name, sev in CONDITIONS:
        for sysname, chain in chains.items():
            dest = logs / f"{sysname}__{cname(name, sev)}.jsonl"
            if dest.exists():
                continue
            nb = log_condition(files, pipe, chain, name, sev, dest, classes, gt_all)
            print(f"{sysname:8s} {cname(name, sev):16s} {nb:5d} boxes  ({time.time() - t0:.0f}s)", flush=True)

    spec = json.loads((R / "week6" / "verifier_deploy.json").read_text())
    img_per_dive = Counter(DIVE_RE.match(f).group(1) for f in files)
    ud = sorted(img_per_dive)
    rng = random.Random(0)
    samples = [[ud[rng.randrange(len(ud))] for _ in ud] for _ in range(N_BOOT)]
    res = {"protocol": "PROTOCOL §22", "images": len(files), "dives": len(ud), "conditions": {}}
    for name, sev in CONDITIONS:
        c = cname(name, sev)
        r, keep = {}, {}
        for sysname in CHAINS:
            r[sysname], keep[sysname] = analyse(logs / f"{sysname}__{c}.jsonl", spec, img_per_dive, samples)
        r["option_d_minus_bandit"] = paired(keep["option_d"], keep["bandit"], img_per_dive, samples)
        h = r["option_d"]["hier_0.2"]
        r["reading_22_5"] = {"holds": h["risk"] <= 0.20, "robust": h["share_resamples_le_alpha"] >= 0.90}
        res["conditions"][c] = r
        b = r["bandit"]["hier_0.2"]
        print(f"{c:16s} D risk {h['risk']:.3f} ({h['share_resamples_le_alpha']:.0%})  correct/img {h['correct_per_image']:.3f}"
              f"  | bandit risk {b['risk']:.3f} correct/img {b['correct_per_image']:.3f}"
              f"  | D re-run {r['option_d']['rerun_share']:.0%}")
    (out_dir / "degradation_d.json").write_text(json.dumps(res, indent=2) + "\n")
    print(f"wrote {out_dir / 'degradation_d.json'}")


if __name__ == "__main__":
    main()
