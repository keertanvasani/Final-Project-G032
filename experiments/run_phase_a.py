"""Phase A — evaluate the frozen aquarium baseline on deep-sea imagery.

Two stages, separated so the expensive one runs once:

  1. CACHE  — run perception over a split, match to GT, write one row per
              emitted detection plus per-image metadata to disk. Resumable:
              re-running skips images already cached.
  2. REPORT — load the cache and compute selective-prediction metrics with
              dive-cluster bootstrap. Cheap; iterate freely.

The motivating question (PROTOCOL.md §10): a system trained in aquaria, run
on real deep-sea imagery where two-thirds of annotations are taxa it has no
label for — does it abstain, or confidently mislabel them?

Usage:
    python experiments/run_phase_a.py cache               # full J_test
    python experiments/run_phase_a.py cache --limit 150   # quick validation
    python experiments/run_phase_a.py report
    python experiments/run_phase_a.py cache --split J_cal
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from experiments.eval import harness, metrics  # noqa: E402

SPLITS_FILE = ROOT / "experiments" / "splits" / "phase_a_splits.json"
CACHE_DIR = ROOT / "experiments" / "results" / "phase_a"
DIVE_RE = re.compile(r"^((?:[A-Z]+|\d+K)\d{3,4})")


def dive_of(fname: str) -> str:
    return DIVE_RE.match(fname).group(1)


def image_path(fname: str) -> Path | None:
    for split in harness.SOURCE_SPLITS:
        p = harness.DATASET / split / "images" / fname
        if p.is_file():
            return p
    return None


# ── Stage 1: cache ───────────────────────────────────────────────────────

def cache(split: str, limit: int | None) -> None:
    import cv2
    from pipeline import EnsemblePipeline
    from agent_controller import run_perception

    manifest = json.loads(SPLITS_FILE.read_text())
    if split not in manifest["files"]:
        raise SystemExit(f"Unknown split {split}; have {list(manifest['files'])}")
    files = manifest["files"][split]
    if limit:
        files = files[:limit]

    gt = harness.load_ground_truth()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    det_path = CACHE_DIR / f"{split}_detections.jsonl"
    img_path = CACHE_DIR / f"{split}_images.jsonl"

    done = set()
    if det_path.exists() and not limit:
        for line in img_path.read_text().splitlines():
            done.add(json.loads(line)["file"])
        print(f"Resuming: {len(done)} images already cached")

    pipe = EnsemblePipeline()
    t0 = time.time()
    mode = "a" if done else "w"
    n_new = 0

    with det_path.open(mode) as df, img_path.open(mode) as imf:
        for i, fname in enumerate(files, 1):
            if fname in done:
                continue
            path = image_path(fname)
            if path is None:
                continue
            img = cv2.imread(str(path))
            if img is None:
                continue

            out = run_perception(img, pipe)
            dets = out["detections"]
            g = gt.get(fname, [])
            records = harness.match_and_classify(dets, g) if dets else []

            for r in records:
                r["file"] = fname
                r["dive"] = dive_of(fname)
                df.write(json.dumps(r) + "\n")

            imf.write(json.dumps({
                "file": fname,
                "dive": dive_of(fname),
                "status": out["status"],
                "n_detections": len(dets),
                "n_gt": len(g),
                "known_gt": harness.known_gt_count(g),
                "blur": round(out["quality"].get("blur_score", 0.0), 2),
            }) + "\n")
            n_new += 1

            if i % 100 == 0:
                rate = n_new / (time.time() - t0 + 1e-9)
                print(f"  {i}/{len(files)}  ({rate:.1f} img/s)", flush=True)

    print(f"\nCached {n_new} new images to {det_path.relative_to(ROOT)}")
    print(f"Wall clock: {time.time() - t0:.0f}s")


# ── Stage 2: report ──────────────────────────────────────────────────────

def report(split: str) -> None:
    det_path = CACHE_DIR / f"{split}_detections.jsonl"
    img_path = CACHE_DIR / f"{split}_images.jsonl"
    if not det_path.exists():
        raise SystemExit(f"No cache for {split}. Run: run_phase_a.py cache --split {split}")

    records = [json.loads(l) for l in det_path.read_text().splitlines()]
    images = [json.loads(l) for l in img_path.read_text().splitlines()]

    total_known_gt = sum(im["known_gt"] for im in images)
    n_images = len(images)
    n_det = len(records)

    # index by dive for bootstrap
    from collections import Counter, defaultdict
    recs_by_dive: dict[str, list] = defaultdict(list)
    kgt_by_dive: dict[str, int] = defaultdict(int)
    for r in records:
        recs_by_dive[r["dive"]].append(r)
    for im in images:
        kgt_by_dive[im["dive"]] += im["known_gt"]

    status = Counter(im["status"] for im in images)
    outcomes = Counter(r["outcome"] for r in records)

    print(f"\n{'='*64}\nPHASE A — frozen aquarium baseline on Jedi {split}\n{'='*64}")
    print(f"images {n_images}   dives {len(recs_by_dive)}   "
          f"detections {n_det}   known GT {total_known_gt}")
    print(f"\nperception status:")
    for k, v in status.most_common():
        print(f"   {k:<12}{v:>6}  ({v/n_images:.1%})")
    print(f"\nemitted-detection outcomes (all confidences):")
    for k in [harness.TP, harness.CLS_ERR, harness.OOD_ERR, harness.LOC_ERR, harness.BG_FP]:
        v = outcomes.get(k, 0)
        print(f"   {k:<10}{v:>6}  ({v/n_det:.1%})" if n_det else f"   {k:<10}{v:>6}")

    curve = metrics.risk_coverage_curve(records, total_known_gt)

    # baseline operating point: its deployed acceptance threshold
    tau = config.REVIEW_CONF_THRESHOLD
    op = metrics.operating_point(records, total_known_gt, tau)
    print(f"\nbaseline operating point (accept combined_conf >= {tau}):")
    print(f"   accepted           {op['accepted']}")
    print(f"   selective risk     {op['selective_risk']:.3f}")
    print(f"   selective recall   {op['selective_recall']:.3f}")
    print(f"   coverage (local)   {op['coverage_policy_local']:.3f}")
    print(f"   OOD-error rate     {op['ood_err_rate']:.3f}  ({op['ood_err_count']} of {op['accepted']} accepted)")

    aurc_recall = metrics.aurc(curve, "selective_recall")
    aurc_cov = metrics.aurc(curve, "coverage_policy_local")
    print(f"\nAURC (lower better):")
    print(f"   vs selective recall (GT-anchored)  {aurc_recall:.4f}")
    print(f"   vs coverage (policy-local)         {aurc_cov:.4f}")

    print(f"\nrisk–recall operating points:")
    for tgt in (0.9, 0.7, 0.5):
        print(f"   recall {tgt:.0%} -> risk {metrics.risk_at_coverage(curve,'selective_recall',tgt):.3f}")
    for mr in (0.10, 0.05):
        print(f"   risk <= {mr:.0%} -> recall {metrics.coverage_at_risk(curve,'selective_recall',mr):.3f}")

    # bootstrap the headline numbers
    print(f"\ndive-cluster bootstrap (95% CI, 1000 resamples):")
    for name, fn in [
        ("OOD-err rate @ baseline tau",
         lambda rs, k: metrics.operating_point(rs, k, tau)["ood_err_rate"]),
        ("selective risk @ baseline tau",
         lambda rs, k: metrics.operating_point(rs, k, tau)["selective_risk"]),
        ("selective recall @ baseline tau",
         lambda rs, k: metrics.operating_point(rs, k, tau)["selective_recall"]),
        ("AURC vs recall",
         lambda rs, k: metrics.aurc(metrics.risk_coverage_curve(rs, k), "selective_recall")),
    ]:
        b = metrics.dive_bootstrap(recs_by_dive, kgt_by_dive, fn, n_boot=1000)
        print(f"   {name:<32} {b['point']:.3f}  [{b['ci_low']:.3f}, {b['ci_high']:.3f}]")

    out = {
        "split": split, "images": n_images, "dives": len(recs_by_dive),
        "detections": n_det, "known_gt": total_known_gt,
        "status": dict(status), "outcomes": dict(outcomes),
        "baseline_operating_point": op,
        "aurc": {"vs_recall": aurc_recall, "vs_coverage": aurc_cov},
        "risk_coverage_curve": curve,
    }
    out_path = CACHE_DIR / f"{split}_report.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {out_path.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["cache", "report"])
    ap.add_argument("--split", default="J_test")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.stage == "cache":
        cache(args.split, args.limit)
    else:
        report(args.split)


if __name__ == "__main__":
    main()
