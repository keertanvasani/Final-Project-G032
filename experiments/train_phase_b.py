"""Phase B — on-domain detector retraining on the dive-disjoint J_fit split.

Trains the PROTOCOL.md §11.7 baseline family on Jedi's 8 known taxa:

    yolov8n, yolov8s   3 seeds each   (the Phase A architectures, retrained)
    rtdetr-l           3 seeds        (transformer family, so the comparison
                                       is not confined to one lineage)

Model selection uses the J_fit val slice carved by `coco_to_yolo.py`; `J_cal`
is never touched during training (§11.5).

Windows note: Ultralytics' dataloader workers deadlock on this platform, so
`workers=0` is the default here (MIGRATION.md).

Usage:
    python experiments/train_phase_b.py --smoke          # 2 epochs, one seed
    python experiments/train_phase_b.py                  # the full 9 runs
    python experiments/train_phase_b.py --arch yolov8n --seeds 0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).parent / "yolo_phase_b" / "data.yaml"
OUTDIR = ROOT / "models" / "phase_b"

# Batch sizes tuned for 8 GB (RTX 5050). RT-DETR's decoder and attention maps
# cost far more activation memory than a YOLO head at the same resolution, so
# it gets a much smaller batch rather than a reduced image size — shrinking
# imgsz instead would change the operating point and break comparability.
ARCHS = {
    "yolov8n": {"weights": "yolov8n.pt", "batch": 16, "kind": "yolo"},
    "yolov8s": {"weights": "yolov8s.pt", "batch": 12, "kind": "yolo"},
    "rtdetr-l": {"weights": "rtdetr-l.pt", "batch": 4, "kind": "rtdetr"},
}


def build(kind: str, weights: str):
    from ultralytics import RTDETR, YOLO
    return (RTDETR if kind == "rtdetr" else YOLO)(weights)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", nargs="*", default=list(ARCHS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=None,
                    help="override the per-architecture default")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--smoke", action="store_true",
                    help="2 epochs, seed 0 only — checks the plumbing")
    args = ap.parse_args()

    if not DATA.is_file():
        raise SystemExit(f"Missing {DATA}. Run experiments/coco_to_yolo.py first.")

    epochs = 2 if args.smoke else args.epochs
    seeds = [0] if args.smoke else args.seeds

    unknown = [a for a in args.arch if a not in ARCHS]
    if unknown:
        raise SystemExit(f"Unknown architecture(s): {unknown}. Choose from {list(ARCHS)}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    runs: list[dict] = []

    for arch in args.arch:
        cfg = ARCHS[arch]
        for seed in seeds:
            name = f"{arch}_seed{seed}" + ("_smoke" if args.smoke else "")
            print(f"\n{'='*64}\n  {arch}  seed={seed}  epochs={epochs}\n{'='*64}")
            model = build(cfg["kind"], cfg["weights"])
            t0 = time.time()
            model.train(
                data=str(DATA.resolve()),
                epochs=epochs,
                imgsz=args.imgsz,
                batch=args.batch or cfg["batch"],
                seed=seed,
                deterministic=True,
                workers=args.workers,
                device=args.device,
                project=str(OUTDIR),
                name=name,
                exist_ok=True,
                patience=args.patience,
                val=True,
                plots=True,
            )
            elapsed = time.time() - t0
            m = model.trainer.metrics if hasattr(model, "trainer") else {}
            rec = {
                "arch": arch,
                "seed": seed,
                "epochs": epochs,
                "imgsz": args.imgsz,
                "batch": args.batch or cfg["batch"],
                "minutes": round(elapsed / 60, 1),
                "weights": str((OUTDIR / name / "weights" / "best.pt").resolve()),
                "metrics": {k: float(v) for k, v in m.items()
                            if isinstance(v, (int, float))},
            }
            runs.append(rec)
            print(f"  done in {rec['minutes']} min -> {rec['weights']}")

            out = OUTDIR / ("runs_smoke.json" if args.smoke else "runs.json")
            out.write_text(json.dumps(runs, indent=2) + "\n")

    print(f"\n{len(runs)} run(s) complete.")
    for r in runs:
        mp = r["metrics"].get("metrics/mAP50(B)")
        print(f"  {r['arch']:<10} seed {r['seed']}  {r['minutes']:>6.1f} min"
              + (f"  mAP50={mp:.4f}" if mp is not None else ""))


if __name__ == "__main__":
    main()
