"""Train the Phase B species classifier on deep-sea crops.

The Phase A classifier is aquarium-trained. Every OOD result so far is computed
from its penultimate features, which means the novelty score is measured in a
representation that has never seen the deep sea — the most likely explanation
for near-OOD sitting at 0.673 out of sample. This retrains the same
architecture on Jedi crops so that score can be recomputed in a domain-relevant
feature space.

Deliberately kept separate from `train_classifier.py`, which reads its paths
from `config` and would overwrite `IMAGE_V1_BASELINE`'s frozen weights. Nothing
here touches the Phase A artefacts.

Architecture, input size and normalisation match the Phase A classifier exactly
(ResNet-18, 224 px, ImageNet statistics) so the two feature spaces are
comparable and any change in OOD performance is attributable to the training
domain rather than to the model.

Crops come from S_fit (train) and S_modelsel (val) — the same two splits the
detector fits on, with images containing held-out taxa already excluded
(PROTOCOL §11.3), so unseen taxa stay unseen in the classifier as well. This
matters: a classifier that had seen shrimp would make the novelty claim false
even if the detector never did.

Usage:
    python experiments/train_classifier_phase_b.py --seeds 0 1 2
    python experiments/train_classifier_phase_b.py --smoke
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATASET = ROOT / "JediOrganismDetectionDataset"
SOURCE_SPLITS = ["coco_train_data", "coco_valid_data", "coco_test_data"]
MANIFEST = Path(__file__).parent / "splits" / "phase_b_splits.json"
CROPS = Path(__file__).parent / "classifier_crops_phase_b"
OUTDIR = ROOT / "models" / "phase_b"
PAD = 0.05
IMG = 224


def image_path(fname: str) -> Path | None:
    for sp in SOURCE_SPLITS:
        p = DATASET / sp / "images" / fname
        if p.is_file():
            return p
    return None


def build_crops(known: set[str], force: bool = False) -> dict[str, int]:
    """Extract GT crops of the KNOWN taxa into an ImageFolder tree."""
    import cv2

    man = json.loads(MANIFEST.read_text())
    clean = set(man["contamination"]["s_fit_images_clean"])
    clean_sel = set(man["contamination"]["s_modelsel_images_clean"])
    want = {"train": clean, "val": clean_sel}

    if CROPS.exists() and not force:
        counts = {s: sum(1 for _ in (CROPS / s).rglob("*.jpg")) for s in want}
        if all(v > 0 for v in counts.values()):
            print(f"Reusing existing crops: {counts}")
            return counts

    boxes: dict[str, list] = defaultdict(list)
    for sp in SOURCE_SPLITS:
        coco = json.loads((DATASET / sp / "coco.json").read_text())
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        imgs = {im["id"]: Path(im["file_name"]).name for im in coco["images"]}
        for a in coco["annotations"]:
            fn = imgs.get(a["image_id"])
            cn = cats.get(a["category_id"])
            if fn and cn in known:
                boxes[fn].append((cn, a["bbox"]))

    counts: Counter = Counter()
    for split, files in want.items():
        for c in sorted(known):
            (CROPS / split / c.replace("/", "_")).mkdir(parents=True, exist_ok=True)
        n = 0
        for fn in sorted(files):
            if fn not in boxes:
                continue
            p = image_path(fn)
            if p is None:
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            H, W = img.shape[:2]
            for k, (cn, bb) in enumerate(boxes[fn]):
                x, y, w, h = bb
                px, py = w * PAD, h * PAD
                x0, y0 = max(int(x - px), 0), max(int(y - py), 0)
                x1, y1 = min(int(x + w + px), W), min(int(y + h + py), H)
                if x1 - x0 < 8 or y1 - y0 < 8:
                    continue
                crop = img[y0:y1, x0:x1]
                out = CROPS / split / cn.replace("/", "_") / f"{Path(fn).stem}_{k}.jpg"
                cv2.imwrite(str(out), crop)
                n += 1
            counts[split] = n
        print(f"  {split}: {n} crops")
    return dict(counts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--freeze-epochs", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rebuild-crops", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, models, transforms

    man = json.loads(MANIFEST.read_text())
    known = set(man["label_space"]["known"])
    print(f"Phase B classifier — {len(known)} known taxa\n")
    build_crops(known, force=args.rebuild_crops)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tf_train = transforms.Compose([
        transforms.Resize((IMG, IMG)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((IMG, IMG)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    ds_tr = datasets.ImageFolder(CROPS / "train", tf_train)
    ds_va = datasets.ImageFolder(CROPS / "val", tf_val)
    print(f"\ntrain {len(ds_tr)} crops · val {len(ds_va)} crops · "
          f"{len(ds_tr.classes)} classes")
    print("per-class train counts:")
    cc = Counter(y for _, y in ds_tr.samples)
    for i, c in enumerate(ds_tr.classes):
        print(f"  {c:<24}{cc[i]:>6}")

    epochs = 2 if args.smoke else args.epochs
    seeds = [0] if args.smoke else args.seeds
    dev = args.device if torch.cuda.is_available() else "cpu"
    runs = []

    for seed in seeds:
        torch.manual_seed(seed)
        random.seed(seed)
        dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=0)
        dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=0)

        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, len(ds_tr.classes))
        model.to(dev)

        # Phase 1: freeze the stem and early blocks, as the Phase A recipe does.
        for name, p in model.named_parameters():
            if name.startswith(("conv1", "bn1", "layer1", "layer2")):
                p.requires_grad = False

        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                               lr=args.lr)
        crit = nn.CrossEntropyLoss()
        best = 0.0
        name = f"classifier_seed{seed}" + ("_smoke" if args.smoke else "")
        dest = OUTDIR / name
        dest.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        for ep in range(epochs):
            if ep == args.freeze_epochs:
                for p in model.parameters():
                    p.requires_grad = True
                opt = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1)

            model.train()
            tot = corr = 0
            for xb, yb in dl_tr:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad()
                out = model(xb)
                loss = crit(out, yb)
                loss.backward()
                opt.step()
                corr += (out.argmax(1) == yb).sum().item()
                tot += len(yb)
            tr_acc = corr / max(tot, 1)

            model.eval()
            tot = corr = 0
            per = Counter(); per_n = Counter()
            with torch.no_grad():
                for xb, yb in dl_va:
                    xb, yb = xb.to(dev), yb.to(dev)
                    pred = model(xb).argmax(1)
                    corr += (pred == yb).sum().item()
                    tot += len(yb)
                    for p_, y_ in zip(pred.cpu().tolist(), yb.cpu().tolist()):
                        per_n[y_] += 1
                        per[y_] += int(p_ == y_)
            va_acc = corr / max(tot, 1)
            print(f"  seed {seed} ep {ep+1:>3}/{epochs}  train {tr_acc:.4f}  val {va_acc:.4f}")
            if va_acc > best:
                best = va_acc
                torch.save(model.state_dict(), dest / "best_classifier.pth")
                (dest / "class_names.json").write_text(json.dumps(
                    {"class_names": ds_tr.classes,
                     "num_classes": len(ds_tr.classes)}, indent=2))

        macro = sum(per[i] / per_n[i] for i in per_n if per_n[i]) / max(len(per_n), 1)
        rec = {"seed": seed, "epochs": epochs, "best_val_acc": best,
               "macro_recall_last": macro, "minutes": round((time.time()-t0)/60, 1),
               "weights": str(dest / "best_classifier.pth")}
        runs.append(rec)
        print(f"  seed {seed}: best val {best:.4f}  macro-recall {macro:.4f}  "
              f"({rec['minutes']} min)\n")
        (OUTDIR / ("classifier_runs_smoke.json" if args.smoke
                   else "classifier_runs.json")).write_text(
            json.dumps(runs, indent=2) + "\n")

    print("done")
    for r in runs:
        print(f"  seed {r['seed']}  val acc {r['best_val_acc']:.4f}  "
              f"macro-recall {r['macro_recall_last']:.4f}")


if __name__ == "__main__":
    main()
