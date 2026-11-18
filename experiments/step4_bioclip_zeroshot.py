"""Step 4 feasibility: can BioCLIP, zero-shot, be a useful second opinion?

BioCLIP (imageomics/bioclip, ViT-B/16, open weights, runs locally) is asked,
with no training, to label crops. Three questions, all on S_cal:

  A. Ground-truth crops of the 9 known classes: zero-shot accuracy vs our
     ResNet-18, how often each is right where the other is wrong, and the
     accuracy of simply averaging the two models' probabilities.
  B. Known vs held-out-taxa ground-truth crops: is BioCLIP a better
     Known/Unknown score than our energy score?
       - closed score: max similarity to the 9 class prompts (no outside names)
       - open score: probability mass on a fixed list of OTHER marine groups.
         That list is generic but unavoidably names groups that include our
         held-out taxa (jellyfish, squid, ...), so the open score is an
         optimistic estimate and is reported as such.
  C. The agent's own boxes (the deployed learned policy: no enhance, no zoom,
     vote), from the step-3 logs: on boxes the agent labels wrongly (CLS_ERR)
     how often BioCLIP has the right class; on correct boxes (TP) how often it
     disagrees; on unseen animals labelled as known (OOD_ERR) how often the
     open score says "other". Cost per crop on the laptop GPU.

Prompts are fixed in this file. S_test is not used.

Usage:
    python experiments/step4_bioclip_zeroshot.py [--split S_cal]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import open_clip  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from ablation_deepsea import MAN, image_path  # noqa: E402
from eval import harness  # noqa: E402
from pipeline import EnsemblePipeline  # noqa: E402

OUT = ROOT / "experiments" / "results" / "phase_b_machine" / "step4"
LOGS = ROOT / "experiments" / "results" / "phase_b_machine" / "step3"
MODEL = "hf-hub:imageomics/bioclip"

TEMPLATES = ["a photo of {}.", "an underwater photo of {}.", "a deep-sea photo of {}."]
CLASS_NAMES = {
    "coral/sea-anemone": ["coral", "sea anemone", "Anthozoa", "Actiniaria"],
    "crab": ["crab", "Brachyura"],
    "crinoid": ["crinoid", "sea lily", "feather star", "Crinoidea"],
    "fish": ["fish", "deep-sea fish", "Actinopterygii"],
    "machine": ["underwater robot equipment", "remotely operated vehicle", "man-made metal object",
                "rope or cable"],
    "sea-cucumber": ["sea cucumber", "Holothuroidea"],
    "shrimp": ["shrimp", "prawn", "Caridea"],
    "sponge": ["sponge", "glass sponge", "Porifera"],
    "starfish": ["starfish", "sea star", "Asteroidea"],
}
# Generic marine groups outside the 9 classes, for the open-set score.
OTHER = ["jellyfish", "comb jelly", "siphonophore", "salp", "squid", "octopus", "cuttlefish",
         "sea urchin", "sea spider", "squat lobster", "lobster", "isopod", "amphipod", "barnacle",
         "polychaete worm", "sea pen", "tunicate", "sea squirt", "nudibranch", "sea snail", "clam",
         "ray", "skate", "shark", "chimaera", "eel", "brachiopod", "bryozoan"]


def auroc(pos, neg):
    """P(score of a positive > score of a negative), ties half."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.r_[pos, neg]
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):                       # average ranks for ties
        m = allv == v
        ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


class BioClip:
    def __init__(self, classes):
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.pre = open_clip.create_model_and_transforms(MODEL)
        self.model = self.model.to(self.dev).eval()
        self.tok = open_clip.get_tokenizer(MODEL)
        self.classes = classes
        self.t_known = self._embed_groups([CLASS_NAMES[c] for c in classes])
        self.t_other = self._embed_groups([[o] for o in OTHER])
        self.scale = float(self.model.logit_scale.exp())

    @torch.no_grad()
    def _embed_groups(self, groups):
        out = []
        for names in groups:
            t = self.tok([tp.format(n) for n in names for tp in TEMPLATES]).to(self.dev)
            e = self.model.encode_text(t)
            e = e / e.norm(dim=-1, keepdim=True)
            e = e.mean(0)
            out.append(e / e.norm())
        return torch.stack(out)

    @torch.no_grad()
    def score(self, crops_bgr, bs=64):
        sims_k, sims_o = [], []
        for i in range(0, len(crops_bgr), bs):
            x = torch.stack([self.pre(Image.fromarray(c[:, :, ::-1]).convert("RGB"))
                             for c in crops_bgr[i:i + bs]]).to(self.dev)
            f = self.model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            sims_k.append((f @ self.t_known.T).float().cpu())
            sims_o.append((f @ self.t_other.T).float().cpu())
        return torch.cat(sims_k).numpy(), torch.cat(sims_o).numpy()


@torch.no_grad()
def resnet_probs(pipe, crops):
    clf = pipe.classifier
    out = []
    for i in range(0, len(crops), 64):
        x = torch.stack([clf.transform(Image.fromarray(c[:, :, ::-1]).convert("RGB"))
                         for c in crops[i:i + 64]]).to(clf.device)
        lg = clf.model(x)
        out.append(torch.cat([torch.softmax(lg, 1), (-torch.logsumexp(lg, 1))[:, None]], 1).cpu())
    a = torch.cat(out).numpy()
    return a[:, :-1], a[:, -1]


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="S_cal", choices=["S_cal", "S_policy"])
    args = ap.parse_args()

    man = json.loads(MAN.read_text())
    known = man["label_space"]["known"]
    harness.KNOWN_CLASSES = set(known)
    held = set(man["label_space"]["held_out_unseen"])
    gt_all = harness.load_ground_truth()
    files = sorted(man["files"][args.split])

    pipe = EnsemblePipeline(profile="deepsea")
    r_classes = list(pipe.classifier.class_names)
    bio = BioClip(r_classes)                          # same class order as the ResNet
    print(f"ResNet classes {r_classes}", flush=True)

    # ── crops: ground truth (A, B) and the agent's boxes (C) ──────────
    gt_crops, gt_lab = [], []
    ag_crops, ag_rec = [], []
    logs = {json.loads(line)["file"]: json.loads(line)
            for line in (LOGS / f"{args.split}_actions.jsonl").open(encoding="utf-8")}
    for fn in files:
        img = cv2.imread(str(image_path(fn)))
        if img is None:
            continue
        for g in gt_all.get(fn, []):
            if g["class"] in known or g["class"] in held:
                c = pipe._extract_crop(img, g["bbox"], padding=0.05)
                if c.size:
                    gt_crops.append(c)
                    gt_lab.append(g["class"])
        var = logs[fn]["variants"]["0"]                # learned policy: no enhancement
        finals = []
        for box in var["boxes"]:
            v = box["z0"]                              # no zoom
            finals.append(v["vote"] if v["vote"] is not None else v["plain"])
        for d, r in zip(finals, harness.match_and_classify(finals, gt_all.get(fn, []))):
            c = pipe._extract_crop(img, d["bbox"], padding=0.05)
            if c.size:
                ag_crops.append(c)
                ag_rec.append({"file": fn, "outcome": r["outcome"], "gt": r["gt_class"],
                               "agent": d["classifier_class"], "unknown": d["unknown"]})
    print(f"{len(gt_crops)} GT crops, {len(ag_crops)} agent boxes", flush=True)

    t = time.perf_counter()
    sk, so = bio.score(gt_crops)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    ms_per_crop = (time.perf_counter() - t) * 1000.0 / max(len(gt_crops), 1)
    pr, en = resnet_probs(pipe, gt_crops)
    pb = softmax(bio.scale * sk)
    lab = np.array(gt_lab)
    is_known = np.isin(lab, known)
    y = np.array([r_classes.index(c) if c in r_classes else -1 for c in lab])

    # ── A: closed-set accuracy on known crops ─────────────────────────
    k = is_known
    r_ok = pr[k].argmax(1) == y[k]
    b_ok = pb[k].argmax(1) == y[k]
    f_ok = (pr[k] + pb[k]).argmax(1) == y[k]
    per_class = {}
    for ci, cname in enumerate(r_classes):
        m = y[k] == ci
        if m.any():
            per_class[cname] = {"n": int(m.sum()), "resnet": float(r_ok[m].mean()),
                                "bioclip": float(b_ok[m].mean()), "average": float(f_ok[m].mean())}
    conf_b = Counter(zip(lab[k], np.array(r_classes)[pb[k].argmax(1)]))
    A = {"n": int(k.sum()), "resnet_acc": float(r_ok.mean()), "bioclip_acc": float(b_ok.mean()),
         "average_acc": float(f_ok.mean()), "both_right": float((r_ok & b_ok).mean()),
         "only_resnet": float((r_ok & ~b_ok).mean()), "only_bioclip": float((~r_ok & b_ok).mean()),
         "both_wrong": float((~r_ok & ~b_ok).mean()), "per_class": per_class,
         "bioclip_top_confusions": [[a, b, n] for (a, b), n in conf_b.most_common(12) if a != b]}

    # ── B: Known/Unknown on GT crops ─────────────────────────────────
    other_mass = softmax(bio.scale * np.c_[sk, so])[:, len(r_classes):].sum(1)
    closed = -sk.max(1)                               # higher = more unknown
    zs = lambda v: (v - v[k].mean()) / (v[k].std() + 1e-9)
    B = {"n_known": int(k.sum()), "n_unseen": int((~k).sum()),
         "unseen_taxa": dict(Counter(lab[~k])),
         "auroc_energy_resnet": auroc(en[~k], en[k]),
         "auroc_bioclip_closed": auroc(closed[~k], closed[k]),
         "auroc_bioclip_open_optimistic": auroc(other_mass[~k], other_mass[k]),
         "auroc_energy_plus_closed": auroc((zs(en) + zs(closed))[~k], (zs(en) + zs(closed))[k]),
         "open_top_is_other_unseen": float((other_mass[~k] > 0.5).mean()),
         "open_top_is_other_known": float((other_mass[k] > 0.5).mean())}

    # ── C: the agent's real boxes ────────────────────────────────────
    ak, ao = bio.score(ag_crops)
    apb = softmax(bio.scale * ak)
    a_other = softmax(bio.scale * np.c_[ak, ao])[:, len(r_classes):].sum(1)
    bio_lab = np.array(r_classes)[apb.argmax(1)]
    by = defaultdict(list)
    for i, r in enumerate(ag_rec):
        by[r["outcome"]].append(i)
    C = {"boxes": len(ag_rec), "outcomes": {o: len(v) for o, v in by.items()}}
    tp, ce, oe = by.get(harness.TP, []), by.get(harness.CLS_ERR, []), by.get(harness.OOD_ERR, [])
    C["tp_bioclip_disagrees"] = float(np.mean([bio_lab[i] != ag_rec[i]["agent"] for i in tp])) if tp else None
    C["cls_err_bioclip_right"] = float(np.mean([bio_lab[i] == ag_rec[i]["gt"] for i in ce])) if ce else None
    C["cls_err_bioclip_disagrees"] = float(np.mean([bio_lab[i] != ag_rec[i]["agent"] for i in ce])) if ce else None
    C["ood_err_open_says_other"] = float(np.mean([a_other[i] > 0.5 for i in oe])) if oe else None
    C["tp_open_says_other"] = float(np.mean([a_other[i] > 0.5 for i in tp])) if tp else None
    # Upper bound: switch to BioCLIP's label exactly when that fixes the box.
    gain = sum(bio_lab[i] == ag_rec[i]["gt"] for i in ce)
    C["oracle_switch_gain_tp"] = int(gain)
    # Naive rule: switch whenever BioCLIP disagrees.
    C["switch_when_disagree_net_tp"] = int(gain - sum(bio_lab[i] != ag_rec[i]["agent"] for i in tp))
    C["bioclip_ms_per_crop"] = ms_per_crop

    res = {"split": args.split, "model": MODEL, "templates": TEMPLATES, "class_prompts": CLASS_NAMES,
           "other_groups": OTHER, "A_closed_set": A, "B_known_unknown": B, "C_agent_boxes": C}
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"bioclip_zeroshot_{args.split}.json"
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k2: v for k2, v in A.items() if k2 not in ("per_class", "bioclip_top_confusions")}))
    for cname, v in per_class.items():
        print(f"  {cname:18s} n {v['n']:4d}  resnet {v['resnet']:.2f}  bioclip {v['bioclip']:.2f}  avg {v['average']:.2f}")
    print("  top BioCLIP confusions:", A["bioclip_top_confusions"][:6])
    print(json.dumps(B))
    print(json.dumps(C))
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
