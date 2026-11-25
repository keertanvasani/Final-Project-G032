"""Step 5 development check (PROTOCOL §17): taxonomy-trained classifiers for the verifier, on S_cal.

The agent's boxes are fixed (the week-3 S_cal log). For each candidate classifier the
verifier's probabilities are recomputed on the same 5% crops; the species candidate stays
the agent's own final class. Then the §14.5 procedure runs unchanged (three dive-fold
rotations: fit combiner / Learn-then-Test / measure). The deployed classifier is the
baseline, and its recomputed probabilities must reproduce the logged ones.

Usage:
    python experiments/step5_dev.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from ablation_deepsea import MAN, image_path  # noqa: E402
from agentic import taxonomy as tx  # noqa: E402
from week0_analyse import logistic_fit  # noqa: E402
from week3_verifier import box_rows, decide, dive_risk, ltt  # noqa: E402

R = ROOT / "experiments" / "results" / "phase_b_machine"
OUT = R / "step5"
MODELS = ROOT / "models" / "phase_b_machine"
CANDIDATES = {"deployed": MODELS / "classifier_seed0",
              "ce": MODELS / "step5" / "ce" / "classifier_seed0",
              "soft1": MODELS / "step5" / "soft1" / "classifier_seed0",
              "soft2": MODELS / "step5" / "soft2" / "classifier_seed0",
              "marginal": MODELS / "step5" / "marginal" / "classifier_seed0"}
RUNS = {k: v.parent / "classifier_runs.json" for k, v in CANDIDATES.items()}
ELIGIBLE_TOP1 = 0.833
ALPHAS = (0.10, 0.20)
N_BOOT = 2000


def load_boxes():
    recs = [json.loads(line) for line in (R / "week3" / "S_cal_verifier.jsonl").open(encoding="utf-8")]
    classes = recs[0]["classes"]
    boxes, dives, files = [], [], []
    for r in recs:
        for b in r["boxes"]:
            boxes.append(b)
            dives.append(r["dive"])
            files.append(r["file"])
    return recs, classes, boxes, dives, files


def crops_for(recs):
    from pipeline import EnsemblePipeline
    out = []
    for r in recs:
        if not r["boxes"]:
            continue
        img = cv2.imread(str(image_path(r["file"])))
        out += [EnsemblePipeline._extract_crop(img, b["bbox"], padding=0.05) for b in r["boxes"]]
    return out


def probs_for(name, crops, classes):
    cache = OUT / f"S_cal_probs_{name}.npy"
    if cache.exists():
        return np.load(cache)
    import torch
    from classifier import SpeciesClassifier
    d = CANDIDATES[name]
    clf = SpeciesClassifier(d / "best_classifier.pth", d / "class_names.json",
                            "cuda" if torch.cuda.is_available() else "cpu")
    assert [c.replace("_", "/") for c in clf.class_names] == classes, clf.class_names
    P = np.concatenate([tx.full_probs(crops[i:i + 128], clf) for i in range(0, len(crops), 128)])
    np.save(cache, P)
    return P


def evaluate(boxes, dives, files, recs, classes, role, excluded):
    """§14.5 rotations for hierarchical and flat; returns summary + per-box arrays."""
    n = len(boxes)
    F, Y, NODES = [], [], []
    for b in boxes:
        f, y, nd = box_rows(b, classes)
        F.append(f)
        Y.append(y)
        NODES.append(nd)
    F, Y = np.array(F), np.array(Y, bool)
    elig = np.array([b["gt_class"] not in excluded for b in boxes])
    heldout = np.array([b["gt_class"] in tx.HELD_OUT for b in boxes])
    cred = np.array([[tx.credit(nd) for nd in row] for row in NODES])
    n_img = len(recs)
    ud = sorted(set(dives))
    random.Random(0).shuffle(ud)
    fold = {d: i % 3 for i, d in enumerate(ud)}
    fz = np.array([fold[d] for d in dives])
    res, arrays = {}, {}
    for alpha in ALPHAS:
        for mode, levels in (("hier", [0, 1, 2, 3]), ("flat", [0])):
            choice = np.full(n, -1)
            rot = []
            for A, B, C in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
                ia = np.where((fz == A) & elig)[0]
                ib = np.where((fz == B) & elig)[0]
                ic = np.where(fz == C)[0]
                P = np.zeros((n, 4))
                for j in levels:
                    mu, sd, w = logistic_fit(F[ia, j], Y[ia, j].astype(float))
                    P[:, j] = 1 / (1 + np.exp(-(np.c_[np.ones(n), (F[:, j] - mu) / sd] @ w)))
                t = ltt(P, Y, dives, ib, alpha, levels)
                ch = decide(P, t if t is not None else 2.0, levels)
                choice[ic] = ch[ic]
                rot.append({"t": t, "risk_C_eligible": dive_risk(ch, Y, dives, ic[elig[ic]])[0]})
            ok = np.array([choice[i] >= 0 and Y[i, choice[i]] for i in range(n)])
            cr = np.array([cred[i, choice[i]] if ok[i] else 0.0 for i in range(n)])
            arrays[(alpha, mode)] = (ok.astype(float), cr)
            ho = [NODES[i][choice[i]] for i in range(n) if ok[i] and heldout[i] and elig[i]]
            res.setdefault(str(alpha), {})[mode] = {
                "thresholds": [x["t"] for x in rot],
                "risk_eligible_mean": float(np.mean([x["risk_C_eligible"] for x in rot])),
                "risk_ok_all_rotations": all(x["risk_C_eligible"] <= alpha for x in rot),
                "reported_per_image": float((choice >= 0).sum() / n_img),
                "correct_per_image": float(ok.sum() / n_img),
                "credit_per_image": float(cr.sum() / n_img),
                "species_correct_per_image": float((ok & (choice == 0)).sum() / n_img),
                "error_among_reported": float(1 - ok.sum() / max((choice >= 0).sum(), 1)),
                "correct_by_level": {lv: int(((choice == j) & ok).sum()) for j, lv in enumerate(tx.LEVELS)},
                "heldout_correct_eligible": len(ho), "heldout_eligible": int((heldout & elig).sum()),
                "heldout_correct_nodes": dict(Counter(ho)),
            }
    # Threshold-free group structure.
    deep = [i for i, b in enumerate(boxes) if b["matched"] and b["gt_class"] in role
            and tx.HELD_OUT[b["gt_class"]] != "Animal"]
    hit = [NODES[i][1] == tx.HELD_OUT[boxes[i]["gt_class"]] or
           NODES[i][2] == tx.HELD_OUT[boxes[i]["gt_class"]] for i in deep]
    wrong_sp = [i for i, b in enumerate(boxes) if b["matched"] and b["gt_class"] in tx.KNOWN and not Y[i, 0]]
    res["ood_dev_deep_node_top_candidate"] = {"boxes": len(deep), "hit": int(sum(hit)),
                                              "by_taxon": dict(Counter(boxes[i]["gt_class"] for i, h in zip(deep, hit) if h))}
    res["better_mistakes"] = {"wrong_species_boxes": len(wrong_sp),
                              "group_right": int(sum(Y[i, 1] for i in wrong_sp))}
    return res, arrays


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    man = json.loads(MAN.read_text())
    role = set(man["label_space"]["ood_roles"]["OOD_dev"])
    excluded = set(man["label_space"]["held_out_unseen"]) - role
    recs, classes, boxes, dives, files = load_boxes()
    names = [k for k, v in CANDIDATES.items() if (v / "best_classifier.pth").exists()]
    crops = None if all((OUT / f"S_cal_probs_{k}.npy").exists() for k in names) else crops_for(recs)
    logged = np.array([b["probs"] for b in boxes])
    out = {"boxes": len(boxes), "images": len(recs), "candidates": {}}
    arrays = {}
    for k in names:
        P = probs_for(k, crops, classes)
        if k == "deployed":
            # Batched GPU inference differs from the per-image log by float noise only
            # (median 4e-6); every candidate, the baseline included, uses recomputed values.
            diff = float(np.abs(P - logged).max())
            flips = int((P.argmax(1) != logged.argmax(1)).sum())
            out["deployed_vs_log"] = {"max_abs_diff": diff, "top_class_changes": flips}
            print(f"deployed classifier vs logged probabilities: max |diff| {diff:.2e}, top-class changes {flips}")
            assert diff < 0.02 and flips == 0, "recomputed crops do not reproduce the week-3 log"
        top1 = 0.8528399311531841 if k == "deployed" else json.loads(RUNS[k].read_text())[0]["best_val_acc"]
        bx = [dict(b, probs=[float(x) for x in p]) for b, p in zip(boxes, P)]
        res, arr = evaluate(bx, dives, files, recs, classes, role, excluded)
        res["S_modelsel_top1"] = top1
        res["eligible"] = top1 >= ELIGIBLE_TOP1 or k == "deployed"
        out["candidates"][k] = res
        arrays[k] = arr
        h = res["0.2"]["hier"]
        print(f"{k:9s} top1 {top1:.3f}  α=0.20 hier: risk {h['risk_eligible_mean']:.3f} "
              f"{'ok' if h['risk_ok_all_rotations'] else 'NO'}  corr {h['correct_per_image']:.3f}  "
              f"credit {h['credit_per_image']:.3f}  species {h['species_correct_per_image']:.3f}  "
              f"held-out {h['heldout_correct_eligible']}/{h['heldout_eligible']} {h['heldout_correct_nodes']}  "
              f"levels {h['correct_by_level']}  deep-node {res['ood_dev_deep_node_top_candidate']['hit']}/"
              f"{res['ood_dev_deep_node_top_candidate']['boxes']}  better-mistakes "
              f"{res['better_mistakes']['group_right']}/{res['better_mistakes']['wrong_species_boxes']}")

    # Bootstrap: credit/img (α = 0.20, hierarchical) of each candidate minus deployed.
    img_by_dive, box_by_dive = defaultdict(int), defaultdict(list)
    for r in recs:
        img_by_dive[r["dive"]] += 1
    for i, d in enumerate(dives):
        box_by_dive[d].append(i)
    dl = sorted(img_by_dive)
    rng = random.Random(0)
    samples = [[dl[rng.randrange(len(dl))] for _ in dl] for _ in range(N_BOOT)]
    base = arrays["deployed"][(0.20, "hier")][1]
    for k in names:
        if k == "deployed":
            continue
        cr = arrays[k][(0.20, "hier")][1]
        v = []
        for s in samples:
            idx = [i for d in s for i in box_by_dive[d]]
            v.append((cr[idx].sum() - base[idx].sum()) / sum(img_by_dive[d] for d in s))
        v = np.sort(v)
        out["candidates"][k]["credit_minus_deployed"] = {
            "diff": float((cr.sum() - base.sum()) / len(recs)),
            "ci95": [float(v[int(0.025 * N_BOOT)]), float(v[int(0.975 * N_BOOT) - 1])]}
        d = out["candidates"][k]["credit_minus_deployed"]
        print(f"  {k:9s} credit − deployed {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]")

    # §17.4 selection and go / no-go.
    pool = [k for k in names if k not in ("deployed", "ce") and out["candidates"][k]["eligible"]]
    if pool and "ce" in names:
        sel = max(pool, key=lambda k: out["candidates"][k]["0.2"]["hier"]["credit_per_image"])
        c = out["candidates"][sel]
        go = {"credit_ci_above_0": c["credit_minus_deployed"]["ci95"][0] > 0,
              "beats_ce_rerun_gain": c["credit_minus_deployed"]["diff"] > out["candidates"]["ce"]["credit_minus_deployed"]["diff"],
              "risk_ok_all_rotations": c["0.2"]["hier"]["risk_ok_all_rotations"]}
        out["selection"] = {"selected": sel, "criteria": go, "go": all(go.values())}
        print(f"\nselected: {sel}  criteria {go}  → {'GO' if all(go.values()) else 'NO-GO'}")
    (OUT / "step5_dev.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {(OUT / 'step5_dev.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
