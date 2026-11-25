"""External validation on FathomNet (PROTOCOL §18). Single use.

Part A (§18.5): labelled objects only. Final boxes of the deployed agent matched to a
scorable FathomNet box; unmatched boxes and boxes matched to excluded concepts are
ignored. Learn-then-Test is re-run on F_cal (frozen combiner) and tested on F_eval (E1);
the JODD thresholds are tested unchanged (E2); hierarchical vs flat (E3); unseen
animals with a correct coarse answer (E4).

Part B (§18.6, descriptive): fully VERIFIED images, full §2 accounting (boxes matched
to excluded concepts ignored): rules / deployed bandit / image agent, and verifier risk.

Refuses to run if a frozen hash differs or the result exists.

Usage:
    python experiments/test_fathomnet.py
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import fit_bandit_policy as fbp  # noqa: E402
import week5_test_spolicy as w5  # noqa: E402
from agentic import taxonomy as tx  # noqa: E402
from eval import harness  # noqa: E402
from week3_verifier import box_rows, decide, dive_risk, ltt  # noqa: E402

R = w5.R
EXT = ROOT / "data_external" / "fathomnet"
RES = ROOT / "experiments" / "results" / "external_fathomnet"
LOGS = RES / "logs"
OUT = RES / "fathomnet_test.json"
THR = R / "week6" / "thresholds_S_policy.json"
FROZEN = dict(w5.FROZEN) | {
    THR: "d583d9d2386e840fb9b08644dc7cabacf4f9d87d755219deaac040fc4fe2f749",
    EXT / "meta" / "manifest.json": "b07caf6bd2576e256096c0436452456c5599bffcf75bd16c356b930d6786ca0a",
    RES / "concept_labels.json": "929db292376677132c330781aa39c9cf4d3e7b86e46111cda5a569d7d77bf19c",
}
N_BOOT = 2000
ALPHAS = (0.10, 0.20)
MODES = (("hierarchical", [0, 1, 2, 3]), ("flat", [0]))

# FathomNet ground-truth labels on the §14.1 tree (§18.2); local to this script
for node in ("Decapoda", "Arthropoda", "Chordata", "Echinodermata", "Cnidaria", "Porifera", "Animal"):
    tx.HELD_OUT[f"unseen:{node}"] = node
tx.HELD_OUT["Man-made"] = "Man-made"
SCORABLE = lambda c: c is not None and c != "excluded"   # noqa: E731

_match = harness.match_and_classify


def match_ignoring_excluded(dets, gts):
    """§2 matching; detections matched to an excluded concept are dropped (§18.2)."""
    return [r for r in _match(dets, gts) if r["gt_class"] != "excluded"]


def combiner_probs(boxes, spec):
    """Combiner P(correct) per level, correctness Y, and the candidate node per level."""
    F, Y, N = [], [], []
    for b in boxes:
        f, y, nodes = box_rows(b, spec["classes"])
        F.append(f)
        Y.append(y)
        N.append(nodes)
    F, Y = np.array(F), np.array(Y, bool)
    P = np.zeros((len(boxes), 4))
    for j, lv in enumerate(tx.LEVELS):
        m = spec["models"][lv]
        P[:, j] = 1 / (1 + np.exp(-(m["w"][0] + ((F[:, j] - np.array(m["mean"])) / np.array(m["std"]))
                                    @ np.array(m["w"][1:]))))
    return P, Y, N


def ci(v):
    v = np.sort(v)
    return [float(v[int(0.025 * len(v))]), float(v[int(0.975 * len(v)) - 1])]


def verifier_block(P, Y, N, dives, idx, img_per_dive, thresholds, samples, unseen):
    """Risk and answers on boxes *idx* for each (alpha, mode) threshold."""
    n_img = sum(img_per_dive.values())
    box_by = defaultdict(list)
    for i in idx:
        box_by[dives[i]].append(i)
    out, choices = {}, {}
    for alpha in ALPHAS:
        for mode, levels in MODES:
            t = thresholds[mode][str(alpha)]
            ch = decide(P, 2.0 if t is None else t, levels)
            choices[(alpha, mode)] = ch
            risk, nd = dive_risk(ch, Y, dives, idx)
            per_dive = {d: np.mean([float(ch[i] >= 0 and not Y[i, ch[i]]) for i in bi]) for d, bi in box_by.items()}
            share = [np.mean(v) <= alpha for v in ([per_dive[d] for d in s if d in per_dive] for s in samples) if v]
            ok = np.array([ch[i] >= 0 and Y[i, ch[i]] for i in range(len(P))])
            sel = np.zeros(len(P), bool)
            sel[idx] = True
            out.setdefault(str(alpha), {})[mode] = {
                "threshold": t, "risk": risk, "dives": nd, "risk_le_alpha": bool(risk <= alpha),
                "share_resamples_le_alpha": float(np.mean(share)) if share else None,
                "reported_per_image": float(((ch >= 0) & sel).sum() / n_img),
                "correct_per_image": float((ok & sel).sum() / n_img),
                "credit_per_image": float(sum(tx.credit(N[i][ch[i]]) for i in idx if ok[i]) / n_img),
                "species_correct_per_image": float((ok & sel & (ch == 0)).sum() / n_img),
                "error_among_reported": float(1 - (ok & sel).sum() / max(((ch >= 0) & sel).sum(), 1)),
                "unseen_correct": int((ok & sel & unseen).sum()),
                "unseen_correct_by_node": dict(Counter(N[i][ch[i]] for i in idx if ok[i] and unseen[i])),
                "correct_by_level": {lv: int(((ch == j) & ok & sel).sum()) for j, lv in enumerate(tx.LEVELS)}}
    return out, choices




def part_a(man, spec, jodd_thr):
    files_a = set(man["parts"]["A"])
    recs = [r for r in (json.loads(l) for l in (LOGS / "F_ext_verifier.jsonl").open(encoding="utf-8"))
            if r["file"] in files_a]
    gt = json.loads((EXT / "meta" / "gt480.json").read_text())
    img_dive = {r["file"]: r["dive"] for r in recs}
    ud = sorted(set(img_dive.values()))
    random.Random(0).shuffle(ud)
    cal = set(ud[: len(ud) // 2])
    boxes, dives = [], []
    for r in recs:
        for b in r["boxes"]:
            if b["matched"] and SCORABLE(b["gt_class"]):
                boxes.append(b)
                dives.append(r["dive"])
    P, Y, N = combiner_probs(boxes, spec)
    unseen = np.array([b["gt_class"].startswith("unseen:") for b in boxes])
    i_cal = [i for i, d in enumerate(dives) if d in cal]
    i_eval = [i for i, d in enumerate(dives) if d not in cal]
    ipd_cal = Counter(d for d in img_dive.values() if d in cal)
    ipd_eval = Counter(d for d in img_dive.values() if d not in cal)

    # Learn-then-Test on F_cal, frozen combiner (E1)
    f_thr = {mode: {str(a): ltt(P, Y, dives, i_cal, a, lv) for a in ALPHAS} for mode, lv in MODES}
    ude = sorted(ipd_eval)
    rng = random.Random(0)
    samples = [[ude[rng.randrange(len(ude))] for _ in ude] for _ in range(N_BOOT)]
    e1, ch1 = verifier_block(P, Y, N, dives, i_eval, ipd_eval, f_thr, samples, unseen)
    e2, _ = verifier_block(P, Y, N, dives, i_eval, ipd_eval, jodd_thr, samples, unseen)

    # E3 / E4 at alpha 0.20, recalibrated thresholds, paired dive bootstrap on F_eval
    ch_h, ch_f = ch1[(0.20, "hierarchical")], ch1[(0.20, "flat")]
    ok_h = np.array([ch_h[i] >= 0 and Y[i, ch_h[i]] for i in range(len(P))], float)
    ok_f = np.array([ch_f[i] >= 0 and Y[i, ch_f[i]] for i in range(len(P))], float)
    by = defaultdict(list)
    for i in i_eval:
        by[dives[i]].append(i)
    d3, c4 = [], []
    for s in samples:
        idx = [i for d in s for i in by.get(d, [])]
        ni = sum(ipd_eval[d] for d in s)
        d3.append((ok_h[idx].sum() - ok_f[idx].sum()) / ni)
        c4.append(float((ok_h * unseen)[idx].sum()))
    d3, c4 = np.sort(d3), np.sort(c4)
    n_eval_img = sum(ipd_eval.values())
    tests = {"E3": {"diff_correct_per_image": float((ok_h[i_eval].sum() - ok_f[i_eval].sum()) / n_eval_img),
                    "ci95": ci(d3), "p": float((d3 <= 0).mean())},
             "E4": {"unseen_correct": int((ok_h * unseen)[i_eval].sum()), "unseen_boxes": int(unseen[i_eval].sum()),
                    "ci95": ci(c4), "p": float((c4 <= 0).mean())}}
    for k, v in w5.holm({k: tests[k]["p"] for k in tests}).items():
        tests[k]["supported_holm"] = v
    descriptive_only = len(ipd_eval) < 30 or len(i_eval) < 300

    # descriptive: recall and classifier top-1 on all Part A images
    gt_n, hit, top1 = Counter(), Counter(), Counter()
    for r in recs:
        for g in gt.get(r["file"], []):
            if SCORABLE(g["class"]):
                gt_n["unseen" if g["class"].startswith("unseen:") else g["class"]] += 1
        for b in r["boxes"]:
            if b["matched"] and SCORABLE(b["gt_class"]):
                key = "unseen" if b["gt_class"].startswith("unseen:") else b["gt_class"]
                hit[key] += 1
                top1[key] += int(b["classifier_class"] == b["gt_class"])
    recall = {k: {"gt": gt_n[k], "matched": hit[k], "recall": hit[k] / gt_n[k] if gt_n[k] else None,
                  "classifier_top1": top1[k] / hit[k] if hit[k] and not k.startswith("unseen") and k != "Man-made"
                  else None} for k in sorted(gt_n)}
    known_k = [k for k in gt_n if k in tx.KNOWN]
    return {"images": len(recs), "dives": len(ud), "F_cal_dives": len(cal), "F_eval_dives": len(ipd_eval),
            "F_eval_images": n_eval_img, "matched_boxes": {"F_cal": len(i_cal), "F_eval": len(i_eval)},
            "unseen_boxes_F_eval": int(unseen[i_eval].sum()),
            "descriptive_only (§18.7)": descriptive_only,
            "thresholds_F_cal": f_thr, "E1_recalibrated": e1, "E2_jodd_thresholds": e2, "tests": tests,
            "recall": recall,
            "known_recall_overall": sum(hit[k] for k in known_k) / max(sum(gt_n[k] for k in known_k), 1),
            "known_top1_overall": sum(top1[k] for k in known_k) / max(sum(hit[k] for k in known_k), 1)}, f_thr


def part_b(man, spec, jodd_thr, f_thr):
    files_b = set(man["parts"]["B"])
    gt = json.loads((EXT / "meta" / "gt480.json").read_text())
    # stage the frozen policies with the FathomNet logs so w5.part_a runs unchanged
    stage = EXT / "stage"
    for sub, name in (("week2", "image_agent_policy.json"), ("week5", "one_step_policy.json")):
        (stage / sub).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(R / sub / name, stage / sub / name)
    (stage / "week1").mkdir(parents=True, exist_ok=True)
    with (stage / "week1" / "F_B_image_agent.jsonl").open("w", encoding="utf-8") as fh:
        for l in (LOGS / "F_ext_image_agent.jsonl").open(encoding="utf-8"):
            if json.loads(l)["file"] in files_b:
                fh.write(l)
    (stage / "step3").mkdir(parents=True, exist_ok=True)
    with (stage / "step3" / "F_B_actions.jsonl").open("w", encoding="utf-8") as fh:
        for l in (LOGS / "F_ext_actions.jsonl").open(encoding="utf-8"):
            if json.loads(l)["file"] in files_b:
                fh.write(l)
    w5.R, w5.SPLIT, fbp.LOGS = stage, "F_B", stage / "step3"
    harness.match_and_classify = match_ignoring_excluded
    try:
        agents = w5.part_a(gt)
    finally:
        harness.match_and_classify = _match
        w5.R = R
    agents.pop("tests", None)       # §18.6: descriptive, no hypothesis tests

    recs = [r for r in (json.loads(l) for l in (LOGS / "F_ext_verifier.jsonl").open(encoding="utf-8"))
            if r["file"] in files_b]
    boxes, dives = [], []
    for r in recs:
        for b in r["boxes"]:
            if b["gt_class"] != "excluded":
                boxes.append(b)
                dives.append(r["dive"])
    P, Y, N = combiner_probs(boxes, spec)
    unseen = np.array([bool(b["gt_class"]) and b["gt_class"].startswith("unseen:") for b in boxes])
    ipd = Counter(r["dive"] for r in recs)
    ud = sorted(ipd)
    rng = random.Random(0)
    samples = [[ud[rng.randrange(len(ud))] for _ in ud] for _ in range(N_BOOT)]
    idx = list(range(len(boxes)))
    v_f, _ = verifier_block(P, Y, N, dives, idx, ipd, f_thr, samples, unseen)
    v_j, _ = verifier_block(P, Y, N, dives, idx, ipd, jodd_thr, samples, unseen)
    return {"images": len(recs), "dives": len(ud), "agents": agents,
            "verifier_F_cal_thresholds": v_f, "verifier_jodd_thresholds": v_j}


def main() -> None:
    global OUT
    if "--dry-run" in sys.argv:       # plumbing check on partial logs; never the registered run
        OUT = Path(sys.argv[sys.argv.index("--dry-run") + 1])
    elif OUT.exists():
        raise SystemExit(f"{OUT.relative_to(ROOT)} exists: §18 is single-use")
    for f, h in FROZEN.items():
        got = hashlib.sha256(f.read_bytes()).hexdigest()
        if got != h:
            raise SystemExit(f"hash mismatch for {f.name}: {got}")
    man = json.loads((EXT / "meta" / "split_manifest.json").read_text())
    harness.KNOWN_CLASSES = set(man["label_space"]["known"])
    spec = json.loads((R / "week4" / "verifier.json").read_text())
    jodd_thr = json.loads(THR.read_text())["thresholds"]
    a, f_thr = part_a(man, spec, jodd_thr)
    out = {"protocol": "PROTOCOL §18", "part_a": a, "part_b": part_b(man, spec, jodd_thr, f_thr),
           "deviations": ["§18.5 descriptive 'same tables with the image agent on' not computed: the image-agent "
                          "logs hold no classifier probabilities for pass-2 boxes, so the verifier cannot be "
                          "replayed on them."]}
    enc = lambda o: bool(o) if isinstance(o, np.bool_) else float(o)   # noqa: E731
    OUT.write_text(json.dumps(out, indent=2, default=enc) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
