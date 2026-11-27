"""Literature baselines for the verifier (PROTOCOL §20). Descriptive, single use.

Same boxes, same calibration data and the same Learn-then-Test procedure for every
one-parameter method; split conformal + LCA uses its own calibration rule.

Settings:  JODD       calibrate on S_policy (eligible boxes), evaluate on S_test (all boxes)
           FathomNet  calibrate on F_cal, evaluate on F_eval (matched, scorable boxes)

Usage:
    python experiments/test_baselines.py --check     # plumbing: reproduce frozen thresholds only
    python experiments/test_baselines.py             # the single run
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import test_fathomnet as tf  # noqa: E402  (adds FathomNet labels to the tree; no JODD key changes)
from ablation_deepsea import DIVE_RE, MAN  # noqa: E402
from agentic import taxonomy as tx  # noqa: E402
from week3_verifier import DELTA, GRID, hb_pvalue  # noqa: E402

R = tf.R
OUT = tf.ROOT / "experiments" / "results" / "baselines" / "baselines.json"
FROZEN = dict(tf.FROZEN) | {
    R / "week3" / "S_policy_verifier.jsonl": "d36ec85c762ad865b1e4b2190d7593a182141b9f210fb89d08b92dc4c4d910c6",
    R / "week3" / "S_test_verifier.jsonl": "2e9ee743fb0c23115c0f2ddac0f97f10929a64b5154d6d8e9c2e8d8800d009b7",
    tf.RES / "fathomnet_test.json": "fc7faf0fedb8ffb962a3bc22dbf753ef066d960be6460ca3b59ea7b040211883",
    tf.LOGS / "F_ext_verifier.jsonl": "362ff618786aa6fd1ebdb1e101f2181be5164f6883fc85224b20205e985941b0",
}
ALPHAS = (0.10, 0.20)
N_BOOT = 2000
LEAVES = list(tx.KNOWN)
# DARTS nodes, deepest first so np.argmax breaks ties towards the deeper node; root last
INTERNAL = ["Decapoda", "Arthropoda", "Chordata", "Cnidaria", "Porifera", "Man-made", "Echinodermata", "Animal"]
NODES = LEAVES + INTERNAL + [None]


# ── data ──────────────────────────────────────────────────────────────
class Boxes:
    def __init__(self, boxes, dives, spec):
        self.b, self.dives = boxes, dives
        cls = spec["classes"]
        self.P, _, self.N = tf.combiner_probs(boxes, spec)
        self.probs = np.array([b["probs"] for b in boxes], float)            # columns in spec order
        order = [cls.index(c) for c in LEAVES]
        self.leafp = self.probs[:, order]                                     # columns in LEAVES order
        self.final = [b["classifier_class"] for b in boxes]
        self.sr = np.array([self.probs[i, cls.index(f)] for i, f in enumerate(self.final)])
        self.negE = np.array([-b["energy"] if b["energy"] is not None else -np.inf for b in boxes])
        self.conf = np.array([b["combined_confidence"] for b in boxes])
        self.paths = [tx.path(b["gt_class"]) if b["matched"] and b["gt_class"] else set() for b in boxes]
        self.known = np.array([b["matched"] and b["gt_class"] in tx.KNOWN for b in boxes])
        self.unseen = np.array([bool(b["matched"] and b["gt_class"] and b["gt_class"] not in tx.KNOWN
                                     and b["gt_class"] != "Man-made") for b in boxes])

    def correct(self, ans):
        return np.array([a is not None and a in p for a, p in zip(ans, self.paths)])


def jodd(spec):
    man = json.loads(MAN.read_text())
    excluded = set(man["label_space"]["held_out_unseen"]) - set(man["label_space"]["ood_roles"]["OOD_val"])
    boxes, dives, cal_n = [], [], 0
    for split in ("S_policy", "S_test"):
        for line in (R / "week3" / f"{split}_verifier.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            for b in r["boxes"]:
                boxes.append(dict(b, _split=split))
                dives.append(r["dive"])
    B = Boxes(boxes, dives, spec)
    i_cal = [i for i, b in enumerate(boxes) if b["_split"] == "S_policy" and b["gt_class"] not in excluded]
    i_ev = [i for i, b in enumerate(boxes) if b["_split"] == "S_test"]
    ipd = Counter(DIVE_RE.match(f).group(1) for f in man["files"]["S_test"])
    return B, i_cal, i_ev, ipd


def fathomnet(spec):
    man = json.loads((tf.EXT / "meta" / "split_manifest.json").read_text())
    files_a = set(man["parts"]["A"])
    recs = [r for r in (json.loads(l) for l in (tf.LOGS / "F_ext_verifier.jsonl").open(encoding="utf-8"))
            if r["file"] in files_a]
    ud = sorted({r["dive"] for r in recs})
    random.Random(0).shuffle(ud)
    cal = set(ud[: len(ud) // 2])
    boxes, dives = [], []
    for r in recs:
        for b in r["boxes"]:
            if b["matched"] and tf.SCORABLE(b["gt_class"]):
                boxes.append(b)
                dives.append(r["dive"])
    B = Boxes(boxes, dives, spec)
    i_cal = [i for i, d in enumerate(dives) if d in cal]
    i_ev = [i for i, d in enumerate(dives) if d not in cal]
    ipd = Counter(r["dive"] for r in recs if r["dive"] not in cal)
    return B, i_cal, i_ev, ipd


# ── methods: param -> answer node (or None = abstain) per box ─────────
def m_ours(levels):
    def f(B, t):
        out = [None] * len(B.b)
        for i in range(len(B.b)):
            for j in levels:                      # finest passing level wins
                if B.P[i, j] >= t:
                    out[i] = B.N[i][j]
                    break
        return out
    return f


def m_flat_score(attr):
    def f(B, t):
        s = getattr(B, attr)
        return [B.final[i] if s[i] >= t else None for i in range(len(B.b))]
    return f


UNDER = np.array([[1.0 if v is not None and v in tx.path(leaf) else 0.0 for v in NODES] for leaf in LEAVES])
REWARD = np.array([(math.log2(9) - math.log2(UNDER[:, k].sum())) if NODES[k] is not None else 0.0
                   for k in range(len(NODES))])


def m_darts(B, lam):
    p = B.leafp @ UNDER                                       # node posteriors
    p[:, -1] = 1.0                                            # virtual root
    k = np.argmax((REWARD + lam) * p, axis=1)
    return [NODES[j] for j in k]


def lca(labels):
    common = set.intersection(*(tx.path(y) for y in labels)) if labels else set()
    if not common:
        return None
    return min(common, key=lambda v: (UNDER[:, NODES.index(v)].sum(), v not in tx.KNOWN))


def conformal(B, i_cal, alpha):
    cal = [i for i in i_cal if B.known[i]]
    s = np.sort([1 - B.leafp[i, LEAVES.index(B.b[i]["gt_class"])] for i in cal])
    n = len(s)
    q = s[min(math.ceil((n + 1) * (1 - alpha)), n) - 1]
    sets = [[LEAVES[j] for j in np.where(1 - B.leafp[i] <= q)[0]] for i in range(len(B.b))]
    return [lca(st) for st in sets], sets, float(q)


GRID_LAM = [10 ** (3 - 6 * i / 99) for i in range(100)] + [0.0]
QS = [round(1 - 0.01 * i, 2) for i in range(101)]


EXTENDED = "--extended" in sys.argv           # §20.5 post-hoc sensitivity run
GRID_SR_EXT = [1 - 10 ** (-6 + 0.06 * i) for i in range(50)] + GRID
GRID_LAM_EXT = [10 ** (6 - 3 * i / 50) for i in range(50)] + GRID_LAM


def grid_for(name, B, i_cal):
    if EXTENDED and name == "softmax_response":
        return GRID_SR_EXT
    if EXTENDED and name == "darts":
        return GRID_LAM_EXT
    if name == "energy":
        return [float(np.quantile(B.negE[i_cal], q)) for q in QS]
    return GRID_LAM if name == "darts" else GRID


METHODS = {"ours_hierarchical": m_ours([0, 1, 2, 3]), "ours_flat": m_ours([0]),
           "softmax_response": m_flat_score("sr"), "energy": m_flat_score("negE"),
           "calibrated_confidence": m_flat_score("conf"), "darts": m_darts}


# ── risk, LTT, metrics ────────────────────────────────────────────────
def dive_risk(B, ans, ok, idx):
    by = defaultdict(list)
    for i in idx:
        by[B.dives[i]].append(float(ans[i] is not None and not ok[i]))
    return float(np.mean([np.mean(v) for v in by.values()])), len(by)


def ltt(B, f, grid, idx, alpha):
    chosen = None
    for t in grid:
        ans = f(B, t)
        r, n = dive_risk(B, ans, B.correct(ans), idx)
        if hb_pvalue(r, n, alpha) <= DELTA:
            chosen = t
        else:
            break
    return chosen


def metrics(B, ans, idx, ipd, alpha):
    ok = B.correct(ans)
    n_img = sum(ipd.values())
    rep = np.array([a is not None for a in ans])
    sel = np.zeros(len(ans), bool)
    sel[idx] = True
    risk, nd = dive_risk(B, ans, ok, idx)
    return {"risk": risk, "risk_le_alpha": risk <= alpha, "dives_with_boxes": nd,
            "coverage": float(rep[sel].mean()),
            "reported_per_image": float((rep & sel).sum() / n_img),
            "correct_per_image": float((ok & sel).sum() / n_img),
            "credit_per_image": float(sum(tx.credit(ans[i]) for i in idx if ok[i]) / n_img),
            "species_correct_per_image": float(sum(ok[i] and ans[i] in tx.KNOWN for i in idx) / n_img),
            "error_among_reported": float(1 - (ok & sel).sum() / max((rep & sel).sum(), 1)),
            "unseen_boxes": int((B.unseen & sel).sum()),
            "unseen_correct": int((ok & B.unseen & sel).sum()),
            "unseen_wrong": int((rep & ~ok & B.unseen & sel).sum()),
            "correct_by_answer_type": dict(Counter("species" if ans[i] in tx.KNOWN else ans[i]
                                                   for i in idx if ok[i]))}


def setting(B, i_cal, i_ev, ipd):
    ud = sorted(ipd)
    rng = random.Random(0)
    samples = [[ud[rng.randrange(len(ud))] for _ in ud] for _ in range(N_BOOT)]
    by = defaultdict(list)
    for i in i_ev:
        by[B.dives[i]].append(i)
    res = {"boxes_calibration": len(i_cal), "boxes_evaluation": len(i_ev),
           "images_evaluation": sum(ipd.values()), "dives_evaluation": len(ipd),
           "unseen_boxes_evaluation": int(B.unseen[i_ev].sum()), "alphas": {}, "curves": {}}
    for alpha in ALPHAS:
        a, oks = {}, {}
        for name, f in METHODS.items():
            t = ltt(B, f, grid_for(name, B, i_cal), i_cal, alpha)
            ans = f(B, t) if t is not None else [None] * len(B.b)
            a[name] = {"parameter": t, **metrics(B, ans, i_ev, ipd, alpha)}
            oks[name] = B.correct(ans).astype(float)
        ans, sets, q = conformal(B, i_cal, alpha)
        kn = [i for i in i_ev if B.known[i]]
        a["conformal_lca"] = {"parameter": q, **metrics(B, ans, i_ev, ipd, alpha),
                              "known_set_coverage": float(np.mean([B.b[i]["gt_class"] in sets[i] for i in kn])),
                              "mean_set_size": float(np.mean([len(sets[i]) for i in i_ev]))}
        oks["conformal_lca"] = B.correct(ans).astype(float)
        diffs = {}
        for name in oks:
            if name == "ours_hierarchical":
                continue
            d = oks["ours_hierarchical"] - oks[name]
            boot = np.sort([sum(d[i] for dv in s for i in by.get(dv, [])) / sum(ipd[dv] for dv in s)
                            for s in samples])
            diffs[name] = {"ours_minus_baseline_correct_per_image": float(d[i_ev].sum() / sum(ipd.values())),
                           "ci95": tf.ci(boot)}
        a["ours_hierarchical_minus"] = diffs
        res["alphas"][str(alpha)] = a
    for name, f in METHODS.items():
        pts = []
        for t in grid_for(name, B, i_cal):
            ans = f(B, t)
            m = metrics(B, ans, i_ev, ipd, 1.0)
            pts.append([t, m["coverage"], m["risk"], m["correct_per_image"]])
        res["curves"][name] = {"columns": ["parameter", "coverage", "risk", "correct_per_image"], "points": pts}
    return res


def check(spec):
    """Generic LTT must reproduce the frozen thresholds of methods 1–2 (§20 plumbing check)."""
    want_j = json.loads((R / "week6" / "thresholds_S_policy.json").read_text())["thresholds"]
    want_f = json.loads((tf.RES / "fathomnet_test.json").read_text())["part_a"]["thresholds_F_cal"]
    for label, load, want in (("JODD", jodd, want_j), ("FathomNet", fathomnet, want_f)):
        B, i_cal, _, _ = load(spec)
        for mode, key in (("hierarchical", "ours_hierarchical"), ("flat", "ours_flat")):
            for a in ALPHAS:
                got = ltt(B, METHODS[key], GRID, i_cal, a)
                assert got == want[mode][str(a)], (label, mode, a, got, want[mode][str(a)])
        print(f"{label}: frozen thresholds reproduced")


def main() -> None:
    for f, h in FROZEN.items():
        got = hashlib.sha256(f.read_bytes()).hexdigest()
        if got != h:
            raise SystemExit(f"hash mismatch for {f.name}: {got}")
    spec = json.loads((R / "week4" / "verifier.json").read_text())
    tf.harness.KNOWN_CLASSES = set(json.loads(MAN.read_text())["label_space"]["known"])
    if "--check" in sys.argv:
        check(spec)
        return
    out_path = OUT.with_name("baselines_extended_grid.json") if EXTENDED else OUT
    if out_path.exists():
        raise SystemExit(f"{out_path} exists: single-use")
    check(spec)
    out = {"protocol": "PROTOCOL §20.5, post-hoc extended grids" if EXTENDED else "PROTOCOL §20 (descriptive)", "JODD": setting(*jodd(spec)),
           "FathomNet": setting(*fathomnet(spec))}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enc = lambda o: bool(o) if isinstance(o, np.bool_) else float(o)   # noqa: E731
    out_path.write_text(json.dumps(out, indent=1, default=enc) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
