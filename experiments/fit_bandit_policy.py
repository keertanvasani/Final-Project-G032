"""Step 3: fit the contextual-bandit policy on S_cal, evaluate it on S_policy.

Three decisions, each a one-step contextual bandit with two arms (act / skip),
made once per image (D1) or once per box (D3, D4) in a single pass:

    D1 enhance?   state = image quality            reward Δ = R(enhanced) − R(raw),
                                                   rules downstream
    D3 zoom?      state = fused box + image        Δ = r(zoomed box) − r(box)
    D4 vote?      state = box + classifier view    Δ = r(voted) − r(plain)

A box earns +1 if it is a correct known-class identification (TP), −1 if it is
reported and wrong (CLS_ERR, OOD_ERR, LOC_ERR, BG_FP), 0 if it is flagged
Unknown and so not reported. R is the sum over an image after harness matching.

Model: ridge regression of Δ on the state (numpy; sklearn is not a project
dependency). The policy acts iff the predicted Δ exceeds a threshold. The
ridge strength and threshold are chosen on out-of-fold predictions from a
5-fold, dive-grouped cross-fit on S_cal, then the model is refitted on all of
S_cal. The logs hold every arm's outcome (log_bandit_outcomes.py), so policies
are evaluated exactly by replaying the logs: no propensity weighting needed.

S_policy compares: never act, rules (deployed), best constant per decision
(chosen on S_cal), learned, and the per-decision oracle. Paired dive-cluster
bootstrap for learned − rules. S_test is not touched (PROTOCOL §8; H1/H2 run
once, later, with experiments/test_bandit_policy.py).

Usage:
    python experiments/fit_bandit_policy.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from ablation_deepsea import MAN  # noqa: E402
from agentic.features import image_features, vote_features, zoom_features  # noqa: E402
from eval import harness  # noqa: E402

LOGS = ROOT / "experiments" / "results" / "phase_b_machine" / "step3"
POLICY_OUT = ROOT / "models" / "phase_b_machine" / "bandit_policy.json"
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
N_FOLDS = 5
N_BOOT = 1000


# ── Logs and replay ──────────────────────────────────────────────────
def load(split):
    return [json.loads(line) for line in (LOGS / f"{split}_actions.jsonl").open(encoding="utf-8")]


def box_reward(d, gts):
    if d["unknown"]:
        return 0
    return 1 if harness.match_and_classify([d], gts)[0]["outcome"] == harness.TP else -1


def rule_zoom(box):
    return 0.4 <= box["det"]["confidence"] < 0.75


def final_box(box, z, v):
    var = box[f"z{z}"]
    return var["vote"] if (v and var["vote"] is not None) else var["plain"]


def replay(rec, pol):
    """Run one image under policy *pol*; returns (reported dets, cost_ms, actions)."""
    e = 1 if (rec["flagged"] and pol["enhance"](rec)) else 0
    var = rec["variants"][str(e)]
    cost = var["enh_ms"]
    acts = Counter(enhance=e)
    out = []
    for i, box in enumerate(var["boxes"]):
        z = 1 if pol["zoom"](rec, e, i, box) else 0
        cost += box["zoom_ms"] if z else 0.0
        acts["zoom"] += z
        has_vote = box[f"z{z}"]["vote"] is not None
        v = 1 if (has_vote and pol["vote"](rec, e, i, box, z)) else 0
        cost += box[f"z{z}"]["vote_ms"] if v else 0.0
        acts["vote"] += v
        d = final_box(box, z, v)
        if not d["unknown"]:
            out.append(d)
    return out, cost, acts


def evaluate(recs, pol, gt_all):
    """Per-image results under *pol*."""
    rows = []
    for rec in recs:
        dets, cost, acts = replay(rec, pol)
        gts = gt_all.get(rec["file"], [])
        scored = harness.match_and_classify(dets, gts)
        for r in scored:
            r["file"] = rec["file"]
        rows.append({"file": rec["file"], "dive": rec["dive"], "records": scored, "cost": cost,
                     "acts": acts, "kgt": harness.known_gt_count(gts)})
    return rows


def aurc_fast(conf, ok, kgt):
    """AURC with GT-anchored coverage (selective recall), threshold at every distinct score."""
    if len(conf) == 0 or kgt == 0:
        return 0.0
    o = np.argsort(-conf, kind="stable")
    c, t = conf[o], ok[o].astype(float)
    last = np.r_[c[1:] != c[:-1], True]           # last index of each tie group
    n = np.arange(1, len(c) + 1)[last]
    tp = np.cumsum(t)[last]
    risk = np.r_[1 - tp[0] / n[0], 1 - tp / n]
    rec = np.r_[0.0, tp / kgt]
    risk[0] = risk[1]
    span = rec[-1] - rec[0]
    area = float(((rec[1:] - rec[:-1]) * (risk[1:] + risk[:-1]) / 2).sum())
    return area / span if span > 0 else 0.0


def summarise(rows):
    recs = [r for row in rows for r in row["records"]]
    kgt = sum(row["kgt"] for row in rows)
    tp = sum(r["outcome"] == harness.TP for r in recs)
    n = len(recs)
    p = tp / n if n else 0.0
    rcl = tp / kgt if kgt else 0.0
    conf = np.array([r["combined_confidence"] for r in recs])
    ok = np.array([r["outcome"] == harness.TP for r in recs])
    acts = Counter()
    for row in rows:
        acts.update(row["acts"])
    return {"reported": n, "tp": tp, "precision": p, "recall": rcl,
            "f1": 2 * p * rcl / (p + rcl) if p + rcl else 0.0,
            "reward_per_image": (2 * tp - n) / len(rows),
            "aurc": aurc_fast(conf, ok, kgt),
            "discretionary_ms_per_image": sum(row["cost"] for row in rows) / len(rows),
            "actions_per_image": {k: v / len(rows) for k, v in acts.items()},
            "outcomes": dict(Counter(r["outcome"] for r in recs))}


def paired_boot(rows_a, rows_b, seed=0):
    """Dive-cluster bootstrap of (b − a) for each metric."""
    by = defaultdict(lambda: ([], []))
    for ra, rb in zip(rows_a, rows_b):
        by[ra["dive"]][0].append(ra)
        by[ra["dive"]][1].append(rb)
    dives = list(by)
    rng = random.Random(seed)
    keys = ("precision", "recall", "f1", "reward_per_image", "aurc", "discretionary_ms_per_image")
    diffs = {k: [] for k in keys}
    for _ in range(N_BOOT):
        sa, sb = [], []
        for _ in dives:
            d = dives[rng.randrange(len(dives))]
            sa += by[d][0]
            sb += by[d][1]
        ma, mb = summarise(sa), summarise(sb)
        for k in keys:
            diffs[k].append(mb[k] - ma[k])
    out = {}
    for k, v in diffs.items():
        v = np.sort(v)
        p = 2 * min((v <= 0).mean(), (v >= 0).mean())
        out[k] = {"ci95": [float(v[int(0.025 * N_BOOT)]), float(v[int(0.975 * N_BOOT) - 1])],
                  "p_two_sided": float(min(p, 1.0))}
    return out


# ── Training data per decision ───────────────────────────────────────
RULES = {"enhance": lambda rec: True, "zoom": lambda rec, e, i, box: rule_zoom(box),
         "vote": lambda rec, e, i, box, z: True}
NEVER = {"enhance": lambda rec: False, "zoom": lambda rec, e, i, box: False,
         "vote": lambda rec, e, i, box, z: False}


def image_reward(rec, e, gts, pol):
    sub = dict(rec, flagged=True)
    p = dict(pol, enhance=lambda r: bool(e))
    dets, _, _ = replay(sub, p)
    return sum(1 if r["outcome"] == harness.TP else -1 for r in harness.match_and_classify(dets, gts))


def build(recs, gt_all, classes):
    """Rows of (features, Δ, cost, dive) for each decision."""
    data = {"enhance": [], "zoom": [], "vote": []}
    for rec in recs:
        gts = gt_all.get(rec["file"], [])
        q = rec["quality"]
        if rec["flagged"]:
            dlt = image_reward(rec, 1, gts, RULES) - image_reward(rec, 0, gts, RULES)
            data["enhance"].append((image_features(q), dlt, rec["variants"]["1"]["enh_ms"], rec["dive"]))
        for e, var in rec["variants"].items():
            n = var["n_dets"]
            for box in var["boxes"]:
                vr = lambda z: final_box(box, z, 1)                       # rules vote downstream
                dz = box_reward(vr(1), gts) - box_reward(vr(0), gts)
                data["zoom"].append((zoom_features(box["det"], q, n, e == "1", classes), dz,
                                     box["zoom_ms"], rec["dive"]))
                for z in (0, 1):
                    v = box[f"z{z}"]
                    if v["vote"] is None:
                        continue
                    src = box["zoomed"] if z else box["det"]
                    dv = box_reward(v["vote"], gts) - box_reward(v["plain"], gts)
                    data["vote"].append((vote_features(src, v["cls"], q, n, e == "1", classes), dv,
                                         v["vote_ms"], rec["dive"]))
    return data


# ── Ridge with dive-grouped cross-fitting ───────────────────────────
def ridge_fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    A = Z.T @ Z + alpha * np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z.T @ (y - y.mean()))
    return {"mean": mu.tolist(), "std": sd.tolist(), "w": w.tolist(), "b": float(y.mean())}


def ridge_predict(m, X):
    return ((X - np.array(m["mean"])) / np.array(m["std"])) @ np.array(m["w"]) + m["b"]


def oof(X, y, dives, alpha):
    ud = sorted(set(dives))
    rng = random.Random(0)
    rng.shuffle(ud)
    fold = {d: i % N_FOLDS for i, d in enumerate(ud)}
    f = np.array([fold[d] for d in dives])
    pred = np.zeros(len(y))
    for k in range(N_FOLDS):
        tr, te = f != k, f == k
        if te.any() and tr.sum() > X.shape[1]:
            pred[te] = ridge_predict(ridge_fit(X[tr], y[tr], alpha), X[te])
    return pred


def fit_decision(rows):
    X = np.array([r[0] for r in rows], dtype=float)
    y = np.array([r[1] for r in rows], dtype=float)
    dives = [r[3] for r in rows]
    best = None
    for a in ALPHAS:
        p = oof(X, y, dives, a)
        mse = float(((p - y) ** 2).mean())
        if best is None or mse < best[0]:
            best = (mse, a, p)
    mse, alpha, p = best
    # Threshold: maximise total OOF Δ of the acted-on rows. Candidates are the
    # OOF predictions themselves plus "never" (+inf) and "always" (−inf).
    cands = np.r_[np.inf, -np.inf, np.unique(p)]
    gain = [float(y[p > t].sum()) for t in cands]
    t = float(cands[int(np.argmax(gain))])
    base_mse = float(((y - y.mean()) ** 2).mean())
    corr = float(np.corrcoef(p, y)[0, 1]) if y.std() > 0 and p.std() > 0 else 0.0
    m = ridge_fit(X, y, alpha)
    m.update(alpha=alpha, threshold=t if np.isfinite(t) else (1e9 if t > 0 else -1e9))
    diag = {"rows": len(y), "mean_delta": float(y.mean()), "share_positive": float((y > 0).mean()),
            "share_negative": float((y < 0).mean()), "alpha": alpha, "oof_mse": mse,
            "constant_mse": base_mse, "oof_corr": corr, "threshold": m["threshold"],
            "oof_gain_learned": max(gain), "oof_gain_always": float(y.sum()), "oof_gain_never": 0.0,
            "oof_act_rate": float((p > t).mean())}
    return m, diag


# ── Policies for replay ──────────────────────────────────────────────
def learned(models, classes):
    def enh(rec):
        return float(ridge_predict(models["enhance"], np.array([image_features(rec["quality"])]))[0]) \
            > models["enhance"]["threshold"]

    def zoom(rec, e, i, box):
        x = zoom_features(box["det"], rec["quality"], rec["variants"][str(e)]["n_dets"], e == 1, classes)
        return float(ridge_predict(models["zoom"], np.array([x]))[0]) > models["zoom"]["threshold"]

    def vote(rec, e, i, box, z):
        src = box["zoomed"] if z else box["det"]
        x = vote_features(src, box[f"z{z}"]["cls"], rec["quality"], rec["variants"][str(e)]["n_dets"],
                          e == 1, classes)
        return float(ridge_predict(models["vote"], np.array([x]))[0]) > models["vote"]["threshold"]
    return {"enhance": enh, "zoom": zoom, "vote": vote}


def constant(diag):
    """Best context-free choice per decision, from S_cal: always if mean Δ > 0."""
    a = {k: diag[k]["oof_gain_always"] > 0 for k in diag}
    return {"enhance": lambda rec: a["enhance"], "zoom": lambda rec, e, i, box: a["zoom"],
            "vote": lambda rec, e, i, box, z: a["vote"]}, a


def oracle(gt_all):
    """Per-decision best in hindsight (ties go to the cheaper arm)."""
    def best_box(box, gts):
        opts = []
        for z in (0, 1):
            for v in ((0, 1) if box[f"z{z}"]["vote"] is not None else (0,)):
                opts.append((box_reward(final_box(box, z, v), gts), -(z + v), z, v))
        return max(opts)[2:]

    cache = {}

    def choice(rec, e, i, box):
        k = (rec["file"], e, i)
        if k not in cache:
            cache[k] = best_box(box, gt_all.get(rec["file"], []))
        return cache[k]

    def enh(rec):
        gts = gt_all.get(rec["file"], [])
        pol = {"enhance": None, "zoom": lambda r, e, i, b: choice(r, e, i, b)[0] == 1,
               "vote": lambda r, e, i, b, z: choice(r, e, i, b)[1] == 1}
        return image_reward(rec, 1, gts, pol) > image_reward(rec, 0, gts, pol)
    return {"enhance": enh, "zoom": lambda r, e, i, b: choice(r, e, i, b)[0] == 1,
            "vote": lambda r, e, i, b, z: choice(r, e, i, b)[1] == 1}


def main() -> None:
    man = json.loads(MAN.read_text())
    classes = sorted(man["label_space"]["known"])
    harness.KNOWN_CLASSES = set(classes)
    gt_all = harness.load_ground_truth()
    cal, pol = load("S_cal"), load("S_policy")

    data = build(cal, gt_all, classes)
    models, diag = {}, {}
    for k, rows in data.items():
        models[k], diag[k] = fit_decision(rows)
        d = diag[k]
        print(f"S_cal {k:8s} rows {d['rows']:5d}  meanΔ {d['mean_delta']:+.3f}  "
              f"+{d['share_positive']:.1%} −{d['share_negative']:.1%}  α {d['alpha']:g}  "
              f"OOF corr {d['oof_corr']:+.3f}  OOF gain learned {d['oof_gain_learned']:+.0f} "
              f"always {d['oof_gain_always']:+.0f}  act {d['oof_act_rate']:.1%}", flush=True)

    const_pol, const_choice = constant(diag)
    policies = {"never": NEVER, "rules": RULES, "constant": const_pol,
                "learned": learned(models, classes), "oracle": oracle(gt_all)}
    res, rows_by = {}, {}
    for name, p in policies.items():
        rows_by[name] = evaluate(pol, p, gt_all)
        res[name] = summarise(rows_by[name])
        s = res[name]
        a = s["actions_per_image"]
        print(f"S_pol {name:8s} rep {s['reported']:5d} P {s['precision']:.3f} R {s['recall']:.3f} "
              f"F1 {s['f1']:.3f} AURC {s['aurc']:.3f} reward/img {s['reward_per_image']:+.3f} "
              f"disc {s['discretionary_ms_per_image']:6.1f} ms/img  "
              f"enh {a.get('enhance', 0):.2f} zoom {a.get('zoom', 0):.2f} vote {a.get('vote', 0):.2f}",
              flush=True)

    comps = {f"{b}_minus_rules": paired_boot(rows_by["rules"], rows_by[b])
             for b in ("never", "constant", "learned")}
    for k, c in comps.items():
        print(f"{k:22s} " + "  ".join(f"{m} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] p={v['p_two_sided']:.3f}"
                                      for m, v in c.items()), flush=True)

    POLICY_OUT.write_text(json.dumps({
        "kind": "ridge contextual bandit (act iff predicted Δreward > threshold)",
        "fitted_on": "S_cal", "classes": classes, "models": models}, indent=2) + "\n")
    out = LOGS / "bandit_results.json"
    out.write_text(json.dumps({"reward": "+1 TP, −1 reported error, 0 flagged Unknown",
                               "s_cal_diagnostics": diag, "constant_choice": const_choice,
                               "S_policy": res, "paired_vs_rules": comps}, indent=2) + "\n")
    print(f"wrote {POLICY_OUT.relative_to(ROOT)} and {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
