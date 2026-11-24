"""Week 0 (PROTOCOL §13): headroom and probe signal on S_cal, go/no-go.

Reads week0/S_cal_week0.jsonl (week0_log_probes.py). For every action:

    Δ per unit (image or box) = reward(action) − reward(base)
    (a) ridge on the Step 3 features, (b) ridge on Step 3 features + probes,
    5-fold dive-grouped out-of-fold predictions, ridge strength 10;
    policy: act iff OOF prediction > 0; gain = Σ Δ where it acts.

Dive-cluster bootstrap (2,000, seed 0) for gain(b), gain(a) and gain(b) − gain(a).
Correctness: OOF AUROC for "box would be TP if reported": calibrated combined
confidence vs logistic(base fields) vs logistic(base fields + probes).

Decision rule exactly as registered in §13.5.

Usage:
    python experiments/week0_analyse.py
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from ablation_deepsea import MAN  # noqa: E402
from agentic.features import image_features, vote_features, zoom_features  # noqa: E402
from eval import harness  # noqa: E402
from fit_bandit_policy import box_reward, oof  # noqa: E402

DIR = ROOT / "experiments" / "results" / "phase_b_machine" / "week0"
ALPHA = 10.0
N_BOOT = 2000
IMAGE_ACTIONS = ("clahe1.5", "clahe2.0", "clahe3.0", "gamma", "greyworld", "det960", "tile2x2")
BOX_ACTIONS = ("zoom", "vote", "fliptta")
PROBE_KEYS = ("p_top", "margin", "entropy", "energy", "p_det_class", "flip_change", "flip_p_own",
              "shift_agree", "shift_p_own", "shift_p_min", "cent_dist_own", "cent_dist_other",
              "cent_gap", "edge", "max_iou_other")


def image_reward(finals, gts):
    dets = [d for d in finals if not d["unknown"]]
    return sum(1 if r["outcome"] == harness.TP else -1 for r in harness.match_and_classify(dets, gts))


def box_probe_vec(b):
    return [b["probes"][k] for k in PROBE_KEYS]


def image_probe_vec(rec):
    c = rec["colour"]
    v = [c["lab_a"] / 20, c["lab_b"] / 20, c["sat"], c["sat_std"], math.log1p(len(rec["boxes"]))]
    if rec["boxes"]:
        P = np.array([box_probe_vec(b) for b in rec["boxes"]])
        conf = [b["base"]["combined_confidence"] for b in rec["boxes"]]
        v += list(P.mean(0)) + list(P.min(0)) + list(P.max(0))
        v += [float(np.mean(conf)), float(np.min(conf)), float(np.mean([b["disagree"] for b in rec["boxes"]])),
              float(np.mean([b["base"]["unknown"] for b in rec["boxes"]])), 1.0]
    else:
        v += [0.0] * (3 * len(PROBE_KEYS) + 4) + [0.0]
    v.append(float(rec["base"]["refused"]))
    return v


def build(recs, gt_all, classes):
    """Units per action: (features_a, features_b, Δ, cost_ms, dive)."""
    units = {a: [] for a in IMAGE_ACTIONS + BOX_ACTIONS}
    for rec in recs:
        gts = gt_all.get(rec["file"], [])
        q = rec["quality"]
        fa = image_features(q)
        fb = fa + image_probe_vec(rec)
        r0 = image_reward(rec["base"]["finals"], gts)
        for a in IMAGE_ACTIONS:
            act = rec["actions"][a]
            units[a].append((fa, fb, image_reward(act["finals"], gts) - r0, act["ms"], rec["dive"]))
        n = rec["base"]["n_dets"]
        for b in rec["boxes"]:
            za = zoom_features(b["det"], q, n, False, classes)
            pv = box_probe_vec(b)
            rb = box_reward(b["base"], gts)
            units["zoom"].append((za, za + pv, box_reward(b["zoom"], gts) - rb, b["zoom_ms"], rec["dive"]))
            units["fliptta"].append((za, za + pv, box_reward(b["fliptta"], gts) - rb, b["fliptta_ms"],
                                     rec["dive"]))
            if b["disagree"]:
                va = vote_features(b["det"], b["cls"], q, n, False, classes)
                units["vote"].append((va, va + pv, rb - box_reward(b["novote"], gts), 0.0, rec["dive"]))
    return units


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) > 0 and np.std(ry) > 0 else 0.0


def dive_boot(dives, cols, n_images_by_dive, seed=0):
    """Bootstrap per-image means of each column of *cols* (dict name → per-unit values)."""
    ud = sorted(set(n_images_by_dive))
    sums = {k: defaultdict(float) for k in cols}
    for k, v in cols.items():
        for d, x in zip(dives, v):
            sums[k][d] += x
    rng = random.Random(seed)
    out = {k: [] for k in cols}
    for _ in range(N_BOOT):
        pick = [ud[rng.randrange(len(ud))] for _ in ud]
        n = sum(n_images_by_dive[d] for d in pick)
        for k in cols:
            out[k].append(sum(sums[k][d] for d in pick) / n)
    res = {}
    for k, v in out.items():
        v = np.sort(v)
        res[k] = [float(v[int(0.025 * N_BOOT)]), float(v[int(0.975 * N_BOOT) - 1])]
    return res


def auroc(score, y):
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return float("nan")
    s = np.asarray(score)                       # Mann–Whitney with average ranks for ties
    order = np.argsort(s, kind="stable")
    ss = s[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    npos = y.sum()
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos)))


def logistic_fit(X, y, l2=1.0, iters=50):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    R = l2 * np.eye(Z.shape[1])
    R[0, 0] = 0.0
    for _ in range(iters):                       # Newton / IRLS
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y) + R @ w
        H = (Z * (p * (1 - p))[:, None]).T @ Z + R
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return mu, sd, w


def logistic_oof(X, y, dives, folds=5):
    ud = sorted(set(dives))
    rng = random.Random(0)
    rng.shuffle(ud)
    fold = {d: i % folds for i, d in enumerate(ud)}
    f = np.array([fold[d] for d in dives])
    pred = np.zeros(len(y))
    for k in range(folds):
        tr, te = f != k, f == k
        mu, sd, w = logistic_fit(X[tr], y[tr])
        pred[te] = np.c_[np.ones(te.sum()), (X[te] - mu) / sd] @ w
    return pred


def main() -> None:
    man = json.loads(MAN.read_text())
    classes = sorted(man["label_space"]["known"])
    harness.KNOWN_CLASSES = set(classes)
    gt_all = harness.load_ground_truth()
    recs = [json.loads(line) for line in (DIR / "S_cal_week0.jsonl").open(encoding="utf-8")]
    n_img = len(recs)
    imgs_by_dive = defaultdict(int)
    for r in recs:
        imgs_by_dive[r["dive"]] += 1
    units = build(recs, gt_all, classes)

    results = {"split": "S_cal", "images": n_img, "dives": len(imgs_by_dive), "actions": {}}
    print(f"S_cal: {n_img} images, {len(imgs_by_dive)} dives; gains are reward per image "
          f"(+1 TP, −1 reported error)\n")
    hdr = (f"{'action':10s} {'units':>5s} {'Δ≠0':>5s} {'always':>7s} {'oracle':>7s} "
           f"{'feat(a)':>7s} {'probe(b)':>8s} {'b 95% CI':>17s} {'b−a 95% CI':>17s} {'ρ(b)':>5s} "
           f"{'act%':>5s} {'ms/img':>6s} pass")
    print(hdr)
    any_pass, probes_credit = [], []
    for a, rows in units.items():
        if not rows:
            continue
        Xa = np.array([r[0] for r in rows], float)
        Xb = np.array([r[1] for r in rows], float)
        y = np.array([r[2] for r in rows], float)
        cost = np.array([r[3] for r in rows], float)
        dv = [r[4] for r in rows]
        pa, pb = oof(Xa, y, dv, ALPHA), oof(Xb, y, dv, ALPHA)
        ga, gb = y * (pa > 0), y * (pb > 0)
        ci = dive_boot(dv, {"b": gb, "a": ga, "diff": gb - ga, "always": y,
                            "oracle": np.maximum(y, 0)}, imgs_by_dive)
        res = {"units": len(y), "nonzero": int((y != 0).sum()),
               "always_gain": float(y.sum() / n_img), "oracle_gain": float(np.maximum(y, 0).sum() / n_img),
               "gain_features": float(ga.sum() / n_img), "gain_probes": float(gb.sum() / n_img),
               "gain_probes_ci95": ci["b"], "gain_features_ci95": ci["a"],
               "probes_minus_features": float((gb - ga).sum() / n_img), "probes_minus_features_ci95": ci["diff"],
               "always_ci95": ci["always"], "oracle_ci95": ci["oracle"],
               "spearman_features": spearman(pa, y), "spearman_probes": spearman(pb, y),
               "act_rate_probes": float((pb > 0).mean()), "act_rate_features": float((pa > 0).mean()),
               "ms_per_image_if_acting_probes": float((cost * (pb > 0)).sum() / n_img),
               "ms_per_image_always": float(cost.sum() / n_img)}
        res["passes"] = ci["b"][0] > 0
        res["probes_credited"] = res["passes"] and ci["diff"][0] > 0
        results["actions"][a] = res
        if res["passes"]:
            any_pass.append(a)
        if res["probes_credited"]:
            probes_credit.append(a)
        print(f"{a:10s} {len(y):5d} {res['nonzero']:5d} {res['always_gain']:+7.3f} {res['oracle_gain']:7.3f} "
              f"{res['gain_features']:+7.3f} {res['gain_probes']:+8.3f} "
              f"[{ci['b'][0]:+.3f},{ci['b'][1]:+.3f}] [{ci['diff'][0]:+.3f},{ci['diff'][1]:+.3f}] "
              f"{res['spearman_probes']:5.2f} {100 * res['act_rate_probes']:5.1f} "
              f"{res['ms_per_image_if_acting_probes']:6.1f} {'PASS' if res['passes'] else '-'}")

    # Correctness signal for the learned combiner.
    Y, S, Fa, Fb, D = [], [], [], [], []
    for rec in recs:
        gts = gt_all.get(rec["file"], [])
        for b in rec["boxes"]:
            f = b["base"]
            Y.append(box_reward(dict(f, unknown=False), gts) == 1)
            S.append(f["combined_confidence"])
            base = [f["raw_combined_confidence"], f["detector_confidence"], f["classifier_confidence"],
                    float(f["agreement"]), f["energy"] if f["energy"] is not None else 0.0, float(f["unknown"])]
            Fa.append(base)
            Fb.append(base + box_probe_vec(b))
            D.append(rec["dive"])
    Y = np.array(Y, float)
    S = np.array(S)
    la = logistic_oof(np.array(Fa), Y, D)
    lb = logistic_oof(np.array(Fb), Y, D)
    au = {"calibrated_conf": auroc(S, Y), "logistic_base": auroc(la, Y), "logistic_probes": auroc(lb, Y)}
    by = defaultdict(list)
    for i, d in enumerate(D):
        by[d].append(i)
    ud = sorted(by)
    rng = random.Random(0)
    diffs = []
    for _ in range(N_BOOT):
        idx = [i for _ in ud for i in by[ud[rng.randrange(len(ud))]]]
        diffs.append(auroc(lb[idx], Y[idx]) - auroc(S[idx], Y[idx]))
    diffs = np.sort([x for x in diffs if not math.isnan(x)])
    ci = [float(diffs[int(0.025 * len(diffs))]), float(diffs[int(0.975 * len(diffs)) - 1])]
    gain = au["logistic_probes"] - au["calibrated_conf"]
    results["combiner"] = {"boxes": len(Y), "tp_rate": float(Y.mean()), "auroc": au,
                           "gain_vs_calibrated": gain, "gain_ci95": ci,
                           "adopt": bool(gain >= 0.01 and ci[0] > 0)}
    print(f"\nCombiner, {len(Y)} base boxes (TP-if-reported {Y.mean():.3f}): AUROC calibrated "
          f"{au['calibrated_conf']:.3f}, logistic base {au['logistic_base']:.3f}, + probes "
          f"{au['logistic_probes']:.3f}; gain {gain:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] → "
          f"{'ADOPT' if results['combiner']['adopt'] else 'keep calibrated confidence'}")

    results["decision"] = {"go": bool(any_pass), "passing_actions": any_pass,
                           "probes_credited_for": probes_credit}
    print(f"\nDecision (§13.5): {'GO' if any_pass else 'NO-GO'} — passing actions: {any_pass or 'none'}; "
          f"probes credited for: {probes_credit or 'none'}")
    (DIR / "week0_results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {(DIR / 'week0_results.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
