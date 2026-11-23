"""Week 3 development check (PROTOCOL §14.5): taxonomy-aware, risk-controlled verifier on S_cal.

S_cal dives → three folds A / B / C, rotated three times:
    A  fit the per-level combiner (logistic) for P(candidate at this level is correct)
    B  Learn-then-Test: threshold t so that dive-level risk ≤ α with prob ≥ 1 − δ
    C  measure: realised risk, correct answers, specificity credit, unseen animals

Hierarchical (deepest level with P ≥ t) vs flat (species or abstain), same
pipeline and calibration. Boxes matched to held-out taxa outside S_cal's role
(§14.3) are excluded from A and B; on C both "eligible" and "all" are reported.
Descriptive only.

Usage:
    python experiments/week3_verifier.py
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
from agentic import taxonomy as tx  # noqa: E402
from week0_analyse import logistic_fit  # noqa: E402

DIR = ROOT / "experiments" / "results" / "phase_b_machine" / "week3"
ALPHAS = (0.10, 0.15, 0.20, 0.25)
DELTA = 0.10
GRID = [round(0.99 - 0.01 * i, 2) for i in range(100)]          # 0.99 … 0.00, fixed order
N_BOOT = 2000


# ── Features and labels ──────────────────────────────────────────────
def box_rows(b, classes):
    p = np.asarray(b["probs"])
    ent = float(-(p * np.log(np.clip(p, 1e-12, 1))).sum())
    cands = tx.candidates(p, classes, b["classifier_class"])
    feats, ys, nodes = [], [], []
    for lv in tx.LEVELS:
        node, s = cands[lv]
        feats.append([s, float(p.max()), ent, b["energy"] if b["energy"] is not None else 0.0,
                      float(b["unknown"]), b["detector_confidence"], b["raw_combined_confidence"],
                      float(b["agreement"]), float(tx.under(node, b["detector_class"])),
                      math.log(max(b["area"], 1e-6))])
        ys.append(tx.is_correct(node, b["gt_class"], b["matched"]))
        nodes.append(node)
    return feats, ys, nodes


def hb_pvalue(rhat, n, alpha):
    """Hoeffding–Bentkus p-value for H0: mean loss > alpha (losses in [0, 1], n samples)."""
    if rhat >= alpha:
        return 1.0
    a = rhat
    h = (a * math.log(a / alpha) if a > 0 else 0.0) + (1 - a) * math.log((1 - a) / (1 - alpha))
    p_h = math.exp(-n * h)
    k = math.ceil(n * rhat)
    cdf = sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                       + i * math.log(alpha) + (n - i) * math.log(1 - alpha)) for i in range(k + 1))
    return min(p_h, math.e * cdf, 1.0)


def decide(P, t, levels):
    """Index of the deepest level (in *levels*) with P ≥ t, or −1 = abstain."""
    out = np.full(len(P), -1)
    for j in reversed(levels):           # coarse → fine, so the finest passing level wins
        out[P[:, j] >= t] = j
    return out


def dive_risk(choice, Y, dives, idx):
    by = defaultdict(list)
    for i in idx:
        wrong = choice[i] >= 0 and not Y[i, choice[i]]
        by[dives[i]].append(float(wrong))
    return float(np.mean([np.mean(v) for v in by.values()])), len(by)


def ltt(P, Y, dives, idx, alpha, levels):
    chosen = None
    for t in GRID:
        r, n = dive_risk(decide(P, t, levels), Y, dives, idx)
        if hb_pvalue(r, n, alpha) <= DELTA:
            chosen = t
        else:
            break
    return chosen


def decide2(P, ts, tc):
    """Species if P_species ≥ ts, else the deepest coarse level with P ≥ tc, else abstain."""
    out = decide(P, tc, [1, 2, 3])
    out[P[:, 0] >= ts] = 0
    return out


def ltt2(P, Y, dives, idx, alpha, seed=0):
    """Two thresholds on independent halves of the calibration dives (dev variant).

    Half 1: ts by Learn-then-Test on species-or-abstain. Half 2: with ts fixed,
    tc by Learn-then-Test on the total risk. The guarantee is the half-2 test,
    valid because ts does not depend on half 2."""
    ud = sorted({dives[i] for i in idx})
    random.Random(seed + 7).shuffle(ud)
    h1 = set(ud[: len(ud) // 2])
    i1 = [i for i in idx if dives[i] in h1]
    i2 = [i for i in idx if dives[i] not in h1]
    ts = ltt(P, Y, dives, i1, alpha, [0])
    ts = 2.0 if ts is None else ts
    tc = None
    for t in GRID:
        r, n = dive_risk(decide2(P, ts, t), Y, dives, i2)
        if hb_pvalue(r, n, alpha) <= DELTA:
            tc = t
        else:
            break
    return ts, (2.0 if tc is None else tc)


def main() -> None:
    man = json.loads(MAN.read_text())
    role = set(man["label_space"]["ood_roles"]["OOD_dev"])
    excluded = set(man["label_space"]["held_out_unseen"]) - role
    recs = [json.loads(line) for line in (DIR / "S_cal_verifier.jsonl").open(encoding="utf-8")]
    classes = recs[0]["classes"]
    boxes, dives, files = [], [], []
    for r in recs:
        for b in r["boxes"]:
            boxes.append(b)
            dives.append(r["dive"])
            files.append(r["file"])
    n = len(boxes)
    F, Y, NODES = [], [], []
    for b in boxes:
        f, y, nd = box_rows(b, classes)
        F.append(f)
        Y.append(y)
        NODES.append(nd)
    F = np.array(F)                       # n × 4 levels × features
    Y = np.array(Y, bool)                 # n × 4
    elig = np.array([b["gt_class"] not in excluded for b in boxes])
    heldout = np.array([b["gt_class"] in tx.HELD_OUT for b in boxes])
    cred = np.array([[tx.credit(nd) for nd in row] for row in NODES])
    n_img = len(recs)
    print(f"S_cal: {n_img} images, {n} boxes, {len(set(dives))} dives; excluded from fitting "
          f"(other roles' taxa): {int((~elig).sum())} boxes; held-out boxes eligible: "
          f"{int((heldout & elig).sum())}")
    print("Correct-at-level rate:", {lv: round(float(Y[:, j].mean()), 3) for j, lv in enumerate(tx.LEVELS)})

    ud = sorted(set(dives))
    random.Random(0).shuffle(ud)
    fold = {d: i % 3 for i, d in enumerate(ud)}
    fz = np.array([fold[d] for d in dives])

    results = {"images": n_img, "boxes": n, "excluded_boxes": int((~elig).sum()), "alphas": {}}
    per_box = {}                              # (alpha, mode) → arrays over boxes (each box once, from C)
    for alpha in ALPHAS:
        for mode, levels in (("hier", [0, 1, 2, 3]), ("flat", [0]), ("hier2", [0, 1, 2, 3]), ("hier_nok", [0, 1, 2])):
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
                if mode == "hier2":
                    t = ltt2(P, Y, dives, ib, alpha)
                    ch = decide2(P, *t)
                else:
                    t = ltt(P, Y, dives, ib, alpha, levels)
                    ch = decide(P, t if t is not None else 2.0, levels)
                choice[ic] = ch[ic]
                r_all, _ = dive_risk(ch, Y, dives, ic)
                r_el, _ = dive_risk(ch, Y, dives, ic[elig[ic]])
                rot.append({"t": t, "risk_C_all": r_all, "risk_C_eligible": r_el})
            per_box[(alpha, mode)] = choice
            rep = choice >= 0
            ok = np.array([rep[i] and Y[i, choice[i]] for i in range(n)])
            cr = np.array([cred[i, choice[i]] if ok[i] else 0.0 for i in range(n)])
            lvl = {lv: int(((choice == j) & ok).sum()) for j, lv in enumerate(tx.LEVELS)}
            results["alphas"].setdefault(str(alpha), {})[mode] = {
                "rotations": rot,
                "risk_all_mean": float(np.mean([x["risk_C_all"] for x in rot])),
                "risk_eligible_mean": float(np.mean([x["risk_C_eligible"] for x in rot])),
                "risk_ok_all_rotations": all(x["risk_C_eligible"] <= alpha for x in rot),
                "reported_per_image": float(rep.sum() / n_img),
                "correct_per_image": float(ok.sum() / n_img),
                "credit_per_image": float(cr.sum() / n_img),
                "error_among_reported": float(1 - ok.sum() / max(rep.sum(), 1)),
                "correct_by_level": lvl,
                "heldout_correct": int((ok & heldout).sum()), "heldout_boxes": int(heldout.sum()),
                "heldout_correct_eligible": int((ok & heldout & elig).sum()),
                "heldout_eligible": int((heldout & elig).sum()),
            }

    # Today's agent for reference: every non-Unknown box at species level.
    today = np.array([0 if not b["unknown"] else -1 for b in boxes])
    tok = np.array([today[i] >= 0 and Y[i, 0] for i in range(n)])
    r_today, _ = dive_risk(today, Y, dives, np.arange(n))
    results["today"] = {"risk_all": r_today, "reported_per_image": float((today >= 0).sum() / n_img),
                        "correct_per_image": float(tok.sum() / n_img),
                        "error_among_reported": float(1 - tok.sum() / max((today >= 0).sum(), 1))}

    # Hierarchical − flat, dive bootstrap of correct answers and credit per image.
    img_by_dive = defaultdict(set)
    for d, f in zip(dives, files):
        img_by_dive[d].add(f)
    for r in recs:
        img_by_dive[r["dive"]].add(r["file"])
    box_by_dive = defaultdict(list)
    for i, d in enumerate(dives):
        box_by_dive[d].append(i)
    dl = sorted(img_by_dive)
    rng = random.Random(0)
    samples = [[dl[rng.randrange(len(dl))] for _ in dl] for _ in range(N_BOOT)]
    for alpha in ALPHAS:
        vals = {}
        for mode in ("hier", "flat", "hier2", "hier_nok"):
            ch = per_box[(alpha, mode)]
            ok = np.array([ch[i] >= 0 and Y[i, ch[i]] for i in range(n)], float)
            cr = np.array([cred[i, ch[i]] if ok[i] else 0.0 for i in range(n)])
            vals[mode] = (ok, cr)
        a = results["alphas"][str(alpha)]
        for hm in ("hier", "hier2", "hier_nok"):
            diffs = {"correct": [], "credit": []}
            for s in samples:
                idx = [i for d in s for i in box_by_dive[d]]
                ni = sum(len(img_by_dive[d]) for d in s)
                for k, m in (("correct", 0), ("credit", 1)):
                    diffs[k].append((vals[hm][m][idx].sum() - vals["flat"][m][idx].sum()) / ni)
            for k in diffs:
                v = np.sort(diffs[k])
                m = 0 if k == "correct" else 1
                a[f"{hm}_minus_flat_{k}_per_image"] = {
                    "diff": float((vals[hm][m].sum() - vals["flat"][m].sum()) / n_img),
                    "ci95": [float(v[int(0.025 * N_BOOT)]), float(v[int(0.975 * N_BOOT) - 1])]}

    print(f"\nToday's agent: risk {r_today:.3f}, reported {results['today']['reported_per_image']:.2f}/img, "
          f"correct {results['today']['correct_per_image']:.3f}/img, error among reported "
          f"{results['today']['error_among_reported']:.3f}")
    print(f"\n{'α':>5s} {'mode':5s} {'t (3 rot.)':>28s} {'risk C':>7s} {'≤α?':>4s} {'rep/img':>7s} "
          f"{'corr/img':>8s} {'credit':>6s} {'err|rep':>7s} {'held-out ok':>11s}  by level")
    for alpha in ALPHAS:
        a = results["alphas"][str(alpha)]
        for mode in ("hier", "flat", "hier2", "hier_nok"):
            m = a[mode]
            fmt = lambda t: "-" if t is None else ("/".join(f"{v:.2f}" for v in t) if isinstance(t, tuple)
                                                   else f"{t:.2f}")
            ts = ",".join(fmt(x["t"]) for x in m["rotations"])
            print(f"{alpha:5.2f} {mode:5s} {ts:>28s} {m['risk_eligible_mean']:7.3f} "
                  f"{'yes' if m['risk_ok_all_rotations'] else 'NO':>4s} {m['reported_per_image']:7.2f} "
                  f"{m['correct_per_image']:8.3f} {m['credit_per_image']:6.3f} {m['error_among_reported']:7.3f} "
                  f"{m['heldout_correct_eligible']:>4d}/{m['heldout_eligible']:<4d}  {m['correct_by_level']}")
        for hm in ("hier", "hier2", "hier_nok"):
            for k in ("correct", "credit"):
                d = a[f"{hm}_minus_flat_{k}_per_image"]
                print(f"      {hm:5s} − flat {k:7s} {d['diff']:+.3f} [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}]")
    (DIR / "week3_results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {(DIR / 'week3_results.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
