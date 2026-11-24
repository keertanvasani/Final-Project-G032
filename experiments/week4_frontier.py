"""Week 4: budget control and the accuracy–compute frontier on S_cal.

Policies, all replayed exactly from full-information logs of the same S_cal images:
    plain      no discretionary action
    rules      the phase-I rule agent (enhance if flagged, zoom 0.40–0.75, vote)
    deployed   the Step 3 learned bandit (no enhance, no zoom, vote) — the app today
    agent(λ)   deployed + the two-step image agent (no-probe state, CLAHE 3.0),
               out-of-fold (nested dive-grouped cross-fitting, as fit_image_agent.py)

Cost = discretionary ms per image over the plain models: enhance + zoom + vote
from the Step 3 logs, plus the image agent's second pass when it re-runs.

Budget control (dual ascent): for a budget B ms/img, bisection on λ for the
smallest λ whose mean cost ≤ B. Matched-compute comparison (original H1): the
agent at B = the rule agent's cost vs the rules, and vs the deployed bandit.
Paired dive-cluster bootstrap. S_cal only; descriptive.

Usage:
    python experiments/week4_frontier.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from ablation_deepsea import MAN  # noqa: E402
from calibrate import apply_isotonic, fit_isotonic  # noqa: E402
from fit_deepsea_calibration import compress  # noqa: E402
from agentic.probes import state_columns  # noqa: E402
from eval import harness  # noqa: E402
from fit_bandit_policy import NEVER, RULES, evaluate, load, paired_boot, summarise  # noqa: E402
from fit_image_agent import FOLDS, fit_pred, folds_of, inner_oof, reward  # noqa: E402

W1 = ROOT / "experiments" / "results" / "phase_b_machine" / "week1"
OUT = ROOT / "experiments" / "results" / "phase_b_machine" / "week4"
ARM = "clahe3.0"
DEPLOYED = {"enhance": lambda rec: False, "zoom": lambda rec, e, i, box: False,
            "vote": lambda rec, e, i, box, z: True}


def agent_preds(recs, gt):
    """Out-of-fold Q1, Q2 (no-probe state), Δ and re-run cost per image."""
    a = [r["arms"][ARM] for r in recs]
    dives = [r["dive"] for r in recs]
    S1 = np.array([x["s1"] for x in a], float)[:, state_columns([], 1)]
    S2 = np.array([x["s2"] for x in a], float)[:, state_columns([], 2)]
    D = np.array([reward(x["passes"][1]["finals"], gt.get(r["file"], []))
                  - reward(x["passes"][0]["finals"], gt.get(r["file"], [])) for x, r in zip(a, recs)], float)
    C = np.array([x["pass2_ms"] for x in a])
    n = len(recs)
    q1, q2 = np.zeros(n), np.zeros(n)
    outer = folds_of(dives, FOLDS)
    for k in range(FOLDS):
        tr, te = np.where(outer != k)[0], np.where(outer == k)[0]
        q2[te] = fit_pred(S2, D, tr, te)
        y1 = np.zeros(n)
        y1[tr] = D[tr] * (inner_oof(S2, D, dives, tr) > 0)
        q1[te] = fit_pred(S1, y1, tr, te)
    return q1, q2, C


def pass2_recalibration(recs, gt):
    """Out-of-fold isotonic map for second-pass boxes (raw combined confidence → P(TP)).

    The deployed map was fitted on first-pass boxes; enhanced-pass boxes sit on a
    different scale. Fitted per dive fold on the other folds' second-pass boxes,
    so no image is recalibrated by a map that saw it."""
    dives = [r["dive"] for r in recs]
    f = folds_of(dives, FOLDS, seed=3)
    rows = []                                   # (image index, box index, raw, is_tp)
    for i, r in enumerate(recs):
        fin = r["arms"][ARM]["passes"][1]["finals"]
        rep = [k for k, d in enumerate(fin) if not d["unknown"]]
        m = harness.match_and_classify([fin[k] for k in rep], gt.get(r["file"], []))
        rows += [(i, k, fin[k]["raw_combined_confidence"], mm["outcome"] == harness.TP) for k, mm in zip(rep, m)]
    out = {}
    for k in range(FOLDS):
        tr = [x for x in rows if f[x[0]] != k]
        xs, ys = fit_isotonic(np.clip([x[2] for x in tr], 1e-6, 1 - 1e-6), np.array([x[3] for x in tr], float))
        xs, ys = compress(xs, ys)
        for x in rows:
            if f[x[0]] == k:
                out[(x[0], x[1])] = float(apply_isotonic(np.array(xs), np.array(ys), np.array([x[2]]))[0])
    return out


def agent_rows(recs, q1, q2, C, lam, base_rows, gt, recal=None):
    """Rows in fit_bandit_policy.evaluate's format for the agent at λ.

    *recal*: optional {(image, box): confidence} for second-pass boxes."""
    act = q1 - lam * C.mean() > 0
    keep2 = act & (q2 > 0)
    rows = []
    for i, (rec, base) in enumerate(zip(recs, base_rows)):
        if keep2[i]:
            fin = [dict(d, combined_confidence=recal[(i, k)]) if recal else d
                   for k, d in enumerate(rec["arms"][ARM]["passes"][1]["finals"]) if not d["unknown"]]
            recs_ = harness.match_and_classify(fin, gt.get(rec["file"], []))
        else:
            recs_ = base["records"]
        rows.append({"file": rec["file"], "dive": rec["dive"], "records": recs_,
                     "cost": base["cost"] + (C[i] if act[i] else 0.0),
                     "acts": Counter(base["acts"]) + Counter(rerun=int(act[i]), kept=int(keep2[i])),
                     "kgt": base["kgt"]})
    return rows


def main() -> None:
    man = json.loads(MAN.read_text())
    harness.KNOWN_CLASSES = set(man["label_space"]["known"])
    gt = harness.load_ground_truth()
    s3 = {r["file"]: r for r in load("S_cal")}
    recs = [json.loads(line) for line in (W1 / "S_cal_image_agent.jsonl").open(encoding="utf-8")]
    recs = [r for r in recs if not r["arms"][ARM].get("skipped") and r["file"] in s3]
    order = [s3[r["file"]] for r in recs]
    rows = {"plain": evaluate(order, NEVER, gt), "rules": evaluate(order, RULES, gt),
            "deployed": evaluate(order, DEPLOYED, gt)}
    q1, q2, C = agent_preds(recs, gt)

    recal = pass2_recalibration(recs, gt)

    def at(lam, rc=None):
        return agent_rows(recs, q1, q2, C, lam, rows["deployed"], gt, rc)

    def cost(lam):
        return float(np.mean([r["cost"] for r in at(lam)]))

    def lam_for(budget):
        if cost(0.0) <= budget:
            return 0.0
        lo, hi = 0.0, 0.2
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if cost(mid) <= budget else (mid, hi)
        return hi

    out = {"images": len(recs), "policies": {}, "frontier": [], "matched": {}}
    keys = ("reward_per_image", "f1", "precision", "recall", "aurc", "discretionary_ms_per_image")

    def show(name, s):
        print(f"{name:22s} reward/img {s['reward_per_image']:+.3f}  F1 {s['f1']:.3f}  P {s['precision']:.3f}  "
              f"R {s['recall']:.3f}  AURC {s['aurc']:.3f}  ms/img {s['discretionary_ms_per_image']:6.1f}")

    for name, r in rows.items():
        out["policies"][name] = {k: summarise(r)[k] for k in keys}
        show(name, out["policies"][name])
    rules_ms = out["policies"]["rules"]["discretionary_ms_per_image"]
    dep_ms = out["policies"]["deployed"]["discretionary_ms_per_image"]
    print("\nFrontier (agent on top of deployed; budget → λ by bisection):")
    budgets = sorted({round(dep_ms, 1)} | {round(dep_ms + b, 1) for b in (2, 5, 10, 15, 20, 25)}
                     | {round(rules_ms, 1)})
    for b in budgets:
        lam = lam_for(b)
        s = summarise(at(lam))
        row = {"budget_ms": b, "lambda": lam, **{k: s[k] for k in keys},
               "rerun_rate": s["actions_per_image"].get("rerun", 0.0)}
        out["frontier"].append(row)
        row["aurc_pass2_recalibrated"] = summarise(at(lam, recal))["aurc"]
        print(f"  budget {b:5.1f} ms  λ {lam:.4f}  re-run {100 * row['rerun_rate']:4.1f}%  "
              f"reward/img {s['reward_per_image']:+.3f}  F1 {s['f1']:.3f}  AURC {s['aurc']:.3f} "
              f"(pass-2 recalibrated {row['aurc_pass2_recalibrated']:.3f})  ms {s['discretionary_ms_per_image']:5.1f}")

    lam_m = lam_for(rules_ms)
    matched = at(lam_m)
    s = summarise(matched)
    show(f"agent @ rules' {rules_ms:.1f} ms", s)
    out["matched"] = {"budget_ms": rules_ms, "lambda": lam_m, **{k: s[k] for k in keys}}
    matched_rc = at(lam_m, recal)
    out["matched"]["aurc_pass2_recalibrated"] = summarise(matched_rc)["aurc"]
    for ref in ("rules", "deployed"):
        for tag, mrows in (("", matched), ("_recal", matched_rc)):
            c = paired_boot(rows[ref], mrows)
            out["matched"][f"minus_{ref}{tag}"] = c
            print(f"agent{tag:6s} − {ref:8s} " + "  ".join(f"{m} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]"
                                                          for m, v in c.items()))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "frontier.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {(OUT / 'frontier.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
