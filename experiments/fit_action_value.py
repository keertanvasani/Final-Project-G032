"""Was the ZOOM worth it, and can we predict when? (report Phases 15 & 16)

`log_zoom_outcomes.py` forced the ZOOM on every detection and recorded what
happened. This answers the two questions that data exists to settle:

  1. Does the action help at all? The deployed rule fires it on every detection
     scoring 0.40-0.75 and has never been measured. Reported as a contingency
     table over PROTOCOL §2 outcomes plus the net change in correctness.

  2. Can the benefit be predicted from the state? A value model
     Q(s, ZOOM) = E[correct_after - correct_before | s] is fitted on the fit
     split and evaluated on the eval split. If it carries signal, a policy can
     fire the action only where it pays and skip it elsewhere — which is the
     compute reduction H2 claims.

The comparison that matters is against the deployed rule, not against never
zooming: the rule already spends this compute, so a policy that matches its
accuracy while firing far less often is the result.

Fitted on --fit, reported on --eval. Never J_test.

Usage:
    python experiments/fit_action_value.py --fit J_cal --eval J_policy
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).parent
ACT = ROOT / "experiments" / "results" / "actions"
sys.path.insert(0, str(EXP))

FEATURES = ["conf_before", "cls_conf_before", "agreement_before",
            "log_area", "aspect_ratio", "log_blur", "log_n_dets"]


def load(split: str) -> list[dict]:
    p = ACT / f"{split}_zoom_outcomes.jsonl"
    if not p.exists():
        raise SystemExit(f"Missing {p.name}. Run log_zoom_outcomes.py --split {split}")
    return [json.loads(l) for l in p.read_text().splitlines()]


def applied_outcome(r: dict) -> str:
    """Outcome after applying the deployed rule's own keep condition."""
    if not r.get("candidate_found"):
        return r["outcome_before"]
    return r["outcome_after"] if r.get("rule_would_keep") else r["outcome_before"]


def matrix(rows: list[dict]) -> np.ndarray:
    return np.column_stack([
        [r["conf_before"] for r in rows],
        [r["cls_conf_before"] for r in rows],
        [1.0 if r["agreement_before"] else 0.0 for r in rows],
        np.log1p([r["box_area"] for r in rows]),
        np.clip([r["aspect_ratio"] for r in rows], 0, 10),
        np.log1p([r["blur"] for r in rows]),
        np.log1p([r["n_dets"] for r in rows]),
    ])


def fit_ridge(X, y, l2=1.0):
    Xb = np.column_stack([np.ones(len(X)), X])
    A = Xb.T @ Xb + l2 * np.eye(Xb.shape[1])
    A[0, 0] -= l2
    return np.linalg.solve(A, Xb.T @ y)


def predict(X, w):
    return np.column_stack([np.ones(len(X)), X]) @ w


def boot_mean(dives, vals, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(dives)
    idx = {d: np.flatnonzero(dives == d) for d in uniq}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx[d] for d in pick])
        if len(ii):
            out.append(vals[ii].mean())
    o = np.array(out)
    return float(o.mean()), (float(np.percentile(o, 2.5)),
                             float(np.percentile(o, 97.5)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="J_cal")
    ap.add_argument("--eval", dest="ev", default="J_policy")
    args = ap.parse_args()
    if "test" in args.fit.lower() or "test" in args.ev.lower():
        raise SystemExit("Refusing to touch a test split (PROTOCOL §8).")

    F, E = load(args.fit), load(args.ev)

    # ---- 1. does the action do anything? --------------------------------
    print(f"\n=== Did the ZOOM help? ({args.ev}, {len(E)} forced actions) ===\n")
    n = len(E)
    found = [r for r in E if r.get("candidate_found")]
    kept = [r for r in found if r.get("rule_would_keep")]
    fired = [r for r in E if r["rule_would_fire"]]
    fired_kept = [r for r in fired if r.get("candidate_found") and r.get("rule_would_keep")]
    print(f"  forced on                     {n}")
    print(f"  candidate box found           {len(found)} ({len(found)/n:.1%})")
    print(f"  ...and confidence improved    {len(kept)} ({len(kept)/n:.1%})  <- the rule keeps these")
    print(f"  rule would have fired on      {len(fired)} ({len(fired)/n:.1%})")
    print(f"  rule would have changed a box {len(fired_kept)} ({len(fired_kept)/n:.1%})")

    corr_b = np.array([r["outcome_before"] == "TP" for r in E], dtype=float)
    corr_applied = np.array([applied_outcome(r) == "TP" for r in E], dtype=float)
    corr_always = np.array([
        (r["outcome_after"] if r.get("candidate_found") else r["outcome_before"]) == "TP"
        for r in E], dtype=float)
    dive = np.array([r["dive"] for r in E])

    print(f"\n  correct, no zoom at all       {corr_b.mean():.4f}")
    print(f"  correct, deployed rule         {corr_applied.mean():.4f}")
    print(f"  correct, zoom always+keep all  {corr_always.mean():.4f}")

    d_rule = corr_applied - corr_b
    m, ci = boot_mean(dive, d_rule)
    print(f"\n  net effect of the deployed rule: {m:+.5f}  95% CI [{ci[0]:+.5f}, {ci[1]:+.5f}]")
    if ci[0] <= 0 <= ci[1]:
        print("  -> CI includes zero: the rule's zoom does not measurably change correctness.")
    else:
        print("  -> CI excludes zero.")

    changed = [r for r in E if applied_outcome(r) != r["outcome_before"]]
    print(f"\n  detections whose outcome the rule actually changed: {len(changed)}")
    if changed:
        t = Counter((r["outcome_before"], applied_outcome(r)) for r in changed)
        print(f"  {'before':<10}{'after':<10}{'n':>5}")
        for (b, a), c in t.most_common(10):
            mark = "  better" if a == "TP" else ("  worse" if b == "TP" else "")
            print(f"  {b:<10}{a:<10}{c:>5}{mark}")

    cost = np.array([r["cost_ms"] for r in E], dtype=float)
    print(f"\n  measured cost per forced action: {cost.mean():.1f} ms "
          f"(median {np.median(cost):.1f})")
    print(f"  rule spends {cost[[r['rule_would_fire'] for r in E]].sum()/1000:.1f} s "
          f"across {len(fired)} firings on this split")

    # ---- 2. can the benefit be predicted? -------------------------------
    yf = np.array([(applied_outcome(r) == "TP") - (r["outcome_before"] == "TP")
                   for r in F], dtype=float)
    ye = np.array([(applied_outcome(r) == "TP") - (r["outcome_before"] == "TP")
                   for r in E], dtype=float)
    print(f"\n=== Q(s, ZOOM): can the gain be predicted? ===\n")
    print(f"  fit split  {args.fit}: {int((yf>0).sum())} improved, "
          f"{int((yf<0).sum())} worsened, {int((yf==0).sum())} unchanged")
    print(f"  eval split {args.ev}: {int((ye>0).sum())} improved, "
          f"{int((ye<0).sum())} worsened, {int((ye==0).sum())} unchanged")

    # Persist the measurement before the fittability check. The action being
    # inert is a result in its own right, and an early return that skipped the
    # write left downstream consumers (the figure script) with no Phase A file.
    dest = ACT / f"{args.ev}_action_value.json"
    summary = {
        "fit": args.fit, "eval": args.ev,
        "n_actions": n, "candidate_found": len(found), "rule_kept": len(kept),
        "rule_fired": len(fired),
        "correct_no_zoom": float(corr_b.mean()),
        "correct_rule": float(corr_applied.mean()),
        "rule_net_effect": m, "rule_net_ci95": ci,
        "mean_cost_ms": float(cost.mean()),
        "value_model_fitted": False,
    }
    dest.write_text(json.dumps(summary, indent=2) + "\n")

    if (yf != 0).sum() < 20:
        print("\n  Too few outcome changes to fit a value model. The action is")
        print("  near-inert on this data, which is itself the finding: a policy")
        print("  cannot learn to allocate an action that does nothing.")
        print(f"\nWrote {dest.relative_to(ROOT)} (measurement only, no model)")
        return

    Xf_raw, Xe_raw = matrix(F), matrix(E)
    mu, sd = Xf_raw.mean(0), Xf_raw.std(0) + 1e-9
    Xf, Xe = (Xf_raw - mu) / sd, (Xe_raw - mu) / sd
    w = fit_ridge(Xf, yf)
    q = predict(Xe, w)

    print("\n  standardised weights (positive = zoom more likely to help)")
    for name, c in sorted(zip(FEATURES, w[1:]), key=lambda t: -abs(t[1])):
        print(f"    {name:<20}{c:>+9.5f}")

    order = np.argsort(-q)
    print(f"\n  realised mean gain by predicted-value decile ({args.ev})")
    print(f"    {'decile':<9}{'n':>6}{'mean gain':>12}{'cum. gain':>12}")
    dec = np.array_split(order, 10)
    cum = 0.0
    for i, ix in enumerate(dec, 1):
        g = ye[ix].sum()
        cum += g
        print(f"    {i:<9}{len(ix):>6}{ye[ix].mean():>12.5f}{cum:>12.2f}")

    # ---- 3. budget-aware policy ------------------------------------------
    print(f"\n=== Budget-aware policy vs the deployed rule ({args.ev}) ===\n")
    rule_fire = np.array([r["rule_would_fire"] for r in E])
    rule_gain = ye[rule_fire].sum()
    rule_cost = cost[rule_fire].sum()
    print(f"  deployed rule: fires {rule_fire.sum():>5} times, "
          f"net gain {rule_gain:+.2f} corrections, cost {rule_cost/1000:.1f} s")

    print(f"\n  {'budget':<10}{'fires':>7}{'net gain':>11}{'cost s':>9}"
          f"{'gain/rule':>11}{'cost/rule':>11}")
    for frac in (0.1, 0.2, 0.3, 0.5, 1.0):
        k = int(len(E) * frac)
        sel = order[:k]
        g, c = ye[sel].sum(), cost[sel].sum()
        print(f"  top {frac:<6.0%}{k:>7}{g:>+11.2f}{c/1000:>9.1f}"
              f"{(g/rule_gain if rule_gain else float('nan')):>11.2f}"
              f"{c/rule_cost:>11.2f}")

    print("\n  A policy reaching the rule's net gain at a fraction of its cost is")
    print("  the H2 claim: same reliability, less computation.")

    summary.update({"features": FEATURES, "weights": list(map(float, w)),
                    "value_model_fitted": True})
    dest.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
