"""Week 1: fit the two-step image agent by fitted Q-iteration on S_cal.

Decisions (per image, after the deployed first pass):
    D5  re-run on an enhanced image with arm a ∈ ARMS, or stop
    D6  after the re-run, keep the first or the second pass

Reward per image: +1 TP, −1 reported error, 0 Unknown (Step 3). Relative to
stopping, Δ_a = R(second pass) − R(first pass).

Fitted Q-iteration, two steps, exact full-information targets:
    Q2_a(s2) ≈ Δ_a                       ridge;   D6 keeps the second pass iff Q2 > 0
    g_a      = Δ_a · 1[Q2_a(s2) > 0]      realised gain of the D6 policy (out of fold)
    Q1_a(s1) ≈ g_a                        ridge;   D5 re-runs with argmax_a Q1_a − λ c_a if > 0

Evaluation on S_cal uses nested, dive-grouped cross-fitting: for each outer
fold the Q2 targets for Q1 are themselves out-of-fold within the training
folds, so no image is scored by a model that saw it, directly or through its
targets. Compared, as reward gain per image over the first pass:
never, always re-run (keep second), always re-run + learned D6, one-step
(re-run and keep iff a single model on s1 predicts Δ > 0), two-step FQI,
oracle. λ sweep for the gain–compute curve. Dive-cluster bootstrap CIs.

The final policy is refitted on all of S_cal (λ = 0 here; the λ used for an
evaluation is registered before it) and written to
models/phase_b_machine/image_agent_policy.json (+ a committed copy).

Usage:
    python experiments/fit_image_agent.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from ablation_deepsea import MAN  # noqa: E402
from eval import harness  # noqa: E402
from fit_bandit_policy import ridge_fit, ridge_predict  # noqa: E402

DIR = ROOT / "experiments" / "results" / "phase_b_machine" / "week1"
POLICY_OUT = ROOT / "models" / "phase_b_machine" / "image_agent_policy.json"
ALPHA = 10.0
FOLDS = 5
N_BOOT = 2000
LAMBDAS = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05]       # reward per ms


def reward(finals, gts):
    dets = [d for d in finals if not d["unknown"]]
    return sum(1 if r["outcome"] == harness.TP else -1 for r in harness.match_and_classify(dets, gts))


def folds_of(dives, k, seed=0):
    ud = sorted(set(dives))
    random.Random(seed).shuffle(ud)
    f = {d: i % k for i, d in enumerate(ud)}
    return np.array([f[d] for d in dives])


def fit_pred(X, y, tr, te):
    return ridge_predict(ridge_fit(X[tr], y[tr], ALPHA), X[te])


def inner_oof(X, y, dives, idx):
    """OOF predictions on rows *idx* using folds of their own dives."""
    f = folds_of([dives[i] for i in idx], FOLDS, seed=1)
    out = np.zeros(len(idx))
    for k in range(FOLDS):
        tr, te = idx[f != k], idx[f == k]
        if len(te) and len(tr) > X.shape[1]:
            out[f == k] = fit_pred(X, y, tr, te)
    return out


def main() -> None:
    man = json.loads(MAN.read_text())
    harness.KNOWN_CLASSES = set(man["label_space"]["known"])
    gt_all = harness.load_ground_truth()
    recs = [json.loads(line) for line in (DIR / "S_cal_image_agent.jsonl").open(encoding="utf-8")]
    arms = sorted(recs[0]["arms"])
    recs = [r for r in recs if all(not r["arms"][a].get("skipped") for a in arms)]
    n = len(recs)
    dives = [r["dive"] for r in recs]
    S1 = np.array([recs[i]["arms"][arms[0]]["s1"] for i in range(n)], float)
    S2 = {a: np.array([r["arms"][a]["s2"] for r in recs], float) for a in arms}
    D = {a: np.array([reward(r["arms"][a]["passes"][1]["finals"], gt_all.get(r["file"], []))
                      - reward(r["arms"][a]["passes"][0]["finals"], gt_all.get(r["file"], []))
                      for r in recs], float) for a in arms}
    C = {a: np.array([r["arms"][a]["pass2_ms"] + r["arms"][a]["probe_ms"][1] for r in recs]) for a in arms}
    probe0 = np.array([r["arms"][arms[0]]["probe_ms"][0] for r in recs])
    base_r = np.array([reward(r["arms"][arms[0]]["passes"][0]["finals"], gt_all.get(r["file"], []))
                       for r in recs], float)
    print(f"S_cal: {n} images, {len(set(dives))} dives, arms {arms}; first pass reward/img "
          f"{base_r.mean():+.3f}; probe cost (pass 1) {probe0.mean():.1f} ms/img")

    # ── Nested cross-fitted predictions ────────────────────────────
    outer = folds_of(dives, FOLDS)
    q2, q1, one = ({a: np.zeros(n) for a in arms} for _ in range(3))
    for k in range(FOLDS):
        tr, te = np.where(outer != k)[0], np.where(outer == k)[0]
        for a in arms:
            q2[a][te] = fit_pred(S2[a], D[a], tr, te)
            g_tr = D[a][tr] * (inner_oof(S2[a], D[a], dives, tr) > 0)
            y1 = np.zeros(n)
            y1[tr] = g_tr
            q1[a][te] = fit_pred(S1, y1, tr, te)
            one[a][te] = fit_pred(S1, D[a], tr, te)

    def two_step(lam, use=arms):
        val = np.stack([q1[a] - lam * C[a].mean() for a in use], 1)
        pick = val.argmax(1)
        act = val.max(1) > 0
        gain = np.array([D[use[p]][i] * (q2[use[p]][i] > 0) if act[i] else 0.0 for i, p in enumerate(pick)])
        cost = np.array([C[use[p]][i] if act[i] else 0.0 for i, p in enumerate(pick)])
        return gain, cost, act

    policies = {"never": (np.zeros(n), np.zeros(n), np.zeros(n, bool))}
    for a in arms:
        policies[f"always {a}"] = (D[a], C[a], np.ones(n, bool))
        policies[f"always {a} + D6"] = (D[a] * (q2[a] > 0), C[a], np.ones(n, bool))
        act = one[a] > 0
        policies[f"one-step {a}"] = (D[a] * act, C[a] * act, act)
        policies[f"two-step {a}"] = two_step(0.0, [a])
    policies["two-step all arms"] = two_step(0.0)
    best = np.stack([D[a] for a in arms], 1)
    policies["oracle"] = (np.maximum(best.max(1), 0), np.where(best.max(1) > 0, C[arms[0]], 0.0),
                          best.max(1) > 0)

    by_dive = defaultdict(list)
    for i, d in enumerate(dives):
        by_dive[d].append(i)
    ud = sorted(by_dive)
    rng = random.Random(0)
    samples = [np.array([i for _ in ud for i in by_dive[ud[rng.randrange(len(ud))]]]) for _ in range(N_BOOT)]

    def ci(v):
        b = np.sort([v[s].mean() for s in samples])
        return [float(b[int(0.025 * N_BOOT)]), float(b[int(0.975 * N_BOOT) - 1])]

    res = {"images": n, "dives": len(ud), "arms": arms, "first_pass_reward_per_image": float(base_r.mean()),
           "probe_ms_per_image": float(probe0.mean()), "policies": {}, "comparisons": {}, "lambda_sweep": []}
    print(f"\n{'policy':26s} {'gain/img':>9s} {'95% CI':>18s} {'re-run %':>8s} {'extra ms/img':>12s}")
    for name, (g, c, act) in policies.items():
        res["policies"][name] = {"gain": float(g.mean()), "ci95": ci(g), "rerun_rate": float(act.mean()),
                                 "extra_ms": float(c.mean())}
        print(f"{name:26s} {g.mean():+9.3f} [{ci(g)[0]:+.3f},{ci(g)[1]:+.3f}] {100 * act.mean():8.1f} "
              f"{c.mean():12.1f}")

    main_arm = max(arms, key=lambda a: res["policies"][f"two-step {a}"]["gain"])
    ts = policies[f"two-step {main_arm}"][0]
    for other in (f"one-step {main_arm}", f"always {main_arm} + D6", "never"):
        d = ts - policies[other][0]
        res["comparisons"][f"two-step {main_arm} − {other}"] = {"diff": float(d.mean()), "ci95": ci(d)}
        print(f"two-step {main_arm} − {other:22s} {d.mean():+.3f} [{ci(d)[0]:+.3f},{ci(d)[1]:+.3f}]")

    print(f"\nλ sweep (two-step {main_arm}): λ, gain/img, re-run %, extra ms/img")
    for lam in LAMBDAS:
        g, c, act = two_step(lam, [main_arm])
        res["lambda_sweep"].append({"lambda": lam, "gain": float(g.mean()), "ci95": ci(g),
                                    "rerun_rate": float(act.mean()), "extra_ms": float(c.mean())})
        print(f"  {lam:6.3f}  {g.mean():+.3f} [{ci(g)[0]:+.3f},{ci(g)[1]:+.3f}]  {100 * act.mean():5.1f}%  "
              f"{c.mean():5.1f}")

    # ── Final policy on all of S_cal ───────────────────────────────
    allidx = np.arange(n)
    spec = {"fitted_on": "S_cal", "method": "fitted Q-iteration, 2 steps, ridge alpha 10",
            "arms_evaluated": arms, "lambda": 0.0, "cost_ms": {}, "q1": {}, "q2": {},
            "state": "s1 = image_features + probes.pass_state(pass 1); s2 = s1 + pass_state(pass 2)"}
    for a in [main_arm]:
        spec["q2"][a] = ridge_fit(S2[a], D[a], ALPHA)
        g = D[a] * (inner_oof(S2[a], D[a], dives, allidx) > 0)
        spec["q1"][a] = ridge_fit(S1, g, ALPHA)
        spec["cost_ms"][a] = float(C[a].mean())
    res["deployed_arm"] = main_arm
    POLICY_OUT.parent.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(spec, indent=2) + "\n"
    POLICY_OUT.write_text(txt)
    (DIR / "image_agent_policy.json").write_text(txt)
    (DIR / "fit_results.json").write_text(json.dumps(res, indent=2) + "\n")
    print(f"\nwrote {POLICY_OUT.relative_to(ROOT)} (arm {main_arm}) and week1/fit_results.json")


if __name__ == "__main__":
    main()
