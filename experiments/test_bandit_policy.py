"""Confirmatory test of H1′ / H2 on S_test (PROTOCOL §12). Run once.

Replays the S_test full-information action logs under the rule agent and the
frozen learned policy (plus descriptive never-act and oracle rows), then tests:

    H1′  AURC(learned) < AURC(rules), with discretionary ms(learned) <= ms(rules)
    H2   ms(learned) < ms(rules), with selective risk at full coverage
         non-inferior (upper 95% bound of learned − rules < +0.01)

Paired dive-cluster bootstrap, 2,000 resamples, seed 0, one-sided; Holm over the
§7.4 family of three.

Usage:
    python experiments/log_bandit_outcomes.py --split S_test --confirmatory
    python experiments/test_bandit_policy.py
"""

from __future__ import annotations

import hashlib
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
from fit_bandit_policy import (LOGS, NEVER, RULES, evaluate, learned, load,  # noqa: E402
                               oracle, summarise)

POLICY = LOGS / "bandit_policy.json"
SHA256 = "03ff9e7d8f46abfe2bc53ee2a7a01bb469c2046688551e8ab67b5dec0bd37f32"
N_BOOT = 2000
NI_MARGIN = 0.01


def boot(rows_r, rows_l):
    by = defaultdict(lambda: ([], []))
    for a, b in zip(rows_r, rows_l):
        by[a["dive"]][0].append(a)
        by[a["dive"]][1].append(b)
    dives = list(by)
    rng = random.Random(0)
    keys = ("aurc", "discretionary_ms_per_image", "precision", "recall", "f1", "reward_per_image")
    d = {k: [] for k in keys}
    for _ in range(N_BOOT):
        sr, sl = [], []
        for _ in dives:
            k = dives[rng.randrange(len(dives))]
            sr += by[k][0]
            sl += by[k][1]
        mr, ml = summarise(sr), summarise(sl)
        for k in keys:
            d[k].append(ml[k] - mr[k])
    return {k: np.array(v) for k, v in d.items()}, len(dives)


def ci(v):
    s = np.sort(v)
    return [float(s[int(0.025 * len(s))]), float(s[int(0.975 * len(s)) - 1])]


def main() -> None:
    digest = hashlib.sha256(POLICY.read_bytes()).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"policy file changed since registration ({digest}); refusing to run")
    spec = json.loads(POLICY.read_text())

    man = json.loads(MAN.read_text())
    classes = sorted(man["label_space"]["known"])
    assert classes == spec["classes"]
    harness.KNOWN_CLASSES = set(classes)
    gt_all = harness.load_ground_truth()
    recs = load("S_test")
    assert {r["file"] for r in recs} == set(man["files"]["S_test"]), "S_test log incomplete"

    pols = {"never": NEVER, "rules": RULES, "learned": learned(spec["models"], classes),
            "oracle": oracle(gt_all)}
    rows, res = {}, {}
    for name, p in pols.items():
        rows[name] = evaluate(recs, p, gt_all)
        res[name] = s = summarise(rows[name])
        a = s["actions_per_image"]
        print(f"S_test {name:8s} rep {s['reported']:5d} P {s['precision']:.3f} R {s['recall']:.3f} "
              f"F1 {s['f1']:.3f} AURC {s['aurc']:.4f} reward/img {s['reward_per_image']:+.3f} "
              f"disc {s['discretionary_ms_per_image']:6.1f} ms/img  enh {a.get('enhance', 0):.2f} "
              f"zoom {a.get('zoom', 0):.2f} vote {a.get('vote', 0):.2f}", flush=True)

    d, n_dives = boot(rows["rules"], rows["learned"])
    # One-sided p-values: share of resamples on the wrong side of zero.
    p_h1 = float((d["aurc"] >= 0).mean())
    cost_ok_h1 = res["learned"]["discretionary_ms_per_image"] <= res["rules"]["discretionary_ms_per_image"]
    p_h2 = float((d["discretionary_ms_per_image"] >= 0).mean())
    risk_diff = -d["precision"]                                   # risk = 1 − precision
    risk_ub = float(np.sort(risk_diff)[int(0.95 * N_BOOT) - 1])
    ni_h2 = risk_ub < NI_MARGIN

    # Holm over the family of three (H3 untested here: counted, assumed not smaller).
    m = 3
    ps = sorted([("H1′", p_h1), ("H2", p_h2)], key=lambda t: t[1])
    adj, prev = {}, 0.0
    for i, (h, p) in enumerate(ps):
        prev = max(prev, min(1.0, (m - i) * p))
        adj[h] = prev

    out = {
        "split": "S_test", "images": len(recs), "dives": n_dives, "policy_sha256": digest,
        "results": res,
        "learned_minus_rules": {k: {"mean": float(v.mean()), "ci95": ci(v)} for k, v in d.items()},
        "H1_prime": {"aurc_rules": res["rules"]["aurc"], "aurc_learned": res["learned"]["aurc"],
                     "p_one_sided": p_h1, "p_holm": adj["H1′"], "cost_condition_met": cost_ok_h1,
                     "supported": bool(adj["H1′"] < 0.05 and cost_ok_h1)},
        "H2": {"ms_rules": res["rules"]["discretionary_ms_per_image"],
               "ms_learned": res["learned"]["discretionary_ms_per_image"],
               "p_one_sided": p_h2, "p_holm": adj["H2"],
               "risk_diff_upper95": risk_ub, "noninferiority_margin": NI_MARGIN,
               "risk_noninferior": bool(ni_h2), "supported": bool(adj["H2"] < 0.05 and ni_h2)},
    }
    for h in ("H1_prime", "H2"):
        print(h, json.dumps(out[h]))
    for k, v in out["learned_minus_rules"].items():
        print(f"  learned − rules {k:28s} {v['mean']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
    (LOGS / "S_test_confirmatory.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {(LOGS / 'S_test_confirmatory.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
