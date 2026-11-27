"""Option D in the app: verifier risk on image agent + verifier together (PROTOCOL §21).

Descriptive. Compares the verifier on the deployed bandit's S_policy boxes
(week3/S_policy_verifier.jsonl) with the verifier on option D's S_policy boxes
(option_d/S_policy_verifier.jsonl), same frozen combiner and S_policy thresholds.

Usage:
    python experiments/option_d_check.py
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from ablation_deepsea import DIVE_RE, MAN  # noqa: E402
from agentic import taxonomy as tx  # noqa: E402
from week3_verifier import box_rows, decide, dive_risk  # noqa: E402

R = ROOT / "experiments" / "results" / "phase_b_machine"
LOGS = {"bandit": R / "week3" / "S_policy_verifier.jsonl", "option_d": R / "option_d" / "S_policy_verifier.jsonl"}
BANDIT_HASH = "d36ec85c"
OUT = R / "option_d" / "option_d_check.json"
N_BOOT = 2000


def load(path, spec):
    recs = [json.loads(line) for line in path.open(encoding="utf-8")]
    boxes, dives = [], []
    for r in recs:
        for b in r["boxes"]:
            boxes.append(b)
            dives.append(r["dive"])
    F, Y = [], []
    for b in boxes:
        f, y, _ = box_rows(b, spec["classes"])
        F.append(f)
        Y.append(y)
    F, Y = np.array(F), np.array(Y, bool)
    P = np.zeros((len(boxes), 4))
    for j, lv in enumerate(tx.LEVELS):
        m = spec["models"][lv]
        P[:, j] = 1 / (1 + np.exp(-(m["w"][0] + ((F[:, j] - np.array(m["mean"])) / np.array(m["std"]))
                                    @ np.array(m["w"][1:]))))
    return boxes, dives, P, Y


def per_dive(ch, Y, dives):
    """dive → (sum wrong / n boxes, n correct)."""
    wr, ok = defaultdict(list), defaultdict(float)
    for i, d in enumerate(dives):
        wr[d].append(float(ch[i] >= 0 and not Y[i, ch[i]]))
        ok[d] += float(ch[i] >= 0 and Y[i, ch[i]])
    return {d: float(np.mean(v)) for d, v in wr.items()}, ok


def main() -> None:
    got = hashlib.sha256(LOGS["bandit"].read_bytes()).hexdigest()
    if not got.startswith(BANDIT_HASH):
        raise SystemExit(f"bandit log hash {got[:8]} != {BANDIT_HASH}")
    man = json.loads(MAN.read_text())
    img_per_dive = Counter(DIVE_RE.match(f).group(1) for f in man["files"]["S_policy"])
    n_img = sum(img_per_dive.values())
    spec = json.loads((R / "week6" / "verifier_deploy.json").read_text())
    thr = spec["thresholds"]
    data = {k: load(p, spec) for k, p in LOGS.items()}
    ud = sorted(img_per_dive)
    rng = random.Random(0)
    samples = [[ud[rng.randrange(len(ud))] for _ in ud] for _ in range(N_BOOT)]
    out = {"protocol": "PROTOCOL §21", "images": n_img, "dives": len(ud),
           "boxes": {k: len(v[0]) for k, v in data.items()}, "alphas": {}}
    for alpha in (0.10, 0.20):
        a = {}
        for mode, levels in (("hierarchical", [0, 1, 2, 3]), ("flat", [0])):
            t = thr[mode][str(alpha)]
            res, pd = {}, {}
            for k, (boxes, dives, P, Y) in data.items():
                ch = decide(P, t, levels)
                ok = np.array([ch[i] >= 0 and Y[i, ch[i]] for i in range(len(ch))])
                res[k] = {"risk": dive_risk(ch, Y, dives, np.arange(len(ch)))[0],
                          "answers_per_image": float((ch >= 0).sum() / n_img),
                          "correct_per_image": float(ok.sum() / n_img),
                          "species_correct_per_image": float((ok & (ch == 0)).sum() / n_img),
                          "error_among_reported": float(1 - ok.sum() / max((ch >= 0).sum(), 1))}
                pd[k] = per_dive(ch, Y, dives)
            dr, dc = [], []
            for s in samples:
                ni = sum(img_per_dive[d] for d in s)
                risks = {k: [pd[k][0][d] for d in s if d in pd[k][0]] for k in pd}
                dr.append(np.mean(risks["option_d"]) - np.mean(risks["bandit"]))
                dc.append((sum(pd["option_d"][1].get(d, 0) for d in s) - sum(pd["bandit"][1].get(d, 0) for d in s)) / ni)
            dr, dc = np.sort(dr), np.sort(dc)
            ci = lambda v: [float(v[int(0.025 * N_BOOT)]), float(v[int(0.975 * N_BOOT) - 1])]  # noqa: E731
            res["option_d_minus_bandit"] = {
                "risk": res["option_d"]["risk"] - res["bandit"]["risk"], "risk_ci95": ci(dr),
                "correct_per_image": res["option_d"]["correct_per_image"] - res["bandit"]["correct_per_image"],
                "correct_ci95": ci(dc)}
            a[mode] = dict(res, threshold=t)
        out["alphas"][str(alpha)] = a
    h = out["alphas"]["0.2"]["hierarchical"]
    out["reading_21_4"] = {"risk_le_alpha": h["option_d"]["risk"] <= 0.20,
                           "risk_diff_ci_lower_le_0": h["option_d_minus_bandit"]["risk_ci95"][0] <= 0}
    out["guarantee_carries_over"] = all(out["reading_21_4"].values())
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["alphas"]["0.2"]["hierarchical"], indent=1))
    print("guarantee carries over:", out["guarantee_carries_over"])


if __name__ == "__main__":
    main()
