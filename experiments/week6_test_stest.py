"""Week 6: the confirmatory S_test run of §15 (PROTOCOL §16). Single use.

Part A reuses week5_test_spolicy.part_a unchanged, on S_test. Part B uses the
thresholds fixed on S_policy (week6/thresholds_S_policy.json) on all S_test dives,
with no calibration on S_test. Refuses to run if a hash differs or the result exists.

Usage:
    python experiments/week6_test_stest.py
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

import week5_test_spolicy as w5  # noqa: E402
from ablation_deepsea import DIVE_RE, MAN  # noqa: E402
from agentic import taxonomy as tx  # noqa: E402
from eval import harness  # noqa: E402
from week3_verifier import box_rows, decide, dive_risk  # noqa: E402

R = w5.R
OUT = R / "week6" / "S_test_confirmatory.json"
THR = R / "week6" / "thresholds_S_policy.json"
FROZEN = dict(w5.FROZEN) | {THR: "d583d9d2386e840fb9b08644dc7cabacf4f9d87d755219deaac040fc4fe2f749"}


def part_b(man, split, role):
    spec = json.loads((R / "week4" / "verifier.json").read_text())
    thr = json.loads(THR.read_text())["thresholds"]
    role_taxa = set(man["label_space"]["ood_roles"][role])
    recs = [json.loads(line) for line in (R / "week3" / f"{split}_verifier.jsonl").open(encoding="utf-8")]
    img_per_dive = Counter(DIVE_RE.match(f).group(1) for f in man["files"][split])
    n_img = sum(img_per_dive.values())
    boxes, dives = [], []
    for r in recs:
        for b in r["boxes"]:
            boxes.append(b)
            dives.append(r["dive"])
    n = len(boxes)
    F, Y = [], []
    for b in boxes:
        f, y, _ = box_rows(b, spec["classes"])
        F.append(f)
        Y.append(y)
    F, Y = np.array(F), np.array(Y, bool)
    P = np.zeros((n, 4))
    for j, lv in enumerate(tx.LEVELS):
        m = spec["models"][lv]
        P[:, j] = 1 / (1 + np.exp(-(m["w"][0] + ((F[:, j] - np.array(m["mean"])) / np.array(m["std"]))
                                    @ np.array(m["w"][1:]))))
    allidx = np.arange(n)
    role_box = np.array([b["gt_class"] in role_taxa for b in boxes])
    today = np.array([0 if not b["unknown"] else -1 for b in boxes])
    res = {"images": n_img, "dives": len(img_per_dive), "boxes": n, "role_boxes": int(role_box.sum()),
           "today_risk": dive_risk(today, Y, dives, allidx)[0], "alphas": {}}
    box_by = defaultdict(list)
    for i, d in enumerate(dives):
        box_by[d].append(i)
    ud = sorted(img_per_dive)
    rng = random.Random(0)
    samples = [[ud[rng.randrange(len(ud))] for _ in ud] for _ in range(w5.N_BOOT)]
    choices = {}
    for alpha in (0.10, 0.20):
        a = {}
        for mode, levels in (("hierarchical", [0, 1, 2, 3]), ("flat", [0])):
            t = thr[mode][str(alpha)]
            ch = decide(P, 2.0 if t is None else t, levels)
            choices[(alpha, mode)] = ch
            risk, _ = dive_risk(ch, Y, dives, allidx)
            per_dive = {d: np.mean([float(ch[i] >= 0 and not Y[i, ch[i]]) for i in bi]) for d, bi in box_by.items()}
            share = [np.mean(v) <= alpha for v in ([per_dive[d] for d in s if d in per_dive] for s in samples) if v]
            ok = np.array([ch[i] >= 0 and Y[i, ch[i]] for i in range(n)])
            a[mode] = {"threshold": t, "risk": risk, "risk_le_alpha": risk <= alpha,
                       "share_resamples_le_alpha": float(np.mean(share)),
                       "reported_per_image": float((ch >= 0).sum() / n_img),
                       "correct_per_image": float(ok.sum() / n_img),
                       "species_correct_per_image": float((ok & (ch == 0)).sum() / n_img),
                       "error_among_reported": float(1 - ok.sum() / max((ch >= 0).sum(), 1)),
                       "role_correct": int((ok & role_box).sum()),
                       "role_correct_nodes": dict(Counter(tx.LEVELS[ch[i]] for i in range(n) if ok[i] and role_box[i])),
                       "correct_by_level": {lv: int(((ch == j) & ok).sum()) for j, lv in enumerate(tx.LEVELS)}}
        res["alphas"][str(alpha)] = a
    ch_h, ch_f = choices[(0.20, "hierarchical")], choices[(0.20, "flat")]
    ok_h = np.array([ch_h[i] >= 0 and Y[i, ch_h[i]] for i in range(n)], float)
    ok_f = np.array([ch_f[i] >= 0 and Y[i, ch_f[i]] for i in range(n)], float)
    sp_h, sp_f = ok_h * (ch_h == 0), ok_f * (ch_f == 0)
    d4, c5, d6 = [], [], []
    for s in samples:
        idx = [i for d in s for i in box_by[d]]
        ni = sum(img_per_dive[d] for d in s)
        d4.append((ok_h[idx].sum() - ok_f[idx].sum()) / ni)
        c5.append(float((ok_h * role_box)[idx].sum()))
        d6.append((sp_h[idx].sum() - sp_f[idx].sum()) / ni)
    d4, c5, d6 = np.sort(d4), np.sort(c5), np.sort(d6)
    ci = lambda v: [float(v[int(0.025 * w5.N_BOOT)]), float(v[int(0.975 * w5.N_BOOT) - 1])]   # noqa: E731
    tests = {"H4": {"diff_correct_per_image": float((ok_h.sum() - ok_f.sum()) / n_img), "ci95": ci(d4),
                    "p": float((d4 <= 0).mean())},
             "H5": {"role_correct": int((ok_h * role_box).sum()), "role_boxes": int(role_box.sum()),
                    "ci95": ci(c5), "p": float((c5 <= 0).mean())},
             "H6": {"diff_species_correct_per_image": float((sp_h.sum() - sp_f.sum()) / n_img),
                    "lower95": float(d6[int(0.05 * w5.N_BOOT)]), "p": float((d6 <= -0.05).mean())}}
    for k, v in w5.holm({k: tests[k]["p"] for k in tests}).items():
        tests[k]["supported_holm"] = v
    res["tests"] = tests
    return res


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT.relative_to(ROOT)} exists: §16 is single-use")
    for f, h in FROZEN.items():
        got = hashlib.sha256(f.read_bytes()).hexdigest()
        if got != h:
            raise SystemExit(f"hash mismatch for {f.name}: {got}")
    man = json.loads(MAN.read_text())
    harness.KNOWN_CLASSES = set(man["label_space"]["known"])
    gt = harness.load_ground_truth()
    w5.SPLIT, w5.ROLE = "S_test", "OOD_test"
    out = {"protocol": "PROTOCOL §16", "part_a": w5.part_a(gt), "part_b": part_b(man, "S_test", "OOD_test")}
    enc = lambda o: bool(o) if isinstance(o, np.bool_) else float(o)   # noqa: E731
    OUT.write_text(json.dumps(out, indent=2, default=enc) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
