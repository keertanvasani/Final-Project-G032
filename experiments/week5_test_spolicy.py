"""Week 5: the registered S_policy test of the image agent and the verifier (PROTOCOL §15).

Run once. Refuses to run if a frozen file's sha256 differs from §15.1, or if the
result file already exists (single use).

Part A — image agent, exact replay of full-information logs:
    H1-R  agent at λ_m vs rule agent: reward/img and F1 higher (p = larger of the two)
    H1-A  agent at λ_m vs rules: AURC upper 95% bound of the difference < +0.01
    H7    two-step vs one-step at λ = 0.002: reward/img higher
    H8    agent at λ = 0.002 vs deployed bandit: reward/img higher
Part B — verifier, S_policy dives split in half (calibration / evaluation):
    H-risk  dive-level risk on the evaluation half ≤ α (hier and flat), α ∈ {0.10, 0.20}
    H4      α = 0.20: hierarchical correct answers/img > flat
    H5      α = 0.20: hierarchical correct answers on OOD_val boxes > 0
    H6      α = 0.20: species-level correct/img, hier − flat > −0.05
Paired dive bootstrap, 2,000 resamples, seed 0; Holm at 0.05 within each part.

Usage:
    python experiments/week5_test_spolicy.py
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
from agentic.probes import state_columns  # noqa: E402
from eval import harness  # noqa: E402
from fit_bandit_policy import RULES, evaluate, load, ridge_predict, summarise  # noqa: E402
from week3_verifier import box_rows, decide, dive_risk, ltt  # noqa: E402

R = ROOT / "experiments" / "results" / "phase_b_machine"
OUT = R / "week5" / "S_policy_test.json"
FROZEN = {R / "week2" / "image_agent_policy.json": "fe1f5439a27f0a76c6be93455835db079fd0dbbcb8a323028349d804484f61bb",
          R / "week5" / "one_step_policy.json": "5a77c48ec0e1d371e23c1854e179ba90aeeb73174f9e47d8c144b1b482383fa2",
          R / "week4" / "verifier.json": "fed9559f23128b18c0b270bad5c5029670a8579d26ca6876233f0e040927dc9b"}
LAM_M = 0.00892128865561972
LAM = 0.002
ARM = "clahe3.0"
N_BOOT = 2000
FW = 0.05
SPLIT = "S_policy"
ROLE = "OOD_val"
DEPLOYED = {"enhance": lambda rec: False, "zoom": lambda rec, e, i, box: False,
            "vote": lambda rec, e, i, box, z: True}


def holm(ps):
    order = sorted(ps, key=ps.get)
    out, ok = {}, True
    for i, k in enumerate(order):
        ok = ok and ps[k] <= FW / (len(order) - i)
        out[k] = ok
    return out


# ── Part A ───────────────────────────────────────────────────────────
def part_a(gt):
    agent = json.loads((R / "week2" / "image_agent_policy.json").read_text())
    one = json.loads((R / "week5" / "one_step_policy.json").read_text())
    s3 = {r["file"]: r for r in load(SPLIT)}
    logs = [json.loads(line) for line in (R / "week1" / f"{SPLIT}_image_agent.jsonl").open(encoding="utf-8")]
    logs = [r for r in logs if r["file"] in s3]
    order = [s3[r["file"]] for r in logs]
    rules, deployed = evaluate(order, RULES, gt), evaluate(order, DEPLOYED, gt)
    c1, c2 = state_columns(agent["probe_keys"], 1), state_columns(agent["probe_keys"], 2)
    ok = [not r["arms"][ARM].get("skipped") for r in logs]
    S1 = np.array([r["arms"][ARM]["s1"] if k else [0.0] * 64 for r, k in zip(logs, ok)], float)[:, c1]
    S2 = np.array([r["arms"][ARM]["s2"] if k else [0.0] * 120 for r, k in zip(logs, ok)], float)[:, c2]
    q1 = ridge_predict(agent["q1"][ARM], S1)
    q2 = ridge_predict(agent["q2"][ARM], S2)
    qo = ridge_predict(one["q"], S1)
    ok = np.array(ok)

    def rows_for(act, keep):
        out = []
        for i, (rec, base) in enumerate(zip(logs, deployed)):
            if keep[i]:
                fin = [d for d in rec["arms"][ARM]["passes"][1]["finals"] if not d["unknown"]]
                recs_ = harness.match_and_classify(fin, gt.get(rec["file"], []))
            else:
                recs_ = base["records"]
            extra = rec["arms"][ARM]["pass2_ms"] if act[i] else 0.0
            out.append({"file": rec["file"], "dive": rec["dive"], "records": recs_, "cost": base["cost"] + extra,
                        "acts": Counter(base["acts"]) + Counter(rerun=int(act[i]), kept=int(keep[i])),
                        "kgt": base["kgt"]})
        return out

    def two_step(lam):
        act = ok & (q1 - lam * agent["cost_ms"][ARM] > 0)
        return rows_for(act, act & (q2 > 0))

    act_o = ok & (qo - LAM * one["cost_ms"] > 0)
    arms = {"rules": rules, "deployed": deployed, "agent_lam_m": two_step(LAM_M),
            "agent_lam": two_step(LAM), "one_step_lam": rows_for(act_o, act_o)}
    keys = ("reward_per_image", "f1", "precision", "recall", "aurc", "discretionary_ms_per_image")
    res = {"images": len(logs), "skipped_arm_images": int((~ok).sum()), "policies": {}}
    for k, v in arms.items():
        s = summarise(v)
        res["policies"][k] = {m: s[m] for m in keys} | {"rerun_rate": s["actions_per_image"].get("rerun", 0.0)}

    by = defaultdict(list)
    for i, r in enumerate(logs):
        by[r["dive"]].append(i)
    ud = sorted(by)
    rng = random.Random(0)
    samples = [[i for _ in ud for i in by[ud[rng.randrange(len(ud))]]] for _ in range(N_BOOT)]

    def boot(a, b, metrics):
        """Differences b − a per resample for each metric."""
        d = {m: [] for m in metrics}
        for s in samples:
            sa, sb = summarise([arms[a][i] for i in s]), summarise([arms[b][i] for i in s])
            for m in metrics:
                d[m].append(sb[m] - sa[m])
        return {m: np.sort(v) for m, v in d.items()}

    def ci(v):
        return [float(v[int(0.025 * N_BOOT)]), float(v[int(0.975 * N_BOOT) - 1])]

    def p_gt0(v):
        return float((v <= 0).mean())

    full = lambda a, b, m: res["policies"][b][m] - res["policies"][a][m]   # noqa: E731
    h1 = boot("rules", "agent_lam_m", ("reward_per_image", "f1", "aurc", "precision", "recall",
                                       "discretionary_ms_per_image"))
    h7 = boot("one_step_lam", "agent_lam", ("reward_per_image", "f1", "discretionary_ms_per_image"))
    h8 = boot("deployed", "agent_lam", ("reward_per_image", "f1", "aurc", "discretionary_ms_per_image"))
    tests = {
        "H1-R": {"diff_reward": full("rules", "agent_lam_m", "reward_per_image"),
                 "ci_reward": ci(h1["reward_per_image"]), "p_reward": p_gt0(h1["reward_per_image"]),
                 "diff_f1": full("rules", "agent_lam_m", "f1"), "ci_f1": ci(h1["f1"]), "p_f1": p_gt0(h1["f1"])},
        "H7": {"diff_reward": full("one_step_lam", "agent_lam", "reward_per_image"),
               "ci_reward": ci(h7["reward_per_image"]), "p": p_gt0(h7["reward_per_image"])},
        "H8": {"diff_reward": full("deployed", "agent_lam", "reward_per_image"),
               "ci_reward": ci(h8["reward_per_image"]), "p": p_gt0(h8["reward_per_image"])},
    }
    tests["H1-R"]["p"] = max(tests["H1-R"]["p_reward"], tests["H1-R"]["p_f1"])
    ub = float(h1["aurc"][int(0.95 * N_BOOT) - 1])
    tests["H1-A"] = {"diff_aurc": full("rules", "agent_lam_m", "aurc"), "upper95": ub, "supported": ub < 0.01}
    rej = holm({k: tests[k]["p"] for k in ("H1-R", "H7", "H8")})
    for k, v in rej.items():
        tests[k]["supported_holm"] = v
    res["tests"] = tests
    res["descriptive"] = {"agent_lam_m_minus_rules": {m: ci(v) for m, v in h1.items()},
                          "agent_lam_minus_deployed": {m: ci(v) for m, v in h8.items()},
                          "agent_lam_minus_one_step": {m: ci(v) for m, v in h7.items()}}
    return res


# ── Part B ───────────────────────────────────────────────────────────
def part_b(man):
    spec = json.loads((R / "week4" / "verifier.json").read_text())
    excluded = set(man["label_space"]["held_out_unseen"]) - set(man["label_space"]["ood_roles"][ROLE])
    val_taxa = set(man["label_space"]["ood_roles"][ROLE])
    recs = [json.loads(line) for line in (R / "week3" / f"{SPLIT}_verifier.jsonl").open(encoding="utf-8")]
    classes = spec["classes"]
    all_dives = sorted({DIVE_RE.match(f).group(1) for f in man["files"][SPLIT]})
    sh = list(all_dives)
    random.Random(0).shuffle(sh)
    cal = set(sh[: len(sh) // 2])
    img_per_dive = Counter(DIVE_RE.match(f).group(1) for f in man["files"][SPLIT])
    boxes, dives = [], []
    for r in recs:
        for b in r["boxes"]:
            boxes.append(b)
            dives.append(r["dive"])
    n = len(boxes)
    F, Y = [], []
    for b in boxes:
        f, y, _ = box_rows(b, classes)
        F.append(f)
        Y.append(y)
    F, Y = np.array(F), np.array(Y, bool)
    P = np.zeros((n, 4))
    for j, lv in enumerate(tx.LEVELS):
        m = spec["models"][lv]
        z = (F[:, j] - np.array(m["mean"])) / np.array(m["std"])
        P[:, j] = 1 / (1 + np.exp(-(m["w"][0] + z @ np.array(m["w"][1:]))))
    in_cal = np.array([d in cal for d in dives])
    elig = np.array([b["gt_class"] not in excluded for b in boxes])
    i_cal = np.where(in_cal & elig)[0]
    i_ev = np.where(~in_cal)[0]
    ev_dives = sorted(set(all_dives) - cal)
    n_img_ev = sum(img_per_dive[d] for d in ev_dives)
    val_box = np.array([b["gt_class"] in val_taxa for b in boxes])
    today = np.array([0 if not b["unknown"] else -1 for b in boxes])
    res = {"dives_total": len(all_dives), "dives_calibration": len(cal), "dives_evaluation": len(ev_dives),
           "boxes_calibration_eligible": int(len(i_cal)), "boxes_evaluation": int(len(i_ev)),
           "images_evaluation": n_img_ev, "OOD_val_boxes_evaluation": int(val_box[i_ev].sum()),
           "today_risk_evaluation": dive_risk(today, Y, dives, i_ev)[0], "alphas": {}}
    rng = random.Random(0)
    bdives = sorted({dives[i] for i in i_ev} | set(ev_dives))
    box_by = defaultdict(list)
    for i in i_ev:
        box_by[dives[i]].append(i)
    samples = [[bdives[rng.randrange(len(bdives))] for _ in bdives] for _ in range(N_BOOT)]
    choices = {}
    for alpha in (0.10, 0.20):
        a = {}
        for mode, levels in (("hierarchical", [0, 1, 2, 3]), ("flat", [0])):
            t = ltt(P, Y, dives, i_cal, alpha, levels)
            ch = decide(P, 2.0 if t is None else t, levels)
            choices[(alpha, mode)] = ch
            risk, nd = dive_risk(ch, Y, dives, i_ev)
            share = []
            per_dive = {d: np.mean([float(ch[i] >= 0 and not Y[i, ch[i]]) for i in bi])
                        for d, bi in box_by.items() if bi}
            for s in samples:
                # Risk over the resampled dives that have boxes (duplicates counted), as dive_risk.
                vals = [per_dive[d] for d in s if d in per_dive]
                if vals:
                    share.append(np.mean(vals) <= alpha)
            ok = np.array([ch[i] >= 0 and Y[i, ch[i]] for i in range(n)])
            a[mode] = {"threshold": t, "risk_evaluation": risk, "risk_le_alpha": risk <= alpha,
                       "share_resamples_le_alpha": float(np.mean(share)),
                       "reported_per_image": float((ch[i_ev] >= 0).sum() / n_img_ev),
                       "correct_per_image": float(ok[i_ev].sum() / n_img_ev),
                       "species_correct_per_image": float((ok & (ch == 0))[i_ev].sum() / n_img_ev),
                       "error_among_reported": float(1 - ok[i_ev].sum() / max((ch[i_ev] >= 0).sum(), 1)),
                       "OOD_val_correct": int((ok & val_box)[i_ev].sum()),
                       "correct_by_level": {lv: int(((ch == j) & ok)[i_ev].sum()) for j, lv in enumerate(tx.LEVELS)}}
        res["alphas"][str(alpha)] = a

    # H4–H6 at α = 0.20.
    ch_h, ch_f = choices[(0.20, "hierarchical")], choices[(0.20, "flat")]
    ok_h = np.array([ch_h[i] >= 0 and Y[i, ch_h[i]] for i in range(n)], float)
    ok_f = np.array([ch_f[i] >= 0 and Y[i, ch_f[i]] for i in range(n)], float)
    sp_h = ok_h * (ch_h == 0)
    sp_f = ok_f * (ch_f == 0)
    d4, c5, d6 = [], [], []
    for s in samples:
        idx = [i for d in s for i in box_by[d]]
        ni = sum(img_per_dive[d] for d in s)
        d4.append((ok_h[idx].sum() - ok_f[idx].sum()) / ni)
        c5.append(float((ok_h * val_box)[idx].sum()))
        d6.append((sp_h[idx].sum() - sp_f[idx].sum()) / ni)
    d4, c5, d6 = np.sort(d4), np.sort(c5), np.sort(d6)
    ci = lambda v: [float(v[int(0.025 * N_BOOT)]), float(v[int(0.975 * N_BOOT) - 1])]   # noqa: E731
    tests = {
        "H4": {"diff_correct_per_image": float((ok_h[i_ev].sum() - ok_f[i_ev].sum()) / n_img_ev),
               "ci95": ci(d4), "p": float((d4 <= 0).mean())},
        "H5": {"OOD_val_correct": int((ok_h * val_box)[i_ev].sum()),
               "OOD_val_boxes": int(val_box[i_ev].sum()), "ci95": ci(c5), "p": float((c5 <= 0).mean())},
        "H6": {"diff_species_correct_per_image": float((sp_h[i_ev].sum() - sp_f[i_ev].sum()) / n_img_ev),
               "lower95": float(d6[int(0.05 * N_BOOT)]), "p": float((d6 <= -0.05).mean())},
    }
    for k, v in holm({k: tests[k]["p"] for k in tests}).items():
        tests[k]["supported_holm"] = v
    res["tests"] = tests
    return res


def main() -> None:
    global SPLIT, ROLE, OUT
    if "--dry-run-on-S_cal" in sys.argv:          # same code on the fit split, to catch bugs only
        SPLIT, ROLE, OUT = "S_cal", "OOD_dev", R / "week5" / "dry_run_S_cal.json"
        if OUT.exists():
            OUT.unlink()
    if OUT.exists():
        raise SystemExit(f"{OUT.relative_to(ROOT)} exists: §15 is single-use")
    for f, h in FROZEN.items():
        got = hashlib.sha256(f.read_bytes()).hexdigest()
        if got != h:
            raise SystemExit(f"hash mismatch for {f.name}: {got}")
    man = json.loads(MAN.read_text())
    harness.KNOWN_CLASSES = set(man["label_space"]["known"])
    gt = harness.load_ground_truth()
    out = {"protocol": "PROTOCOL §15", "part_a": part_a(gt), "part_b": part_b(man)}
    OUT.write_text(json.dumps(out, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o)) + "\n")
    print(json.dumps({"part_a_tests": out["part_a"]["tests"], "part_b_tests": out["part_b"]["tests"]}, indent=1,
                     default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o)))
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
