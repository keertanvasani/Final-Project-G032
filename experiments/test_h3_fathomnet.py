"""H3 on FathomNet Part A F_eval (PROTOCOL §19). Single use.

Verifier (hierarchical, F_cal threshold) vs a confidence gate at matched coverage (H3) and
vs the confidence gate plus the deployed energy Unknown flag (H3-U). OOD_ERR rate among
accepted = accepted unseen-animal boxes with a wrong answer / all accepted boxes.

Usage:
    python experiments/test_h3_fathomnet.py [--dry-run OUT]
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import test_fathomnet as tf  # noqa: E402  (sets the FathomNet labels on the tree)
from week3_verifier import decide  # noqa: E402

RES = tf.RES
OUT = RES / "h3_test.json"
FROZEN = dict(tf.FROZEN) | {
    RES / "fathomnet_test.json": "fc7faf0fedb8ffb962a3bc22dbf753ef066d960be6460ca3b59ea7b040211883",
    tf.LOGS / "F_ext_verifier.jsonl": "362ff618786aa6fd1ebdb1e101f2181be5164f6883fc85224b20205e985941b0",
}


def load():
    """F_eval boxes exactly as in §18.5 (same split code as test_fathomnet.part_a)."""
    man = json.loads((tf.EXT / "meta" / "split_manifest.json").read_text())
    tf.harness.KNOWN_CLASSES = set(man["label_space"]["known"])
    spec = json.loads((tf.R / "week4" / "verifier.json").read_text())
    files_a = set(man["parts"]["A"])
    recs = [r for r in (json.loads(l) for l in (tf.LOGS / "F_ext_verifier.jsonl").open(encoding="utf-8"))
            if r["file"] in files_a]
    ud = sorted({r["dive"] for r in recs})
    random.Random(0).shuffle(ud)
    cal = set(ud[: len(ud) // 2])
    boxes, dives = [], []
    for r in recs:
        for b in r["boxes"]:
            if b["matched"] and tf.SCORABLE(b["gt_class"]) and r["dive"] not in cal:
                boxes.append(b)
                dives.append(r["dive"])
    P, Y, N = tf.combiner_probs(boxes, spec)
    return boxes, dives, P, Y


def gates(boxes, P, Y, t):
    """Accepted / wrong masks for the verifier, the confidence gate and the deployed gate."""
    n = len(boxes)
    ch = decide(P, t, [0, 1, 2, 3])
    acc_v = ch >= 0
    wrong_v = np.array([ch[i] >= 0 and not Y[i, ch[i]] for i in range(n)])
    k = int(acc_v.sum())
    key = lambda i: (boxes[i]["combined_confidence"], boxes[i]["raw_combined_confidence"])  # noqa: E731
    order = sorted(range(n), key=key, reverse=True)
    acc_c = np.zeros(n, bool)
    acc_c[order[:k]] = True
    flagged = np.array([b["unknown"] for b in boxes])
    rest = [i for i in order if not flagged[i]]
    acc_u = np.zeros(n, bool)
    acc_u[rest[:k]] = True
    wrong_species = ~Y[:, 0]
    return {"verifier": (acc_v, wrong_v), "confidence": (acc_c, acc_c & wrong_species),
            "confidence_unknown": (acc_u, acc_u & wrong_species)}, k


def rate(acc, wrong, unseen, idx):
    a = acc[idx].sum()
    return float((wrong & unseen)[idx].sum() / a) if a else 0.0


def evaluate(boxes, dives, P, Y, t, samples, test):
    unseen = np.array([b["gt_class"].startswith("unseen:") for b in boxes])
    g, k = gates(boxes, P, Y, t)
    allidx = np.arange(len(boxes))
    by = defaultdict(list)
    for i, d in enumerate(dives):
        by[d].append(i)
    out = {"k_matched": k, "coverage_target": k / len(boxes), "gates": {}}
    for name, (acc, wrong) in g.items():
        out["gates"][name] = {
            "accepted": int(acc.sum()), "coverage": float(acc.mean()),
            "ood_err_rate_among_accepted": rate(acc, wrong, unseen, allidx),
            "unseen_wrong": int((wrong & unseen).sum()), "unseen_boxes": int(unseen.sum()),
            "unseen_wrong_share_of_unseen": float((wrong & unseen).sum() / unseen.sum()),
            "known_err_among_accepted": float((wrong & ~unseen).sum() / max(acc.sum(), 1)),
            "error_among_accepted": float(wrong.sum() / max(acc.sum(), 1))}
    if test:
        av, wv = g["verifier"]
        for h, base in (("H3", "confidence"), ("H3-U", "confidence_unknown")):
            ab, wb = g[base]
            diffs = []
            for s in samples:
                idx = np.array([i for d in s for i in by[d]], int)
                diffs.append(rate(ab, wb, unseen, idx) - rate(av, wv, unseen, idx))
            diffs = np.sort(diffs)
            out[h] = {"diff_baseline_minus_verifier": out["gates"][base]["ood_err_rate_among_accepted"]
                      - out["gates"]["verifier"]["ood_err_rate_among_accepted"],
                      "ci95": tf.ci(diffs), "p": float((diffs <= 0).mean())}
    return out


def main() -> None:
    global OUT
    if "--dry-run" in sys.argv:
        OUT = Path(sys.argv[sys.argv.index("--dry-run") + 1])
    elif OUT.exists():
        raise SystemExit(f"{OUT} exists: §19 is single-use")
    for f, h in FROZEN.items():
        got = hashlib.sha256(f.read_bytes()).hexdigest()
        if got != h:
            raise SystemExit(f"hash mismatch for {f.name}: {got}")
    thr = json.loads((RES / "fathomnet_test.json").read_text())["part_a"]["thresholds_F_cal"]["hierarchical"]
    boxes, dives, P, Y = load()
    ude = sorted(set(dives))
    rng = random.Random(0)
    samples = [[ude[rng.randrange(len(ude))] for _ in ude] for _ in range(tf.N_BOOT)]
    a20 = evaluate(boxes, dives, P, Y, thr["0.2"], samples, test=True)
    a10 = evaluate(boxes, dives, P, Y, thr["0.1"], samples, test=False)
    for h, ok in tf.w5.holm({h: a20[h]["p"] for h in ("H3", "H3-U")}).items():
        a20[h]["supported_holm"] = ok
    out = {"protocol": "PROTOCOL §19", "f_eval_boxes": len(boxes), "f_eval_dives": len(ude),
           "unseen_boxes": int(sum(b["gt_class"].startswith("unseen:") for b in boxes)),
           "flagged_unknown": dict(Counter(("unseen" if b["gt_class"].startswith("unseen:") else "other",
                                            b["unknown"]) for b in boxes).items()),
           "alpha_0.20": a20, "alpha_0.10_descriptive": a10}
    out["flagged_unknown"] = {f"{a}|unknown={u}": v for (a, u), v in out["flagged_unknown"].items()}
    OUT.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
