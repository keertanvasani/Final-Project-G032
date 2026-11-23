"""Week 4: fit the deployable risk-controlled verifier on S_cal.

* Combiner: one logistic model per level (species, group, phylum, kingdom) for
  P(the level's candidate is correct), fitted on every S_cal box eligible under
  PROTOCOL §14.3 (other splits' held-out taxa excluded).
* Thresholds (PROVISIONAL): Learn-then-Test on S_cal with out-of-fold
  (5 dive folds) combiner probabilities, per α, hierarchical and flat. These
  approximate the final model's thresholds. They carry no guarantee for it,
  because the combiner was fitted on the same split. A guaranteed threshold
  needs calibration on data the combiner never saw (the registered S_policy
  calibration half).

Writes models/phase_b_machine/verifier.json and a committed copy in week4/.

Usage:
    python experiments/week4_fit_verifier.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from ablation_deepsea import MAN  # noqa: E402
from agentic import taxonomy as tx  # noqa: E402
from week0_analyse import logistic_fit  # noqa: E402
from week3_verifier import ALPHAS, DELTA, box_rows, decide, ltt  # noqa: E402

W3 = ROOT / "experiments" / "results" / "phase_b_machine" / "week3"
OUT = ROOT / "experiments" / "results" / "phase_b_machine" / "week4"
MODEL = ROOT / "models" / "phase_b_machine" / "verifier.json"


def predict(m, X):
    return 1 / (1 + np.exp(-(np.c_[np.ones(len(X)), (X - np.array(m["mean"])) / np.array(m["std"])]
                             @ np.array(m["w"]))))


def main() -> None:
    man = json.loads(MAN.read_text())
    excluded = set(man["label_space"]["held_out_unseen"]) - set(man["label_space"]["ood_roles"]["OOD_dev"])
    recs = [json.loads(line) for line in (W3 / "S_cal_verifier.jsonl").open(encoding="utf-8")]
    classes = recs[0]["classes"]
    rows = [(r["dive"], b) for r in recs for b in r["boxes"]]
    F, Y = [], []
    for _, b in rows:
        f, y, _ = box_rows(b, classes)
        F.append(f)
        Y.append(y)
    F, Y = np.array(F), np.array(Y, bool)
    dives = [d for d, _ in rows]
    elig = np.array([b["gt_class"] not in excluded for _, b in rows])
    n = len(rows)

    models = {}
    for j, lv in enumerate(tx.LEVELS):
        mu, sd, w = logistic_fit(F[elig, j], Y[elig, j].astype(float))
        models[lv] = {"mean": mu.tolist(), "std": sd.tolist(), "w": w.tolist()}

    # Out-of-fold probabilities for the provisional thresholds.
    ud = sorted(set(dives))
    random.Random(0).shuffle(ud)
    fold = {d: i % 5 for i, d in enumerate(ud)}
    fz = np.array([fold[d] for d in dives])
    P = np.zeros((n, 4))
    for k in range(5):
        tr = np.where((fz != k) & elig)[0]
        te = np.where(fz == k)[0]
        for j in range(4):
            mu, sd, w = logistic_fit(F[tr, j], Y[tr, j].astype(float))
            P[te, j] = predict({"mean": mu, "std": sd, "w": w}, F[te, j])
    idx = np.where(elig)[0]
    thresholds = {}
    for mode, levels in (("hierarchical", [0, 1, 2, 3]), ("flat", [0])):
        thresholds[mode] = {}
        for a in ALPHAS:
            t = ltt(P, Y, dives, idx, a, levels)
            thresholds[mode][str(a)] = t
    spec = {"kind": "taxonomy-aware risk-controlled verifier (PROTOCOL §14)",
            "fitted_on": "S_cal (eligible boxes, §14.3)", "classes": classes,
            "levels": list(tx.LEVELS), "features": list(tx.FEATURES), "models": models,
            "thresholds": thresholds, "delta": DELTA,
            "thresholds_status": "PROVISIONAL — Learn-then-Test on S_cal out-of-fold probabilities; "
                                 "no guarantee for this combiner until calibrated on unseen dives",
            "source": "experiments/week4_fit_verifier.py"}
    txt = json.dumps(spec, indent=2) + "\n"
    MODEL.write_text(txt)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "verifier.json").write_text(txt)
    print(f"fitted on {int(elig.sum())} boxes; provisional thresholds: {json.dumps(thresholds)}")
    # In-sample sanity: decisions with the final models at the provisional thresholds.
    Pf = np.stack([predict(models[lv], F[:, j]) for j, lv in enumerate(tx.LEVELS)], 1)
    for a in ALPHAS:
        t = thresholds["hierarchical"][str(a)]
        ch = decide(Pf, 2.0 if t is None else t, [0, 1, 2, 3])
        print(f"α {a}: t {t}, reported {np.mean(ch >= 0):.2f} of boxes (in-sample, final combiner)")


if __name__ == "__main__":
    main()
