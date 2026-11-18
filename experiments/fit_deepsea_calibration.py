"""Fit the deep-sea app's confidence calibration and refit its Known/Unknown threshold.

Both are fitted on S_cal detections that the deployed agent itself produced
(cached by step1_deepsea_uncertainty.py) and checked on S_policy. S_test is
never touched.

* Calibration: a global isotonic map from the agent's combined confidence to
  the probability that the detection is correct (harness TP). Step 1 found it
  cuts ECE on S_policy from 0.269 to 0.051; a per-class map did no better.
  Written to models/phase_b_machine/calibration.json.
* Known/Unknown: the energy threshold is refitted as the 95th percentile of
  S_cal detections whose ground truth is a known class. The previous threshold
  was fitted on hand-drawn ground-truth crops, which are tighter than detector
  boxes, and flagged only ~3% of known and ~9% of unseen-taxon detections on
  S_policy instead of the intended 5% false-alarm budget. The old value is kept
  in novelty.json as `threshold_gt_crops`.

Usage:
    python experiments/fit_deepsea_calibration.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

from calibrate import apply_isotonic, brier, ece, fit_isotonic  # noqa: E402

CACHE = ROOT / "experiments" / "results" / "phase_b_machine" / "step1"
DS = ROOT / "models" / "phase_b_machine"
KEEP_KNOWN = 0.95
EPS = 1e-6


def load(split):
    return [json.loads(l) for l in (CACHE / f"{split}_detections.jsonl").read_text().splitlines()]


def compress(xs, ys):
    """One knot per distinct x, then drop knots inside constant runs of y.

    Tied inputs must map to one value. Combined confidence saturates at 1.0
    whenever detector and classifier agree strongly (hundreds of detections),
    and the pool-adjacent-violators fit can leave those ties spread over
    several blocks; np.interp would then return the last block (1.0). Averaging
    the fitted values over each tie gives the pooled accuracy instead, and the
    result stays monotone.
    """
    ux, inv = np.unique(np.asarray(xs), return_inverse=True)
    uy = np.bincount(inv, weights=np.asarray(ys)) / np.bincount(inv)
    xs, ys = ux, uy
    if len(xs) < 3:
        return [round(float(v), 6) for v in xs], [round(float(v), 6) for v in ys]
    keep = [0]
    for i in range(1, len(xs) - 1):
        if ys[i] != ys[i - 1] or ys[i] != ys[i + 1]:
            keep.append(i)
    keep.append(len(xs) - 1)
    return [round(float(xs[i]), 6) for i in keep], [round(float(ys[i]), 6) for i in keep]


def main() -> None:
    man = json.loads((ROOT / "experiments" / "splits" / "phase_b_machine_splits.json").read_text())
    known = set(man["label_space"]["known"])
    rc, rp = load("S_cal"), load("S_policy")

    # ---- calibration
    cc = np.clip([r["combined_confidence"] for r in rc], EPS, 1 - EPS)
    yc = np.array([r["outcome"] == "TP" for r in rc], float)
    cp = np.clip([r["combined_confidence"] for r in rp], EPS, 1 - EPS)
    yp = np.array([r["outcome"] == "TP" for r in rp], float)
    xs, ys = fit_isotonic(cc, yc)
    x, y = compress(xs, ys)
    pp = apply_isotonic(np.array(x), np.array(y), cp)
    cal = {
        "method": "isotonic",
        "maps": "agent combined_confidence -> P(detection is correct)",
        "x": x, "y": y,
        "fit": {"split": "S_cal", "n": len(rc)},
        "eval": {"split": "S_policy", "n": len(rp),
                 "ece_raw": ece(cp, yp)[0], "ece_calibrated": ece(pp, yp)[0],
                 "brier_raw": brier(cp, yp), "brier_calibrated": brier(pp, yp)},
        "source": "experiments/fit_deepsea_calibration.py",
    }
    (DS / "calibration.json").write_text(json.dumps(cal, indent=2) + "\n")
    print(f"calibration: {len(x)} knots | S_policy ECE {cal['eval']['ece_raw']:.3f} -> "
          f"{cal['eval']['ece_calibrated']:.3f}")

    # ---- Known/Unknown threshold
    e_fit = np.array([r["energy"] for r in rc if r["gt_class"] in known])
    thr = float(np.quantile(e_fit, KEEP_KNOWN))
    ek = np.array([r["energy"] for r in rp if r["gt_class"] in known])
    eu = np.array([r["energy"] for r in rp if r["outcome"] == "OOD_ERR"])
    nov_path = DS / "novelty.json"
    nov = json.loads(nov_path.read_text())
    old = nov.get("threshold_gt_crops", nov["threshold"])
    nov.update({
        "threshold": thr,
        "threshold_gt_crops": old,
        "fit": {"split": "S_cal", "on": "agent detections matched to known-class ground truth",
                "n_known": int(len(e_fit)), "known_kept_target": KEEP_KNOWN},
        "eval_detections": {"split": "S_policy", "n_known": int(len(ek)), "n_unseen": int(len(eu)),
                            "unseen_flagged": float((eu > thr).mean()),
                            "known_flagged": float((ek > thr).mean()),
                            "unseen_flagged_old_threshold": float((eu > old).mean()),
                            "known_flagged_old_threshold": float((ek > old).mean())},
    })
    nov_path.write_text(json.dumps(nov, indent=2) + "\n")
    ev = nov["eval_detections"]
    print(f"novelty threshold {old:.3f} -> {thr:.3f} | S_policy unseen flagged "
          f"{ev['unseen_flagged_old_threshold']:.1%} -> {ev['unseen_flagged']:.1%}, known flagged "
          f"{ev['known_flagged_old_threshold']:.1%} -> {ev['known_flagged']:.1%}")


if __name__ == "__main__":
    main()
