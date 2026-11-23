"""Week 5: freeze the one-step comparator for PROTOCOL §15 H7 (S_cal only).

Same no-probe state s1 and arm (CLAHE 3.0) as the image agent, one decision:
re-run and keep the second pass iff Q(s1) − λ·c > 0, with Q a ridge regression
of Δ = R(pass 2) − R(pass 1) on s1, fitted on all of S_cal.

Usage:
    python experiments/week5_freeze_onestep.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from ablation_deepsea import MAN  # noqa: E402
from agentic.probes import state_columns  # noqa: E402
from eval import harness  # noqa: E402
from fit_bandit_policy import ridge_fit  # noqa: E402
from fit_image_agent import ALPHA, reward  # noqa: E402

W1 = ROOT / "experiments" / "results" / "phase_b_machine" / "week1"
OUT = ROOT / "experiments" / "results" / "phase_b_machine" / "week5" / "one_step_policy.json"
ARM = "clahe3.0"


def main() -> None:
    man = json.loads(MAN.read_text())
    harness.KNOWN_CLASSES = set(man["label_space"]["known"])
    gt = harness.load_ground_truth()
    recs = [json.loads(line) for line in (W1 / "S_cal_image_agent.jsonl").open(encoding="utf-8")]
    a = [r["arms"][ARM] for r in recs if not r["arms"][ARM].get("skipped")]
    files = [r["file"] for r in recs if not r["arms"][ARM].get("skipped")]
    S1 = np.array([x["s1"] for x in a], float)[:, state_columns([], 1)]
    D = np.array([reward(x["passes"][1]["finals"], gt.get(f, [])) - reward(x["passes"][0]["finals"], gt.get(f, []))
                  for x, f in zip(a, files)], float)
    spec = {"kind": "one-step comparator (PROTOCOL §15 H7)", "fitted_on": "S_cal", "arm": ARM,
            "probe_keys": [], "q": ridge_fit(S1, D, ALPHA),
            "cost_ms": float(np.mean([x["pass2_ms"] for x in a])),
            "rule": "re-run and keep the second pass iff q(s1) − λ·cost_ms > 0"}
    OUT.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
