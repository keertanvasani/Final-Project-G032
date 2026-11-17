"""Fit the Known/Unknown threshold the deep-sea app uses, and measure it.

The app marks a detection "Unknown" when the classifier's energy score on its
crop is higher than anything typical of the nine trained classes. Energy is
-logsumexp(logits): a crop the network recognises produces a large logit and a
low energy; an unfamiliar one spreads its logits and scores higher.

Fit and evaluation are separate, as in `ood_features.py`:

* **Fit on S_cal**, known-class ground-truth crops only. The threshold is the
  95th percentile of their energy, so by construction about 5% of known
  animals are flagged Unknown. That is the false-alarm budget, chosen up front.
* **Evaluate on S_policy**, whose held-out taxa (jellyfish, squid, tunicate)
  the classifier never saw. Reported: AUROC, the share of unseen animals
  flagged, and the share of known animals flagged at the fitted threshold.

S_test is never touched (PROTOCOL §8).

Writes models/phase_b_machine/novelty.json, which pipeline.py reads.

Usage:
    python experiments/fit_novelty.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from ood_features import auroc, embed  # noqa: E402

MAN = ROOT / "experiments" / "splits" / "phase_b_machine_splits.json"
CLF_DIR = ROOT / "models" / "phase_b_machine" / "classifier_seed0"
OUT = ROOT / "models" / "phase_b_machine" / "novelty.json"
KNOWN_KEEP = 0.95   # share of known crops that must stay "Known" on the fit split
CAP = 1500


def energy(L: np.ndarray) -> np.ndarray:
    if len(L) == 0:
        return np.zeros(0)
    m = L.max(axis=1, keepdims=True)
    return -(m[:, 0] + np.log(np.exp(L - m).sum(axis=1)))


def main() -> None:
    import torch.nn as nn
    from classifier import SpeciesClassifier

    man = json.loads(MAN.read_text())
    ls = man["label_space"]
    known = set(ls["known"])
    hosts = ls["split_hosts_ood_role"]
    roles = {k: set(v) for k, v in ls["ood_roles"].items()}

    clf = SpeciesClassifier(CLF_DIR / "best_classifier.pth", CLF_DIR / "class_names.json", "cpu")
    model, fc = clf.model, clf.model.fc
    model.fc = nn.Identity()
    model.eval()

    def groups(split):
        return {"known": known, "unseen": roles[hosts[split]]}

    _, Lf = embed("S_cal", man, groups("S_cal"), clf, model, fc, CAP)
    _, Le = embed("S_policy", man, groups("S_policy"), clf, model, fc, CAP)

    e_fit = energy(Lf["known"])
    thr = float(np.quantile(e_fit, KNOWN_KEEP))

    ek, eu = energy(Le["known"]), energy(Le["unseen"])
    s = np.concatenate([ek, eu])
    y = np.concatenate([np.zeros(len(ek)), np.ones(len(eu))])
    res = {
        "score": "energy",
        "definition": "-logsumexp(classifier logits) on the 5%-padded crop; higher = less familiar",
        "threshold": thr,
        "fit": {"split": "S_cal", "n_known": int(len(e_fit)), "known_kept_target": KNOWN_KEEP},
        "eval": {
            "split": "S_policy",
            "unseen_taxa": sorted(roles[hosts["S_policy"]]),
            "n_known": int(len(ek)), "n_unseen": int(len(eu)),
            "auroc": auroc(s, y),
            "unseen_flagged": float((eu > thr).mean()) if len(eu) else float("nan"),
            "known_flagged": float((ek > thr).mean()) if len(ek) else float("nan"),
        },
        "classifier": str(CLF_DIR.relative_to(ROOT)),
    }
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    ev = res["eval"]
    print(f"\nthreshold (95th pct of S_cal known energy) = {thr:.3f}")
    print(f"S_policy: AUROC {ev['auroc']:.3f} | unseen flagged {ev['unseen_flagged']:.1%} "
          f"(n={ev['n_unseen']}) | known flagged {ev['known_flagged']:.1%} (n={ev['n_known']})")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
