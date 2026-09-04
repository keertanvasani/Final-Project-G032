"""Selective-prediction metrics and dive-cluster bootstrap.

Implements PROTOCOL.md §3 (risk / coverage / recall, AURC, both
denominators) and §7.2 (cluster bootstrap over dives). Operates on the
flat per-detection records produced by the harness plus per-image metadata.

An acceptance policy here is a **confidence threshold** applied to
``combined_confidence``: a detection is accepted iff its confidence >= tau.
Sweeping tau traces the risk–coverage curve. The deployed baseline's
operating point is one such tau (its REVIEW_CONF_THRESHOLD), reported
separately.
"""

from __future__ import annotations

import random
from typing import Any

from .harness import TP, OOD_ERR

# A detection is "correct" iff its outcome is TP (matched, correct known class).
# Everything else counts against selective risk when accepted.
CORRECT = {TP}


def _accepted(records: list[dict[str, Any]], tau: float) -> list[dict[str, Any]]:
    return [r for r in records if r["combined_confidence"] >= tau]


def operating_point(
    records: list[dict[str, Any]],
    total_known_gt: int,
    tau: float,
) -> dict[str, float]:
    """Metrics at a single acceptance threshold.

    Parameters
    ----------
    records : every detection emitted across the split (the perception output).
    total_known_gt : number of in-label-space GT objects across the split
        — the recall denominator, which includes GT on images that were
        abstained or produced no detection.
    tau : acceptance threshold on combined_confidence.
    """
    accepted = _accepted(records, tau)
    n_acc = len(accepted)
    n_all = len(records)

    n_correct = sum(1 for r in accepted if r["outcome"] in CORRECT)
    n_ood = sum(1 for r in accepted if r["outcome"] == OOD_ERR)

    # GT recalled = distinct known GT matched by an accepted TP. matched_gt is
    # a per-IMAGE index, so it must be namespaced by file to be a global GT
    # identity — otherwise index 0 from every image collapses to one element
    # and recall is undercounted by orders of magnitude.
    recalled = {(r.get("file"), r["matched_gt"]) for r in accepted
                if r["outcome"] == TP and r["matched_gt"] is not None}

    risk = (n_acc - n_correct) / n_acc if n_acc else 0.0
    return {
        "tau": tau,
        "accepted": n_acc,
        "selective_risk": risk,
        "coverage_policy_local": n_acc / n_all if n_all else 0.0,
        "selective_recall": len(recalled) / total_known_gt if total_known_gt else 0.0,
        "ood_err_rate": n_ood / n_acc if n_acc else 0.0,
        "ood_err_count": n_ood,
    }


def risk_coverage_curve(
    records: list[dict[str, Any]],
    total_known_gt: int,
    n_points: int = 100,
) -> list[dict[str, float]]:
    """Sweep tau over the observed confidence range; one row per threshold."""
    if not records:
        return []
    confs = sorted({round(r["combined_confidence"], 4) for r in records})
    if len(confs) > n_points:
        step = len(confs) / n_points
        confs = [confs[int(i * step)] for i in range(n_points)] + [confs[-1]]
    # Include a tau just below the minimum so full coverage is represented.
    taus = [0.0] + confs
    return [operating_point(records, total_known_gt, t) for t in sorted(set(taus))]


def aurc(curve: list[dict[str, float]], x_key: str) -> float:
    """Area under the risk vs {coverage|recall} curve, by trapezoid.

    Lower is better. ``x_key`` selects the coverage definition:
    ``coverage_policy_local`` or ``selective_recall``.
    """
    pts = sorted(((p[x_key], p["selective_risk"]) for p in curve))
    area = 0.0
    for (x0, r0), (x1, r1) in zip(pts, pts[1:]):
        area += (x1 - x0) * (r0 + r1) / 2.0
    span = pts[-1][0] - pts[0][0] if len(pts) > 1 else 0.0
    return area / span if span > 0 else 0.0


def risk_at_coverage(curve: list[dict[str, float]], x_key: str, target: float) -> float:
    """Selective risk at the operating point nearest a coverage/recall target."""
    if not curve:
        return float("nan")
    return min(curve, key=lambda p: abs(p[x_key] - target))["selective_risk"]


def coverage_at_risk(curve: list[dict[str, float]], x_key: str, max_risk: float) -> float:
    """Max coverage/recall achievable at or below a risk ceiling."""
    ok = [p for p in curve if p["selective_risk"] <= max_risk]
    return max((p[x_key] for p in ok), default=0.0)


# ── Dive-cluster bootstrap (PROTOCOL.md §7.2) ────────────────────────────

def dive_bootstrap(
    records_by_dive: dict[str, list[dict[str, Any]]],
    known_gt_by_dive: dict[str, int],
    metric_fn,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap a scalar metric by resampling **dives** with replacement.

    Detections within a dive are correlated (same camera, lighting, often the
    same animal), so the resampling unit is the dive, not the detection.
    ``metric_fn(records, total_known_gt) -> float`` is evaluated on each
    resample. Returns the point estimate and a 95% percentile interval.
    """
    rng = random.Random(seed)
    dives = list(records_by_dive)
    all_records = [r for rs in records_by_dive.values() for r in rs]
    point = metric_fn(all_records, sum(known_gt_by_dive.values()))

    boots: list[float] = []
    n = len(dives)
    for _ in range(n_boot):
        recs: list[dict[str, Any]] = []
        kgt = 0
        # Each draw is an independent cluster copy. A dive drawn twice must
        # count twice in BOTH the summed denominator and any set-based
        # numerator (e.g. recalled GT), so its records are re-keyed per draw:
        # without this, recall's numerator dedupes duplicated dives while the
        # denominator sums them, biasing the interval below the point
        # estimate. Ratio-of-count metrics are unaffected; re-keying is
        # harmless to them.
        for k in range(n):
            d = dives[rng.randrange(n)]
            tag = f"{k}\x00"
            for r in records_by_dive[d]:
                rc = dict(r)
                if rc.get("file") is not None:
                    rc["file"] = tag + rc["file"]
                recs.append(rc)
            kgt += known_gt_by_dive[d]
        boots.append(metric_fn(recs, kgt))

    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[min(n_boot - 1, int(0.975 * n_boot))]
    return {"point": point, "ci_low": lo, "ci_high": hi, "n_boot": n_boot}
