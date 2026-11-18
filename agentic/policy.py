"""Policies decide which discretionary tools run.

Three decision points, one per discretionary tool:

    enhance(obs)            D1  run CLAHE / sharpening on this image?
    zoom(obs)               D3  which boxes to re-detect on a zoomed crop?
    vote(obs, i, det, cls)  D4  on a detector/classifier disagreement, take a
                                third opinion from a looser crop and vote?

A policy only chooses among these. It cannot switch off the verifier (text
filter, refuse gate, calibration, Known/Unknown) and cannot call a tool twice.

FallbackChain asks each policy in turn. If one raises or returns something
invalid, the next one answers, and the trace says so. The default chain is
rules → plain models; step 3 puts the learned policy in front.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentic import tools


@dataclass
class Observation:
    """What a policy may look at when it decides."""
    quality: dict[str, Any]
    detections: list[dict[str, Any]] = field(default_factory=list)
    elapsed_ms: float = 0.0
    budget_ms: float | None = None
    enhanced: bool = False


class Policy:
    name = "policy"

    def enhance(self, obs: Observation) -> bool:
        raise NotImplementedError

    def zoom(self, obs: Observation) -> set[int]:
        raise NotImplementedError

    def vote(self, obs: Observation, idx: int, det: dict[str, Any],
             cls_res: dict[str, Any]) -> bool:
        raise NotImplementedError


class RulePolicy(Policy):
    """The hand-written rules the app has used since phase I."""
    name = "rules"

    def enhance(self, obs):
        q = obs.quality
        return bool(q["needs_enhancement"] or q["is_blurry"])

    def zoom(self, obs):
        return {i for i, d in enumerate(obs.detections) if tools.in_zoom_band(d)}

    def vote(self, obs, idx, det, cls_res):
        return True


class PlainPolicy(Policy):
    """Plain models: never act. The last rung of the fallback chain."""
    name = "plain"

    def enhance(self, obs):
        return False

    def zoom(self, obs):
        return set()

    def vote(self, obs, idx, det, cls_res):
        return False


class LearnedPolicy(Policy):
    """Contextual bandit fitted on S_cal (experiments/fit_bandit_policy.py).

    For each decision a ridge model predicts the change in reward from acting
    (+1 per correct identification, −1 per reported error) given the state;
    the policy acts iff the prediction clears a threshold chosen on S_cal.
    Frozen: it never learns from uploads. If its file is missing it raises,
    and the fallback chain hands over to the rules.
    """
    name = "learned"

    def __init__(self, path=None) -> None:
        import json
        from pathlib import Path

        import config
        p = Path(path or getattr(config, "AGENT_LEARNED_POLICY", ""))
        self.spec = json.loads(p.read_text()) if p.is_file() else None
        self.classes = self.spec["classes"] if self.spec else []

    def _act(self, decision, x):
        import numpy as np
        if self.spec is None:
            raise FileNotFoundError("no fitted bandit policy")
        m = self.spec["models"][decision]
        z = (np.asarray(x, dtype=float) - np.asarray(m["mean"])) / np.asarray(m["std"])
        return bool(float(z @ np.asarray(m["w"]) + m["b"]) > m["threshold"])

    def enhance(self, obs):
        from agentic.features import image_features
        return self._act("enhance", image_features(obs.quality))

    def zoom(self, obs):
        from agentic.features import zoom_features
        n = len(obs.detections)
        return {i for i, d in enumerate(obs.detections)
                if self._act("zoom", zoom_features(d, obs.quality, n, obs.enhanced, self.classes))}

    def vote(self, obs, idx, det, cls_res):
        from agentic.features import vote_features
        return self._act("vote", vote_features(det, cls_res, obs.quality, len(obs.detections),
                                               obs.enhanced, self.classes))


POLICIES: dict[str, type[Policy]] = {"learned": LearnedPolicy, "rules": RulePolicy,
                                     "plain": PlainPolicy}


def _valid(decision: str, value: Any, obs: Observation) -> bool:
    if decision in ("enhance", "vote"):
        return isinstance(value, bool)
    if decision == "zoom":
        n = len(obs.detections)
        return isinstance(value, set) and all(isinstance(i, int) and 0 <= i < n for i in value)
    return False


class FallbackChain:
    """Ask each policy in order; the first valid answer wins."""

    def __init__(self, policies: list[Policy]) -> None:
        if not policies:
            raise ValueError("FallbackChain needs at least one policy")
        self.policies = policies
        self.name = " → ".join(p.name for p in policies)

    def decide(self, decision: str, obs: Observation, *args: Any
               ) -> tuple[Any, str, list[str]]:
        """Return (value, policy_name, trace_lines)."""
        trace: list[str] = []
        for i, p in enumerate(self.policies):
            try:
                value = getattr(p, decision)(obs, *args)
                if _valid(decision, value, obs):
                    return value, p.name, trace
                problem = f"returned an invalid {decision} decision ({value!r})"
            except Exception as e:  # a broken policy must never break the image
                problem = f"failed on {decision} ({type(e).__name__}: {e})"
            nxt = self.policies[i + 1].name if i + 1 < len(self.policies) else None
            trace.append(f"Guardrail: policy '{p.name}' {problem}; "
                         + (f"falling back to '{nxt}'" if nxt else "no fallback left"))
        # Every rung failed: do nothing discretionary.
        return getattr(PlainPolicy(), decision)(obs, *args), "plain", trace


def build_chain(names: tuple[str, ...] | list[str]) -> FallbackChain:
    return FallbackChain([POLICIES[n]() for n in names])
