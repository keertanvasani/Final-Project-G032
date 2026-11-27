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
from agentic.probes import PROBE_KEYS


@dataclass
class Observation:
    """What a policy may look at when it decides."""
    quality: dict[str, Any]
    detections: list[dict[str, Any]] = field(default_factory=list)
    elapsed_ms: float = 0.0
    budget_ms: float | None = None
    enhanced: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class Policy:
    name = "policy"

    def enhance(self, obs: Observation) -> bool:
        raise NotImplementedError

    def zoom(self, obs: Observation) -> set[int]:
        raise NotImplementedError

    def vote(self, obs: Observation, idx: int, det: dict[str, Any],
             cls_res: dict[str, Any]) -> bool:
        raise NotImplementedError

    def rerun(self, obs: Observation) -> str | None:
        """D5: name of an enhancement arm to re-run the image with, or None."""
        raise NotImplementedError

    def commit(self, obs: Observation) -> str:
        """D6: "first" or "second" pass."""
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

    def rerun(self, obs):
        return None

    def commit(self, obs):
        return "first"


class PlainPolicy(Policy):
    """Plain models: never act. The last rung of the fallback chain."""
    name = "plain"

    def enhance(self, obs):
        return False

    def zoom(self, obs):
        return set()

    def vote(self, obs, idx, det, cls_res):
        return False

    def rerun(self, obs):
        return None

    def commit(self, obs):
        return "first"


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


def pass_states(obs: Observation) -> list[list[float]]:
    """State after each finished pass: s1 = quality + pass-1 probes, s2 = s1 + pass-2 probes.

    Cached in obs.extra["state"] so a probe is computed once per pass."""
    from agentic.features import image_features
    from agentic.probes import PROBE_KEYS, pass_state
    ex = obs.extra
    cache = ex["state"]
    keys = ex.get("probe_keys", PROBE_KEYS)
    for k, p in enumerate(ex["passes"]):
        if k not in cache:
            v, ms = pass_state(p["work"], p["detections"], ex["pipeline"], p["status"] == "needs_review",
                               keys)
            cache[k] = v
            cache[f"probe_ms_{k}"] = ms
    s1 = image_features(obs.quality) + cache[0]
    return [s1] + ([s1 + cache[1]] if len(ex["passes"]) > 1 else [])


def _ridge(m: dict[str, Any], x: list[float]) -> float:
    import numpy as np
    z = (np.asarray(x, dtype=float) - np.asarray(m["mean"])) / np.asarray(m["std"])
    return float(z @ np.asarray(m["w"]) + m["b"])


class ImageAgentPolicy(Policy):
    """Two-step image agent fitted by fitted Q-iteration (experiments/fit_image_agent.py).

    D5 after the first pass:  Q1(s1, arm) = predicted gain of re-running with
                              *arm* (when the D6 policy then picks the better
                              pass) minus λ × the arm's mean cost; re-run with
                              the best arm iff its Q1 > 0.
    D6 after the second pass: Q2(s2) = predicted reward(second) − reward(first);
                              keep the second pass iff Q2 > 0.
    Makes only D5 and D6; the other decisions go to the next policy in the chain.
    Frozen; raises if its file is missing, and the chain falls back (no re-run).
    """
    name = "image"

    def __init__(self, path=None) -> None:
        import json
        from pathlib import Path

        import config
        p = Path(path or getattr(config, "AGENT_IMAGE_POLICY", ""))
        self.spec = json.loads(p.read_text()) if p.is_file() else None
        lam = getattr(config, "AGENT_IMAGE_LAMBDA", None)
        if self.spec is not None and lam is not None and path is None:
            # Operating point chosen in config (e.g. λ_m); the fitted Q-functions are unchanged.
            self.spec = dict(self.spec, **{"lambda": float(lam)})

    def _spec(self):
        if self.spec is None:
            raise FileNotFoundError("no fitted image-agent policy")
        return self.spec

    def rerun(self, obs):
        sp = self._spec()
        obs.extra["probe_keys"] = sp.get("probe_keys", PROBE_KEYS)
        s1 = pass_states(obs)[0]
        best, arm = 0.0, None
        for a, m in sp["q1"].items():
            v = _ridge(m, s1) - sp["lambda"] * sp["cost_ms"][a]
            if v > best:
                best, arm = v, a
        obs.extra["state"]["arm"] = arm
        return arm

    def commit(self, obs):
        sp = self._spec()
        arm = obs.extra["state"].get("arm")
        if arm is None:
            raise ValueError("commit asked without a re-run arm")
        obs.extra["probe_keys"] = sp.get("probe_keys", PROBE_KEYS)
        return "second" if _ridge(sp["q2"][arm], pass_states(obs)[1]) > 0 else "first"


POLICIES: dict[str, type[Policy]] = {"image": ImageAgentPolicy, "learned": LearnedPolicy,
                                     "rules": RulePolicy, "plain": PlainPolicy}


def _valid(decision: str, value: Any, obs: Observation) -> bool:
    if decision in ("enhance", "vote"):
        return isinstance(value, bool)
    if decision == "rerun":
        from agentic.probes import ARMS
        return value is None or value in ARMS
    if decision == "commit":
        return value in ("first", "second")
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
                try:
                    value = getattr(p, decision)(obs, *args)
                except NotImplementedError:
                    continue        # this policy does not make this decision; ask the next
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
