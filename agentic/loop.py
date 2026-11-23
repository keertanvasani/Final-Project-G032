"""The agent loop: Observe → Decide → Act → Verify, with an optional second pass.

    Observe   image quality
    Decide    D1 enhance?                           (policy)
    Act       enhance; detect + fuse                (tools)
    Verify    text filter; refuse gate              (verifier)
    Decide    D3 which boxes to zoom?               (policy)
    Act       zoom; classify every box              (tools)
    Decide    D4 vote on each disagreement?         (policy)
    Act       vote, or plain scoring                (tools)
    Verify    calibration; Known/Unknown            (verifier)
    Decide    D5 re-run on an enhanced image?       (image agent, app mode only)
    Act       the same pass on the enhanced image
    Decide    D6 keep the first or the second pass?  (image agent)

Guardrails:
    * at most two passes per image (the second only if the image agent
      asks for it); each discretionary tool runs at most
      AGENT_MAX_CALLS_PER_TOOL times per image (per box for the vote);
    * a per-image time budget (AGENT_TIME_BUDGET_MS): once spent, the
      remaining discretionary steps are skipped, i.e. the image finishes on
      the plain models;
    * a fallback chain of policies (policy.FallbackChain);
    * the verifier always runs and the policy cannot switch it off.

With the rule policy and a budget that does not bind, the output is identical
to the controller before this refactor (experiments/verify_agent_loop.py).
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import config
import ensemble
from agentic import probes, tools, verifier
from agentic.policy import FallbackChain, Observation, build_chain


class Guard:
    """Time budget and call limits for one image."""

    def __init__(self, budget_ms: float | None, max_calls: int, t_start: float) -> None:
        self.budget_ms = budget_ms
        self.max_calls = max_calls
        self.t_start = t_start
        self.calls: Counter = Counter()
        self.events: list[str] = []
        self._noted: set[str] = set()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.t_start) * 1000.0

    def allow(self, tool: str, key: Any = None) -> tuple[bool, str | None]:
        """May *tool* run now? Returns (ok, trace_line_or_None)."""
        el = self.elapsed_ms()
        if self.budget_ms is not None and el >= self.budget_ms:
            return False, self._note(
                f"budget:{tool}",
                f"Guardrail: {el:.0f} ms used of the {self.budget_ms:.0f} ms budget; "
                f"skipping {tool} for the rest of this image (plain models)")
        if self.calls[(tool, key)] >= self.max_calls:
            return False, self._note(
                f"calls:{tool}:{key}",
                f"Guardrail: {tool} already ran {self.calls[(tool, key)]} time(s) "
                f"on {'this image' if key is None else f'object {key + 1}'}; not repeated")
        self.calls[(tool, key)] += 1
        return True, None

    def _note(self, k: str, line: str) -> str | None:
        if k in self._noted:
            return None
        self._noted.add(k)
        self.events.append(line)
        return line


@dataclass
class AgentRun:
    """Everything one pass of the loop produced."""
    status: str = "complete"            # complete | needs_review | ok | unreadable | empty
    quality: dict[str, Any] = field(default_factory=dict)
    work_image: np.ndarray | None = None
    detections: list[dict[str, Any]] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    det_timings: dict[str, float] = field(default_factory=dict)
    max_det_conf: float = 0.0
    decisions: list[dict[str, Any]] = field(default_factory=list)
    guardrails: list[str] = field(default_factory=list)
    policy: str = ""


def default_chain() -> FallbackChain:
    return build_chain(getattr(config, "AGENT_POLICY_CHAIN", ("rules", "plain")))


def run_loop(
    image: np.ndarray,
    pipeline: Any,
    *,
    enhance: bool = True,
    mode: str = "app",
    chain: FallbackChain | None = None,
    budget_ms: float | None = "config",
    max_calls: int | None = None,
) -> AgentRun:
    """Run one image through the loop.

    mode="app"         the deployed behaviour: refuse gate, calibration and
                       Known/Unknown.
    mode="perception"  evaluation: no confidence gate (a downstream sweep sets
                       the acceptance threshold), raw confidence, no
                       Known/Unknown; an unreadable image returns no boxes.
    """
    if mode not in ("app", "perception"):
        raise ValueError(f"unknown mode {mode!r}")
    t_start = time.perf_counter()
    chain = chain or default_chain()
    if budget_ms == "config":
        budget_ms = getattr(config, "AGENT_TIME_BUDGET_MS", None)
    guard = Guard(budget_ms, max_calls or getattr(config, "AGENT_MAX_CALLS_PER_TOOL", 1), t_start)
    run = AgentRun(policy=chain.name)
    trace = run.trace
    tm = run.timings
    for k in ("preprocessing_ms", "wbf_ms", "text_filter_ms", "zoom_ms",
              "classification_ms", "reclassification_ms"):
        tm[k] = 0.0

    def decide(decision: str, obs: Observation, *args: Any) -> Any:
        value, who, lines = chain.decide(decision, obs, *args)
        trace.extend(lines)
        run.guardrails.extend(lines)
        run.decisions.append({"decision": decision, "policy": who,
                              "args": args[:1], "value": sorted(value) if isinstance(value, set) else value})
        return value

    def allowed(tool: str, key: Any = None) -> bool:
        ok, line = guard.allow(tool, key)
        if line:
            trace.append(line)
            run.guardrails.append(line)
        return ok

    # ── Observe ────────────────────────────────────────────────────
    t_pre = time.perf_counter()
    q = tools.assess_image_quality(image)
    run.quality = q
    if mode == "app":
        trace.append(
            f"Quality check: blur score = {q['blur_score']:.1f}, "
            f"brightness = {q['brightness']:.1f}, "
            f"contrast = {q['contrast']:.1f}, "
            f"resolution = {q['resolution']}"
        )

    # ── Decide + Act: D1 enhance ───────────────────────────────────
    work = image
    wants = decide("enhance", Observation(q, elapsed_ms=guard.elapsed_ms(), budget_ms=budget_ms))
    if wants and not enhance:
        # Switched off by the caller, so the effect of D1 can be seen side by side.
        trace.append("Action: Enhancement switched off by the user; analysing the original image "
                     "(the quality check would have enhanced it)")
    elif wants and allowed("enhance"):
        work, lines = tools.enhance_image(image, q)
        if mode == "app":
            trace.extend(lines)
    elif not wants:
        trace.append("Action: Image is well-formed, skipping pre-enhancements")
    if mode == "perception":
        trace.clear()
    tm["preprocessing_ms"] = (time.perf_counter() - t_pre) * 1000.0

    # Perception: no trustworthy evidence below the blur floor, so stop here.
    if mode == "perception" and not verifier.is_readable(q):
        run.work_image = work
        run.status = "unreadable"
        return run

    one = _one_pass(work, image, q, pipeline, mode, trace, tm, decide, allowed, guard, budget_ms)
    _apply(run, one)
    if mode != "app":
        return run

    # ── Image agent (two-step): D5 re-run on an enhanced image? ────
    # Rules and plain models never re-run, so this block is a no-op for them.
    state = {}

    def obs_after(passes):
        return Observation(q, run.detections, guard.elapsed_ms(), budget_ms, work is not image,
                           extra={"image": image, "pipeline": pipeline, "passes": passes,
                                  "state": state})

    arm = decide("rerun", obs_after([one]))
    if arm is None or not allowed("rerun"):
        return run
    t_arm = time.perf_counter()
    alt = probes.ARMS[arm](image)
    t_arm = (time.perf_counter() - t_arm) * 1000.0
    trace.append(f"Action: Image agent re-runs the analysis on a {arm} image "
                 f"(the lines above are the first pass: {len(one['detections'])} boxes, {one['status']})")
    trace2: list[str] = []
    # Pass 2 has its own call limits (one zoom, one vote per box), same budget.
    allowed2 = lambda tool, key=None: allowed(f"{tool} (pass 2)", key)
    two = _one_pass(alt, image, q, pipeline, mode, trace2, tm, decide, allowed2, guard, budget_ms)
    two["arm_ms"] = t_arm

    # ── D6: keep which pass? ───────────────────────────────────────
    keep = decide("commit", obs_after([one, two]))
    if keep == "second":
        trace.extend(trace2)
        trace.append(f"Action: Image agent keeps the {arm} pass "
                     f"({len(two['detections'])} boxes, {two['status']}; the lines above it are that pass)")
        _apply(run, two)
    else:
        trace.append(f"Action: Image agent keeps the first pass; the {arm} pass "
                     f"({len(two['detections'])} boxes) is discarded")
    return run


def _apply(run: AgentRun, p: dict[str, Any]) -> None:
    run.work_image = p["work"]
    run.det_timings = p["det_timings"]
    run.max_det_conf = p["max_det_conf"]
    run.detections = p["detections"]
    run.status = p["status"]


def _one_pass(work, image, q, pipeline, mode, trace, tm, decide, allowed, guard, budget_ms):
    """Detect → verify → zoom → classify → vote → verify on *work*.

    Returns {work, det_timings, max_det_conf, detections, status}."""
    out = {"work": work, "det_timings": {}, "max_det_conf": 0.0, "detections": [], "status": ""}

    # ── Act: detect + fuse (core) ──────────────────────────────────
    det_timings: dict[str, float] = {}
    if mode == "perception":
        detections = ensemble.run_detection(work, pipeline, timings=det_timings)
    else:
        res_a, res_b = ensemble.run_dual_detection(
            work, pipeline.model_a, pipeline.model_b, timings=det_timings)
        t_wbf = time.perf_counter()
        detections = ensemble.fuse_detections(res_a, res_b, work.shape[:2],
                                              class_names=getattr(pipeline, "class_names", None),
                                              weights=getattr(pipeline, "wbf_weights", None))
        tm["wbf_ms"] += (time.perf_counter() - t_wbf) * 1000.0
    out["det_timings"] = det_timings

    # ── Verify: text filter (safety) ───────────────────────────────
    t_txt = time.perf_counter()
    detections, lines = verifier.text_filter(work, detections)
    tm["text_filter_ms"] += (time.perf_counter() - t_txt) * 1000.0
    if mode == "app":
        trace.extend(lines)
    out["max_det_conf"] = max((d["confidence"] for d in detections), default=0.0)

    if mode == "perception" and not detections:
        out["status"] = "empty"
        return out

    if mode == "app":
        avg = float(np.mean([d["confidence"] for d in detections])) if detections else 0.0
        trace.append(
            f"Detection results: Found {len(detections)} boxes "
            f"(avg conf: {avg:.2f}, best: {out['max_det_conf']:.2f})"
        )

        # ── Verify: refuse gate (safety) ──────────────────────────
        accepted, lines = verifier.refuse_gate(q, detections)
        trace.extend(lines)
        if not accepted:
            out["status"] = "needs_review"
            return out

    # ── Decide + Act: D3 zoom ──────────────────────────────────────
    enhanced = work is not image
    t_zoom = time.perf_counter()
    obs = Observation(q, detections, guard.elapsed_ms(), budget_ms, enhanced)
    chosen = decide("zoom", obs)
    if chosen and allowed("zoom"):
        detections, lines = tools.zoom_boxes(work, detections, pipeline, chosen)
        if mode == "app":
            trace.extend(lines)
    tm["zoom_ms"] += (time.perf_counter() - t_zoom) * 1000.0

    # ── Act: classify every box (core, the second opinion) ─────────
    t_cls = time.perf_counter()
    cls_results = tools.classify_boxes(work, detections, pipeline)
    tm["classification_ms"] += (time.perf_counter() - t_cls) * 1000.0

    # ── Decide + Act: D4 vote, then Verify each box ────────────────
    resolved_all = []
    for idx, (det, cls_res) in enumerate(zip(detections, cls_results)):
        disagree = det["class_name"] != cls_res["class_name"]
        do_vote = disagree and decide(
            "vote", Observation(q, detections, guard.elapsed_ms(), budget_ms, enhanced),
            idx, det, cls_res)
        if do_vote and allowed("vote", idx):
            resolved, lines, ms = tools.resolve_detection(idx, det, cls_res, pipeline, work)
            tm["reclassification_ms"] += ms
            if mode == "app":
                trace.extend(lines)
        else:
            resolved = tools.score_plain(idx, det, cls_res)
        if mode == "app":
            trace.extend(verifier.finalise(idx, resolved, cls_res, pipeline))
            spec = verifier.load_taxon_spec(pipeline)
            if spec is not None:
                cfg = config.AGENT_VERIFIER
                tl = verifier.taxon_label(resolved, cls_res, work.shape, spec, cfg["mode"], cfg["alpha"])
                resolved["taxon"] = tl
                sp = resolved["classifier_class"]
                if tl["level"] is None:
                    trace.append(f"Verifier: object {idx + 1} — no level is safe at the {cfg['alpha']:.0%} "
                                 f"error target (threshold {tl['threshold']}); needs review (nearest: '{sp}')")
                elif tl["level"] != "species":
                    trace.append(f"Verifier: object {idx + 1} reported as '{tl['node']}' "
                                 f"(P correct {tl['p_correct']:.2f} ≥ {tl['threshold']}); species "
                                 f"'{sp}' is not safe at the {cfg['alpha']:.0%} error target")
        resolved_all.append(resolved)

    out["detections"] = resolved_all
    out["status"] = "complete" if mode == "app" else "ok"
    return out
