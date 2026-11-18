"""Step 1 of the next-semester plan: cheap uncertainty improvements, deep-sea stack.

Everything here works on the deployed 9-class deep-sea agent (enhancement, text
filter, refuse gate, zoom, three-opinion vote; the Known/Unknown flag is
recorded but nothing is dropped) and needs no retraining.

Stage 1 - cache. Every detection the agent emits on S_cal and S_policy is
scored against the JEDI ground truth with the harness taxonomy and stored with
the classifier's 512-d feature and logits for the 5%-padded crop and for its
horizontal mirror. Re-run with --recache to rebuild.

Stage 2 - analysis. Fit on S_cal, report on S_policy only (PROTOCOL §4.2);
S_test is never touched (§8).

  (1) Re-measure: AURC, ECE, Brier and correctness AUROC of the deployed
      combined_confidence; novelty AUROC of the deployed energy score.
  (2) Flip TTA: classifier accuracy on crops whose ground truth is a known
      class, single crop vs mean of crop and mirror logits.
  (3) Novelty: energy vs feature-space Mahalanobis vs their sum (each
      z-scored on S_cal known crops), at a threshold that keeps 95% of S_cal
      known crops - the same false-alarm budget the app uses.
  (4) Calibration: temperature and isotonic fitted globally, and isotonic
      fitted per predicted class (fallback to global under MIN_PER_CLASS).

Deltas get paired 95% dive-cluster bootstrap intervals.

Usage:
    python experiments/step1_deepsea_uncertainty.py [--recache] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP))

import config  # noqa: E402
import ensemble  # noqa: E402
import text_overlay  # noqa: E402
from ablation_deepsea import DIVE_RE, MAN, image_path  # noqa: E402
from agent_controller import (assess_image_quality, enhance_image,  # noqa: E402
                              rerun_zoomed_detection, resolve_detection)
from calibrate import (apply_isotonic, brier, ece, fit_isotonic,  # noqa: E402
                       fit_temperature, logit, sigmoid)
from eval import harness  # noqa: E402
from ood_features import auroc, fpr_at_tpr  # noqa: E402
from risk_coverage import aurc, curve  # noqa: E402

OUTDIR = ROOT / "experiments" / "results" / "phase_b_machine" / "step1"
KEEP_KNOWN = 0.95
MIN_PER_CLASS = 50
N_BOOT = 1000
EPS = 1e-6


# ------------------------------------------------------------------ stage 1
def agent_detections(img, pipe):
    """The deployed agent minus the Unknown drop. Returns (work image, resolved dets)."""
    q = assess_image_quality(img)
    work = img
    if q["needs_enhancement"] or q["is_blurry"]:
        work, _ = enhance_image(img, q)
    if q["blur_score"] < config.UNUSABLE_BLUR_THRESHOLD:
        return work, []
    dets = ensemble.run_detection(work, pipe)
    dets, _ = text_overlay.filter_text_detections(work, dets)
    if not any(d["confidence"] >= config.REVIEW_CONF_THRESHOLD for d in dets):
        return work, []
    dets, _ = rerun_zoomed_detection(work, dets, pipe)
    crops = [pipe._extract_crop(work, d["bbox"], padding=0.05) for d in dets]
    cls = pipe.classifier.classify_crops(crops)
    out = []
    for i, (d, c) in enumerate(zip(dets, cls)):
        r, _, _ = resolve_detection(i, d, c, pipe, work)
        r["energy"] = c.get("energy")
        out.append(r)
    return work, out


def features(pipe, crops):
    """512-d features and logits for each crop and its mirror."""
    import torch
    from PIL import Image
    clf = pipe.classifier
    m = clf.model
    backbone = torch.nn.Sequential(*list(m.children())[:-1])
    xs = [clf.transform(Image.fromarray(c[:, :, ::-1]).convert("RGB")) for c in crops]
    x = torch.stack(xs).to(clf.device)
    with torch.no_grad():
        f = torch.flatten(backbone(x), 1)
        ff = torch.flatten(backbone(torch.flip(x, dims=[3])), 1)
        return (f.cpu().numpy(), m.fc(f).cpu().numpy(), ff.cpu().numpy(), m.fc(ff).cpu().numpy())


def build_cache(split, files, pipe, gt_all):
    rows, F, L, FF, LF = [], [], [], [], []
    t0 = time.time()
    for k, fn in enumerate(files):
        p = image_path(fn)
        img = cv2.imread(str(p)) if p else None
        if img is None:
            continue
        work, dets = agent_detections(img, pipe)
        if not dets:
            continue
        recs = harness.match_and_classify(dets, gt_all.get(fn, []))
        crops = [pipe._extract_crop(work, d["bbox"], padding=0.05) for d in dets]
        f, lg, ff, lf = features(pipe, crops)
        for r, d in zip(recs, [dets[r["det_index"]] for r in recs]):
            i = r["det_index"]
            r.update(file=fn, dive=DIVE_RE.match(fn).group(1), energy=d["energy"])
            rows.append(r)
            F.append(f[i]); L.append(lg[i]); FF.append(ff[i]); LF.append(lf[i])
        if (k + 1) % 200 == 0:
            print(f"  {split}: {k + 1}/{len(files)} images, {len(rows)} detections "
                  f"({time.time() - t0:.0f}s)", flush=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / f"{split}_detections.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    np.savez_compressed(OUTDIR / f"{split}_features.npz", feat=np.array(F, np.float32),
                        logits=np.array(L, np.float32), feat_flip=np.array(FF, np.float32),
                        logits_flip=np.array(LF, np.float32))
    print(f"  {split}: cached {len(rows)} detections from {len(files)} images", flush=True)


def load_cache(split):
    rows = [json.loads(l) for l in (OUTDIR / f"{split}_detections.jsonl").read_text().splitlines()]
    z = np.load(OUTDIR / f"{split}_features.npz")
    return rows, {k: z[k] for k in z.files}


# ------------------------------------------------------------------ helpers
def softmax(L):
    e = np.exp(L - L.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def energy(L):
    m = L.max(axis=1, keepdims=True)
    return -(m[:, 0] + np.log(np.exp(L - m).sum(axis=1)))


def pooled_isotonic(p, y):
    """Isotonic fit with tied inputs mapped to one value (see fit_deepsea_calibration.compress)."""
    xs, ys = fit_isotonic(p, y)
    ux, inv = np.unique(xs, return_inverse=True)
    return ux, np.bincount(inv, weights=ys) / np.bincount(inv)


def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def maha_fit(X):
    Xn = unit(X)
    mu = Xn.mean(axis=0)
    Xc = Xn - mu
    cov = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    cov += np.eye(Xn.shape[1]) * (0.10 * np.trace(cov) / Xn.shape[1])   # as ood_features.py
    P = np.linalg.pinv(cov)

    def score(Y):
        d = unit(Y) - mu
        return np.einsum("ij,jk,ik->i", d, P, d)
    return score


def boot(dives, stat, n_boot=N_BOOT, seed=0):
    """Paired dive-cluster bootstrap of stat(idx) -> float; returns [lo, hi]."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(dives)
    by = {d: np.flatnonzero(dives == d) for d in uniq}
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([by[d] for d in rng.choice(uniq, size=len(uniq))])
        v = stat(idx)
        if np.isfinite(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def sel_metrics(score, conf_prob, correct):
    pts = curve(score, correct, np.zeros_like(correct))
    e, mx = ece(conf_prob, correct)
    return {"aurc": aurc(pts), "ece": e, "mce": mx, "brier": brier(conf_prob, correct),
            "auroc_correct": auroc(score, correct), "accuracy": float(correct.mean())}


# ------------------------------------------------------------------ stage 2
def analyse(known):
    rc, zc = load_cache("S_cal")
    rp, zp = load_cache("S_policy")
    out = {"n_detections": {"S_cal": len(rc), "S_policy": len(rp)}}
    dive_p = np.array([r["dive"] for r in rp])

    # ---- (1) re-measure the deployed scores
    conf_c = np.clip(np.array([r["combined_confidence"] for r in rc]), EPS, 1 - EPS)
    conf_p = np.clip(np.array([r["combined_confidence"] for r in rp]), EPS, 1 - EPS)
    ok_c = np.array([r["outcome"] == harness.TP for r in rc], float)
    ok_p = np.array([r["outcome"] == harness.TP for r in rp], float)
    out["1_deployed_confidence_S_policy"] = sel_metrics(conf_p, conf_p, ok_p)
    out["1_outcomes_S_policy"] = {o: int(sum(r["outcome"] == o for r in rp))
                                  for o in sorted({r["outcome"] for r in rp})}

    # ---- (4) calibration (fit S_cal -> eval S_policy)
    T = fit_temperature(conf_c, ok_c)
    p_temp = sigmoid(logit(conf_p) / T)
    xs, ys = pooled_isotonic(conf_c, ok_c)
    p_iso = apply_isotonic(xs, ys, conf_p)
    pred_c = np.array([r["predicted_class"] for r in rc])
    pred_p = np.array([r["predicted_class"] for r in rp])
    p_pc = p_iso.copy()
    per_class_fitted = {}
    for c in np.unique(pred_c):
        m = pred_c == c
        if m.sum() >= MIN_PER_CLASS and 0 < ok_c[m].mean() < 1:
            cx, cy = pooled_isotonic(conf_c[m], ok_c[m])
            mp = pred_p == c
            p_pc[mp] = apply_isotonic(cx, cy, conf_p[mp])
            per_class_fitted[str(c)] = int(m.sum())
    cal = {"temperature_T": T, "per_class_fitted": per_class_fitted}
    for name, pr in (("raw", conf_p), ("temperature", p_temp), ("isotonic", p_iso),
                     ("isotonic_per_class", p_pc)):
        cal[name] = sel_metrics(pr, pr, ok_p)
    cal["delta_aurc_per_class_vs_raw_ci"] = boot(dive_p, lambda i: (
        aurc(curve(p_pc[i], ok_p[i], np.zeros(len(i)))) - aurc(curve(conf_p[i], ok_p[i], np.zeros(len(i))))))
    cal["delta_ece_isotonic_vs_raw_ci"] = boot(dive_p, lambda i: ece(p_iso[i], ok_p[i])[0] - ece(conf_p[i], ok_p[i])[0])
    out["4_calibration_S_policy"] = cal

    # ---- (2) flip TTA on the classifier, crops whose GT is a known class
    gt_p = np.array([r["gt_class"] or "" for r in rp])
    raw = json.loads(Path(config.MODEL_PROFILES["deepsea"]["classifier_names"]).read_text())
    names = raw["class_names"] if isinstance(raw, dict) else raw
    kn = np.isin(gt_p, list(known))
    L1, L2 = zp["logits"], (zp["logits"] + zp["logits_flip"]) / 2
    a1 = np.array([names[i] for i in L1.argmax(1)]) == gt_p
    a2 = np.array([names[i] for i in L2.argmax(1)]) == gt_p
    idx_k = np.flatnonzero(kn)
    dk = dive_p[idx_k]
    out["2_flip_tta_S_policy"] = {
        "n_known_crops": int(kn.sum()),
        "acc_single": float(a1[kn].mean()), "acc_flip": float(a2[kn].mean()),
        "delta": float(a2[kn].mean() - a1[kn].mean()),
        "delta_ci": boot(dk, lambda i: float(a2[idx_k][i].mean() - a1[idx_k][i].mean())),
        "changed_predictions": int((L1.argmax(1) != L2.argmax(1))[kn].sum()),
        "maxprob_auroc_correct_single": auroc(softmax(L1)[kn].max(1), a1[kn].astype(float)),
        "maxprob_auroc_correct_flip": auroc(softmax(L2)[kn].max(1), a2[kn].astype(float)),
    }

    # ---- (3) novelty: matched detections, unseen taxon (OOD_ERR) vs known GT
    gt_c = np.array([r["gt_class"] or "" for r in rc])
    kc = np.isin(gt_c, list(known))
    mp_ = np.array([r["matched_gt"] is not None for r in rp])
    y = np.array([r["outcome"] == harness.OOD_ERR for r in rp], float)
    ev = mp_                                   # known-GT or unseen-GT detections
    maha = maha_fit(zc["feat"][kc])
    scores_c = {"energy": energy(zc["logits"])[kc], "maha": maha(zc["feat"][kc]),
                "energy_flip": energy((zc["logits"] + zc["logits_flip"]) / 2)[kc]}
    scores_p = {"energy": energy(zp["logits"]), "maha": maha(zp["feat"]),
                "energy_flip": energy((zp["logits"] + zp["logits_flip"]) / 2)}
    st = {k: (v.mean(), v.std() + 1e-9) for k, v in scores_c.items()}
    z = lambda k, v: (v - st[k][0]) / st[k][1]  # noqa: E731
    scores_c["energy+maha"] = z("energy", scores_c["energy"]) + z("maha", scores_c["maha"])
    scores_p["energy+maha"] = z("energy", scores_p["energy"]) + z("maha", scores_p["maha"])
    nov = {"n_known": int((ev & (y == 0)).sum()), "n_unseen": int((ev & (y == 1)).sum()),
           "unseen_taxa_S_policy": sorted({r["gt_class"] for r in rp if r["outcome"] == harness.OOD_ERR})}
    ie = np.flatnonzero(ev)
    for k in scores_p:
        thr = float(np.quantile(scores_c[k], KEEP_KNOWN))
        s, yy = scores_p[k][ie], y[ie]
        nov[k] = {"auroc": auroc(s, yy), "fpr_at_95tpr": fpr_at_tpr(s, yy),
                  "threshold": thr, "unseen_flagged": float((s[yy == 1] > thr).mean()),
                  "known_flagged": float((s[yy == 0] > thr).mean())}
    de = dive_p[ie]
    for k in ("maha", "energy_flip", "energy+maha"):
        nov[f"delta_auroc_{k}_vs_energy_ci"] = boot(de, lambda i, k=k: (
            auroc(scores_p[k][ie][i], y[ie][i]) - auroc(scores_p["energy"][ie][i], y[ie][i])))
    out["3_novelty_S_policy"] = nov
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recache", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    man = json.loads(MAN.read_text())
    known = set(man["label_space"]["known"])
    harness.KNOWN_CLASSES = known
    if any(not (OUTDIR / f"{s}_detections.jsonl").exists() for s in ("S_cal", "S_policy")) or args.recache:
        from pipeline import EnsemblePipeline
        pipe = EnsemblePipeline(profile="deepsea")
        gt_all = harness.load_ground_truth()
        for split in ("S_cal", "S_policy"):
            build_cache(split, sorted(man["files"][split])[: args.limit], pipe, gt_all)

    res = analyse(known)
    (OUTDIR / "step1_results.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
    print(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {(OUTDIR / 'step1_results.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
