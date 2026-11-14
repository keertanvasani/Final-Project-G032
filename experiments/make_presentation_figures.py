"""Collect and generate the figures for the project presentation.

Gathers every figure produced by the experiment scripts into one folder with
names that sort in presentation order, and generates two that no script emits
but that the talk needs: the OOD feature-space comparison and the learned
policy's feature weights.

Writes an INDEX.md saying what each figure shows and the one sentence it
supports, so the folder is usable without going back through the analysis.

Usage:
    python experiments/make_presentation_figures.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "experiments" / "results" / "phase_a"
DEG = ROOT / "experiments" / "results" / "degradation"
MODELS = ROOT / "models" / "phase_b"
OUT = ROOT / "experiments" / "presentation_figures"

INK = "#12232B"
ACCENT = "#0A6E8A"
CRITICAL = "#B3402F"
GOOD = "#3F7A5E"
MUTED = "#5C7078"

# (source, destination, caption)
COPIES = [
    (CACHE / "J_test_risk_coverage.png",
     "01_the_problem_risk_coverage.png",
     "Phase A. The frozen aquarium pipeline on 3,261 deep-sea frames. Risk stays "
     "near 0.89 across the useful coverage range and the OOD-error panel RISES "
     "toward low coverage — the most confident detections are the most likely to "
     "be confident misidentifications. No threshold fixes this."),
    (CACHE / "J_policy_reliability.png",
     "02_calibration_reliability.png",
     "Calibration. Raw confidence averaged 0.558 against 9.1% actual accuracy. "
     "Isotonic regression brings the curve onto the diagonal (ECE 0.467 -> 0.016). "
     "Note the trivial constant predictor also scores well on ECE — the value is "
     "that calibration keeps the ranking."),
    (CACHE / "J_policy_risk_coverage_gates.png",
     "03_gate_comparison.png",
     "Adding novelty-awareness to the gate. At matched coverage the OOD-aware "
     "gate halves the OOD-error rate among accepted detections (0.373 -> 0.182). "
     "Its curve stops early because the hard mask caps coverage at 14.7%."),
    (CACHE / "J_policy_learned_gate.png",
     "04_learned_policy.png",
     "The learned state-conditional policy. It reaches full coverage instead of "
     "the fixed gate's 14.7% ceiling, and lowers risk at matched coverage "
     "(0.846 -> 0.776 at 30%, CI excluding zero)."),
    (DEG / "degradation_curves.png",
     "06_degradation_does_it_notice.png",
     "Controlled in-domain degradation. Blur is caught (abstention 1.6% -> 83%, "
     "risk nearly flat). Colour shift and compression are NOT (abstention barely "
     "moves while risk climbs). The system notices only the degradation it was "
     "explicitly programmed to check."),
    (MODELS / "yolov8n_seed0" / "results.png",
     "07_detector_training_curves.png",
     "Phase B detector training, seed 0 of 3. 80 epochs, no early stopping."),
    (MODELS / "yolov8n_seed0" / "BoxPR_curve.png",
     "08_detector_pr_curve.png",
     "Precision-recall by class after on-domain retraining. mAP@50 0.688 "
     "(3 seeds, sd 0.005)."),
    (MODELS / "yolov8n_seed0" / "confusion_matrix_normalized.png",
     "09_detector_confusion_matrix.png",
     "Normalised confusion matrix, 8 known deep-sea taxa."),
    (MODELS / "yolov8n_seed0" / "val_batch0_labels.jpg",
     "10_qualitative_ground_truth.jpg",
     "Ground truth on a validation batch — what the detector should find."),
    (MODELS / "yolov8n_seed0" / "val_batch0_pred.jpg",
     "11_qualitative_predictions.jpg",
     "The retrained detector's predictions on the same batch. Pair with the "
     "previous figure side by side."),
    (ROOT / "outputs" / "annotated_jellyfish.png",
     "12_qualitative_pipeline_output.png",
     "End-to-end pipeline output with the annotated detections it returns."),
    (ROOT / "outputs" / "annotated_blurry_stock_test.png",
     "13_qualitative_abstention.png",
     "A deliberately blurry input. The refusal gate fires and the system "
     "declines to answer rather than guessing."),
]


def fig_ood_comparison():
    """Aquarium vs retrained features, out-of-sample on S_policy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    aq = json.loads((CACHE / "aquarium-feats_S_policy_ood_features.json").read_text())
    rt = json.loads((CACHE / "retrained-feats_S_policy_ood_features.json").read_text())

    def get(d, regime, score):
        return d["results"][f"S_policy:{regime}:{score}"]["auroc"]

    labels = ["near-OOD\nenergy", "near-OOD\nMahalanobis",
              "far-OOD\nenergy", "far-OOD\nMahalanobis"]
    keys = [("near", "energy"), ("near", "mahalanobis"),
            ("far", "energy"), ("far", "mahalanobis")]
    a = [get(aq, r, s) for r, s in keys]
    b = [get(rt, r, s) for r, s in keys]

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=150)
    ax.bar(x - w / 2, a, w, label="aquarium-trained features", color=MUTED)
    ax.bar(x + w / 2, b, w, label="deep-sea retrained features", color=ACCENT)
    ax.axhline(0.5, ls="--", lw=1, color=CRITICAL)
    ax.text(len(labels) - 0.42, 0.515, "chance", color=CRITICAL, fontsize=9)
    for i, (va, vb) in enumerate(zip(a, b)):
        ax.text(i - w / 2, va + 0.012, f"{va:.3f}", ha="center", fontsize=8.5, color=INK)
        ax.text(i + w / 2, vb + 0.012, f"{vb:.3f}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("AUROC (out-of-sample)"); ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_title("Does on-domain training improve novelty detection?", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(OUT / "05_ood_feature_comparison.png")
    plt.close(fig)


def fig_feature_weights():
    """What the learned policy actually relies on."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    d = json.loads((CACHE / "J_policy_learned_gate.json").read_text())
    names, w = d["features"], d["weights"][1:]
    order = np.argsort([abs(v) for v in w])
    names = [names[i].replace("_", " ") for i in order]
    vals = [w[i] for i in order]
    colors = [CRITICAL if v < 0 else GOOD for v in vals]

    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=150)
    ax.barh(names, vals, color=colors)
    ax.axvline(0, color=INK, lw=1)
    for i, v in enumerate(vals):
        ax.text(v + (0.08 if v >= 0 else -0.08), i, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right",
                fontsize=9, color=INK)
    ax.set_xlabel("standardised weight  (negative = pushes toward ABSTAIN)")
    ax.set_title("What the learned policy relies on", fontsize=11.5)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    ax.margins(x=0.16)
    fig.tight_layout()
    fig.savefig(OUT / "14_learned_policy_feature_weights.png")
    plt.close(fig)


def fig_action_effects():
    """Every discretionary action, effect on correctness, both platforms."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    A = ROOT / "experiments" / "results" / "actions"
    za = json.loads((A / "J_policy_action_value.json").read_text())
    zb = json.loads((A / "S_policy_phaseB_action_value.json").read_text())
    ra = json.loads((A / "J_policy_reclassify_analysis.json").read_text())
    rb = json.loads((A / "S_policy_phaseB_reclassify_analysis.json").read_text())
    ia = json.loads((A / "J_policy_image_actions_analysis.json").read_text())
    ib = json.loads((A / "S_policy_phaseB_image_actions_analysis.json").read_text())

    # (label, phase A (delta, lo, hi), phase B (delta, lo, hi))
    def rc(d):
        k = "classifier_only|all detections"
        return d["bootstrap"][k]["delta"], *d["bootstrap"][k]["ci95"]

    def im(d, name):
        k = f"{name}|all images"
        return d["bootstrap"][k]["delta_tp_per_image"], *d["bootstrap"][k]["ci95"]

    # The two pairs are measured in different units and must not share an axis:
    # the whole-frame actions change how many correct detections an IMAGE
    # yields, while the per-box actions change whether a DETECTION is correct.
    # Plotting them on one scale would invite a comparison that is not defined.
    panels = [
        ("Whole-frame actions", "change in correct detections per image",
         [("ENHANCE\n(vs no enhancement)", im(ia, "ENHANCE"), im(ib, "ENHANCE")),
          ("REDETECT @1280\n(vs 640)", im(ia, "FULL_IMAGE_REDETECT"),
           im(ib, "FULL_IMAGE_REDETECT"))]),
        ("Per-detection actions", "change in correctness rate per detection",
         [("ZOOM\n(vs no zoom)",
           (za["rule_net_effect"], *za["rule_net_ci95"]),
           (zb["rule_net_effect"], *zb["rule_net_ci95"])),
          ("RECLASSIFY\n(vs classifier only)", rc(ra), rc(rb))]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), dpi=150)
    h = 0.34
    for ax, (title, xlabel, items) in zip(axes, panels):
        y = np.arange(len(items))
        for off, key, colr, lab in ((+h / 2, 1, MUTED, "frozen aquarium stack"),
                                    (-h / 2, 2, ACCENT, "retrained deep-sea stack")):
            vals = [it[key][0] for it in items]
            lo = [it[key][0] - it[key][1] for it in items]
            hi = [it[key][2] - it[key][0] for it in items]
            ax.barh(y + off, vals, h, xerr=[lo, hi], color=colr, label=lab,
                    error_kw={"lw": 1, "ecolor": INK, "capsize": 2.5})
        ax.axvline(0, color=INK, lw=1.2)
        ax.set_yticks(y)
        ax.set_yticklabels([it[0] for it in items], fontsize=9.5)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel, fontsize=9.5)
        ax.set_title(title, fontsize=10.5)
        ax.grid(axis="x", alpha=0.25, lw=0.6)
        ax.margins(y=0.35)
    axes[0].legend(fontsize=8.5, frameon=False, loc="lower left")
    fig.suptitle("Does any discretionary action earn its cost?  "
                 "(bars left of zero = the action makes things worse)",
                 fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "15_action_effects.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    index = []

    for src, dst, cap in COPIES:
        if src.is_file():
            shutil.copy2(src, OUT / dst)
            index.append((dst, cap))
            print(f"  copied  {dst}")
        else:
            print(f"  MISSING {src.relative_to(ROOT)} -> skipped")

    try:
        fig_ood_comparison()
        index.append((
            "05_ood_feature_comparison.png",
            "The OOD experiment. Confidence-based scores sit at chance. Retraining "
            "on deep-sea data lifts near-OOD energy from 0.553 to 0.806 — the "
            "novelty signal moves into the logits once the classes are relevant."))
        print("  generated 05_ood_feature_comparison.png")
    except Exception as e:
        print(f"  OOD comparison figure failed: {e}")

    try:
        fig_feature_weights()
        index.append((
            "14_learned_policy_feature_weights.png",
            "The most quotable result. The policy weights novelty distance about "
            "5x more heavily than calibrated confidence, and ignores classifier "
            "confidence entirely — it learned from data what the Phase A analysis "
            "argued: confidence is not the signal, novelty is."))
        print("  generated 14_learned_policy_feature_weights.png")
    except Exception as e:
        print(f"  feature-weight figure failed: {e}")

    try:
        fig_action_effects()
        index.append((
            "15_action_effects.png",
            "Every discretionary action, measured against its counterfactual on "
            "both platforms, with 95% CIs. None is above zero. ENHANCE is "
            "significantly harmful on the retrained stack; ZOOM is inert; "
            "RECLASSIFY does not beat simply trusting the classifier. The "
            "hand-written adaptive layer is a net negative."))
        print("  generated 15_action_effects.png")
    except Exception as e:
        print(f"  action-effects figure failed: {e}")

    index.sort(key=lambda t: t[0])
    lines = ["# Presentation figures", "",
             "Generated by `experiments/make_presentation_figures.py`. Each entry "
             "gives the figure and the single claim it supports.", ""]
    for name, cap in index:
        lines += [f"### {name}", "", cap, ""]
    (OUT / "INDEX.md").write_text("\n".join(lines))

    print(f"\n{len(index)} figures in {OUT.relative_to(ROOT)}")
    print(f"Index written to {(OUT / 'INDEX.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
