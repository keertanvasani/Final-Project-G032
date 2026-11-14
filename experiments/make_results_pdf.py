"""Build the presentation results guide as a PDF.

One page per figure: the figure itself, what it shows in plain terms, the line
to say when presenting it, and — where a number invites an obvious challenge —
the answer to that challenge. Written for someone explaining the work to an
audience that does not share the vocabulary.

Usage:
    python experiments/make_results_pdf.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "experiments" / "presentation_figures"
OUT = ROOT / "experiments" / "RESULTS_GUIDE.pdf"

INK = colors.HexColor("#12232B")
ACCENT = colors.HexColor("#0A6E8A")
MUTED = colors.HexColor("#5C7078")
CRIT = colors.HexColor("#B3402F")
RULE = colors.HexColor("#C9D6D6")
SOFT = colors.HexColor("#EDF3F2")

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
CONTENT_W = PAGE_W - 2 * MARGIN

ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=ss["Title"], fontName="Times-Bold",
                            fontSize=26, leading=30, textColor=INK, spaceAfter=6),
    "sub": ParagraphStyle("s", parent=ss["Normal"], fontName="Helvetica",
                          fontSize=11.5, leading=16, textColor=MUTED, spaceAfter=18),
    "h": ParagraphStyle("h", parent=ss["Heading1"], fontName="Times-Bold",
                        fontSize=17, leading=21, textColor=INK, spaceAfter=2),
    "eyebrow": ParagraphStyle("e", parent=ss["Normal"], fontName="Helvetica-Bold",
                              fontSize=8, leading=11, textColor=ACCENT,
                              spaceAfter=3),
    "body": ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica",
                           fontSize=10, leading=14.5, textColor=INK,
                           alignment=TA_LEFT, spaceAfter=7),
    "say": ParagraphStyle("say", parent=ss["Normal"], fontName="Helvetica-Oblique",
                          fontSize=10, leading=14.5, textColor=INK,
                          leftIndent=8, spaceAfter=6),
    "small": ParagraphStyle("sm", parent=ss["Normal"], fontName="Helvetica",
                            fontSize=8.5, leading=12, textColor=MUTED),
    "label": ParagraphStyle("lb", parent=ss["Normal"], fontName="Helvetica-Bold",
                            fontSize=9, leading=12.5, textColor=ACCENT,
                            spaceAfter=2),
}


def box(text: str, style, bg, border):
    t = Table([[Paragraph(text, style)]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LINEBEFORE", (0, 0), (0, -1), 2, border),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return t


def figure(name: str, max_h: float):
    p = FIGS / name
    if not p.is_file():
        return Paragraph(f"[missing figure: {name}]", S["small"])
    from PIL import Image as PILImage
    with PILImage.open(p) as im:
        w, h = im.size
    scale = min(CONTENT_W / w, max_h / h)
    return Image(str(p), width=w * scale, height=h * scale)


# (file, max image height mm, eyebrow, heading, what it shows, what to say, note)
SLIDES = [
    ("01_the_problem_risk_coverage.png", 72, "THE SETUP",
     "Why this project exists",
     "Two curves. The left shows how often the system is wrong as you make it "
     "pickier about which answers it will commit to. The right shows how often "
     "it confidently names an organism it was never trained on.",
     "Normally, if a model is unreliable, you raise the confidence bar and accept "
     "fewer answers. This shows that does not work here — the error rate stays "
     "flat at about 89% no matter how selective we are. And the right panel goes "
     "the wrong way: the more confident the system is, the more likely it is to "
     "be confidently naming something it has never seen.",
     "This is the slide that justifies everything after it. Without it, the "
     "improvements have no problem to solve."),

    ("02_calibration_reliability.png", 78, "RESULT 1 — HONEST CONFIDENCE",
     "Making the confidence score mean something",
     "The dashed diagonal is a perfectly honest model: when it says 80% "
     "confident, it is right 80% of the time. The raw system sits far away from "
     "that line. After calibration it sits close to it.",
     "The system was claiming 56% confidence while being right 9% of the time. "
     "After calibration, its confidence number means what it says — which matters "
     "because everything downstream has to act on that number.",
     "If asked <i>“isn’t that just predicting the average?”</i> — partly, yes, and "
     "we measured exactly that. The value is that it stays honest while still "
     "ranking detections correctly, which a constant cannot do."),

    ("05_ood_feature_comparison.png", 76, "RESULT 2 — THE STRONGEST FINDING",
     "The system can learn to recognise the unfamiliar",
     "Four pairs of bars. Grey is the original model, blue is the model retrained "
     "on deep-sea data. The dashed line marks chance — a coin flip.",
     "This asks whether the system can tell “I have never seen this creature” "
     "from “I know this one”. Using its confidence score, it is at chance. But "
     "the information is still inside the network — once retrained on the right "
     "data, novelty detection rises from 0.55 to 0.81.",
     "The insight worth stating: the information needed to recognise something "
     "new survives inside the model and is destroyed by the final layer that "
     "turns it into a species name. That is a precise diagnosis, not “the model "
     "is bad”."),

    ("14_learned_policy_feature_weights.png", 76, "RESULT 3 — INDEPENDENT CONFIRMATION",
     "The model discovered the same thing on its own",
     "What the trained decision model relies on when deciding whether to trust a "
     "detection. Longer bar means more influence. Novelty distance dominates "
     "everything else.",
     "We trained a model to decide when to trust a detection, and let it work out "
     "for itself what to pay attention to. It weighted novelty about five times "
     "more heavily than confidence — and ignored classifier confidence almost "
     "entirely. It found from data exactly what our analysis argued separately.",
     "Two independent lines of evidence agreeing is much stronger than either "
     "alone. This is the slide that proves the thesis rather than asserting it."),

    ("04_learned_policy.png", 74, "RESULT 4 — THE SYSTEM IMPROVES",
     "A learned decision beats a hand-tuned threshold",
     "Error rate against how many answers the system gives. Lower is better. The "
     "learned policy sits below the alternatives and spans the full width of the "
     "chart.",
     "At the same number of answers given, the learned policy is wrong less "
     "often — 84.6% down to 77.6%, with a confidence interval that excludes zero, "
     "so this is a real effect and not noise. It also works across the whole "
     "operating range, where the fixed rule only worked at one very restrictive "
     "setting.",
     "Being straight about scale: the system is improved, not solved. It is still "
     "wrong most of the time, because the underlying detector is working far "
     "outside the domain it was trained for."),

    ("08_detector_pr_curve.png", 82, "RESULT 5 — THE RETRAINED DETECTOR",
     "Trained on the right data, it works",
     "Precision against recall for each of the eight deep-sea species the "
     "retrained detector knows. Curves toward the top-right are better.",
     "Retrained on deep-sea imagery, the detector reaches 0.688 mAP across three "
     "independent training runs, with a standard deviation of 0.005. That "
     "tightness matters — it means this is a stable result, not one lucky run.",
     "Pair this with the side-by-side of ground truth and predictions if the "
     "audience wants to see the boxes rather than the curve."),

    ("15_action_effects.png", 62, "RESULT 6 — A METHOD RESULT",
     "We checked whether the clever parts actually help",
     "Each of the system's four adaptive behaviours, measured against simply not "
     "doing it, on both models, with confidence intervals. Bars left of zero mean "
     "the action makes things worse.",
     "The system has four adaptive behaviours — enhance the image, zoom in, "
     "re-classify, re-detect. Nobody had ever checked whether they help. We forced "
     "each one on every detection and scored it against not doing it. None earns "
     "its cost, and image enhancement actively hurts — worst on exactly the images "
     "its own rule selects.",
     "Every bar is negative, and this is still a positive result. It says these "
     "rules were never validated, we validated them, and the learned policy now "
     "has a measured baseline to beat instead of assumptions to imitate. Removing "
     "the layer makes the system both more accurate and cheaper."),

    ("06_degradation_does_it_notice.png", 58, "RESULT 7 — DOES IT KNOW ITS LIMITS?",
     "A controlled test of self-awareness",
     "We took clean aquarium images and damaged them deliberately — blur, motion "
     "blur, noise, darkness, low contrast, colour cast, compression — at three "
     "increasing strengths, with the correct answers unchanged. Left panel: how "
     "often the system is wrong. Middle: how often it declines to answer. Right: "
     "how confident it stays. Severity 0 is the undamaged image.",
     "The question is not whether accuracy falls as images get worse — of course "
     "it does. The question is whether the system <i>notices</i>. A system aware "
     "of its limits refuses more as evidence degrades, holding its error rate "
     "roughly flat. A system blind to it keeps answering and is simply wrong more "
     "often. Blur is the success case: refusals rise from 2% to 83% and the error "
     "rate barely moves. But under colour shift and compression the error rate "
     "climbs while refusals stay flat — the system does not notice at all.",
     "Blur is also the <i>only</i> damage the pipeline explicitly tests for. So "
     "the finding is precise: hand-written quality rules catch exactly what their "
     "author thought to check for, and nothing else. Colour shift matters most, "
     "because red light fades fastest with depth — it is the dominant distortion "
     "in real underwater imagery, and gives a mechanical explanation for part of "
     "the failure on the first slide."),
]


def build():
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title="Results Guide — Marine Organism Recognition",
        author="Final Year Project",
    )
    story = []

    # ---- cover ----
    story.append(Spacer(1, 26 * mm))
    story.append(Paragraph("Results Guide", S["title"]))
    story.append(Paragraph(
        "Marine organism recognition under domain shift and open-set conditions<br/>"
        "Every figure, what it shows, and what to say about it.", S["sub"]))

    rows = [
        ["The problem", "An in-domain-validated system is wrong 88.8% of the "
                        "time on real deep-sea footage"],
        ["Why", "Two-thirds of the organisms are outside its vocabulary, and its "
                "confidence carries no information about correctness"],
        ["What we built", "Calibrated confidence, feature-space novelty detection, "
                          "and a learned accept/abstain policy"],
        ["What improved", "Novelty detection 0.55 → 0.81; confident errors on "
                          "unknown organisms cut by more than half"],
        ["What we learned", "None of the system's four hand-written adaptive "
                            "behaviours earns its computational cost"],
    ]
    t = Table([[Paragraph(f"<b>{a}</b>", S["body"]), Paragraph(b, S["body"])]
               for a, b in rows], colWidths=[38 * mm, CONTENT_W - 38 * mm])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (0, -1), 0),
    ]))
    story.append(t)
    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph(
        "All figures are measured on development splits with confidence intervals "
        "from a bootstrap over whole dives. The held-out test set has been read "
        "once, to establish the baseline, and not since — so none of these numbers "
        "has been tuned against the data they will finally be judged on. "
        "Full numbers and caveats: <font face='Courier'>experiments/RESULTS.md</font>.",
        S["small"]))
    story.append(PageBreak())

    # ---- one page per figure ----
    for i, (fname, maxh, eyebrow, head, shows, say, note) in enumerate(SLIDES, 1):
        story.append(Paragraph(eyebrow, S["eyebrow"]))
        story.append(Paragraph(head, S["h"]))
        story.append(Spacer(1, 5))
        story.append(figure(fname, maxh * mm))
        story.append(Spacer(1, 9))

        story.append(Paragraph("WHAT THE CHART SHOWS", S["label"]))
        story.append(Paragraph(shows, S["body"]))

        story.append(Paragraph("WHAT TO SAY", S["label"]))
        story.append(box(say, S["say"], SOFT, ACCENT))
        story.append(Spacer(1, 7))

        story.append(Paragraph("WORTH ADDING", S["label"]))
        story.append(Paragraph(note, S["body"]))

        story.append(Spacer(1, 6))
        story.append(Paragraph(
            f"Figure file: <font face='Courier'>{fname}</font>", S["small"]))
        if i < len(SLIDES):
            story.append(PageBreak())

    # ---- closing ----
    story.append(PageBreak())
    story.append(Paragraph("SUGGESTED ORDER", S["eyebrow"]))
    story.append(Paragraph("Putting the deck together", S["h"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<b>Problem</b> (01) → <b>confidence is meaningless</b> (02) → "
        "<b>but the signal survives</b> (05) → <b>the model finds it "
        "independently</b> (14) → <b>the system improves</b> (04) → "
        "<b>the detector works</b> (08) → <b>what we learned about the rules</b> "
        "(15) → <b>and about its self-awareness</b> (06).", S["body"]))
    story.append(Spacer(1, 5))
    story.append(Paragraph(
        "The degradation result (06) can also be moved directly after the problem "
        "slide, where it explains <i>why</i> the system fails rather than adding "
        "another finding at the end. It needs about two minutes either way — it is "
        "a three-panel chart and the argument depends on reading the middle panel "
        "against the left one, so do not rush it or put it last if time is tight.",
        S["body"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "One figure is left out: <b>03_gate_comparison</b> is superseded by 04 and "
        "would repeat it.", S["body"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("QUESTIONS TO EXPECT", S["label"]))
    qa = [
        ("Why are your numbers so low?",
         "Because they are measured honestly. Two-thirds of the test organisms are "
         "outside the model's vocabulary, and we removed the data leakage that "
         "inflates published results on this dataset — 57.3% of its supplied test "
         "images shared a dive with training images."),
        ("Why not just retrain and be done?",
         "We did, and it reaches 0.688 mAP. But retraining does not solve the core "
         "problem: the ocean will always contain organisms the model was not "
         "trained on. Knowing when you are looking at one is a separate capability."),
        ("Isn't this just YOLO with extra steps?",
         "YOLO decides what is in the image. This decides whether it has looked "
         "hard enough to answer at all, and whether the answer is worth the "
         "computation. That is a different question."),
        ("Have you compared against published work?",
         "Not directly, and deliberately. Published numbers on this dataset largely "
         "use the leaking splits and different label spaces, so the comparison "
         "would mislead. We compare against our own frozen baseline under "
         "identical conditions."),
    ]
    for q, a in qa:
        story.append(Paragraph(f"<b>{q}</b>", S["body"]))
        story.append(Paragraph(a, S["body"]))
        story.append(Spacer(1, 3))

    doc.build(story)
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    build()
