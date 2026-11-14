"""Render the system architecture as a slide-sized PNG.

Two bands: the perception path that always executes, and the decision layer
that governs it. Decision points are numbered D1-D4 so the diagram can be
narrated in order. Kept deliberately sparse: the target canvas is 6.3 x 3.55
inches, so anything denser is unreadable when projected.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, ACCENT, MUTED = "#12232B", "#0A6E8A", "#5C7078"
GOOD, CRIT, WARN = "#3F7A5E", "#B3402F", "#9A6B18"
SOFT, LIGHT = "#DCEBF0", "#EDF3F2"

fig, ax = plt.subplots(figsize=(13.5, 6.2), dpi=200)
ax.set_xlim(0, 135); ax.set_ylim(0, 62); ax.axis("off")

def box(x, y, w, h, label, sub=None, fc=LIGHT, ec=INK, lw=1.4, fs=10.5, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                facecolor=fc, edgecolor=ec, linewidth=lw))
    ax.text(x + w/2, y + h/2 + (1.6 if sub else 0), label, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", color=INK)
    if sub:
        ax.text(x + w/2, y + h/2 - 2.4, sub, ha="center", va="center",
                fontsize=8.2, color=MUTED)

def arrow(x1, y1, x2, y2, color=INK, style="-|>", lw=1.5, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=13, color=color, linewidth=lw,
                                 linestyle=ls, shrinkA=0, shrinkB=0))

# ---------------- band labels ----------------
ax.text(1, 57.5, "PERCEPTION  (always executes)", fontsize=9.5, fontweight="bold",
        color=MUTED)
ax.text(1, 24.5, "DECISION LAYER  (governs the above)", fontsize=9.5, fontweight="bold",
        color=ACCENT)

# ---------------- perception row ----------------
y = 38
box(1,   y, 15, 11, "Input image", "JPEG / PNG / WebP", fc="#E8EEED")
box(20,  y, 18, 11, "D1  Quality", "blur · brightness\n· contrast", fc=SOFT, ec=ACCENT, lw=2)
box(42,  y, 19, 11, "Dual detection", "YOLOv8-n  +  YOLOv8-s", fc="#FFFFFF")
box(65,  y, 15, 11, "WBF", "IoU 0.55 · [2,1]", fc="#FFFFFF")
box(84,  y, 18, 11, "D2  Review gate", "any box ≥ 0.40 ?", fc=SOFT, ec=ACCENT, lw=2)
box(106, y, 18, 11, "ResNet-18", "224 px · 8 classes", fc="#FFFFFF")

for x1, x2 in ((16,20), (38,42), (61,65), (80,84), (102,106)):
    arrow(x1, y+5.5, x2, y+5.5)

# enhance branch (above D1)
box(20, 51.5, 18, 7, "ENHANCE", "CLAHE · sharpen", fc="#F5E9D2", ec=WARN, lw=1.4, fs=9)
arrow(29, y+11, 29, 51.5, color=WARN)
ax.text(30.5, 49.5, "no", fontsize=8, color=WARN)

# abstain branch (below D2)
box(84, 27.5, 18, 7, "ABSTAIN", "flag for review", fc="#F4E1DC", ec=CRIT, lw=1.4, fs=9)
arrow(93, y, 93, 34.5, color=CRIT)
ax.text(94.5, 35.5, "no", fontsize=8, color=CRIT)

# ---------------- decision layer ----------------
yd = 8
box(20, yd, 24, 12, "Uncertainty", "calibration  +  OOD score", fc=SOFT, ec=ACCENT, lw=2, fs=10)
box(50, yd, 26, 12, "Agent", "max  Q(s,a) − λ·C(a)", fc=SOFT, ec=ACCENT, lw=2.4, fs=10)
box(82, yd, 20, 12, "D3 / D4", "zoom · re-classify", fc=SOFT, ec=ACCENT, lw=2, fs=10)
box(108, yd, 20, 12, "Accept / Abstain", "+ decision trace", fc="#DDEBE3", ec=GOOD, lw=1.8, fs=9.5)

arrow(44, yd+6, 50, yd+6, color=ACCENT)
arrow(76, yd+6, 82, yd+6, color=ACCENT)
arrow(102, yd+6, 108, yd+6, color=ACCENT)

# perception -> uncertainty
arrow(115, y, 115, 24, color=ACCENT, ls="--")
ax.text(116, 30, "scored\ndetections", fontsize=8, color=ACCENT)
arrow(112, 24, 32, 24, color=ACCENT, ls="--", style="-")
arrow(32, 24, 32, yd+12, color=ACCENT, ls="--")

# evidence loop back into perception
arrow(92, yd+12, 92, 24.5, color=ACCENT, ls=":", lw=1.6)
ax.text(46, 21.6, "acquire more evidence → re-observe", fontsize=8.2,
        color=ACCENT, style="italic")

ax.text(1, 2.0, "D1–D4 are the four runtime decision points.  Every branch taken is "
                "written to the decision trace.",
        fontsize=8.6, color=MUTED)

fig.tight_layout(pad=0.3)
fig.savefig("experiments/presentation_figures/arch_diagram_slide.png",
            bbox_inches="tight", facecolor="white")
print("wrote experiments/presentation_figures/arch_diagram_slide.png")
