"""Before/after figure for the burned-in overlay-text filter.

Frame 03 of the demo set carries a single line of telemetry across the top. The
detector boxes the digits "02/10/", the classifier calls them fish at 0.99, and
the agreement boost carries the result out at 61% — a species record with a
timestamp inside the box. This renders the same frame with the filter off and on.

Both panels are produced by running the real detection path, not by redrawing a
remembered result, so the figure fails loudly if the fix regresses.

Usage:
    python experiments/make_text_filter_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ensemble  # noqa: E402
import text_overlay as T  # noqa: E402
from agent_controller import assess_image_quality, enhance_image  # noqa: E402
from pipeline import EnsemblePipeline  # noqa: E402

SRC = ROOT / "demo_images" / "03_seacucumber_called_fish_100pct.jpg"
OUT = ROOT / "experiments" / "presentation_figures" / "17_text_filter.png"

INK = (0x12, 0x23, 0x2B)
PAPER = (0xFF, 0xFF, 0xFF)
MUTED = (0x5C, 0x70, 0x78)
CRITICAL = (0xB3, 0x40, 0x2F)
GOOD = (0x3F, 0x7A, 0x5E)

SCALE = 2
MARGIN, GAP = 44, 36


def font(size, bold=False):
    for n in (("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")):
        try:
            return ImageFont.truetype(f"C:/Windows/Fonts/{n}", size)
        except OSError:
            continue
    return ImageFont.load_default()


F_TITLE, F_HEAD, F_BODY, F_TAG = font(30, True), font(20, True), font(17), font(16, True)


def tag(dr, x, y, text, colour, w_lim):
    w = dr.textlength(text, font=F_TAG)
    pad, h = 9, 25
    x = max(2, min(x, w_lim - w - 2 * pad - 2))
    y = max(2, y)
    dr.rounded_rectangle((x, y, x + w + 2 * pad, y + h), radius=5, fill=colour)
    dr.text((x + pad, y + 4), text, font=F_TAG, fill=PAPER)


def draw(base_bgr, dets, suppressed):
    """Panel image with kept boxes in green and suppressed ones in red."""
    rgb = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb).resize(
        (rgb.shape[1] * SCALE, rgb.shape[0] * SCALE), Image.LANCZOS)
    dr = ImageDraw.Draw(im)
    items = [(d, GOOD, False) for d in dets] + [(d, CRITICAL, True) for d in suppressed]
    for d, colour, is_txt in items:
        x0, y0, x1, y1 = [v * SCALE for v in d["bbox"]]
        dr.rectangle((x0, y0, x1, y1), outline=colour, width=4)
    for d, colour, is_txt in items:
        x0, y0, x1, y1 = [v * SCALE for v in d["bbox"]]
        # Plain ASCII only: the system UI font has no dingbats, and a missing
        # glyph renders as a tofu box.
        label = (f'{d["label"]}  {d["shown"]}' if not is_txt
                 else f'{d["label"]}  {d["shown"]}  REJECTED: TEXT')
        tag(dr, x0, y0 - 29 if y0 > 34 else y1 + 5, label, colour, im.size[0])
    return im


def main():
    bgr = cv2.imread(str(SRC))
    q = assess_image_quality(bgr)
    enh = bgr
    if q["needs_enhancement"] or q["is_blurry"]:
        enh, _ = enhance_image(bgr, q)

    pipe = EnsemblePipeline()
    ra, rb = ensemble.run_dual_detection(enh, pipe.model_a, pipe.model_b)
    fused = ensemble.fuse_detections(ra, rb, enh.shape[:2])

    crops = [pipe._extract_crop(enh, d["bbox"], padding=0.05) for d in fused]
    cls = pipe.classifier.classify_crops(crops)
    import config
    for d, c in zip(fused, cls):
        comb = (config.DETECTOR_WEIGHT * d["confidence"]
                + config.CLASSIFIER_WEIGHT * c["confidence"])
        if d["class_name"] == c["class_name"]:
            comb = min(comb * (1 + config.AGREEMENT_BOOST), 1.0)
        d["label"] = c["class_name"]
        d["shown"] = f"{comb:.0%}"
        d["text"] = T.text_score(enh, d["bbox"])

    kept, dropped = T.filter_text_detections(enh, fused)
    print(f"before {len(fused)} boxes -> after {len(kept)}; "
          f"suppressed {[(d['label'], round(d['text'], 2)) for d in dropped]}")
    if not dropped:
        raise SystemExit("no box was suppressed — the fix has regressed")

    left = draw(enh, fused, [])                 # everything, as before the fix
    right = draw(enh, kept, dropped)            # suppression marked

    PW, PH = left.size
    top, body_y = 96, 96 + 34 + PH + 26
    W = MARGIN * 2 + PW * 2 + GAP
    H = body_y + 150
    canvas = Image.new("RGB", (W, H), PAPER)
    c = ImageDraw.Draw(canvas)
    c.text((MARGIN, 30), "The detector was reading the clock", font=F_TITLE, fill=INK)
    c.text((MARGIN, 68),
           "Demo frame 03. Contrast 20.4 triggers CLAHE, which sharpens the "
           "burned-in telemetry along with the seabed.", font=F_BODY, fill=MUTED)

    for i, (img, head) in enumerate([(left, "Before \u2014 every fused box"),
                                     (right, "After \u2014 text boxes rejected")]):
        x = MARGIN + i * (PW + GAP)
        c.text((x, top), head, font=F_HEAD, fill=INK)
        canvas.paste(img, (x, top + 34))
        c.rectangle((x, top + 34, x + PW, top + 34 + PH),
                    outline=(0xD5, 0xDD, 0xE0), width=1)

    worst = max(d["text"] for d in dropped)
    notes = [
        (CRITICAL, "A timestamp reported as an organism.",
         "The detector scores the digits 0.26; the classifier calls them fish at "
         "0.99; agreement lifts the pair to 61%."),
        (GOOD, "Glyph regularity separates text from seabed.",
         f"One pen width, one glyph height, a shared baseline. The rejected box "
         f"is {worst:.0%} glyphs; every organism tested scores 0.00."),
    ]
    col = (W - MARGIN * 2 - GAP) // 2
    for i, (colour, head, body) in enumerate(notes):
        x = MARGIN + i * (col + GAP)
        c.rectangle((x, body_y + 3, x + 4, body_y + 74), fill=colour)
        c.text((x + 16, body_y), head, font=F_HEAD, fill=INK)
        line, lines = "", []
        for wd in body.split():
            t = f"{line} {wd}".strip()
            if c.textlength(t, font=F_BODY) > col - 20:
                lines.append(line)
                line = wd
            else:
                line = t
        lines.append(line)
        for j, ln in enumerate(lines[:3]):
            c.text((x + 16, body_y + 28 + j * 22), ln, font=F_BODY, fill=MUTED)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT)
    print("wrote:", OUT, canvas.size)


if __name__ == "__main__":
    main()
