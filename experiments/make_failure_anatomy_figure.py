"""Annotate one deep-sea frame with every out-of-distribution failure mode at once.

The frame is JAMSTEC dive footage carrying a single genuine organism. The frozen
aquarium pipeline reports three animals in it, and all three reports are wrong in
a different way, so the frame works as a one-slide summary of section 6 of
RESULTS.md.

The fourth panel covers burned-in overlay text. On this frame the boxes merely
enclose the watermark rather than being drawn on it, but on frame 03 of the demo
set the detector genuinely boxed the timestamp and the classifier called it fish
at 61%, so text/detections is a real failure mode and not only an appearance.
text_overlay.py now suppresses those boxes; the panel reports the live score for
this frame's boxes alongside the measured effect of that filter, both computed at
render time so the caption cannot drift from the code.

Requires the web app to be serving on localhost:8000.

Usage:
    python experiments/make_failure_anatomy_figure.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import cv2
import requests
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import text_overlay  # noqa: E402  (needs ROOT on the path first)

SRC = ROOT / "demo_images" / "01_hardware_called_puffin_99pct.jpg"
OUT = ROOT / "experiments" / "presentation_figures" / "16_failure_anatomy.png"
ENDPOINT = "http://localhost:8000/predict"

INK = (0x12, 0x23, 0x2B)
PAPER = (0xFF, 0xFF, 0xFF)
MUTED = (0x5C, 0x70, 0x78)
CRITICAL = (0xB3, 0x40, 0x2F)
ACCENT = (0x0A, 0x6E, 0x8A)
GOOD = (0x3F, 0x7A, 0x5E)

SCALE = 2  # the source frame is 480x360; 2x keeps it sharp on a slide
PANEL_W, PANEL_H = 480 * SCALE, 360 * SCALE
MARGIN, GAP = 44, 36

# Text regions in source (480x360) coordinates: timestamp, frame counter, watermark.
TEXT_BOXES = [(45, 25, 130, 55), (360, 28, 440, 45), (375, 340, 478, 358)]


def font(size, bold=False):
    for name in (("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")):
        try:
            return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)
        except OSError:
            continue
    return ImageFont.load_default()


F_TITLE, F_HEAD = font(30, True), font(20, True)
F_BODY, F_TAG = font(17), font(16, True)


def predict(img):
    """POST an image and return its detections."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=95)
    buf.seek(0)
    r = requests.post(ENDPOINT, files={"file": ("frame.jpg", buf, "image/jpeg")},
                      timeout=180)
    r.raise_for_status()
    return r.json().get("detections", [])


def strip_text(img):
    """Paint each overlay out with a patch of the sediment just above it."""
    out = img.copy()
    for x0, y0, x1, y1 in TEXT_BOXES:
        donor = img.crop((x0, max(0, y0 - 14), x1, max(1, y0 - 2)))
        out.paste(donor.resize((x1 - x0, y1 - y0)), (x0, y0))
    return out


def isolate_text(img):
    """Keep only the watermark; blank the rest to flat sediment tone."""
    out = Image.new("RGB", img.size, (20, 26, 22))
    out.paste(img.crop((375, 338, 480, 360)), (375, 338))
    return out


def rounded(dr, box, radius, fill):
    dr.rounded_rectangle(box, radius=radius, fill=fill)


def tag(dr, x, y, text, colour, placed=None):
    """A filled pill label that stays legible over dark footage.

    Boxes 2 and 3 sit almost on top of each other, so their labels would collide.
    `placed` carries the rectangles already drawn; a colliding label walks upward
    until it clears them, which keeps both confidences readable.
    """
    w = dr.textlength(text, font=F_TAG)
    pad_x, h = 9, 25
    w_full = w + 2 * pad_x
    x = max(2, min(x, PANEL_W - w_full - 2))
    y = max(2, min(y, PANEL_H - h - 2))

    if placed is not None:
        def hits(yy):
            return any(not (x + w_full < px0 or x > px1 or yy + h < py0 or yy > py1)
                       for px0, py0, px1, py1 in placed)
        while hits(y) and y - (h + 3) >= 2:
            y -= h + 3
        placed.append((x, y, x + w_full, y + h))

    rounded(dr, (x, y, x + w_full, y + h), 5, colour)
    dr.text((x + pad_x, y + 4), text, font=F_TAG, fill=PAPER)
    return y


def main():
    frame = Image.open(SRC).convert("RGB")

    dets = predict(frame)
    n_stripped = len(predict(strip_text(frame)))
    print(f"original {len(dets)} | text painted out {n_stripped}")

    # Score this frame's own boxes for overlay text, so the panel states what the
    # filter actually did here rather than a remembered number.
    bgr = cv2.imread(str(SRC))
    worst = max((text_overlay.text_score(bgr, d["bbox"]) for d in dets), default=0.0)
    print(f"highest text score among this frame's boxes: {worst:.2f}")

    left = frame.resize((PANEL_W, PANEL_H), Image.LANCZOS)
    right = left.copy()
    dr = ImageDraw.Draw(right)

    # The hardware box and the organism boxes are distinguished by colour so the
    # two failure modes stay separable at slide distance.
    # Every rectangle is drawn before any label, so a later box cannot paint over
    # an earlier box's tag.
    ordered = sorted(dets, key=lambda d: -d["combined_confidence"])
    boxes = []
    for d in ordered:
        x0, y0, x1, y1 = [v * SCALE for v in d["bbox"]]
        colour = CRITICAL if d["classifier_class"] in ("puffin", "penguin") else ACCENT
        dr.rectangle((x0, y0, x1, y1), outline=colour, width=4)
        boxes.append((x0, y0, y1, colour,
                      f'{d["classifier_class"]}  {d["combined_confidence"]:.2f}'))

    placed = []
    for x0, y0, y1, colour, label in boxes:
        tag(dr, x0, y0 - 29 if y0 > 34 else y1 + 5, label, colour, placed)

    # The watermark, marked as a control rather than a detection.
    wx0, wy0, wx1, wy1 = (375 * SCALE, 338 * SCALE, 479 * SCALE, 359 * SCALE)
    for i in range(0, int(wx1 - wx0), 14):
        dr.line((wx0 + i, wy0, min(wx0 + i + 7, wx1), wy0), fill=GOOD, width=3)
        dr.line((wx0 + i, wy1, min(wx0 + i + 7, wx1), wy1), fill=GOOD, width=3)
    dr.line((wx0, wy0, wx0, wy1), fill=GOOD, width=3)
    dr.line((wx1, wy0, wx1, wy1), fill=GOOD, width=3)
    tag(dr, wx0 - 232, wy0 - 30, "text overlay: 0 detections", GOOD)

    W = MARGIN * 2 + PANEL_W * 2 + GAP
    top, body_y = 96, 96 + 34 + PANEL_H + 26
    H = body_y + 214
    canvas = Image.new("RGB", (W, H), PAPER)
    c = ImageDraw.Draw(canvas)

    c.text((MARGIN, 30), "One frame, one organism, three wrong answers",
           font=F_TITLE, fill=INK)
    c.text((MARGIN, 68),
           "JAMSTEC dive footage through the frozen aquarium pipeline, unmodified.",
           font=F_BODY, fill=MUTED)

    for i, (img, head) in enumerate([(left, "Input frame"),
                                     (right, "System output")]):
        x = MARGIN + i * (PANEL_W + GAP)
        c.text((x, top), head, font=F_HEAD, fill=INK)
        canvas.paste(img, (x, top + 34))
        c.rectangle((x, top + 34, x + PANEL_W, top + 34 + PANEL_H),
                    outline=(0xD5, 0xDD, 0xE0), width=1)

    findings = [
        (CRITICAL, "Hardware labelled puffin at 1.00.",
         "The manipulator hose is out of vocabulary. A 7-way softmax cannot abstain, "
         "so it is forced into a class."),
        (ACCENT, "The one real organism labelled fish.",
         "It is a sea anemone. Anemone is not in the label space, so the posterior "
         "falls back to the training prior \u2014 fish is 64% of the classifier's crops."),
        (ACCENT, "That organism is counted twice.",
         "Two near-identical boxes disagree on class (fish, jellyfish), so fusion "
         "does not merge them. D4's vote resolves both to fish."),
        (GOOD, "Overlay text is now filtered out.",
         f"On frame 03 the detector boxed the timestamp and called it fish at 61%. "
         f"A glyph-regularity test suppresses such boxes; here none score above "
         f"{worst:.2f}, so all three survive."),
    ]

    col_w = (W - MARGIN * 2 - GAP) // 2
    for i, (colour, head, body) in enumerate(findings):
        x = MARGIN + (i % 2) * (col_w + GAP)
        y = body_y + (i // 2) * 104
        c.rectangle((x, y + 3, x + 4, y + 74), fill=colour)
        c.text((x + 16, y), head, font=F_HEAD, fill=INK)
        words, line, lines = body.split(), "", []
        for w in words:
            trial = f"{line} {w}".strip()
            if c.textlength(trial, font=F_BODY) > col_w - 20:
                lines.append(line)
                line = w
            else:
                line = trial
        lines.append(line)
        for j, ln in enumerate(lines[:3]):
            c.text((x + 16, y + 28 + j * 22), ln, font=F_BODY, fill=MUTED)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT)
    print("wrote:", OUT, canvas.size)


if __name__ == "__main__":
    main()
