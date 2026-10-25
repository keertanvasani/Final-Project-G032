"""Invariant checks for the burned-in overlay-text filter.

The filter exists because the detector boxes video telemetry — timestamps, depth
readouts, the copyright watermark — and the classifier, unable to abstain, names
those boxes as organisms. Frame 03 of the demo set produced "fish" at 61%
confidence on the digits "02/10/".

Two invariants matter, and they pull against each other:

  1. Boxes drawn on overlay text are suppressed.
  2. No box drawn on anything else is suppressed. Losing a real organism is a
     worse failure than keeping a text box, so the threshold sits nearer the
     text end of the measured gap.

Run after any change to text_overlay.py or to the enhancement step, since
enhancement is what makes overlay glyphs detectable in the first place.

Usage:
    python experiments/verify_text_filter.py
"""

from __future__ import annotations

import glob
import random
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ensemble  # noqa: E402
import text_overlay as T  # noqa: E402
from agent_controller import assess_image_quality, enhance_image  # noqa: E402
from pipeline import EnsemblePipeline  # noqa: E402

DEMO = ROOT / "demo_images"
JEDI = ROOT / "JediOrganismDetectionDataset"
SAMPLE = 200

# Hand-marked regions, read off the frames at 2x magnification.
TEXT_BOXES = [
    ("07_crowded_30_detections.jpg", (33, 10, 140, 36), "timestamp block"),
    ("07_crowded_30_detections.jpg", (85, 12, 140, 24), "one timestamp line"),
    ("07_crowded_30_detections.jpg", (380, 252, 478, 270), "(C) JAMSTEC"),
    ("01_hardware_called_puffin_99pct.jpg", (48, 27, 132, 56), "timestamp block"),
]
NON_TEXT_BOXES = [
    ("01_hardware_called_puffin_99pct.jpg", (200, 127, 255, 184), "the anemone"),
    ("01_hardware_called_puffin_99pct.jpg", (310, 220, 480, 360),
     "hardware, watermark inside the box"),
    ("07_crowded_30_detections.jpg", (350, 130, 430, 200), "bare seabed"),
]

# The demo copy of frame 06 was later cropped to 290 rows, which removed its
# telemetry, so the uncropped original is read from the dataset. It carries
# DARK text on a pale seabed, a polarity the first version of the filter never
# looked for: four of its six detections were boxes on "D 1041" and "A 1,7", each
# about one character wide, and all four were kept.
ORIG_06 = (JEDI / "coco_train_data" / "images" / "3K0380CCDSV1011_480.jpg")
# (box, is_text): hand-labelled from the frame at 3x magnification.
ORIG_06_BOXES = [
    ((375, 27, 389, 41), True),    # "D"
    ((376, 51, 390, 66), True),    # "A"
    ((413, 50, 420, 64), True),    # "1"
    ((434, 52, 447, 64), True),    # "7"
    ((221, 97, 256, 113), False),  # fish
    ((256, 126, 311, 169), False),  # fish
]
# Frame 02's timestamp digits were boxed too, and likewise kept.
FRAME_02_TEXT = [(273, 38, 289, 64), (258, 38, 272, 64)]

failures: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    if not ok:
        failures.append(msg)


def detect(pipe, path):
    """Boxes as production sees them: quality check, enhancement, detect, fuse."""
    im = cv2.imread(str(path))
    if im is None:
        return None, []
    q = assess_image_quality(im)
    enh = im
    if q["needs_enhancement"] or q["is_blurry"]:
        enh, _ = enhance_image(im, q)
    ra, rb = ensemble.run_dual_detection(enh, pipe.model_a, pipe.model_b)
    return enh, ensemble.fuse_detections(ra, rb, enh.shape[:2])


print("1. Marked overlay-text regions score above the suppression threshold")
for name, box, note in TEXT_BOXES:
    img = cv2.imread(str(DEMO / name))
    s = T.text_score(img, list(box))
    check(s >= T.SUPPRESS_COVERAGE, f"{note:34s} score={s:.2f} >= {T.SUPPRESS_COVERAGE}")

print("\n2. Organisms, hardware and seabed score below it")
for name, box, note in NON_TEXT_BOXES:
    img = cv2.imread(str(DEMO / name))
    s = T.text_score(img, list(box))
    check(s < T.SUPPRESS_COVERAGE, f"{note:34s} score={s:.2f} <  {T.SUPPRESS_COVERAGE}")

print("\n3. The reported failure no longer reaches the output")
pipe = EnsemblePipeline()
enh, fused = detect(pipe, DEMO / "03_seacucumber_called_fish_100pct.jpg")
kept, dropped = T.filter_text_detections(enh, fused)
check(len(dropped) == 1, f"exactly one box suppressed on frame 03 (got {len(dropped)})")
check(len(kept) == len(fused) - len(dropped), "kept + suppressed accounts for every box")
if dropped:
    d = dropped[0]
    on_text = 90 <= d["bbox"][0] <= 105 and 35 <= d["bbox"][1] <= 45
    check(on_text, f"the suppressed box is the timestamp at {[round(v) for v in d['bbox']]}")
check(any(190 <= k["bbox"][0] <= 205 for k in kept), "the real organism survives")

print("\n3b. Dark-on-light overlay and character-sized boxes (the original miss)")
if not ORIG_06.exists():
    check(False, f"original frame 06 not found at {ORIG_06}")
else:
    im06 = cv2.imread(str(ORIG_06))
    q6 = assess_image_quality(im06)
    enh06 = enhance_image(im06, q6)[0] if (q6["needs_enhancement"] or q6["is_blurry"]) else im06
    boxes = [{"bbox": list(b), "confidence": 0.5, "class_name": "fish"}
             for b, _ in ORIG_06_BOXES]
    _, dropped06 = T.filter_text_detections(enh06, boxes)
    dropped_boxes = {tuple(d["bbox"]) for d in dropped06}
    for b, is_text in ORIG_06_BOXES:
        was = tuple(b) in dropped_boxes
        check(was == is_text,
              f"06 original {str(b):22s} {'text' if is_text else 'fish'}: "
              f"{'suppressed' if was else 'kept'}")

    # Same boxes, through the full detector, to confirm the real path agrees.
    ra, rb = ensemble.run_dual_detection(enh06, pipe.model_a, pipe.model_b)
    fused06 = ensemble.fuse_detections(ra, rb, enh06.shape[:2])
    kept06, drop06 = T.filter_text_detections(enh06, fused06)
    text_left = [k for k in kept06 if k["bbox"][0] >= 370 and k["bbox"][1] < 70]
    check(not text_left, f"no detection left in the 06 telemetry corner "
                         f"(found {len(text_left)})")
    check(len(kept06) == 2, f"06 original keeps exactly its 2 fish (got {len(kept06)})")

img02 = cv2.imread(str(DEMO / "02_hardware_called_fish_92pct.jpg"))
q2 = assess_image_quality(img02)
enh02 = enhance_image(img02, q2)[0] if (q2["needs_enhancement"] or q2["is_blurry"]) else img02
b02 = [{"bbox": list(b), "confidence": 0.5, "class_name": "fish"} for b in FRAME_02_TEXT]
_, d02 = T.filter_text_detections(enh02, b02)
check(len(d02) == len(FRAME_02_TEXT),
      f"frame 02 timestamp digits suppressed ({len(d02)}/{len(FRAME_02_TEXT)})")

print("\n4. Aquarium frames carry no overlay, so nothing may be suppressed")
for name in ["11_aquarium_sharks.jpg", "12_aquarium_fish_school_44.jpg",
             "13_aquarium_sharks_closeup.jpg", "14_sample_fish.jpg",
             "15_sample_shark.jpg"]:
    enh, fused = detect(pipe, DEMO / name)
    _, dropped = T.filter_text_detections(enh, fused)
    check(not dropped, f"{name:38s} {len(fused):3d} boxes, {len(dropped)} suppressed")

print(f"\n5. Suppression stays rare across {SAMPLE} JEDI frames")
random.seed(1)
paths: list[str] = []
for split in ("coco_test_data", "coco_valid_data", "coco_train_data"):
    paths += sorted(glob.glob(str(JEDI / split / "images" / "*")))
random.shuffle(paths)

n_boxes = n_dropped = 0
for p in paths[:SAMPLE]:
    enh, fused = detect(pipe, p)
    if enh is None:
        continue
    _, dropped = T.filter_text_detections(enh, fused)
    n_boxes += len(fused)
    n_dropped += len(dropped)
rate = n_dropped / max(n_boxes, 1)
print(f"  {n_boxes} boxes, {n_dropped} suppressed ({rate:.1%})")
# Overlay text accounts for roughly one detector box in ten on JEDI frames: all
# 24 suppressions in this sample were inspected tile by tile and every one was a
# digit, a letter or the (C) mark; none was an organism. The first version of the
# filter caught 1 of those 24. This bound is a tripwire, not a target: a rate far
# above the measured one means the filter has started eating organisms, and the
# suppressed boxes must be re-inspected before the change is accepted.
check(rate <= 0.15, f"suppression rate {rate:.1%} <= 15% (measured 9.6%, all "
                    f"inspected as text)")

print("\n" + ("ALL CHECKS PASSED" if not failures
             else f"{len(failures)} CHECK(S) FAILED:"))
for f in failures:
    print("  -", f)
sys.exit(1 if failures else 0)
