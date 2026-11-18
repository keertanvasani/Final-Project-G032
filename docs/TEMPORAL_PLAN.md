# Temporal evidence — plan for when video is available (parked 2026-11-18)

Not started. The current dataset cannot support it: 8,151 images from 7,061 clips,
about one frame per clip, with the nearest frames of the same clip a median of
360 frames apart. This needs the source dive videos, or our own annotated video.

## Approach: tracking by detection

1. **Detector:** keep YOLO, run per frame (~50 ms/frame on the laptop GPU).
2. **Tracker:** ByteTrack or BoT-SORT through Ultralytics `model.track(...)`.
   * ByteTrack keeps low-confidence boxes that line up with an existing track,
     recovering weak or blurry detections.
   * BoT-SORT adds camera-motion compensation (moving ROV) and appearance
     re-identification; the ResNet-18 features we already compute can serve.
3. **Evidence per track, not per frame:**
   * label = classifier vote over the track;
   * confidence rises for long, stable tracks and falls for one-frame boxes or
     flickering labels;
   * Known/Unknown uses the novelty score averaged over the track (catches only
     14.5% of unseen animals from a single frame today);
   * track length and label stability become bandit state features (step 3 found
     no usable per-frame signal).

## Fit with the agent loop

| Loop part | Addition |
|---|---|
| Core step | tracker (associate boxes with tracks) |
| Tool | temporal check: pool opinions along the track |
| Verifier | track-level confidence and Known/Unknown |
| State | small per-video track memory carried between frames |
| Guardrails | unchanged: one pass per frame, per-frame time budget, frozen policy |

## Training and evaluation

* Detector training stays per frame; sample frames sparsely (1–2 per second) and
  keep dive-level splits so near-duplicate frames never cross splits.
* Labelling: annotate keyframes, propagate boxes with the tracker or SAM 2, spot-check.
* Metrics: IDF1 / HOTA for tracking, per-track identification accuracy, and
  precision / recall / AURC computed per track.

## Alternatives

| Option | Verdict |
|---|---|
| YOLO + tracker | Start here. Cheap, measurable, fits the framework. |
| RT-DETR | Detector comparison (already planned); also supports tracking. |
| Video-native detectors (multi-frame feature aggregation, video transformers) | Only if YOLO + tracker leaves a measured gap; heavier, need dense video labels. |
| SAM 2 | For labelling, not deployment. |

First step when a video arrives: run YOLO + tracking on it and measure track stability.
