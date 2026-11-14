# API reference

The FastAPI application in [`main.py`](../main.py) exposes two endpoints. `webapp.py`
serves the same two endpoints plus the web interface from one process.

```bash
uvicorn webapp:app --port 8000     # web interface + API
uvicorn main:app   --port 8000     # API only
```

## `GET /health`

Reports whether the models have finished loading.

```json
{
  "status": "healthy",
  "models_loaded": true,
  "device": "cpu",
  "model_details": { "yolo_n": true, "yolo_s": true, "classifier": true }
}
```

Models load during startup, so `models_loaded` is `false` for the first few seconds.

## `POST /predict`

Upload one image and receive the annotated result.

**Request:** multipart form data with a `file` field. JPEG, PNG or WebP, up to 20 MB.

```bash
curl -X POST http://localhost:8000/predict -F "file=@image.jpg" -o response.json
```

**Response:**

```json
{
  "annotated_image": "<base64-encoded PNG>",
  "detections": [
    {
      "id": 1,
      "bbox": [120.5, 80.3, 340.2, 290.7],
      "detector_class": "fish",
      "detector_confidence": 0.87,
      "classifier_class": "fish",
      "classifier_confidence": 0.92,
      "combined_confidence": 0.89,
      "agreement": true
    }
  ],
  "summary": {
    "total_objects": 1,
    "average_confidence": 0.89,
    "inference_time_ms": 342.5,
    "inference_breakdown": {
      "preprocessing_ms": 7.1,
      "detector_nano_ms": 52.3,
      "detector_small_ms": 36.7,
      "detection_ms": 89.0,
      "wbf_ms": 0.4,
      "text_filter_ms": 5.2,
      "zoom_ms": 0.0,
      "classification_ms": 21.8,
      "reclassification_ms": 0.0,
      "annotation_ms": 1.0,
      "agent_decision_ms": 0.2
    },
    "models_used": {
      "detector_1": "YOLOv8n",
      "detector_2": "YOLOv8s",
      "classifier": "ResNet18",
      "fusion": "Weighted Boxes Fusion"
    }
  },
  "decision_trace": [
    "Quality check: blur score = 120.4, brightness = 42.1, contrast = 28.5, resolution = (640, 480)",
    "Action: Applied CLAHE enhancement due to low brightness: 42.1, low contrast: 28.5",
    "Detection results: Found 1 boxes (avg conf: 0.58, best: 0.58)",
    "Action: 1 of 1 boxes cleared 0.40; proceeding with all of them",
    "Final label for Object 1: fish (combined conf: 0.89)"
  ]
}
```

The values above are illustrative. The `inference_breakdown` components are mutually
exclusive and reconcile to `inference_time_ms`; `agent_decision_ms` is the residual, so
any unaccounted-for time shows up there.

### Overlay text

Boxes drawn on burned-in telemetry (timestamps, depth readouts, watermarks) are removed
before anything else uses them, and the removal is recorded:

```
Action: Rejected 1 detection(s) drawn on burned-in overlay text — 1 on glyphs
(up to 96% of the box). These are video telemetry, not organisms
```

### When the image cannot be analysed

If the image is unreadable, or no box reaches the review threshold, `detections` is
empty, `summary.total_objects` is `0`, and `annotated_image` carries a red banner in
place of boxes. The reason is the last line of `decision_trace`.

Every call also saves its annotated output to `outputs/` as a uniquely named PNG.

### Errors

| Code | Reason |
|------|--------|
| 400 | Not an accepted image type, or a corrupt or unreadable image |
| 400 | Image larger than 20 MB |
| 503 | Models not loaded yet (the server is still starting) |
| 500 | Internal inference error |

## Configuration

Every parameter lives in [`config.py`](../config.py).

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `YOLO_CONF_THRESHOLD` | 0.25 | Minimum detection confidence |
| `REVIEW_CONF_THRESHOLD` | 0.40 | A frame is analysed only if a box reaches this |
| `UNUSABLE_BLUR_THRESHOLD` | 5.0 | Below this Laplacian variance the frame is unreadable |
| `WBF_IOU_THRESHOLD` | 0.55 | IoU threshold for box fusion |
| `WBF_WEIGHTS` | [2, 1] | Fusion weights (nano, small) |
| `DETECTOR_WEIGHT` | 0.6 | Detector share of combined confidence |
| `CLASSIFIER_WEIGHT` | 0.4 | Classifier share of combined confidence |
| `AGREEMENT_BOOST` | 0.10 | Confidence boost when detector and classifier agree |
| `YOLO_TRAIN_EPOCHS` | 50 | Detector training epochs |
| `CLASSIFIER_TRAIN_EPOCHS` | 25 | Classifier training epochs |

Combined confidence is `min[(0.6·c_det + 0.4·c_cls) × (1 + boost), 1.0]`.
