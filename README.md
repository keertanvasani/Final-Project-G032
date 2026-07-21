# Deep-Sea Creature Detection & Classification System

A demo-ready AI pipeline that detects and classifies underwater/deep-sea creatures using a **multi-model ensemble** approach. The system combines dual YOLOv8 object detectors, Weighted Boxes Fusion (WBF), and a ResNet18 species classifier to deliver high-confidence, trustworthy results.

Ships with a **web interface** — drop in a photo and get the annotated image back
alongside the reasoning the system used to produce it. See
[Web Interface](#web-interface), or [WEBAPP.md](WEBAPP.md) for the full details.

**Repository:** https://github.com/keertanvasani/Final-Project-Phase-1

## Architecture

```
                         ┌─────────────────┐
                         │  Input Image     │
                         └────────┬────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
              ┌─────▼─────┐             ┌──────▼──────┐
              │  YOLOv8n   │             │  YOLOv8s    │
              │  (nano)    │             │  (small)    │
              └─────┬─────┘             └──────┬──────┘
                    │                           │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │  Weighted Boxes Fusion     │
                    │  (ensemble-boxes library)  │
                    └─────────────┬─────────────┘
                                  │
                         ┌────────▼────────┐
                         │  Fused Bounding  │
                         │  Boxes           │
                         └────────┬────────┘
                                  │
                         ┌────────▼────────┐
                         │  Crop Regions    │
                         │  from Image      │
                         └────────┬────────┘
                                  │
                         ┌────────▼────────┐
                         │  ResNet18        │
                         │  Species         │
                         │  Classifier      │
                         └────────┬────────┘
                                  │
                         ┌────────▼────────┐
                         │  Combined        │
                         │  Confidence      │
                         │  Scoring         │
                         └────────┬────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │  Annotated Image +         │
                    │  Detection JSON Results    │
                    └───────────────────────────┘
```

## Agentic Decision-Making

The pipeline is not a fixed sequence of steps. `agent_controller.py` inspects image quality and intermediate model confidence at runtime and decides what to do next:

1. **Quality check** — every image is scored for blur (Laplacian variance), brightness, and contrast.
2. **Conditional enhancement** — CLAHE contrast/brightness correction and/or a sharpening filter are applied only when the quality check flags a problem, not on every image.
3. **Review gate** — two independent checks decide whether the image is worth reporting on at all. An image whose Laplacian blur variance falls below `UNUSABLE_BLUR_THRESHOLD` (5.0) is unreadable, so no detection from it can be trusted however confident the detector claims to be. Otherwise, at least one fused box must reach `REVIEW_CONF_THRESHOLD` (0.40). If either check fails the request is flagged for review: classification is skipped, and the returned image gets a red warning banner instead of bounding boxes (e.g. *"Image too blurry to analyze reliably. Please try another photo."*). The same message is appended to `decision_trace`, so the system never silently returns a low-quality guess. The gate decides *whether to analyse the image*, not which boxes survive — weak detections are kept so the classifier still gets its say, and their uncertainty reaches the user through the combined confidence and the annotation colours.
4. **Zoom-and-recheck** — ambiguous detections (confidence 0.4-0.75) are re-cropped and re-run through detection + fusion at effectively higher resolution; the result is only kept if it improves confidence and geometrically matches the original box.
5. **Disagreement resolution** — when the detector and classifier disagree on species, a second classification pass runs on a looser crop, and the final class is chosen by majority vote across detector / classifier-tight / classifier-loose.

Every decision taken is logged in the `decision_trace` array returned by `/predict`. Every annotated output — a successful detection or a warning banner — is also automatically saved to `outputs/` as a PNG, so results are reviewable after the fact without any extra client-side code.

## Quick Start

### Prerequisites

- **Python 3.9+** (3.10 or 3.11 recommended)
- **pip** package manager
- **Roboflow account** (free tier) for dataset download — [Sign up here](https://roboflow.com)
- (Optional) NVIDIA GPU with CUDA for faster training/inference

### 1. Clone & Setup

```bash
# Clone the repository
git clone https://github.com/keertanvasani/Final-Project-Phase-1.git
cd Final-Project-Phase-1

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

> **Note:** `data/`, `classifier_data/` and `models/` are gitignored, so a fresh
> clone contains code but no dataset or weights. Follow steps 2–4 to produce them.

### 2. Download Dataset

Get your free API key from [Roboflow Settings](https://app.roboflow.com/settings/api) and run:

```bash
# Option A: Using environment variable
export ROBOFLOW_API_KEY="your_api_key_here"
python download_dataset.py

# Option B: Using CLI argument
python download_dataset.py --api-key your_api_key_here

# Option C: See manual download instructions
python download_dataset.py --manual
```

This downloads the **Aquarium Combined** dataset (638 images, 7 species) in YOLO format.

### 3. Prepare Classifier Data

Extract cropped species images from the detection dataset:

```bash
python prepare_classifier_data.py
```

This creates `classifier_data/` with crops organized by species for training the classifier.

### 4. Train Models

```bash
# Train both YOLOv8 models (nano + small) — ~30-60 min on CPU each
python train_yolo.py

# Train only one model if needed
python train_yolo.py --model n   # Just YOLOv8n
python train_yolo.py --model s   # Just YOLOv8s

# Train the species classifier — ~15-20 min on CPU
python train_classifier.py

# Override training parameters
python train_yolo.py --epochs 30 --batch 8
python train_classifier.py --epochs 15 --lr 0.0005
```

### 5. Start the Server

```bash
# Web interface + API (recommended)
uvicorn webapp:app --reload --host 0.0.0.0 --port 8000

# API only, no front-end
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Either starts at `http://localhost:8000`. With `webapp:app` that address serves the
browser interface; `/health` and `/predict` behave identically in both cases.

> **Note:** If you haven't trained models yet, the system will fall back to pretrained COCO models (generic object detection, not species-specific). Train the models for species-specific detection.

### 6. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Run prediction on an image
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/underwater_image.jpg" \
  -o response.json

# Pretty-print the response (without the base64 image)
python -c "
import json
with open('response.json') as f:
    data = json.load(f)
# Show detections and summary (skip the large base64 image string)
print(json.dumps({
    'detections': data['detections'],
    'summary': data['summary']
}, indent=2))
"
```

## Web Interface

Start with `uvicorn webapp:app --port 8000` and open `http://localhost:8000`.

`webapp.py` imports the FastAPI app from `main.py` and mounts the static site in
`frontend/` at the root. Because `/health` and `/predict` are registered on the
router before the mount, they still match first and everything else falls through
to the static files. Site and API therefore share one origin — no CORS round-trip,
and one process to run. Nothing in the model code is modified by any of this.

The page covers:

- **Architecture** as a flowchart with all four agentic decision points drawn
  inline, each showing its condition and both branches
- **A comparison** against a typical single-model pipeline, and the mechanisms
  that differ
- **Live inference** — drag in a photo, or use one of six one-click samples,
  including a deliberately blurry one that trips the confidence gate
- **The `decision_trace`** rendered as a timeline, so the route your image took
  through the pipeline is visible
- **Measured performance** — validation metrics and training curves

Accepted uploads are JPEG, PNG and WebP up to 20 MB, matching
`config.ALLOWED_IMAGE_TYPES` and `config.MAX_IMAGE_SIZE_MB`. HEIC — the iPhone
default — is rejected with a hint to export as JPEG first.

See [WEBAPP.md](WEBAPP.md) for the file layout and the full table of handled UI states.

## API Reference

### `GET /health`

Returns system health and model loading status.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "device": "cpu",
  "model_details": {
    "yolo_n": true,
    "yolo_s": true,
    "classifier": true
  }
}
```

### `POST /predict`

Upload an image and get annotated detection results.

**Request:** Multipart form data with `file` field (JPEG, PNG, or WebP image).

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
    "total_objects": 3,
    "average_confidence": 0.84,
    "inference_time_ms": 342.5,
    "inference_breakdown": {
      "detection_ms": 180.2,
      "fusion_ms": 4.8,
      "classification_ms": 150.1,
      "annotation_ms": 7.4
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
    "Detection results: Found 1 boxes (avg conf: 0.58)",
    "Action: Zoomed-in crop for original class 'fish' (0.58) improved confidence to 0.87 (new class 'fish')",
    "Final label for Object 1: fish (combined conf: 0.89)"
  ]
}
```

**When the image cannot be reliably analyzed** (e.g. still too blurry after CLAHE/sharpening), `detections` is empty, `summary.total_objects` is `0`, and `annotated_image` contains a red warning banner instead of bounding boxes:

```json
{
  "decision_trace": [
    "Quality check: blur score = 0.7, brightness = 26.9, contrast = 6.3, resolution = (768, 1024)",
    "Action: Applied CLAHE enhancement due to low brightness: 26.9, low contrast: 6.3",
    "Action: Applied sharpening filter (blur variance 0.7 < 100.0)",
    "Detection results: Found 8 boxes (avg conf: 0.31)",
    "Action: Combined detection confidence too low (< 0.4) or no objects found. Flagged for human review",
    "Final state: Image too blurry to analyze reliably. Please try another photo."
  ]
}
```

Every call to `/predict` also saves its annotated output (success or warning banner) to `outputs/` as a uniquely-named PNG, regardless of the outcome.

**Error Responses:**

| Code | Reason |
|------|--------|
| 400  | Invalid file type (not an image) |
| 400  | Corrupt or unreadable image |
| 400  | Image exceeds 20MB size limit |
| 503  | Models not loaded (server still starting) |
| 500  | Internal inference error |

## Project Structure

```
├── main.py                    # FastAPI application
├── webapp.py                  # Mounts the web UI onto the app from main.py
├── agent_controller.py        # Agentic decision-making orchestrator
├── pipeline.py                # Full ensemble pipeline orchestrator
├── ensemble.py                # WBF fusion logic module
├── classifier.py              # Species classifier inference module
├── config.py                  # Central configuration
├── train_yolo.py              # Dual YOLO model training script
├── train_classifier.py        # Species classifier training script
├── download_dataset.py        # Dataset download & conversion
├── prepare_classifier_data.py # Extract crops for classifier training
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── WEBAPP.md                  # Web interface guide
├── DEMO_SCRIPT.md             # Demo talking points
├── frontend/                  # Browser interface (no build step)
│   ├── index.html
│   ├── css/styles.css
│   ├── js/app.js
│   └── assets/
│       ├── samples/           # One-click demo images
│       └── metrics/           # PR curve, confusion matrix, training curves
├── models/                    # Trained model weights
│   ├── yolov8n_aquarium/
│   ├── yolov8s_aquarium/
│   └── classifier/
├── data/                      # Downloaded dataset
│   ├── images/{train,val}/
│   ├── labels/{train,val}/
│   └── data.yaml
├── classifier_data/           # Cropped species images
│   ├── train/{class_name}/
│   └── val/{class_name}/
└── outputs/                   # Auto-saved annotated predictions (PNG)
```

## Configuration

All parameters are centralized in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `YOLO_CONF_THRESHOLD` | 0.25 | Minimum detection confidence |
| `WBF_IOU_THRESHOLD` | 0.55 | IoU threshold for box fusion |
| `WBF_WEIGHTS` | [2, 1] | Model weights (nano, small) |
| `DETECTOR_WEIGHT` | 0.6 | Combined confidence: detector weight |
| `CLASSIFIER_WEIGHT` | 0.4 | Combined confidence: classifier weight |
| `AGREEMENT_BOOST` | 0.10 | Confidence boost when models agree |
| `YOLO_TRAIN_EPOCHS` | 50 | YOLO training epochs |
| `CLASSIFIER_TRAIN_EPOCHS` | 25 | Classifier training epochs |

## Troubleshooting

**"Models not loaded" error (503)**
- The server is still loading models, or model files are missing. Check server logs.
- Run the training scripts first, or let the system fall back to pretrained COCO models.

**Slow inference on CPU**
- Expected: ~300-500ms per image on CPU for the full pipeline.
- Reduce `YOLO_IMG_SIZE` in `config.py` to 416 or 320 for faster inference.

**CUDA out of memory**
- Reduce `YOLO_TRAIN_BATCH` in `config.py` or use `--batch 8` flag.
- The system auto-detects GPU but can be forced to CPU with `--device cpu`.

**Import errors**
- Ensure you activated the virtual environment.
- Run `pip install -r requirements.txt` again.

**`RuntimeError: Front-end directory not found` on startup**
- `webapp.py` needs `frontend/` beside it. Run uvicorn from the project root, or
  start the API on its own with `uvicorn main:app` if you don't need the web UI.

**Web page loads but shows "Backend offline"**
- The static files are being served but `/health` isn't answering. This happens if
  the page is opened directly from disk (`file://`) instead of through the server.
  Use `http://localhost:8000`.

**"Image too blurry to analyze reliably" warning banner**
- The agent could not recover enough detail after CLAHE + sharpening (blur variance stayed below the 100.0 threshold in `config.py`). Retake the photo with better focus/lighting, or adjust the blur/brightness/contrast thresholds in `assess_image_quality()` in `agent_controller.py` if this triggers too often on acceptable images.

## Dataset

Uses the [Aquarium Combined](https://universe.roboflow.com/brad-dwyer/aquarium-combined) dataset from Roboflow Universe:
- **638 images** from Henry Doorly Zoo and National Aquarium
- **7 species**: fish, jellyfish, penguin, puffin, shark, starfish, stingray
- Pre-annotated with YOLO-format bounding boxes

## License

This project is for academic/demo purposes. The Aquarium Combined dataset is provided under its original license on Roboflow Universe.
