# Deep-Sea Creature Detection & Classification System

A demo-ready AI pipeline that detects and classifies underwater/deep-sea creatures using a **multi-model ensemble** approach. The system combines dual YOLOv8 object detectors, Weighted Boxes Fusion (WBF), and a ResNet18 species classifier to deliver high-confidence, trustworthy results.

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
3. **Low-confidence gate** — if detection confidence stays below 0.4 after enhancement (or nothing is detected at all), the request is flagged for review: classification is skipped, and the returned image gets a red warning banner instead of bounding boxes (e.g. *"Image too blurry to analyze reliably. Please try another photo."*). The same message is appended to `decision_trace`, so the system never silently returns a low-quality guess.
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
# Navigate to the project directory
cd "Final Year UI part 1"

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

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
# Start with auto-reload (development)
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or run directly
python main.py
```

The server starts at `http://localhost:8000`.

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
├── DEMO_SCRIPT.md             # Demo talking points
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

**"Image too blurry to analyze reliably" warning banner**
- The agent could not recover enough detail after CLAHE + sharpening (blur variance stayed below the 100.0 threshold in `config.py`). Retake the photo with better focus/lighting, or adjust the blur/brightness/contrast thresholds in `assess_image_quality()` in `agent_controller.py` if this triggers too often on acceptable images.

## Dataset

Uses the [Aquarium Combined](https://universe.roboflow.com/brad-dwyer/aquarium-combined) dataset from Roboflow Universe:
- **638 images** from Henry Doorly Zoo and National Aquarium
- **7 species**: fish, jellyfish, penguin, puffin, shark, starfish, stingray
- Pre-annotated with YOLO-format bounding boxes

## License

This project is for academic/demo purposes. The Aquarium Combined dataset is provided under its original license on Roboflow Universe.
