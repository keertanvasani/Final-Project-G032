"""
Deep-Sea Creature Detection & Classification System — FastAPI Backend

Provides REST API endpoints for running the multi-model ensemble pipeline
on uploaded underwater images. Supports dual YOLOv8 detection with WBF
fusion and ResNet18 species classification.

Endpoints:
    GET  /health   — System health check and model status
    POST /predict  — Upload an image and get annotated detections
"""

import base64
import io
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image

import config

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("deep_sea_api")


# ──────────────────────────────────────────────
# Response Schemas
# ──────────────────────────────────────────────
class DetectionResult(BaseModel):
    """Single detection result with both detector and classifier outputs."""
    id: int = Field(..., description="Detection index (1-based)")
    bbox: list[float] = Field(..., description="Bounding box [x1, y1, x2, y2] in pixels")
    detector_class: str = Field(..., description="Class predicted by YOLO ensemble")
    detector_confidence: float = Field(..., description="YOLO ensemble confidence score")
    classifier_class: str = Field(..., description="Class predicted by species classifier")
    classifier_confidence: float = Field(..., description="Species classifier confidence score")
    combined_confidence: float = Field(..., description="Weighted combined confidence score")
    agreement: bool = Field(..., description="Whether detector and classifier agree on class")


class InferenceBreakdown(BaseModel):
    """Measured wall-clock cost of each pipeline stage, in milliseconds.

    Components are mutually exclusive and reconcile to ``inference_time_ms``.
    ``detection_ms`` is the sum of the two detector arms, which are also
    reported individually. ``zoom_ms`` covers the whole zoom-and-recheck
    action including the detector and fusion passes it triggers internally,
    because that total is the cost of choosing that action.
    """
    preprocessing_ms: float = Field(..., description="Quality assessment + conditional enhancement")
    detector_nano_ms: float = Field(..., description="YOLOv8n forward pass (initial detection)")
    detector_small_ms: float = Field(..., description="YOLOv8s forward pass (initial detection)")
    detection_ms: float = Field(..., description="Both detector arms combined")
    wbf_ms: float = Field(..., description="Weighted Boxes Fusion (initial)")
    text_filter_ms: float = Field(0.0, description="Rejecting boxes drawn on burned-in overlay text")
    zoom_ms: float = Field(..., description="Zoom-and-recheck action, including nested detection")
    classification_ms: float = Field(..., description="First classification pass over all crops")
    reclassification_ms: float = Field(..., description="Disagreement tie-break passes")
    annotation_ms: float = Field(..., description="Drawing boxes or the warning banner")
    agent_decision_ms: float = Field(..., description="Controller overhead, recovered as residual")


class ModelsUsed(BaseModel):
    """Models used in the pipeline."""
    detector_1: str = "YOLOv8n"
    detector_2: str = "YOLOv8s"
    classifier: str = "ResNet18"
    fusion: str = "Weighted Boxes Fusion"


class PredictionSummary(BaseModel):
    """Summary statistics for the prediction."""
    total_objects: int
    average_confidence: float
    inference_time_ms: float
    inference_breakdown: InferenceBreakdown
    models_used: ModelsUsed


class PredictionResponse(BaseModel):
    """Full prediction response with annotated image, detections, and summary."""
    annotated_image: str = Field(..., description="Base64-encoded annotated PNG image")
    detections: list[DetectionResult]
    summary: PredictionSummary
    decision_trace: list[str] = Field(..., description="Trace of agentic decision points")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: bool
    device: str
    model_details: dict[str, bool]


# ──────────────────────────────────────────────
# Global pipeline reference
# ──────────────────────────────────────────────
pipeline = None


# ──────────────────────────────────────────────
# Application lifespan (model loading)
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models on startup, clean up on shutdown."""
    global pipeline

    logger.info("=" * 60)
    logger.info("  Deep-Sea Creature Detection System — Starting Up")
    logger.info("=" * 60)
    logger.info(f"  Device: {config.DEVICE}")
    logger.info(f"  Project root: {config.PROJECT_ROOT}")

    load_start = time.perf_counter()

    try:
        from pipeline import EnsemblePipeline
        pipeline = EnsemblePipeline()
        load_time = (time.perf_counter() - load_start) * 1000
        logger.info(f"  All models loaded in {load_time:.0f}ms")
        logger.info(f"  Model status: {pipeline.get_models_status()}")
    except Exception as e:
        logger.error(f"  Failed to load models: {e}", exc_info=True)
        logger.warning("  Server will start but /predict will return 503")

    logger.info("=" * 60)
    logger.info("  Server ready — accepting requests")
    logger.info("=" * 60)

    yield

    # Cleanup
    logger.info("Shutting down — releasing model resources")
    pipeline = None


# ──────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────
app = FastAPI(
    title="Deep-Sea Creature Detection & Classification API",
    description=(
        "Multi-model ensemble pipeline for underwater species detection. "
        "Uses dual YOLOv8 detectors with Weighted Boxes Fusion (WBF) "
        "and a ResNet18 species classifier for high-confidence predictions."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for demo flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────
def validate_image(file: UploadFile) -> None:
    """Validate that the uploaded file is an acceptable image."""
    if file.content_type not in config.ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid file type: '{file.content_type}'. "
                f"Accepted types: {', '.join(config.ALLOWED_IMAGE_TYPES)}"
            ),
        )


def decode_image(contents: bytes, filename: str) -> np.ndarray:
    """Decode raw bytes into an OpenCV BGR image."""
    try:
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("cv2.imdecode returned None")
        return image
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode image '{filename}': {str(e)}",
        )


def encode_image_base64(image: np.ndarray) -> str:
    """Encode an OpenCV BGR image to base64 PNG string."""
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to encode annotated image to PNG",
        )
    return base64.b64encode(buffer).decode("utf-8")


def save_annotated_output(image: np.ndarray, original_filename: Optional[str]) -> Path:
    """Persist an annotated image to config.OUTPUTS_DIR as a timestamped PNG.

    Every call gets a unique filename (original stem + timestamp + short
    random suffix) so concurrent/rapid requests never collide.
    """
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(original_filename).stem if original_filename else "prediction"
    safe_stem = re.sub(r"[^A-Za-z0-9_-]", "_", stem)[:60] or "prediction"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = config.OUTPUTS_DIR / f"{safe_stem}_{timestamp}_{uuid.uuid4().hex[:6]}.png"
    cv2.imwrite(str(output_path), image)
    return output_path


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Check system health and model loading status.

    Returns the current device (CPU/CUDA), whether all models are loaded,
    and individual model status (yolo_n, yolo_s, classifier).
    """
    if pipeline is None:
        return HealthResponse(
            status="degraded",
            models_loaded=False,
            device=config.DEVICE,
            model_details={"yolo_n": False, "yolo_s": False, "classifier": False},
        )

    status = pipeline.get_models_status()
    all_loaded = all(status.values())

    return HealthResponse(
        status="healthy" if all_loaded else "partial",
        models_loaded=all_loaded,
        device=config.DEVICE,
        model_details=status,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Detection"])
async def predict(
    file: UploadFile = File(..., description="Image file to analyze (JPEG, PNG, or WebP)"),
):
    """
    Run the full ensemble detection + classification pipeline on an uploaded image.

    **Pipeline stages:**
    1. Dual YOLOv8 detection (nano + small variants)
    2. Weighted Boxes Fusion to merge predictions
    3. Per-crop ResNet18 species classification
    4. Combined confidence scoring with agreement detection

    **Returns:**
    - Base64-encoded annotated image with bounding boxes and labels
    - Detailed detection results with both detector and classifier outputs
    - Performance summary with per-stage timing breakdown
    """
    # Check that models are loaded
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Models are not loaded. The server may still be starting up, "
                "or model loading failed. Check server logs for details."
            ),
        )

    # Validate file type
    validate_image(file)

    # Read and check file size
    contents = await file.read()
    file_size_mb = len(contents) / (1024 * 1024)
    if file_size_mb > config.MAX_IMAGE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Image too large: {file_size_mb:.1f}MB. "
                f"Maximum allowed: {config.MAX_IMAGE_SIZE_MB}MB."
            ),
        )

    logger.info(
        f"Processing image: {file.filename} "
        f"({file_size_mb:.2f}MB, {file.content_type})"
    )

    # Decode image
    image = decode_image(contents, file.filename)
    h, w = image.shape[:2]
    logger.info(f"Image dimensions: {w}x{h}")

    # Run full agentic pipeline
    try:
        from agent_controller import run_agentic_pipeline
        annotated_image, detections, summary, decision_trace = run_agentic_pipeline(image, pipeline)
    except Exception as e:
        logger.error(f"Agentic pipeline inference failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {str(e)}",
        )

    # Persist the annotated output (detections or a review-warning banner)
    try:
        output_path = save_annotated_output(annotated_image, file.filename)
        logger.info(f"Annotated output saved to {output_path}")
    except Exception as e:
        logger.warning(f"Failed to save annotated output: {e}")

    # Encode annotated image to base64
    annotated_b64 = encode_image_base64(annotated_image)

    # Build response
    detection_results = [
        DetectionResult(
            id=det["id"],
            bbox=det["bbox"],
            detector_class=det["detector_class"],
            detector_confidence=round(det["detector_confidence"], 4),
            classifier_class=det["classifier_class"],
            classifier_confidence=round(det["classifier_confidence"], 4),
            combined_confidence=round(det["combined_confidence"], 4),
            agreement=det["agreement"],
        )
        for det in detections
    ]

    prediction_summary = PredictionSummary(
        total_objects=summary["total_objects"],
        average_confidence=round(summary["average_confidence"], 4),
        inference_time_ms=round(summary["inference_time_ms"], 1),
        inference_breakdown=InferenceBreakdown(
            **{k: round(v, 2) for k, v in summary["inference_breakdown"].items()}
        ),
        models_used=ModelsUsed(**summary["models_used"]),
    )

    logger.info(
        f"Prediction complete: {summary['total_objects']} objects detected, "
        f"avg confidence {summary['average_confidence']:.2%}, "
        f"total time {summary['inference_time_ms']:.0f}ms"
    )

    return PredictionResponse(
        annotated_image=annotated_b64,
        detections=detection_results,
        summary=prediction_summary,
        decision_trace=decision_trace,
    )


# ──────────────────────────────────────────────
# Error handlers
# ──────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Catch-all exception handler for unexpected errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Check server logs for details."},
    )


# ──────────────────────────────────────────────
# Run directly
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting server on {config.API_HOST}:{config.API_PORT}")
    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
        log_level="info",
    )
