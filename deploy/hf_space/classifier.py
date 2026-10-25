"""
Species classifier inference module for the Deep-Sea Creature Detection
& Classification System.

Provides :class:`SpeciesClassifier` — a lightweight wrapper around a
ResNet-18 model fine-tuned for aquatic-species classification.  Crops
extracted from YOLO detections are passed through this classifier to
obtain species-level labels and confidence scores.

If the model weights are not available on disk the factory function
:func:`build_classifier` returns a :class:`_DummyClassifier` that
always predicts *"unknown"* with 0.0 confidence, so the rest of the
pipeline can still run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

import config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Main classifier
# ──────────────────────────────────────────────────────────────────────


class SpeciesClassifier:
    """ResNet-18-based species classifier for aquatic-creature crops.

    Parameters
    ----------
    weights_path : str | Path
        Path to a ``.pth`` file containing the fine-tuned model state
        dict.
    class_names_path : str | Path
        Path to a JSON file holding a list of class-name strings.
    device : str
        PyTorch device string (``"cpu"`` or ``"cuda"``).

    Raises
    ------
    FileNotFoundError
        If *weights_path* or *class_names_path* do not exist.
    """

    def __init__(
        self,
        weights_path: str | Path,
        class_names_path: str | Path,
        device: str,
    ) -> None:
        self.device: torch.device = torch.device(device)

        # --- Load class names -------------------------------------------
        class_names_path = Path(class_names_path)
        if not class_names_path.exists():
            raise FileNotFoundError(
                f"Class names file not found: {class_names_path}"
            )
        with open(class_names_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        # Handle both formats: plain list or {"class_names": [...], "num_classes": N}
        if isinstance(raw, dict):
            self.class_names: list[str] = raw["class_names"]
        else:
            self.class_names: list[str] = raw
        num_classes: int = len(self.class_names)
        logger.info("Loaded %d class names from %s.", num_classes, class_names_path)

        # --- Build model ------------------------------------------------
        self.model: nn.Module = models.resnet18(weights=None)
        in_features: int = self.model.fc.in_features  # 512 for ResNet-18
        self.model.fc = nn.Linear(in_features, num_classes)

        # --- Load weights -----------------------------------------------
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Classifier weights not found: {weights_path}"
            )
        state_dict = torch.load(
            weights_path, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(state_dict)
        logger.info("Loaded classifier weights from %s.", weights_path)

        self.model.to(self.device)
        self.model.eval()

        # --- Validation transform (matches training) --------------------
        self.transform: transforms.Compose = transforms.Compose(
            [
                transforms.Resize(
                    (config.CLASSIFIER_IMG_SIZE, config.CLASSIFIER_IMG_SIZE)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=config.IMAGENET_MEAN,
                    std=config.IMAGENET_STD,
                ),
            ]
        )
        logger.info(
            "SpeciesClassifier ready (device=%s, classes=%d).",
            self.device, num_classes,
        )

    # ------------------------------------------------------------------ #
    # Single-crop inference
    # ------------------------------------------------------------------ #

    def classify_crop(self, crop_image: np.ndarray) -> dict[str, Any]:
        """Classify a single crop image.

        Parameters
        ----------
        crop_image : np.ndarray
            BGR NumPy array (OpenCV format) of the cropped region.

        Returns
        -------
        dict[str, Any]
            ``{"class_name": str, "confidence": float,
            "top3": [{"class_name": str, "confidence": float}, ...]}``
        """
        pil_img: Image.Image = Image.fromarray(
            crop_image[:, :, ::-1]  # BGR → RGB
        ).convert("RGB")
        tensor: torch.Tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits: torch.Tensor = self.model(tensor)
            probs: torch.Tensor = torch.softmax(logits, dim=1)[0]

        top3_vals, top3_idxs = probs.topk(min(3, len(self.class_names)))

        return {
            "class_name": self.class_names[top3_idxs[0].item()],
            "confidence": float(top3_vals[0].item()),
            "top3": [
                {
                    "class_name": self.class_names[idx.item()],
                    "confidence": float(val.item()),
                }
                for val, idx in zip(top3_vals, top3_idxs)
            ],
        }

    # ------------------------------------------------------------------ #
    # Batch-crop inference
    # ------------------------------------------------------------------ #

    def classify_crops(self, crops: list[np.ndarray]) -> list[dict[str, Any]]:
        """Classify multiple crops in a single batched forward pass.

        Parameters
        ----------
        crops : list[np.ndarray]
            List of BGR NumPy arrays (OpenCV format).

        Returns
        -------
        list[dict[str, Any]]
            One classification result per crop (same schema as
            :meth:`classify_crop`).
        """
        if not crops:
            return []

        tensors: list[torch.Tensor] = []
        for crop in crops:
            pil_img = Image.fromarray(crop[:, :, ::-1]).convert("RGB")
            tensors.append(self.transform(pil_img))

        batch: torch.Tensor = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            logits: torch.Tensor = self.model(batch)
            probs: torch.Tensor = torch.softmax(logits, dim=1)

        results: list[dict[str, Any]] = []
        for i in range(probs.size(0)):
            top3_vals, top3_idxs = probs[i].topk(min(3, len(self.class_names)))
            results.append(
                {
                    "class_name": self.class_names[top3_idxs[0].item()],
                    "confidence": float(top3_vals[0].item()),
                    "top3": [
                        {
                            "class_name": self.class_names[idx.item()],
                            "confidence": float(val.item()),
                        }
                        for val, idx in zip(top3_vals, top3_idxs)
                    ],
                }
            )

        logger.debug("Batch-classified %d crops.", len(results))
        return results


# ──────────────────────────────────────────────────────────────────────
# Dummy fallback classifier
# ──────────────────────────────────────────────────────────────────────


class _DummyClassifier:
    """Fallback classifier returned when weights are unavailable.

    Always predicts ``"unknown"`` with 0.0 confidence so that
    downstream code can operate without structural changes.
    """

    _DUMMY_RESULT: dict[str, Any] = {
        "class_name": "unknown",
        "confidence": 0.0,
        "top3": [{"class_name": "unknown", "confidence": 0.0}],
    }

    def classify_crop(self, crop_image: np.ndarray) -> dict[str, Any]:
        """Return a dummy classification result.

        Parameters
        ----------
        crop_image : np.ndarray
            Ignored.

        Returns
        -------
        dict[str, Any]
            Static unknown-class result.
        """
        return dict(self._DUMMY_RESULT)

    def classify_crops(self, crops: list[np.ndarray]) -> list[dict[str, Any]]:
        """Return dummy results for every crop.

        Parameters
        ----------
        crops : list[np.ndarray]
            Ignored except for length.

        Returns
        -------
        list[dict[str, Any]]
            List of static unknown-class results.
        """
        return [dict(self._DUMMY_RESULT) for _ in crops]


# ──────────────────────────────────────────────────────────────────────
# Factory helper
# ──────────────────────────────────────────────────────────────────────


def build_classifier(
    weights_path: str | Path = config.CLASSIFIER_WEIGHTS,
    class_names_path: str | Path = config.CLASSIFIER_CLASS_NAMES,
    device: str = config.DEVICE,
) -> SpeciesClassifier | _DummyClassifier:
    """Create a :class:`SpeciesClassifier` or fall back to a dummy.

    If the model weights or class-name files are missing, a warning is
    logged and a :class:`_DummyClassifier` is returned instead.

    Parameters
    ----------
    weights_path : str | Path
        Path to classifier ``.pth`` weights.
    class_names_path : str | Path
        Path to class-names JSON file.
    device : str
        PyTorch device string.

    Returns
    -------
    SpeciesClassifier | _DummyClassifier
        A ready-to-use classifier instance.
    """
    try:
        return SpeciesClassifier(weights_path, class_names_path, device)
    except FileNotFoundError as exc:
        logger.warning(
            "Classifier weights not found (%s). Using dummy classifier.", exc
        )
        return _DummyClassifier()
