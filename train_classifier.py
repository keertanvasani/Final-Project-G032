#!/usr/bin/env python3
"""
train_classifier.py — Fine-tune a ResNet18 species classifier on cropped images.

The script expects cropped detection patches organised by
``prepare_classifier_data.py`` under ``classifier_data/``::

    classifier_data/
    ├── train/
    │   ├── fish/
    │   ├── jellyfish/
    │   └── …
    └── val/
        ├── fish/
        ├── jellyfish/
        └── …

Training strategy
-----------------
1. Load ImageNet-pretrained ResNet18 and replace the final FC layer.
2. **Phase 1** (first ``CLASSIFIER_FREEZE_EPOCHS`` epochs): freeze ``conv1``
   and ``layer1`` so only the deeper layers adapt.
3. **Phase 2** (remaining epochs): unfreeze everything and continue with a
   cosine-annealing learning-rate schedule.

Usage
-----
    # Defaults from config.py
    python train_classifier.py

    # Override epochs and learning rate
    python train_classifier.py --epochs 40 --lr 5e-4

    # Force CPU
    python train_classifier.py --device cpu
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights

import config

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_classifier")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments, falling back to *config.py* defaults.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune a ResNet18 species classifier on cropped images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.CLASSIFIER_TRAIN_EPOCHS,
        help="Total number of training epochs.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=config.CLASSIFIER_TRAIN_BATCH,
        help="Batch size for train and validation loaders.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config.CLASSIFIER_TRAIN_LR,
        help="Initial learning rate for Adam optimiser.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config.DEVICE,
        help="Device to train on (e.g. 'cpu', 'cuda', 'cuda:0').",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def build_transforms() -> Dict[str, transforms.Compose]:
    """Build train / val data-augmentation pipelines.

    Training uses random flips, rotation, and colour jitter.  Validation
    only resizes and normalises.

    Returns
    -------
    dict
        ``{"train": Compose(…), "val": Compose(…)}``
    """
    img_size = config.CLASSIFIER_IMG_SIZE
    mean = config.IMAGENET_MEAN
    std = config.IMAGENET_STD

    data_transforms: Dict[str, transforms.Compose] = {
        "train": transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.1,
                    hue=0.05,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
    }
    return data_transforms


def build_dataloaders(
    data_dir: Path, batch_size: int
) -> Tuple[Dict[str, DataLoader], Dict[str, int], list[str]]:
    """Create PyTorch ``DataLoader`` objects for train and val splits.

    Parameters
    ----------
    data_dir : Path
        Root of the cropped-image dataset (contains ``train/`` and ``val/``).
    batch_size : int
        Mini-batch size.

    Returns
    -------
    tuple
        ``(dataloaders, dataset_sizes, class_names)``
    """
    data_transforms = build_transforms()

    image_datasets: Dict[str, datasets.ImageFolder] = {
        split: datasets.ImageFolder(
            str(data_dir / split), transform=data_transforms[split]
        )
        for split in ("train", "val")
    }

    dataloaders: Dict[str, DataLoader] = {
        split: DataLoader(
            image_datasets[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=4,
            pin_memory=True,
        )
        for split in ("train", "val")
    }

    dataset_sizes: Dict[str, int] = {
        split: len(image_datasets[split]) for split in ("train", "val")
    }

    class_names: list[str] = image_datasets["train"].classes

    logger.info("Train images : %d", dataset_sizes["train"])
    logger.info("Val images   : %d", dataset_sizes["val"])
    logger.info("Classes (%d) : %s", len(class_names), class_names)

    return dataloaders, dataset_sizes, class_names


# ─────────────────────────────────────────────────────────────────────────────
# Model construction
# ─────────────────────────────────────────────────────────────────────────────
def build_model(num_classes: int, device: torch.device) -> nn.Module:
    """Load ImageNet-pretrained ResNet18 and replace the final FC layer.

    Parameters
    ----------
    num_classes : int
        Number of target species classes.
    device : torch.device
        Target device.

    Returns
    -------
    nn.Module
        Modified ResNet18 model moved to *device*.
    """
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # Replace the final fully-connected layer
    in_features: int = model.fc.in_features  # 512 for ResNet18
    model.fc = nn.Linear(in_features, num_classes)

    model = model.to(device)
    logger.info(
        "Built ResNet18 classifier with %d output classes on '%s'.",
        num_classes,
        device,
    )
    return model


def set_frozen_layers(model: nn.Module, freeze: bool) -> None:
    """Freeze or unfreeze ``conv1`` and ``layer1`` of a ResNet model.

    Parameters
    ----------
    model : nn.Module
        A ResNet-style model.
    freeze : bool
        If ``True``, the parameters of ``conv1``, ``bn1``, and ``layer1``
        will be frozen (``requires_grad=False``).
    """
    for name, param in model.named_parameters():
        if name.startswith(("conv1", "bn1", "layer1")):
            param.requires_grad = not freeze

    state = "frozen" if freeze else "unfrozen"
    logger.info("conv1, bn1, and layer1 are now %s.", state)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train_model(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    dataset_sizes: Dict[str, int],
    *,
    epochs: int,
    lr: float,
    device: torch.device,
    freeze_epochs: int,
) -> Tuple[nn.Module, float]:
    """Run the two-phase training loop.

    Parameters
    ----------
    model : nn.Module
        The classifier model.
    dataloaders : dict
        ``{"train": DataLoader, "val": DataLoader}``.
    dataset_sizes : dict
        ``{"train": int, "val": int}``.
    epochs : int
        Total training epochs.
    lr : float
        Initial learning rate for Adam.
    device : torch.device
        Compute device.
    freeze_epochs : int
        Number of initial epochs during which ``conv1`` and ``layer1`` are
        frozen.

    Returns
    -------
    tuple
        ``(best_model, best_val_accuracy)``
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc: float = 0.0

    # Phase 1: freeze early layers
    set_frozen_layers(model, freeze=True)

    total_start = time.time()

    for epoch in range(epochs):
        # ── Unfreeze at the transition epoch ────────────────────────────
        if epoch == freeze_epochs:
            logger.info(
                "Epoch %d/%d — unfreezing all layers.", epoch + 1, epochs
            )
            set_frozen_layers(model, freeze=False)
            # Re-create optimiser so newly unfrozen params are included
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - freeze_epochs
            )

        epoch_start = time.time()

        for phase in ("train", "val"):
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss: float = 0.0
            running_corrects: int = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            current_lr = optimizer.param_groups[0]["lr"]

            if phase == "train":
                train_loss, train_acc = epoch_loss, epoch_acc
            else:
                val_loss, val_acc = epoch_loss, epoch_acc

            # Track best val accuracy
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        epoch_elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch + 1:>3d}/{epochs}  "
            f"lr={current_lr:.2e}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"[{epoch_elapsed:.1f}s]"
        )

    total_elapsed = time.time() - total_start
    logger.info(
        "Training complete in %.1f min. Best val accuracy: %.4f",
        total_elapsed / 60,
        best_acc,
    )

    model.load_state_dict(best_model_wts)
    return model, best_acc


# ─────────────────────────────────────────────────────────────────────────────
# Saving artefacts
# ─────────────────────────────────────────────────────────────────────────────
def save_model(model: nn.Module, weights_path: Path) -> None:
    """Save the model state-dict to disk.

    Parameters
    ----------
    model : nn.Module
        The trained model.
    weights_path : Path
        Destination ``.pth`` file.
    """
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    logger.info("Model weights saved to: %s", weights_path)


def save_class_names(
    class_names: list[str], output_path: Path
) -> None:
    """Persist the class-name list as JSON for inference.

    Parameters
    ----------
    class_names : list[str]
        Ordered list of class names matching the model's output indices.
    output_path : Path
        Destination ``.json`` file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapping: Dict[str, Any] = {
        "class_names": class_names,
        "num_classes": len(class_names),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    logger.info("Class names saved to: %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    """Entry-point: validate data, build model, train, and save artefacts."""
    args = parse_args()

    # ── Validate dataset directory ──────────────────────────────────────
    data_dir = config.CLASSIFIER_DATA_DIR
    if not data_dir.is_dir():
        logger.error(
            "Classifier dataset directory not found at '%s'.\n"
            "Please run 'python prepare_classifier_data.py' first to "
            "generate cropped training images.",
            data_dir,
        )
        sys.exit(1)

    for split in ("train", "val"):
        split_dir = data_dir / split
        if not split_dir.is_dir():
            logger.error(
                "Expected '%s' split directory at '%s' but it does not exist.\n"
                "Please run 'python prepare_classifier_data.py' first.",
                split,
                split_dir,
            )
            sys.exit(1)

    # ── Build data loaders ──────────────────────────────────────────────
    dataloaders, dataset_sizes, class_names = build_dataloaders(
        data_dir, batch_size=args.batch
    )

    # ── Build model ─────────────────────────────────────────────────────
    device = torch.device(args.device)
    num_classes = len(class_names)
    model = build_model(num_classes, device)

    # ── Train ───────────────────────────────────────────────────────────
    logger.info(
        "Training configuration: epochs=%d, batch=%d, lr=%g, "
        "freeze_epochs=%d, device=%s",
        args.epochs,
        args.batch,
        args.lr,
        config.CLASSIFIER_FREEZE_EPOCHS,
        args.device,
    )

    best_model, best_acc = train_model(
        model,
        dataloaders,
        dataset_sizes,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        freeze_epochs=config.CLASSIFIER_FREEZE_EPOCHS,
    )

    # ── Save artefacts ──────────────────────────────────────────────────
    save_model(best_model, config.CLASSIFIER_WEIGHTS)
    save_class_names(class_names, config.CLASSIFIER_CLASS_NAMES)

    # ── Final summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  CLASSIFIER TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Classes          : {num_classes} — {class_names}")
    print(f"  Best val accuracy: {best_acc:.4f} ({best_acc * 100:.2f}%)")
    print(f"  Model weights    : {config.CLASSIFIER_WEIGHTS}")
    print(f"  Class names JSON : {config.CLASSIFIER_CLASS_NAMES}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
