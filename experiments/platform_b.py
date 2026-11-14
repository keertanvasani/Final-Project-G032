"""Assemble the retrained (Phase B) perception stack for action measurement.

The forced-action scripts were written against the frozen aquarium pipeline.
To test whether ZOOM and RECLASSIFY earn their cost on a *competent* detector
rather than one failing under domain shift, they need the same interface backed
by the Phase B weights.

**One substitution has to be declared.** Phase A's ensemble is YOLOv8-n and
YOLOv8-s — two architectures, different seeds and augmentation. Phase B trained
only YOLOv8-n, so the ensemble here is **YOLOv8-n seed 0 and YOLOv8-n seed 1**:
two independently-trained models with different seeds and different data order,
but the *same* architecture. That preserves the property the ensemble exists for
(independent errors from independently-trained models) while losing
architectural diversity, so the fused box is expected to be somewhat weaker than
a true n+s ensemble would give. Any comparison against Phase A therefore has two
things changing — training domain and ensemble composition — and cannot
attribute a difference to the domain alone.

Also differs from Phase A: the classifier has 8 Jedi-native classes rather than
7 aquarium ones, so the label space, and therefore what counts as OOD_ERR, comes
from the Phase B manifest.

Usage:
    from platform_b import build_platform
    pipe, manifest, known = build_platform("phase_b")
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).parent
sys.path.insert(0, str(ROOT))

PHASE_A_SPLITS = EXP / "splits" / "phase_a_splits.json"
PHASE_B_SPLITS = EXP / "splits" / "phase_b_splits.json"
MODELS_B = ROOT / "models" / "phase_b"


class PhaseBPipeline:
    """Duck-types the parts of EnsemblePipeline the action scripts use."""

    def __init__(self, det_a: Path, det_b: Path, clf_w: Path, clf_names: Path,
                 device: str = "cuda"):
        from ultralytics import YOLO

        import config
        from classifier import SpeciesClassifier
        from pipeline import EnsemblePipeline

        for p in (det_a, det_b, clf_w, clf_names):
            if not Path(p).is_file():
                raise SystemExit(f"Missing Phase B weight file: {p}")

        self.model_a = YOLO(str(det_a))
        self.model_b = YOLO(str(det_b))
        self.classifier = SpeciesClassifier(
            weights_path=clf_w, class_names_path=clf_names,
            device=device if device else config.DEVICE)
        # Identical crop geometry to Phase A, so the action is the same action.
        self._extract_crop = EnsemblePipeline._extract_crop
        self.models_status = {"yolo_a": True, "yolo_b": True, "classifier": True}


def build_platform(platform: str, seed_a: int = 0, seed_b: int = 1,
                   clf_seed: int = 0):
    """Return (pipeline, manifest_dict, known_classes_set)."""
    if platform == "phase_a":
        from pipeline import EnsemblePipeline
        man = json.loads(PHASE_A_SPLITS.read_text())
        return EnsemblePipeline(), man, set(man["label_space"]["known"])

    if platform != "phase_b":
        raise SystemExit(f"Unknown platform {platform!r}")

    import config

    man = json.loads(PHASE_B_SPLITS.read_text())
    known = set(man["label_space"]["known"])
    canonical = sorted(known)

    # `ensemble.fuse_detections` names a detection by indexing the module-level
    # `config.CLASS_NAMES`, which holds the 7 aquarium classes. Left alone, a
    # Phase B detector predicting index 3 is labelled with whatever aquarium
    # class sits at index 3 — the first Phase B run produced detections labelled
    # "puffin", a class the model cannot predict. The detector's own class order
    # is `sorted(known)`, matching data.yaml, so the list is rebound here.
    #
    # This is a deliberate global mutation and the reason the Phase B platform
    # must not be constructed in the same process as a Phase A run.
    config.CLASS_NAMES = canonical
    config.NUM_CLASSES = len(canonical)

    pipe = PhaseBPipeline(
        det_a=MODELS_B / f"yolov8n_seed{seed_a}" / "weights" / "best.pt",
        det_b=MODELS_B / f"yolov8n_seed{seed_b}" / "weights" / "best.pt",
        clf_w=MODELS_B / f"classifier_seed{clf_seed}" / "best_classifier.pth",
        clf_names=MODELS_B / f"classifier_seed{clf_seed}" / "class_names.json",
    )

    # The classifier was trained from an ImageFolder tree, so "/" became "_" in
    # the directory names: it reports "coral_sea-anemone" where the detector and
    # the ground truth both say "coral/sea-anemone". Without this remap the two
    # can never agree on that class, which inflates the disagreement rate and
    # silently corrupts the RECLASSIFY measurement.
    back = {c.replace("/", "_"): c for c in canonical}
    remapped = [back.get(c, c) for c in pipe.classifier.class_names]
    if remapped != list(pipe.classifier.class_names):
        pipe.classifier.class_names = remapped
    if sorted(remapped) != canonical:
        raise SystemExit(
            "Phase B classifier classes do not match the detector label space:\n"
            f"  classifier {sorted(remapped)}\n  detector   {canonical}")

    print(f"  Phase B platform: yolov8n seed{seed_a} + seed{seed_b} "
          f"(same architecture — YOLOv8-s was not trained), "
          f"classifier seed{clf_seed}, {len(canonical)} known taxa")
    return pipe, man, known
