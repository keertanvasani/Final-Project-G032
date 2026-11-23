"""The fixed tree for taxonomy-aware answers (PROTOCOL §14.1).

Each known class has a node at four levels: species → group → phylum → kingdom.
A coarse answer is correct when the true taxon lies under it, so an animal the
models never trained on can still be named correctly at a coarser level
("decapod" for a squat lobster).
"""

from __future__ import annotations

from typing import Any

import numpy as np

LEVELS = ("species", "group", "phylum", "kingdom")

# known leaf → (group, phylum, kingdom)
KNOWN = {
    "fish": ("Chordata", "Chordata", "Animal"),
    "crab": ("Decapoda", "Arthropoda", "Animal"),
    "shrimp": ("Decapoda", "Arthropoda", "Animal"),
    "starfish": ("Echinodermata", "Echinodermata", "Animal"),
    "sea-cucumber": ("Echinodermata", "Echinodermata", "Animal"),
    "crinoid": ("Echinodermata", "Echinodermata", "Animal"),
    "coral/sea-anemone": ("Cnidaria", "Cnidaria", "Animal"),
    "sponge": ("Porifera", "Porifera", "Animal"),
    "machine": ("Man-made", "Man-made", "Man-made"),
}

# held-out taxon → its deepest node in the tree
HELD_OUT = {
    "squat-lobster": "Decapoda", "sea-spider": "Arthropoda", "urchin": "Echinodermata",
    "jellyfish": "Cnidaria", "shark": "Chordata", "ray": "Chordata", "tunicate": "Chordata",
    "octopus": "Animal", "squid": "Animal", "ragworm": "Animal", "comb-jelly": "Animal",
}

PARENT = {"Decapoda": "Arthropoda", "Arthropoda": "Animal", "Chordata": "Animal",
          "Echinodermata": "Animal", "Cnidaria": "Animal", "Porifera": "Animal",
          "Animal": None, "Man-made": None}

CREDIT = {"Decapoda": 0.75, "Animal": 0.25}            # leaves 1.0, phyla / Man-made 0.5


def credit(node: str) -> float:
    if node in KNOWN:
        return 1.0
    return CREDIT.get(node, 0.5)


def path(taxon: str) -> set[str]:
    """Every node an answer may name and still be right for this true taxon."""
    if taxon in KNOWN:
        g, p, k = KNOWN[taxon]
        return {taxon, g, p, k}
    node = HELD_OUT.get(taxon)
    out = set()
    while node is not None:
        out.add(node)
        node = PARENT[node]
    return out


def node_of(leaf: str, level: str) -> str:
    return leaf if level == "species" else KNOWN[leaf][LEVELS.index(level) - 1]


def candidates(probs: np.ndarray, classes: list[str], final_class: str) -> dict[str, tuple[str, float]]:
    """Per level: (node with the highest summed probability, that sum).

    Species level is the agent's own final class (after the vote)."""
    p = {c: float(v) for c, v in zip(classes, probs)}
    out = {"species": (final_class, p.get(final_class, 0.0))}
    for lv in LEVELS[1:]:
        sums: dict[str, float] = {}
        for c, v in p.items():
            n = node_of(c, lv)
            sums[n] = sums.get(n, 0.0) + v
        best = max(sums, key=sums.get)
        out[lv] = (best, sums[best])
    return out


def under(node: str, leaf: str) -> bool:
    return leaf in KNOWN and node in path(leaf)


def is_correct(node: str, gt_class: str | None, matched: bool) -> bool:
    return bool(matched and gt_class is not None and node in path(gt_class))


def full_probs(crops: list[np.ndarray], clf: Any) -> np.ndarray:
    """The classifier's full softmax on each crop, through its own input transform."""
    import torch
    from PIL import Image
    if not crops:
        return np.zeros((0, len(clf.class_names)))
    t = torch.stack([clf.transform(Image.fromarray(np.ascontiguousarray(c[:, :, ::-1])).convert("RGB"))
                     for c in crops]).to(clf.device)
    with torch.no_grad():
        return torch.softmax(clf.model(t), 1).cpu().numpy()
