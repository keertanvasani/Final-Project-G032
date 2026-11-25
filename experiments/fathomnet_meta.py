"""External validation (FathomNet), step 1: metadata only, no images.

For each of our 9 known classes and 11 held-out taxa, collect every FathomNet concept
under the matching WoRMS taxon and count its boxes. Writes
data_external/fathomnet/meta/{concept_counts.json, class_map_counts.json}.
"""
import json
import sys
from pathlib import Path

from fathomnet.api import boundingboxes, worms

OUT = Path(__file__).resolve().parents[1] / "data_external" / "fathomnet" / "meta"

# our label -> WoRMS taxa whose whole sub-tree maps to it (draft; frozen later in §18)
ROOTS = {
    # known (trained) classes
    "coral_sea-anemone": ["Anthozoa"],
    "crab": ["Brachyura"],
    "crinoid": ["Crinoidea"],
    "fish": ["Actinopterygii"],
    "sea-cucumber": ["Holothuroidea"],
    "shrimp": ["Caridea", "Dendrobranchiata"],
    "sponge": ["Porifera"],
    "starfish": ["Asteroidea"],
    # held-out taxa (unseen in training)
    "comb-jelly": ["Ctenophora"],
    "octopus": ["Octopoda"],
    "ray": ["Batoidea"],
    "shark": ["Selachii"],
    "urchin": ["Echinoidea"],
    "jellyfish": ["Scyphozoa"],
    "squid": ["Teuthida", "Oegopsida", "Myopsida"],
    "tunicate": ["Tunicata"],
    "ragworm": ["Polychaeta"],
    "sea-spider": ["Pycnogonida"],
    "squat-lobster": ["Galatheoidea", "Chirostyloidea"],
    # ambiguous look-alikes, counted only to decide the mapping
    "?brittle-star": ["Ophiuroidea"],
    "?king-crab": ["Lithodidae"],
    "?hydromedusa": ["Hydrozoa"],
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("counting boxes per concept ...", flush=True)
    counts = {c.concept: c.count for c in boundingboxes.count_total_by_concept()}
    (OUT / "concept_counts.json").write_text(json.dumps(counts, indent=1, sort_keys=True))
    print(f"{len(counts)} concepts, {sum(counts.values())} boxes in FathomNet")

    lower = {k.lower(): k for k in counts}
    man_made = sorted(k for k in counts if any(w in k.lower() for w in
                      ("trash", "debris", "man-made", "manmade", "anthropogenic", "litter", "equipment")))

    result = {}
    for label, roots in ROOTS.items():
        names = set()
        for r in roots:
            try:
                names |= set(worms.get_descendants_names(r)) | {r}
            except Exception as e:  # noqa: BLE001
                print(f"  {label}: WoRMS lookup failed for {r}: {e}", file=sys.stderr)
        hit = {lower[n.lower()]: counts[lower[n.lower()]] for n in names if n.lower() in lower}
        top = sorted(hit.items(), key=lambda kv: -kv[1])[:8]
        result[label] = {"roots": roots, "n_concepts": len(hit), "boxes": sum(hit.values()),
                         "concepts": hit, "top": top}
        print(f"{label:18s} concepts={len(hit):4d} boxes={sum(hit.values()):7d}  top={top[:4]}", flush=True)
    result["machine"] = {"concepts": {k: counts[k] for k in man_made},
                         "boxes": sum(counts[k] for k in man_made)}
    print(f"{'machine':18s} boxes={result['machine']['boxes']}  concepts={man_made}")
    (OUT / "class_map_counts.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
