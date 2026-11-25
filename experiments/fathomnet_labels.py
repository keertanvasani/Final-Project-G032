"""External validation (FathomNet): place every manifest concept on the §14.1 tree (§18.2).

Uses WoRMS ancestry from the FathomNet taxa service. Output:
data_external/fathomnet/meta/concept_labels.json, concept -> {"label": ..., "why": ...}
where label is a §14.1 leaf, "unseen:<deepest node>", "Man-made" or None (excluded).
"""
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fathomnet.api import taxa, worms

META = Path(__file__).resolve().parents[1] / "data_external" / "fathomnet" / "meta"

LEAF_ROOTS = {
    "Anthozoa": "coral/sea-anemone", "Brachyura": "crab", "Crinoidea": "crinoid",
    "Actinopterygii": "fish", "Holothuroidea": "sea-cucumber", "Caridea": "shrimp",
    "Dendrobranchiata": "shrimp", "Porifera": "sponge", "Asteroidea": "starfish",
}
TYPOS = {"Opiuroidea": "Ophiuroidea"}       # misspelt FathomNet concepts
PHYLA = ("Chordata", "Arthropoda", "Echinodermata", "Cnidaria", "Porifera")


def mbari_chain(concept, max_steps=40):
    """Parents in FathomNet's own (MBARI knowledge-base) hierarchy, nearest first."""
    out, c = [], concept
    for _ in range(max_steps):
        try:
            p = taxa.find_parent("mbari", c)
        except Exception:  # noqa: BLE001
            break
        if p is None or not getattr(p, "name", None) or p.name in out or p.name == c:
            break
        out.append(p.name)
        c = p.name
    return out


def ancestors(concept):
    """WoRMS ancestors; for names WoRMS does not know (morphospecies such as
    "Acanthascinae sp. 1", species complexes, equipment), climb the MBARI hierarchy to
    the nearest name WoRMS knows. Returns (concept, worms_ancestors | None, mbari_chain)."""
    def w(name):
        try:
            return list(worms.get_ancestors_names(name))   # [] when unknown
        except Exception:  # noqa: BLE001
            return []

    a = w(TYPOS.get(concept, concept).strip())
    if a:
        return concept, a, []
    chain = mbari_chain(concept.strip())
    for x in chain:
        a = w(x)
        if a:
            return concept, a + [x], chain
    return concept, (chain or None), chain


def main():
    man = json.loads((META / "manifest.json").read_text())["images"]
    uu = {m["uuid"] for m in man}
    recs = [json.loads(l) for l in (META / "image_records.jsonl").open()]
    concepts = sorted({b["concept"] for r in recs if r["uuid"] in uu for b in r["boxes"]})
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(ancestors, concepts))
    anc = {c: a for c, a, _ in res}
    chains = {c: ch for c, _, ch in res}
    # strict ancestors of any leaf root are too coarse to judge a deeper answer
    coarse = set()
    for root in LEAF_ROOTS:
        a = ancestors(root)[1] or []
        coarse |= set(a) - {root}

    out = {}
    for c in concepts:
        a = anc[c]
        ch = set(chains[c]) | set(a or []) | {c}
        if "equipment" in ch:
            out[c] = {"label": "machine", "why": "man-made leaf (equipment)", "mbari": chains[c]}
        elif ch & {"trash", "anthropogenic", "metal", "plastic", "glass", "rubber", "fabric", "debris"}:
            out[c] = {"label": "Man-made", "why": "man-made, not machine", "mbari": chains[c]}
        elif not a:
            out[c] = {"label": None, "why": "no WoRMS match"}
        elif "Animalia" not in a and c != "Animalia":
            out[c] = {"label": None, "why": "not an animal"}
        elif c in coarse:
            out[c] = {"label": None, "why": "coarser than the tree needs"}
        else:
            chain = set(a) | {c}
            leaf = next((v for k, v in LEAF_ROOTS.items() if k in chain), None)
            if leaf:
                out[c] = {"label": leaf, "why": "known leaf"}
            elif "Decapoda" in chain:
                out[c] = {"label": "unseen:Decapoda", "why": "unseen"}
            else:
                ph = next((p for p in PHYLA if p in chain), None)
                out[c] = {"label": f"unseen:{ph or 'Animal'}", "why": "unseen"}
    (META / "concept_labels.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    n_box = Counter()
    for r in recs:
        if r["uuid"] in uu:
            for b in r["boxes"]:
                n_box[out[b["concept"]]["label"]] += 1
    print(f"{len(concepts)} concepts")
    for k, v in sorted(n_box.items(), key=lambda kv: -kv[1]):
        print(f"  {str(k):28s} {v:6d} boxes")
    exc = [c for c in concepts if out[c]["label"] is None]
    print(f"excluded ({len(exc)}):", [(c, out[c]["why"], chains[c][:3]) for c in exc][:40])
    print("man-made:", [(c, out[c]["label"], chains[c][:3]) for c in concepts
                        if out[c]["label"] in ("machine", "Man-made")])
    print("via MBARI fallback (sample):",
          [(c, out[c]["label"], chains[c][:2]) for c in concepts if chains[c] and out[c]["label"]][:15])


if __name__ == "__main__":
    main()
