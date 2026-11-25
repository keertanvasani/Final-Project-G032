"""External validation (FathomNet), step 3: the frozen image manifest (PROTOCOL §18.3).

Reads the metadata from fathomnet_meta.py / fathomnet_image_meta.py and writes
data_external/fathomnet/meta/manifest.json. No pixels are downloaded.
`--estimate-size` sends HEAD requests for a random sample of the manifest URLs to
estimate the download size.
"""
import argparse
import json
import random
import statistics
import urllib.request
from collections import defaultdict
from pathlib import Path

META = Path(__file__).resolve().parents[1] / "data_external" / "fathomnet" / "meta"

KNOWN = ["coral_sea-anemone", "crab", "crinoid", "fish", "sea-cucumber", "shrimp", "sponge", "starfish"]
PER_LEAF, PER_DIVE_CAP, N_UNSEEN, N_MANMADE = 200, 10, 600, 100
SEED = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimate-size", type=int, default=0, help="HEAD-request this many URLs")
    args = ap.parse_args()

    cmap = json.loads((META / "class_map_counts.json").read_text())
    leaf_of = {c: k for k in KNOWN for c in cmap[k]["concepts"]}
    unseen = {c for k, v in cmap.items() if k not in KNOWN and k != "machine" for c in v["concepts"]}
    manmade = {"equipment", "trash"}

    recs = [json.loads(l) for l in (META / "image_records.jsonl").open()]
    recs = sorted((r for r in recs if r["dive"]), key=lambda r: r["uuid"])
    rng = random.Random(SEED)

    chosen, why = {}, defaultdict(list)

    def take(pool, n, tag, cap=None):
        pool = pool[:]
        rng.shuffle(pool)
        per_dive, got = defaultdict(int), 0
        for r in pool:
            if got >= n:
                break
            if cap and per_dive[r["dive"]] >= cap:
                continue
            per_dive[r["dive"]] += 1
            got += 1
            chosen[r["uuid"]] = r
            why[r["uuid"]].append(tag)
        return got

    counts = {}
    for leaf in KNOWN:
        pool = [r for r in recs if any(leaf_of.get(b["concept"]) == leaf for b in r["boxes"])]
        counts[leaf] = take(pool, PER_LEAF, leaf, PER_DIVE_CAP)
    counts["unseen"] = take([r for r in recs if any(b["concept"] in unseen for b in r["boxes"])],
                            N_UNSEEN, "unseen", PER_DIVE_CAP)
    counts["man-made"] = take([r for r in recs if any(b["concept"] in manmade for b in r["boxes"])],
                              N_MANMADE, "man-made", PER_DIVE_CAP)
    verified = [r for r in recs if r["boxes"] and all(b["review"] == "VERIFIED" for b in r["boxes"])]
    for r in verified:
        chosen[r["uuid"]] = r
        why[r["uuid"]].append("partB")
    counts["partB"] = len(verified)

    manifest = [{"uuid": u, "url": r["url"], "dive": r["dive"], "w": r["w"], "h": r["h"],
                 "sha256": r.get("sha256"), "why": why[u]} for u, r in sorted(chosen.items())]
    (META / "manifest.json").write_text(json.dumps(
        {"seed": SEED, "protocol": "§18.3", "counts": counts, "images": manifest}, indent=1))
    print("selected per stratum:", counts)
    print(f"manifest: {len(manifest)} images, {len({m['dive'] for m in manifest})} dives")
    res = defaultdict(int)
    for m in manifest:
        res[f"{m['w']}x{m['h']}"] += 1
    print("resolution:", sorted(res.items(), key=lambda kv: -kv[1])[:6])

    if args.estimate_size:
        sample = random.Random(1).sample(manifest, min(args.estimate_size, len(manifest)))
        by_px = []
        for m in sample:
            try:
                req = urllib.request.Request(m["url"].replace(" ", "%20"), method="HEAD")
                with urllib.request.urlopen(req, timeout=20) as r:
                    n = int(r.headers.get("Content-Length", 0))
                if n:
                    by_px.append((m["w"] * m["h"], n))
            except Exception as e:  # noqa: BLE001
                print("HEAD failed:", m["url"][:80], e)
        bpp = statistics.median(n / px for px, n in by_px)
        est = sum(m["w"] * m["h"] for m in manifest) * bpp
        print(f"HEAD ok for {len(by_px)}/{len(sample)}; median {bpp:.2f} bytes/pixel; "
              f"estimated total {est / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
