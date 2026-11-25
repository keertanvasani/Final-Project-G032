"""External validation (FathomNet), step 2: per-image metadata sample, no pixels.

For each mapped taxon root: ROV images deeper than MIN_DEPTH, up to LIMIT records,
with descendants included server-side. Stores compact records (url, size, depth,
platform, date path, licences and every box on the image) to
data_external/fathomnet/meta/image_records.jsonl, then prints a planning summary.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from fathomnet import dto
from fathomnet.api import images

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fathomnet_meta import OUT, ROOTS  # noqa: E402

MIN_DEPTH = 200.0
LIMIT = 1000
DIVE_RE = re.compile(r"framegrabs/([^/]+)/images/(\d+)/", re.I)          # real dive number
DAY_RE = re.compile(r"framegrabs/([^/]+)/[^/]+/(\d{4})/(\d{2,3})/", re.I)  # stills: vehicle + day


def dive_of(url):
    m = DIVE_RE.search(url)
    if m:
        return f"{m.group(1).replace('%20', ' ')}_dive{int(m.group(2)):04d}"
    m = DAY_RE.search(url)
    return f"{m.group(1).replace('%20', ' ')}_day{m.group(2)}_{int(m.group(3)):03d}" if m else None


def compact(img):
    tags = {t.key: t.value for t in (img.tags or [])}
    return {
        "uuid": img.uuid, "url": img.url, "w": img.width, "h": img.height,
        "depth": img.depthMeters, "lat": img.latitude, "lon": img.longitude,
        "platform": tags.get("platform"), "source": tags.get("source"),
        "dive": dive_of(img.url or ""), "image_license": img.imageLicense,
        "boxes": [{"concept": b.concept, "x": b.x, "y": b.y, "w": b.width, "h": b.height,
                   "license": b.annotationLicense,
                   "review": getattr(b.reviewState, "value", b.reviewState)}
                  for b in (img.boundingBoxes or [])],
    }


def main():
    recs = {}
    roots = sorted({r for rs in ROOTS.values() for r in rs} | {"anthropogenic", "equipment", "trash"})
    for r in roots:
        c = dto.GeoImageConstraints(concept=r, taxaProviderName="fathomnet", imagingTypes=["ROV"],
                                    minDepth=MIN_DEPTH, limit=LIMIT)
        try:
            got = images.find(c)
        except Exception as e:  # noqa: BLE001
            print(f"{r}: failed {e}", file=sys.stderr)
            continue
        for img in got:
            recs[img.uuid] = compact(img)
        print(f"{r:20s} {len(got):5d} images (total unique {len(recs)})", flush=True)

    path = OUT / "image_records.jsonl"
    with path.open("w") as f:
        for v in recs.values():
            f.write(json.dumps(v) + "\n")
    print(f"\nwrote {path} ({path.stat().st_size / 1e6:.1f} MB)")

    v = list(recs.values())
    print("platforms:", Counter(x["platform"] for x in v).most_common(8))
    print("image licences:", Counter(x["image_license"] for x in v).most_common())
    print("box licences:", Counter(b["license"] for x in v for b in x["boxes"]).most_common())
    print("box review:", Counter(b["review"] for x in v for b in x["boxes"]).most_common())
    print("no dive parsed:", sum(x["dive"] is None for x in v))
    dives = defaultdict(int)
    for x in v:
        dives[x["dive"]] += 1
    print(f"dives: {len(dives)}, images/dive median "
          f"{sorted(dives.values())[len(dives) // 2]}, max {max(dives.values())}")
    print("resolution:", Counter(f"{x['w']}x{x['h']}" for x in v).most_common(6))
    nb = [len(x["boxes"]) for x in v]
    print("boxes/image: mean %.2f, share with 1 box %.2f" % (sum(nb) / len(nb), sum(n == 1 for n in nb) / len(nb)))


if __name__ == "__main__":
    main()
