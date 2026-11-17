"""Record real pipeline outputs for every sample image the static site offers.

The static site cannot run the models, so for its built-in samples it replays
what the pipeline actually returned. Each recording is the unmodified /predict
response, stored under the SHA-256 of the image bytes. The browser hashes the
file it is about to analyse and looks that hash up, so a recording can only be
shown for the exact image it was made from; any other image gets an honest
"not available in recorded mode" message instead of a plausible-looking guess.

Every sample is recorded four times: both model profiles (deep sea, aquarium),
each with conditional enhancement on and off. That way the model switch and the
enhancement toggle on the hosted page replay real results too. Deep-sea frames
that are byte-identical to JEDI frames come back already graded against the
dataset labels by the server (ground_truth.py), so no separate annotation step
is needed.

--extra DIR (repeatable) also records every .jpg under DIR, without adding it as
a sample button. Uploading one of those exact files on the hosted page then
replays its real result, so a prepared test set works without a live backend.
The image files themselves are not copied into the site, only their results.

Requires a server on --base (default http://127.0.0.1:8000) that serves both profiles.

Usage:
    python tools/record_demo.py [--extra path/to/test_images]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "assets" / "samples"
OUT = ROOT / "recorded"
PROFILES = ("deepsea", "aquarium")


def natural_profile(name: str) -> str:
    """Deep-sea frames are named ds_*; the rest are aquarium photos."""
    return "deepsea" if name.startswith("ds_") else "aquarium"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--extra", action="append", default=[], type=Path,
                    help="folder of extra images to record (searched recursively); repeatable")
    args = ap.parse_args()
    extra = [p for d in args.extra for p in sorted(d.rglob("*.jpg"))]

    health = requests.get(f"{args.base}/health", timeout=10).json()
    if not health.get("models_loaded"):
        raise SystemExit(f"server not ready: {health}")
    profiles = health.get("profiles") or {}
    missing = [p for p in PROFILES if p not in profiles]
    if missing:
        raise SystemExit(f"server lacks profile(s) {missing}; it reports {list(profiles)}")
    print("server:", health["status"], "| device:", health["device"], "| profiles:", list(profiles))

    OUT.mkdir(exist_ok=True)
    for f in OUT.glob("*.json"):
        f.unlink()                     # old single-variant recordings

    # The first request per profile pays model warm-up; keep it out of the recorded timings.
    first = sorted(SAMPLES.glob("*.jpg"))[0]
    for prof in PROFILES:
        requests.post(f"{args.base}/predict?profile={prof}",
                      files={"file": (first.name, first.read_bytes(), "image/jpeg")}, timeout=300)

    manifest = {"device": health["device"], "recorded_at": time.strftime("%Y-%m-%d"),
                "profiles": {p: profiles[p] for p in PROFILES},
                "default_profile": "deepsea", "images": {}}
    for p in sorted(SAMPLES.glob("*.jpg")) + extra:
        data = p.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if sha in manifest["images"]:
            continue
        prof0 = "deepsea" if p in extra else natural_profile(p.name)
        entry = {"name": p.name, "default_profile": prof0, "variants": {}}
        if p in extra:
            entry["extra"] = True
        line = []
        for prof in PROFILES:
            for enh in (True, False):
                r = requests.post(f"{args.base}/predict?profile={prof}&enhance={str(enh).lower()}",
                                  files={"file": (p.name, data, "image/jpeg")}, timeout=300)
                r.raise_for_status()
                body = r.json()
                key = f"{prof}|{int(enh)}"
                fname = f"{sha[:16]}_{prof}_{int(enh)}.json"
                (OUT / fname).write_text(json.dumps(body), encoding="utf-8")
                n = body["summary"]["total_objects"]
                ok = sum(1 for d in body["detections"] if (d.get("gt") or {}).get("outcome") == "TP")
                entry["variants"][key] = {"file": fname, "objects": n,
                                          "correct": ok if body.get("ground_truth") else None}
                line.append(f"{prof[:4]}{'+' if enh else '-'}:{n}" + (f"({ok} ok)" if body.get("ground_truth") else ""))
        d = entry["variants"][f"{entry['default_profile']}|1"]
        entry["objects"] = d["objects"]
        manifest["images"][sha] = entry
        print(f"  {p.name:36s} " + "  ".join(line))

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    total = sum(f.stat().st_size for f in OUT.iterdir()) / 1e6
    print(f"recorded {len(manifest['images'])} images x {len(PROFILES) * 2} variants, {total:.1f} MB -> {OUT}")


if __name__ == "__main__":
    main()
