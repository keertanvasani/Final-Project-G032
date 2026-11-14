"""Record real pipeline outputs for every sample image the static site offers.

The static site cannot run the models, so for its built-in samples it replays
what the pipeline actually returned. Each recording is the unmodified /predict
response, stored under the SHA-256 of the image bytes. The browser hashes the
file it is about to analyse and looks that hash up, so a recording can only be
shown for the exact image it was made from; any other image gets an honest
"not available in recorded mode" message instead of a plausible-looking guess.

Requires a server on --base (default http://127.0.0.1:7860).

Usage:
    python tools/record_demo.py
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:7860")
    args = ap.parse_args()

    health = requests.get(f"{args.base}/health", timeout=10).json()
    if not health.get("models_loaded"):
        raise SystemExit(f"server not ready: {health}")
    print("server:", health["status"], "| device:", health["device"])

    OUT.mkdir(exist_ok=True)
    # The first request pays model warm-up; keep it out of the recorded timings.
    first = sorted(SAMPLES.glob("*.jpg"))[0]
    requests.post(f"{args.base}/predict",
                  files={"file": (first.name, first.read_bytes(), "image/jpeg")}, timeout=300)

    manifest = {"device": health["device"], "recorded_at": time.strftime("%Y-%m-%d"),
                "images": {}}
    for p in sorted(SAMPLES.glob("*.jpg")):
        data = p.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        r = requests.post(f"{args.base}/predict",
                          files={"file": (p.name, data, "image/jpeg")}, timeout=300)
        r.raise_for_status()
        body = r.json()
        (OUT / f"{sha}.json").write_text(json.dumps(body), encoding="utf-8")
        n = body["summary"]["total_objects"]
        manifest["images"][sha] = {"name": p.name, "objects": n}
        print(f"  {p.name:36s} {n:3d} objects  {len(json.dumps(body)) / 1024:6.0f} KB")

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    total = sum(f.stat().st_size for f in OUT.iterdir()) / 1e6
    print(f"recorded {len(manifest['images'])} images, {total:.1f} MB -> {OUT}")


if __name__ == "__main__":
    main()
