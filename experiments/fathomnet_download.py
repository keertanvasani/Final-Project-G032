"""External validation (FathomNet), step 4: download the frozen manifest (§18.3).

Resumable: files already present with the right sha256 are skipped. Each download is
checked against FathomNet's sha256; mismatches are deleted and listed. Writes
data_external/fathomnet/images/<uuid>.png and meta/download_report.json.
"""
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data_external" / "fathomnet"
IMG = ROOT / "images"
EXPECTED_MANIFEST_SHA = "b07caf6bd2576e256096c0436452456c5599bffcf75bd16c356b930d6786ca0a"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fetch(m):
    dst = IMG / f"{m['uuid']}.png"
    if dst.exists() and (not m["sha256"] or sha(dst) == m["sha256"]):
        return m["uuid"], "skip", dst.stat().st_size
    tmp = dst.with_suffix(".part")
    for attempt in range(4):
        try:
            req = urllib.request.Request(m["url"].replace(" ", "%20"),
                                         headers={"User-Agent": "research-eval-client"})
            with urllib.request.urlopen(req, timeout=60) as r, tmp.open("wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if m["sha256"] and sha(tmp) != m["sha256"]:
                tmp.unlink()
                return m["uuid"], "sha_mismatch", 0
            tmp.replace(dst)
            return m["uuid"], "ok", dst.stat().st_size
        except Exception as e:  # noqa: BLE001
            err = str(e)
            time.sleep(2 * (attempt + 1))
    if tmp.exists():
        tmp.unlink()
    return m["uuid"], f"failed: {err}", 0


def main():
    mpath = ROOT / "meta" / "manifest.json"
    assert sha(mpath) == EXPECTED_MANIFEST_SHA, "manifest changed since §18 was frozen"
    man = json.loads(mpath.read_text())["images"]
    IMG.mkdir(parents=True, exist_ok=True)
    status, total, t0 = {}, 0, time.time()
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(fetch, m) for m in man]
        for i, fu in enumerate(as_completed(futs), 1):
            u, s, n = fu.result()
            status[u] = s
            total += n
            if i % 100 == 0 or i == len(man):
                bad = sum(not v in ("ok", "skip") for v in status.values())
                print(f"{i}/{len(man)}  {total / 1e9:.2f} GB  problems={bad}  "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
    problems = {u: s for u, s in status.items() if s not in ("ok", "skip")}
    (ROOT / "meta" / "download_report.json").write_text(json.dumps(
        {"n": len(man), "ok": len(man) - len(problems), "bytes": total, "problems": problems}, indent=1))
    print(f"done: {len(man) - len(problems)}/{len(man)} ok, {len(problems)} problems")


if __name__ == "__main__":
    main()
