"""Verify every invariant Phase B depends on. Exits non-zero if any check fails.

This is the guard against silent corruption. Most of the ways this study can
go wrong are invisible in the numbers: a dive leaking between train and test
inflates mAP, a held-out taxon appearing in a training label quietly falsifies
H3, a `J_cal` image used for early stopping biases calibration. None of those
raise an exception on their own — they just produce results that look fine and
are wrong.

Run it after any change to splits, labels or training config, and before
freezing IMAGE_V2_BASELINE.

Usage:
    python experiments/verify_phase_b.py
    python experiments/verify_phase_b.py --strict   # warnings become failures
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).parent
DATASET = ROOT / "JediOrganismDetectionDataset"
SOURCE_SPLITS = ["coco_train_data", "coco_valid_data", "coco_test_data"]

sys.path.insert(0, str(EXP))
from make_splits import dive_of  # noqa: E402

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "", warn_only: bool = False) -> bool:
    results.append((WARN if (not ok and warn_only) else (PASS if ok else FAIL),
                    name, detail))
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    pb_path = EXP / "splits" / "phase_b_splits.json"
    if not pb_path.is_file():
        raise SystemExit("Missing phase_b_splits.json — run make_splits_phase_b.py")
    pb = json.loads(pb_path.read_text())

    files = pb["files"]
    ls = pb["label_space"]
    known, held = set(ls["known"]), set(ls["held_out_unseen"])
    far = set(ls["far_ood"])
    order = ["S_fit", "S_modelsel", "S_cal", "S_policy", "S_test"]

    # ---- 1. the five-way structure of PROTOCOL §4.2 --------------------
    check("all five §4.2 splits present", set(files) == set(order),
          f"{sorted(files)}")
    prop_ok = []
    for s in order:
        r = pb["splits"][s]
        prop_ok.append(abs(r["dive_fraction"] - r["target_dive_fraction"]) < 0.01)
    check("realised dive fractions within 1% of target", all(prop_ok),
          ", ".join(f"{s}={pb['splits'][s]['dive_fraction']:.3f}" for s in order))

    # ---- 2. dive disjointness across all splits ------------------------
    dives = {s: {dive_of(f) for f in files[s]} for s in order}
    bad = []
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            if dives[a] & dives[b]:
                bad.append(f"{a}&{b}:{len(dives[a] & dives[b])}")
    check("all splits pairwise dive-disjoint", not bad, ", ".join(bad) or "4 splits")

    # ---- 3. partition completeness -------------------------------------
    allf = [f for s in order for f in files[s]]
    dup = [f for f, n in Counter(allf).items() if n > 1]
    check("every image assigned exactly once", not dup,
          f"{len(allf)} images, {len(dup)} duplicates")

    # ---- 4. label space partitions the dataset -------------------------
    coco_cls: Counter = Counter()
    dims: dict[str, tuple[int, int]] = {}
    boxes: dict[str, list] = defaultdict(list)
    for sp in SOURCE_SPLITS:
        c = json.loads((DATASET / sp / "coco.json").read_text())
        cats = {x["id"]: x["name"] for x in c["categories"]}
        imgs = {}
        for im in c["images"]:
            fn = Path(im["file_name"]).name
            imgs[im["id"]] = fn
            dims[fn] = (im["width"], im["height"])
        for a in c["annotations"]:
            fn = imgs.get(a["image_id"])
            if fn:
                boxes[fn].append(cats[a["category_id"]])
                coco_cls[cats[a["category_id"]]] += 1

    check("label space covers dataset exactly",
          known | held | far == set(coco_cls),
          f"{len(known)} known + {len(held)} held + {len(far)} far "
          f"= {len(coco_cls)} classes")
    check("known / held-out / far-OOD are disjoint",
          not (known & held) and not (known & far) and not (held & far))

    # ---- 5. OOD roles ---------------------------------------------------
    roles = {r: set(v) for r, v in ls["ood_roles"].items()}
    union = set().union(*roles.values())
    pairwise_ok = all(
        not (roles[a] & roles[b])
        for i, a in enumerate(roles) for b in list(roles)[i + 1:]
    )
    check("OOD roles pairwise disjoint", pairwise_ok)
    check("OOD roles cover exactly the held-out taxa", union == held,
          f"{len(union)} of {len(held)}")
    thin = {r: len(v) for r, v in roles.items() if len(v) < 3}
    check("every OOD role has >=3 taxa", not thin, str(thin) or "3/3/3+")

    # ---- 6. YOLO dataset ------------------------------------------------
    ydir = EXP / "yolo_phase_b"
    yaml_txt = (ydir / "data.yaml").read_text() if (ydir / "data.yaml").is_file() else ""
    names = [l.split(": ", 1)[1].strip()
             for l in yaml_txt.splitlines() if l.strip() and l.startswith("  ")]
    check("data.yaml names == sorted(known), contiguous ids",
          names == sorted(known), f"{len(names)} names")

    def read_list(p: Path) -> list[str]:
        return [Path(l).name for l in p.read_text().split("\n") if l.strip()]

    lists = {n: read_list(ydir / f"{n}.txt") for n in
             ["train", "val", "cal", "policy", "test"] if (ydir / f"{n}.txt").is_file()}

    if "train" in lists and "val" in lists:
        tr, va = set(lists["train"]), set(lists["val"])
        check("train and val share no image", not (tr & va),
              f"train {len(tr)}, val {len(va)}")
        check("train and val are dive-disjoint",
              not ({dive_of(f) for f in tr} & {dive_of(f) for f in va}))

        # The leakage checks that matter most.
        test_set = set(files["S_test"])
        check("NO S_test image in train or val", not ((tr | va) & test_set),
              f"{len((tr | va) & test_set)} leaked")
        cal_pol = set(files["S_cal"]) | set(files["S_policy"])
        check("NO S_cal/S_policy image in train or val (PROTOCOL 11.5)",
              not ((tr | va) & cal_pol), f"{len((tr | va) & cal_pol)} leaked")
        check("train subset of S_fit", tr <= set(files["S_fit"]),
              f"{len(tr - set(files['S_fit']))} outside")
        check("val is S_modelsel, not a slice of S_fit",
              va <= set(files["S_modelsel"]) and not (va & set(files["S_fit"])),
              f"{len(va)} images")

        # 11.3 — no training image may contain a held-out taxon.
        contaminated = [f for f in (tr | va) if any(c in held for c in boxes.get(f, []))]
        check("no training image contains a held-out taxon (PROTOCOL 11.3)",
              not contaminated, f"{len(contaminated)} contaminated")

    # ---- 7. label files contain only known taxa -------------------------
    n_lbl = 0
    bad_id = 0
    mismatch = []
    idx = {c: i for i, c in enumerate(sorted(known))}
    sample = sorted((set(lists.get("train", [])) | set(lists.get("test", []))))
    for fn in sample:
        lp = None
        for sp in SOURCE_SPLITS:
            cand = DATASET / sp / "labels" / (Path(fn).stem + ".txt")
            if cand.is_file():
                lp = cand
                break
        if lp is None:
            continue
        n_lbl += 1
        rows = [r for r in lp.read_text().split("\n") if r.strip()]
        ids = [int(r.split()[0]) for r in rows]
        if any(i < 0 or i >= len(known) for i in ids):
            bad_id += 1
        expect = Counter(idx[c] for c in boxes.get(fn, []) if c in known)
        if Counter(ids) != expect:
            mismatch.append(fn)

    check("all label class ids within [0, n_known)", bad_id == 0,
          f"{bad_id} files out of range")
    check("label files contain exactly the KNOWN annotations "
          "(no held-out / far-OOD written)",
          not mismatch, f"{len(mismatch)} mismatched of {n_lbl} checked")

    # ---- 8. trained runs -------------------------------------------------
    mdir = ROOT / "models" / "phase_b"
    runs = sorted(p for p in mdir.glob("*_seed*") if p.is_dir()) if mdir.is_dir() else []
    seeds_seen: dict[str, set[int]] = defaultdict(set)
    weight_missing, wrong_data = [], []
    for r in runs:
        w = r / "weights" / "best.pt"
        if not w.is_file():
            weight_missing.append(r.name)
        cfg = r / "args.yaml"
        if cfg.is_file():
            txt = cfg.read_text()
            if "yolo_phase_b" not in txt.replace("\\", "/"):
                wrong_data.append(r.name)
            for line in txt.splitlines():
                if line.startswith("seed:"):
                    seeds_seen[r.name.rsplit("_seed", 1)[0]].add(int(line.split(":")[1]))
    check("every finished run has best.pt", not weight_missing,
          ", ".join(weight_missing) or f"{len(runs)} runs")
    check("every run trained on the Phase B data.yaml", not wrong_data,
          ", ".join(wrong_data) or "ok")
    for arch, sd in sorted(seeds_seen.items()):
        check(f"{arch}: seeds distinct", len(sd) == len(runs and sd), str(sorted(sd)))
    check("3 seeds per architecture (PROTOCOL 7.1)",
          all(len(s) >= 3 for s in seeds_seen.values()) if seeds_seen else False,
          {k: len(v) for k, v in seeds_seen.items()} and
          str({k: sorted(v) for k, v in seeds_seen.items()}),
          warn_only=True)

    # ---- report ----------------------------------------------------------
    width = max(len(n) for _, n, _ in results) + 2
    print(f"\n{'':<6}{'check':<{width}}detail")
    print("-" * (width + 40))
    for status, name, detail in results:
        mark = {PASS: "  ok  ", FAIL: " FAIL ", WARN: " warn "}[status]
        print(f"{mark}{name:<{width}}{detail}")

    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    n_warn = sum(1 for s, _, _ in results if s == WARN)
    print("-" * (width + 40))
    print(f"{len(results)} checks: {len(results)-n_fail-n_warn} passed, "
          f"{n_fail} failed, {n_warn} warnings")

    if n_fail or (args.strict and n_warn):
        sys.exit(1)


if __name__ == "__main__":
    main()
