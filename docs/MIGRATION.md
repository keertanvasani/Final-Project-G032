# Migrating the project to the training laptop

Moving the full project — code, trained models, agents, datasets — from the
Mac (development) to the Windows laptop (RTX 5050, for GPU training).

**The move has two streams:** code travels through GitHub; the large assets
(weights + datasets) are **not** in git and travel as a separate bundle.

| Stream | What | How |
|---|---|---|
| Code | `agent_controller.py`, `pipeline.py`, `experiments/`, `PROTOCOL.md`, harness, web app | GitHub (branch `feat/phase1-uncertainty`) |
| Assets | `models/`, `data/`, `classifier_data/`, `JediOrganismDetectionDataset/`, `yolov8n/s.pt` | `fyp-assets.tar` (664 MB), hand-carried |

Why the split: `models/`, `data/`, `classifier_data/`, and the base weights
are gitignored, and `JediOrganismDetectionDataset/` was never committed, so a
`git clone` alone gives you code but **no weights and no data**.

---

## Which machine does what

Almost everything is done on the **laptop**. The Mac's only remaining job is
to copy one file.

| Step | Machine |
|---|---|
| Push code to GitHub | Mac — already done |
| Build `fyp-assets.tar` + checksum | Mac — already done |
| **1. Copy the bundle to USB / cloud** | **Mac** |
| 2. Install Git, Python 3.12, NVIDIA driver | Windows laptop |
| 3. Clone the repo | Windows laptop |
| 4. Verify + extract the bundle | Windows laptop |
| 5. Build venv + CUDA PyTorch | Windows laptop |
| 6. Verify (3 checks) | Windows laptop |

---

## Step 1 — Mac: move the bundle

The bundle and its checksum are in `~/dev/`:

```bash
# to a USB drive:
cp ~/dev/fyp-assets.tar ~/dev/fyp-assets.tar.sha256 /Volumes/YOUR_USB/

# or over the network (find the laptop's IP first):
scp ~/dev/fyp-assets.tar you@LAPTOP_IP:~/
```

Google Drive / Dropbox is fine too. **Carry the `.sha256` with it** — it
verifies the transfer on the other side. AirDrop will not reach a Windows
laptop.

Reference checksum:
```
69eddc6e06206d8a7eda35096ac9d23cb1b3d3d975c5d854b98a240073013069  fyp-assets.tar
```

Everything below runs on the **Windows laptop** (PowerShell).

## Step 2 — Prerequisites

- **Git** — https://git-scm.com/download/win
- **Python 3.12** — *not* 3.13/3.14; PyTorch's CUDA wheels don't exist for
  those yet. Install from python.org and tick "Add to PATH".
- **NVIDIA driver** recent enough for CUDA 12.8 (570-series or newer).
  Confirm with `nvidia-smi` — it prints the GPU and driver version.

## Step 3 — Clone the code

```powershell
cd ~\dev
git clone https://github.com/keertanvasani/Final-Project-Phase-1.git
cd Final-Project-Phase-1
git checkout feat/phase1-uncertainty
```

## Step 4 — Verify and extract the bundle

Copy `fyp-assets.tar` into the repo folder first, then:

```powershell
# must match the reference hash above
Get-FileHash fyp-assets.tar -Algorithm SHA256

# extract into the repo root (Windows 10/11 ships tar)
tar -xf fyp-assets.tar
```

This recreates `models/`, `data/`, `classifier_data/`,
`JediOrganismDetectionDataset/`, and the base weights exactly where
`config.py` expects them.

## Step 5 — Environment with the Blackwell-correct PyTorch

**Order matters.** Install the CUDA build of PyTorch *first* so
`requirements.txt` sees it satisfied and does not pull the CPU build. The
RTX 5050 is Blackwell (compute capability sm_120) and needs the CUDA 12.8
wheel — an older build fails with `no kernel image is available` or silently
runs on CPU.

```powershell
py -3.12 -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Step 6 — Verify (three checks)

```powershell
# 1. GPU visible to PyTorch — must print: True  NVIDIA GeForce RTX 5050 ...
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 2. Assets intact — hashes must match the Mac
python experiments\baseline\freeze_baseline.py --no-measure
python -c "import json; d=json.load(open('experiments/baseline/IMAGE_V1_BASELINE.json')); [print(k, v['sha256'][:16]) for k,v in d['models'].items()]"
#   expect: detector_nano 5e5a9e5c319faec3
#           detector_small 7d576e97bfd17b68
#           classifier a0eed5aeb2abf6b7

# 3. End-to-end on GPU — loads all models, runs perception, prints 'ok'
python -c "import cv2, glob; from pipeline import EnsemblePipeline; from agent_controller import run_perception; p=EnsemblePipeline(); print(run_perception(cv2.imread(glob.glob('data/images/val/*.jpg')[0]), p)['status'])"
```

If check 1 prints `True`, `config.DEVICE` auto-selects `cuda` — no code
changes needed; training and inference use the GPU automatically.

---

## Notes

- **Do not copy the Mac `venv/`** (1.5 GB, CPU-only, wrong platform) — Step 5
  rebuilds it correctly.
- **`outputs/` is not in the bundle** — 172 MB of regenerable prediction
  images; it fills back in as you run.
- **Cached predictions are not bundled** — regenerable in ~7 min each:
  `python experiments\run_phase_a.py cache --split J_test`. The committed
  `RESULTS.md` and figures came with the clone.
- **Windows dataloader:** if Ultralytics hangs on training, pass `workers=0`.
  WSL2 avoids most Windows friction if you prefer that path.

## Where work resumes on the laptop

1. Re-cache the dev splits: `run_phase_a.py cache --split J_cal` and
   `--split J_policy`.
2. Run the abstention-aggregation analysis: `python experiments/analyze_aggregation.py --split J_cal`
   (committed but not yet executed).
3. Continue the Phase 1 uncertainty layer (calibration, OOD, meta-confidence),
   then Phase 2 (agent), then Phase B GPU retraining.

See `experiments/README.md` for the full status and `experiments/PROTOCOL.md`
for the binding evaluation spec.
