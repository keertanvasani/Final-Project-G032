# Troubleshooting

## Server and models

**503 "Models not loaded"**
The server is still loading, which takes roughly a minute on CPU, or weight files are
missing. Watch `GET /health` until `models_loaded` is `true`. If the weights were never
produced, run the training scripts (see the quick start in the README). Without them the
server falls back to generic COCO detectors, which are not species-specific.

**`SyntaxError` near `<<<<<<<` on startup**
A file contains unresolved git merge-conflict markers. Resolve the conflict, or check out
the file again from `main`.

**`RuntimeError: Front-end directory not found` on startup**
`webapp.py` needs `frontend/` beside it. Start `uvicorn` from the project root, or run
the API alone with `uvicorn main:app` if you do not need the web interface.

**The page loads but shows "Backend offline"**
The static files are being served but `/health` is not answering. This happens when the
page is opened straight from disk (`file://`). Use `http://localhost:8000` instead.

**Import errors**
Activate the virtual environment and run `pip install -r requirements.txt` again.

## Performance

**Slow inference on CPU**
Expect about a second for a typical frame, and several seconds for a crowded one with
dozens of objects. Lower `YOLO_IMG_SIZE` in `config.py` to 416 or 320 to trade accuracy
for speed.

**CUDA out of memory during training**
Reduce `YOLO_TRAIN_BATCH` in `config.py` or pass `--batch 8`.

**CPU-only PyTorch on a GPU machine**
Installing `requirements.txt` first lets pip resolve `torch` to the CPU build. Install
the CUDA wheel before everything else. On an RTX 50-series (Blackwell) GPU this needs a
CUDA 12.8 build; see [`MIGRATION.md`](MIGRATION.md). Confirm with
`python -c "import torch; print(torch.cuda.is_available())"`.

**Dataloader hangs on Windows**
Set `workers=0` for training on Windows.

## Results you might not expect

**"Image too blurry to analyze reliably"**
After CLAHE and sharpening the frame was still below the usable blur threshold, or no
detection reached 0.40. This is the system declining to guess.

**Confident wrong labels on deep-sea footage**
Expected. The deployed models are trained on aquarium imagery, and a closed-set
classifier has no "none of these" option, so an unfamiliar object receives its nearest
class. This is the central finding of the project; see `experiments/RESULTS.md`.

**A box on a timestamp or watermark**
The overlay-text filter in `text_overlay.py` removes these. If one slips through, run
`python experiments/verify_text_filter.py` and check whether the frame's overlay uses a
style the filter has not seen.
