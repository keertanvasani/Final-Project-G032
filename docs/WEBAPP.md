# Web Interface

A browser front-end for the detection pipeline. Upload a photo, the full ensemble
runs on it, and the annotated result comes back with the agent's decision trace.

## Running it

```bash
source venv/bin/activate
uvicorn webapp:app --reload --port 8000
```

Then open **http://localhost:8000**.

That single command serves both the site and the API. `python webapp.py` works too.

## How it attaches to the existing system

Nothing in the model code was modified. `webapp.py` imports the FastAPI app from
`main.py` and mounts the static site over it:

```
main.py          →  /health, /predict          (unchanged)
webapp.py        →  imports that app, mounts frontend/ at "/"
frontend/        →  the site
```

Because `/health` and `/predict` are registered before the static mount, they still
match first. Site and API share one origin, so there is no CORS hop and no second
server to run.

To go back to the API on its own, run `uvicorn main:app` exactly as before — the
front-end is purely additive.

## Structure

```
webapp.py                    # mounts the site onto the existing app
frontend/
├── index.html               # single page: overview, upload, results, metrics
├── css/styles.css
├── js/app.js                # health polling, upload, /predict call, rendering
└── assets/
    ├── samples/             # one-click demo images
    └── metrics/             # PR curve, confusion matrix, training curves
```

`assets/` holds **copies** of images from `data/` and `models/` on purpose — both of
those directories are gitignored, so anything served directly from them would be
missing on a fresh clone.

## What the page shows

- A flowchart of the architecture with all four agentic decision points drawn inline,
  including the refusal gate as an explicit terminal branch
- "Why this isn't just a YOLO wrapper" — a comparison against a typical single-model
  pipeline, plus the four mechanisms that differ
- A drop zone plus six one-click samples (including a deliberately blurry one)
- The annotated image, toggleable against the original
- Per-detection detector vs. classifier confidence and whether they agreed
- The `decision_trace` rendered as a timeline — the clearest evidence that the
  pipeline is making runtime decisions rather than following a fixed sequence
- Validation metrics and training curves

## States handled

| State | What you see |
|---|---|
| Models still loading | Status pill turns amber/red, analyse button disabled |
| Backend not running | "Backend offline" pill, with the command to start it |
| Unsupported file | Specific message (HEIC gets its own conversion hint) |
| Over 20 MB | Actual size vs. the limit |
| Detections found | Annotated image, cards, stats, trace |
| Flagged for review | Warning banner, empty detection list, trace explaining why |

## Notes

- Accepted formats are JPEG, PNG and WebP — the same set `config.ALLOWED_IMAGE_TYPES`
  enforces. HEIC (the iPhone default) is rejected with a hint to export as JPEG.
- Box colours in the legend match `config.ANNOTATION_COLORS` as actually rendered:
  green ≥ 0.75, amber ≥ 0.50, orange below.
- Every prediction still auto-saves to `outputs/`, exactly as before.
