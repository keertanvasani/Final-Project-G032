# Deployment

The hosted demo has two halves, and they do different jobs.

| | What it is | Always on | Runs the models |
|---|---|---|---|
| **Static Space** | A static site on Hugging Face that replays recorded pipeline output | Yes | No |
| **Live backend** | This repository's FastAPI app, exposed through a temporary tunnel | Only while running | Yes |

Hugging Face's free tier has no runtime for custom apps, so the always-on part cannot run
the models. It shows what the pipeline actually returned for a fixed set of samples; the
live half exists for when someone needs to try their own image.

Live demo: <https://ajgodwin-deepseaai.static.hf.space/>

## The static Space

[`deploy/static_space/`](../deploy/static_space) is the whole site: the same interface as
`frontend/`, plus a small script, `js/api.js`, that answers the page's two API calls itself.

- **Recorded mode** (default). An image is matched to its recording by the SHA-256 of its
  bytes, so a result is only ever shown for the exact image it was captured from. Any other
  image is refused with an explicit message rather than answered with a guess.
- **Live mode.** Opening the page with `?api=<backend-url>` forwards the calls to a running
  backend. If that backend stops answering, the page falls back to recorded mode instead of
  breaking. `?api=off` forgets the setting.

The status pill always says which mode is active.

### Regenerating the recordings

The recorded results (about 15 MB of JSON) are not committed, because the pipeline
produces them. Regenerate them against any running backend:

```bash
uvicorn webapp:app --port 7860
python deploy/static_space/tools/record_demo.py --base http://127.0.0.1:7860
```

This writes `deploy/static_space/recorded/`, which must exist before the site is uploaded.

### Publishing

Upload the contents of `deploy/static_space/` to a Hugging Face Space created with the
**Static** SDK (**Files → Add file → Upload**). The folder's `README.md` carries the Space
metadata.

## The live backend

[`deploy/run_live_demo.ps1`](../deploy/run_live_demo.ps1) starts the backend, opens a
temporary [Cloudflare Quick Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/)
(no account or domain needed), checks it end to end, and prints the links:

```powershell
winget install --id Cloudflare.cloudflared
.\deploy\run_live_demo.ps1 -SpaceUrl https://ajgodwin-deepseaai.static.hf.space
```

It prints the backend URL, which serves the full site on its own, and a link that opens the
static Space wired to it. It keeps the machine awake while it runs. **Ctrl+C closes the
tunnel and stops the server.**

- The tunnel makes the machine reachable from the internet for as long as the script runs,
  and anyone holding the link can use the GPU. Share it only with the people who need it.
- The machine must stay awake, powered and online. The URL changes every run.

## Alternatives

- **A container host.** Any host that runs Docker with enough memory for PyTorch and two
  YOLO models can serve the full app, such as Google Cloud Run or a small VM. The weights
  are about 70 MB and the code falls back to CPU automatically. Use the CPU PyTorch wheels
  (`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`).
- **In-browser inference.** Converting the models to ONNX and porting the fusion and
  controller logic to JavaScript would remove the server, but it is a large rewrite.
