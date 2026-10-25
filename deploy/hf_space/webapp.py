"""
Web front-end host for the Deep-Sea Creature Detection & Classification System.

This module adds a browser UI on top of the existing API without modifying it.
It imports the FastAPI application defined in `main.py` (routes, lifespan model
loading and all) and mounts the static site in `frontend/` at the site root.

Because `/health` and `/predict` are registered on the router before this mount,
they continue to match first; everything else falls through to the static files.
Serving the UI and the API from one origin means no CORS round-trips and one
process to run for a demo.

Usage:
    uvicorn webapp:app --reload --port 8000
    # or
    python webapp.py

Then open http://localhost:8000
"""

from pathlib import Path

from fastapi.staticfiles import StaticFiles

import config
from main import app, logger

FRONTEND_DIR = Path(__file__).parent.resolve() / "frontend"

if not FRONTEND_DIR.is_dir():
    raise RuntimeError(
        f"Front-end directory not found at {FRONTEND_DIR}. "
        "The static site must be present before starting webapp."
    )

# html=True makes StaticFiles serve index.html for the bare "/" request.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

logger.info(f"Front-end mounted from {FRONTEND_DIR}")


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting web app on {config.API_HOST}:{config.API_PORT}")
    uvicorn.run(
        "webapp:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
        log_level="info",
    )
