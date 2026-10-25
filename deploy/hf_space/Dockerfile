FROM python:3.12-slim

# ultralytics depends on the non-headless opencv-python, which links against
# libGL and glib. They are absent from the slim image, and the failure only
# shows up at import time, so install them up front.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face runs Spaces as uid 1000, and the app writes to ./outputs at
# startup, so the working directory must belong to that user.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics
WORKDIR /home/user/app

# CPU wheels first, from PyTorch's CPU index. The CUDA build is ~2.7 GB and
# useless on a GPU-less Space.
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
RUN mkdir -p outputs

# 7860 is the port Hugging Face routes to (app_port in README.md).
EXPOSE 7860
CMD ["uvicorn", "webapp:app", "--host", "0.0.0.0", "--port", "7860"]
