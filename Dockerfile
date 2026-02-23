# ──────────────────────────────────────────────────────────────
# PPE Detection — Docker Image
# ──────────────────────────────────────────────────────────────
# Build:
#   docker build -t ppe-detection .
#
# Run:
#   docker run --rm \
#     -v /path/to/input:/data/input \
#     -v /path/to/output:/data/output \
#     ppe-detection \
#     --conf 0.5
#
# All CLI args after the image name are forwarded to the scheduler.
# ──────────────────────────────────────────────────────────────

FROM python:3.10-slim

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model weights (required if not using a trained run)
COPY *.pt ./

# Copy project structure
COPY config/ config/
COPY utils/ utils/
COPY scheduler.py .
COPY train.py .
COPY predict.py .
COPY evaluate.py .
COPY docker_entrypoint.py .

# Copy trained model weights if available (from local training)
# Note: This will copy the runs folder if it exists in the build context
COPY runs/ runs/

# Create default mount points and log directory
RUN mkdir -p /data/input /data/output /app/logs && chmod 777 /app/logs

# Default environment variables (can be overridden at runtime)
ENV INPUT_DIR=/data/input
ENV OUTPUT_DIR=/data/output
ENV CONFIDENCE=0.5
ENV FRAME_SKIP=10
ENV POLL_INTERVAL=30
ENV IMGSZ=640
ENV DEVICE=""

# Expose nothing (no web server), just a CLI tool
# metadata
LABEL maintainer="Safety Compliance Team"
LABEL version="1.1.0"
LABEL description="PPE Detection Scheduler for monitoring and processing videos."

ENTRYPOINT ["python3", "docker_entrypoint.py"]
