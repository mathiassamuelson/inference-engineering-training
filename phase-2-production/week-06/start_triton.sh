#!/bin/bash
# Week 6 Experiment 3: Launch Triton Inference Server
# Serves embedding model (all-MiniLM-L6-v2) on GPU 2
#
# Ports:
#   8001 - HTTP inference API
#   8002 - gRPC inference API
#   8003 - Prometheus metrics
#
# Usage:
#   ./start_triton.sh          # foreground (see logs)
#   ./start_triton.sh -d       # detached (background)
#   docker stop triton-week6   # stop the container

set -e

MODEL_REPO="$HOME/triton-models"
CONTAINER_NAME="triton-week6"
TRITON_IMAGE="nvcr.io/nvidia/tritonserver:24.08-py3"

# Stop any existing instance
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

# Check if running detached
DETACH_FLAG=""
if [ "$1" = "-d" ]; then
    DETACH_FLAG="-d"
    echo "Starting Triton in background..."
else
    echo "Starting Triton in foreground (Ctrl+C to stop)..."
fi

docker run --rm $DETACH_FLAG \
    --name "$CONTAINER_NAME" \
    --gpus all \
    -p 8001:8000 \
    -p 8002:8001 \
    -p 8003:8002 \
    -v "$MODEL_REPO":/models \
    "$TRITON_IMAGE" \
    tritonserver \
        --model-repository=/models \
        --log-verbose=1

if [ "$1" = "-d" ]; then
    echo "Container started. View logs with: docker logs -f $CONTAINER_NAME"
    echo "Stop with: docker stop $CONTAINER_NAME"
fi