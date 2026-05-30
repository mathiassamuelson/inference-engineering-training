#!/usr/bin/env bash
#
# start-vllm.sh — launch a vLLM OpenAI-compatible server for the Week 11 TP-vs-PP
# experiment. Parameterized over model and parallelism strategy so the same script
# serves TP=2, PP=2, and the PP=4 stretch run rather than hardcoding one config.
#
# Zero-arg default: RedHatAI/gemma-4-31B-it-FP8-block, TP=2, on the NVLink pair (GPUs 0,2).
#
# Usage:
#   ./start-vllm.sh                          # FP8 31B, TP=2, GPUs 0,2 (default)
#   ./start-vllm.sh --mode pp --size 2       # PP=2 on the NVLink pair
#   ./start-vllm.sh --mode pp --size 4 --gpus all   # PP=4 stretch across all GPUs
#   ./start-vllm.sh --model <hf-id> --mode tp --size 2 --max-model-len 65536
#   ./start-vllm.sh ... -- --enforce-eager   # anything after `--` is passed to vLLM verbatim
#
# Ampere notes (RTX 3090, SM 8.6):
#   - FP8 KV cache requires SM 8.9+, so --kv-cache-dtype auto resolves to BF16. Do not
#     force fp8 KV here.
#   - Marlin FP8 weight emulation works fine on SM 8.6.
#
set -euo pipefail

# ---- Defaults (override via flags) ------------------------------------------------
MODEL="RedHatAI/gemma-4-31B-it-FP8-block"
MODE="tp"                 # tp | pp
SIZE="2"                  # parallel degree
GPUS="0,2"                # comma list of device ids, or the literal "all"
MAX_MODEL_LEN="131072"    # PROVISIONAL — final value set after Day 2 KV characterization
GPU_MEM_UTIL="0.90"
PORT="8000"
IMAGE="vllm/vllm-openai:v0.21.0"
NAME=""                   # container name; default derived below from mode/size
SHM_SIZE="16G"

# ---- Parse flags ------------------------------------------------------------------
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)          MODEL="$2"; shift 2 ;;
    --mode)           MODE="$2"; shift 2 ;;
    --size)           SIZE="$2"; shift 2 ;;
    --gpus)           GPUS="$2"; shift 2 ;;
    --max-model-len)  MAX_MODEL_LEN="$2"; shift 2 ;;
    --gpu-mem-util)   GPU_MEM_UTIL="$2"; shift 2 ;;
    --port)           PORT="$2"; shift 2 ;;
    --image)          IMAGE="$2"; shift 2 ;;
    --name)           NAME="$2"; shift 2 ;;
    --)               shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "[error] unknown argument: $1" >&2; exit 2 ;;
  esac
done

# ---- Resolve parallelism flag -----------------------------------------------------
case "$MODE" in
  tp) PARALLEL_FLAG=(--tensor-parallel-size "$SIZE") ;;
  pp) PARALLEL_FLAG=(--pipeline-parallel-size "$SIZE") ;;
  *)  echo "[error] --mode must be 'tp' or 'pp' (got '$MODE')" >&2; exit 2 ;;
esac

# ---- Resolve GPU selector ---------------------------------------------------------
# Docker wants the literal quoted form '"device=0,2"' for an explicit id list, or the
# bare token 'all' to expose every GPU.
if [[ "$GPUS" == "all" ]]; then
  GPU_ARG="all"
else
  GPU_ARG="\"device=${GPUS}\""
fi

# ---- Container name (descriptive, lets you `docker stop` it) ----------------------
[[ -z "$NAME" ]] && NAME="vllm-${MODE}${SIZE}"

# ---- HF token check (Gemma weights are gated on Hugging Face) ---------------------
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[warn] HF_TOKEN is not set. Gemma weights are gated; the pull may 401." >&2
fi

# ---- Echo resolved config (identity capture) --------------------------------------
echo "=== start-vllm.sh ==="
echo "  model         : ${MODEL}"
echo "  parallelism   : ${MODE}=${SIZE}"
echo "  gpus          : ${GPUS}"
echo "  max-model-len : ${MAX_MODEL_LEN}  (provisional until Day 2 characterization)"
echo "  gpu-mem-util  : ${GPU_MEM_UTIL}"
echo "  image         : ${IMAGE}"
echo "  container     : ${NAME}"
echo "  port          : ${PORT}"
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo "  extra vllm    : ${EXTRA_ARGS[*]}"
echo "====================="

# ---- Launch -----------------------------------------------------------------------
# Foreground (--rm -it): Ctrl-C stops and removes the container.
exec docker run --rm -it \
  --name "${NAME}" \
  --gpus "${GPU_ARG}" \
  --ipc=host --shm-size "${SHM_SIZE}" --network host \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  "${IMAGE}" \
  --model "${MODEL}" \
  "${PARALLEL_FLAG[@]}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --kv-cache-dtype auto \
  --limit-mm-per-prompt '{"image":0,"audio":0}' \
  --host 0.0.0.0 --port "${PORT}" \
  "${EXTRA_ARGS[@]}"
