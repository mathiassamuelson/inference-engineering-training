#!/usr/bin/env bash
#
# start-12b-qat.sh — launch google/gemma-4-12B-it-qat-w4a16-ct on a single RTX 3090
# using the gemma4-unified preview image, for the Week 12 single-card load test and
# context-ceiling characterization.
#
# *** TEMPORARY SCAFFOLDING — retire with Week 13 version convergence. ***
# This launcher exists because the pinned preview image needs two workarounds that the
# stable launcher (tools/start-vllm.sh) deliberately does not carry:
#
#   1. SOURCE PATCH (../patches/gemma4_unified.py): the image's gemma4_unified.py omits
#      prefix threading for the encoder-free vision embedder, so patch_dense reports an
#      empty layer name, never matches the checkpoint's compressed-tensors ignore list,
#      and is constructed packed while the checkpoint ships it bf16 -> shape mismatch at
#      weight load. The patch is a verbatim 3-line backport of the fix already in vLLM
#      upstream main, applied via read-only file mount. See patches/gemma4_unified.py.orig
#      for the pristine image copy and `git log` on the patch for provenance.
#
#   2. HF-OVERRIDES BLOB (mandatory, embedded below): works around two launch bugs.
#      (a) vision_config.num_soft_tokens is read by MM-budget code but missing from the
#          checkpoint config; 256 is an inert placeholder (MM inputs are off).
#      (b) hf-overrides does a SHALLOW REPLACE of quantization_config, not a merge —
#          so the override must restate the checkpoint's ENTIRE quantization_config
#          verbatim. Omitting any of it (e.g. quant_method) silently disables
#          quantization and the model constructs as ~24 GB bf16 -> load OOM.
#          The blob below = checkpoint config.json's quantization_config + regex
#          ignore entries appended. If the checkpoint revision changes, re-derive it.
#
# Verified working config (Week 12 Day 2): loads in 8.28 GiB weights, 11.82 GiB KV pool
# (245,222 tokens) at MML 32768 / util 0.90 / eager, on one 24 GB card.
#
# Usage:
#   ./start-12b-qat.sh                          # MML 32768, GPU 1, cudagraphs ON, port 8001
#   ./start-12b-qat.sh --max-model-len 131072   # ceiling-walk rung
#   ./start-12b-qat.sh --eager on               # fast-startup debug mode (decode floor only)
#   ./start-12b-qat.sh --gpu 3                  # the other PCIe-x1 card
#   ./start-12b-qat.sh --kv-cache-dtype fp8     # Phase-2 KV lever (verify SM 8.6 support
#                                               #  first — Week 11 notes say fp8 KV needs
#                                               #  SM 8.9+; unconfirmed for this image)
#   ./start-12b-qat.sh ... -- --some-vllm-flag  # anything after `--` passed to vLLM verbatim
#
# Single-GPU TP=1: no inter-GPU traffic, so the PCIe-x1 link on GPUs 1/3 is irrelevant.
#
set -euo pipefail

# ---- Locate the patch relative to this script --------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="${SCRIPT_DIR}/../phase-3-optimization-and-quantization/week-12/patches/gemma4_unified.py"
if [[ ! -f "$PATCH_FILE" ]]; then
  echo "[error] patch not found: ${PATCH_FILE}" >&2
  echo "        This launcher requires the gemma4_unified.py backport patch." >&2
  exit 2
fi

# ---- Defaults (override via flags) --------------------------------------------------
MODEL="google/gemma-4-12B-it-qat-w4a16-ct"
GPU="1"                   # single device id (1 or 3 = the PCIe-x1 cards)
MAX_MODEL_LEN="32768"
GPU_MEM_UTIL="0.90"
KV_CACHE_DTYPE="auto"
EAGER="off"               # off = compile + cudagraphs (production intent) | on = --enforce-eager
PORT="8001"
IMAGE="vllm/vllm-openai:gemma4-unified"
# Pinned digest recorded at pull time (verify with: docker inspect --format
# '{{range .RepoDigests}}{{println .}}{{end}}' "$IMAGE"):
#   sha256:e828735fba48bca2cf9701864d41693c91953394c5b1455b4668edd7563ed450
NAME="gemma4-12b-qat"

# ---- The mandatory two-bug override blob (see header) -------------------------------
OVERRIDES='{"vision_config": {"num_soft_tokens": 256}, "quantization_config": {"config_groups": {"group_0": {"format": "pack-quantized", "input_activations": null, "output_activations": null, "targets": ["Linear"], "weights": {"actorder": null, "block_structure": null, "dynamic": false, "group_size": 32, "num_bits": 4, "observer": "memoryless_minmax", "observer_kwargs": {}, "scale_dtype": null, "strategy": "group", "symmetric": true, "type": "int", "zp_dtype": null}}}, "format": "pack-quantized", "global_compression_ratio": null, "ignore": ["model.embed_vision.patch_dense", "model.embed_vision.multimodal_embedder.embedding_projection", "model.embed_audio.embedding_projection", "lm_head", "embed_vision.patch_dense", "embed_vision.multimodal_embedder.embedding_projection", "embed_audio.embedding_projection", "model.embed_vision.embedding_projection", "embed_vision.embedding_projection", "model.vision_embedder.patch_dense", "vision_embedder.patch_dense", "model.vision_embedder.patch_ln1", "vision_embedder.patch_ln1", "model.vision_embedder.patch_ln2", "vision_embedder.patch_ln2", "model.vision_embedder.pos_norm", "vision_embedder.pos_norm", "re:.*vision_embedder.*", "re:.*embed_vision.*", "re:.*embed_audio.*"], "kv_cache_scheme": null, "quant_method": "compressed-tensors", "quantization_status": "compressed", "sparsity_config": {}, "transform_config": {}, "version": "0.17.1.a20260602"}}'

# ---- Parse flags ---------------------------------------------------------------------
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)          MODEL="$2"; shift 2 ;;
    --gpu)            GPU="$2"; shift 2 ;;
    --max-model-len)  MAX_MODEL_LEN="$2"; shift 2 ;;
    --gpu-mem-util)   GPU_MEM_UTIL="$2"; shift 2 ;;
    --kv-cache-dtype) KV_CACHE_DTYPE="$2"; shift 2 ;;
    --eager)          EAGER="$2"; shift 2 ;;
    --port)           PORT="$2"; shift 2 ;;
    --image)          IMAGE="$2"; shift 2 ;;
    --name)           NAME="$2"; shift 2 ;;
    --)               shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "[error] unknown argument: $1" >&2; exit 2 ;;
  esac
done

# ---- Resolve eager flag --------------------------------------------------------------
EAGER_ARGS=()
case "$EAGER" in
  on)  EAGER_ARGS=(--enforce-eager) ;;
  off) : ;;
  *)   echo "[error] --eager must be 'on' or 'off' (got '$EAGER')" >&2; exit 2 ;;
esac

# ---- Echo resolved config (identity capture) ------------------------------------------
echo "=== start-12b-qat.sh (Week 12 temporary launcher) ==="
echo "  model         : ${MODEL}"
echo "  gpu           : device=${GPU}  (single-GPU TP=1; x1 link irrelevant)"
echo "  max-model-len : ${MAX_MODEL_LEN}"
echo "  gpu-mem-util  : ${GPU_MEM_UTIL}"
echo "  kv-cache-dtype: ${KV_CACHE_DTYPE}"
if [[ "$EAGER" == "on" ]]; then
  echo "  eager         : ON  (--enforce-eager; fast startup, decode is a FLOOR)"
else
  echo "  eager         : off (compile + cudagraphs; production intent)"
fi
echo "  image         : ${IMAGE}"
echo "  source patch  : ${PATCH_FILE} (mounted over image's gemma4_unified.py, ro)"
echo "  container     : ${NAME}"
echo "  port          : ${PORT}"
echo "  modalities    : image/audio limited to 0 (held constant w/ Day 2 verified run)"
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo "  extra vllm    : ${EXTRA_ARGS[*]}"
echo "======================================================"

# ---- Launch ----------------------------------------------------------------------------
# Foreground (--rm): Ctrl-C stops and removes the container.
# NOTE: --limit-mm-per-prompt intentionally zeroes image+audio ONLY (not video), matching
# the Day 2 verified configuration. Zeroing video may reclaim the 2496-token encoder-cache
# budget into KV — that is a candidate lever, NOT a default; change it only as a named,
# single-variable step.
exec docker run --rm \
  --name "${NAME}" \
  --gpus "\"device=${GPU}\"" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "${PATCH_FILE}:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4_unified.py:ro" \
  -p "${PORT}:8000" \
  "${IMAGE}" \
  --model "${MODEL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --limit-mm-per-prompt '{"image":0,"audio":0}' \
  --hf-overrides "${OVERRIDES}" \
  "${EAGER_ARGS[@]}"
