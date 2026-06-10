#!/usr/bin/env bash
# Step 5 — Serve the merged 16-bit model via vLLM (ROCm) on an OpenAI-compatible
# endpoint. agents/model_client.py is the matching client; if you change the
# served-model-name here, set PATCHPROOF_VLLM_MODEL to match.
#
# Usage:
#   ./serve/serve.sh                                   # FP16 default, outputs/train/merged_16bit
#   ./serve/serve.sh --model-path PATH                 # override merged-model dir
#   ./serve/serve.sh --port 8000                       # override port
#   ./serve/serve.sh --host 0.0.0.0                    # override bind host
#   ./serve/serve.sh --served-name patchproof-merged   # name advertised on /v1/models
#   ./serve/serve.sh --api-key abc-123                 # require this key on /v1/*
#   ./serve/serve.sh --fp8                             # FP8 quant variant (optimization slide)
#   ./serve/serve.sh --extra "--max-model-len 8192"    # pass-through args to vllm
#
# AMD-only env vars set here (do NOT export these on a CUDA box):
#   HSA_OVERRIDE_GFX_VERSION=9.4.2   ROCm device id (required on MI300X)
#   VLLM_ROCM_USE_AITER=1            AITER kernels — faster on MI300X

set -euo pipefail

MODEL_PATH="outputs/train/merged_16bit"
PORT="8000"
HOST="0.0.0.0"
SERVED_NAME="patchproof-merged"
API_KEY=""
QUANT=""
EXTRA=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)  MODEL_PATH="$2"; shift 2 ;;
    --port)        PORT="$2"; shift 2 ;;
    --host)        HOST="$2"; shift 2 ;;
    --served-name) SERVED_NAME="$2"; shift 2 ;;
    --api-key)     API_KEY="$2"; shift 2 ;;
    --fp8)         QUANT="--quantization fp8"; shift ;;
    --extra)       EXTRA="$2"; shift 2 ;;
    -h|--help)     sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "FAIL: unknown arg: $1" >&2; exit 2 ;;
  esac
done

API_KEY_ARG=""
if [[ -n "$API_KEY" ]]; then
  API_KEY_ARG="--api-key $API_KEY"
fi

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "FAIL: model path not found: $MODEL_PATH" >&2
  echo "      Step 4 (python -m train.finetune) must finish first; expected $MODEL_PATH/" >&2
  exit 1
fi

if ! command -v vllm >/dev/null; then
  echo "FAIL: vllm CLI not on PATH (pip install vllm with the ROCm build)" >&2
  exit 1
fi

export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-9.4.2}"
export VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-1}"

echo "== vLLM serve =="
echo "  model_path = $MODEL_PATH"
echo "  served_as  = $SERVED_NAME"
echo "  host:port  = $HOST:$PORT"
echo "  quant      = ${QUANT:-fp16 (none)}"
echo "  api_key    = ${API_KEY:+set}${API_KEY:-<none>}"
echo "  HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"
echo "  VLLM_ROCM_USE_AITER=$VLLM_ROCM_USE_AITER"

# shellcheck disable=SC2086
exec vllm serve "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$SERVED_NAME" \
  $API_KEY_ARG \
  $QUANT \
  $EXTRA
