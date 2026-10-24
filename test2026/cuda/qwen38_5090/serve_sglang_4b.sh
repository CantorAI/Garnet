#!/usr/bin/env bash
set -euo pipefail
# Refuse an occupied study port before allocating/loading a model.
/opt/cantorai5090/sglang/bin/python - <<'PY'
import socket
with socket.socket() as probe: probe.bind(('127.0.0.1',18081))
PY
study=/mnt/d/CantorAI/5090-study-20261023
export HF_HUB_OFFLINE=1
export HF_HOME=/opt/cantorai5090/huggingface
export CUDA_VISIBLE_DEVICES=0
export PATH="/opt/cantorai5090/sglang/bin:$PATH"
export CUDA_HOME=/opt/cantorai5090/sglang/lib/python3.12/site-packages/nvidia/cu13
export LIBRARY_PATH="/usr/lib/wsl/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
# This candidate is not a tuned baseline until load, quality and timing pass.
# The32768-token pool preserves eight requests at4096 total context.
# Bound cold-start compilation after the recorded WSL RAM/swap exhaustion.
export MAX_JOBS=2
export CMAKE_BUILD_PARALLEL_LEVEL=2
export TRITON_CACHE_DIR=/opt/cantorai5090/triton-cache-sglang
printf '\n=== SGLang full-decode-graph BF16 4B VLM load, %s ===\n' "$(date -u +%FT%TZ)" >> "$study/results/sglang-4b-server.log"
exec /opt/cantorai5090/sglang/bin/python -m sglang.launch_server \
  --model-path "$study/models/Qwen3.5-4B" \
  --served-model-name qwen35-4b-bf16 --host 127.0.0.1 --port 18081 \
  --tp-size 1 --context-length 4096 --max-running-requests 8 \
  --chunked-prefill-size 4096 --mem-fraction-static 0.88 \
  --dtype bfloat16 --kv-cache-dtype bfloat16 --mamba-ssm-dtype float32 \
  --max-mamba-cache-size 8 --disable-radix-cache \
  --disable-flashinfer-autotune \
  --max-total-tokens 32768 \
  --cuda-graph-backend-decode full --cuda-graph-max-bs-decode 8 \
  --cuda-graph-bs-decode 1 2 4 8 --cuda-graph-backend-prefill disabled \
  >> "$study/results/sglang-4b-server.log" 2>&1
