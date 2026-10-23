#!/usr/bin/env bash
set -euo pipefail
study=/mnt/d/CantorAI/5090-study-20261023
export HF_HUB_OFFLINE=1
export HF_HOME=/opt/cantorai5090/huggingface
export CUDA_VISIBLE_DEVICES=0
export PATH="/opt/cantorai5090/sglang/bin:$PATH"
export CUDA_HOME=/opt/cantorai5090/sglang/lib/python3.12/site-packages/nvidia/cu13
export LIBRARY_PATH="/usr/lib/wsl/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
# Bound cold-start compilation after the recorded WSL RAM/swap exhaustion.
export MAX_JOBS=2
export CMAKE_BUILD_PARALLEL_LEVEL=2
export TRITON_CACHE_DIR=/opt/cantorai5090/triton-cache-sglang
printf '\n=== SGLang initial NVFP4 VLM load, %s ===\n' "$(date -u +%FT%TZ)" >> "$study/results/sglang-server.log"
exec /opt/cantorai5090/sglang/bin/python -m sglang.launch_server \
  --model-path "$study/models/Qwen3.8-27B-NVFP4" \
  --served-model-name qwen38-27b-nvfp4 --host 127.0.0.1 --port 18081 \
  --tp-size 1 --context-length 4096 --max-running-requests 8 \
  --chunked-prefill-size 4096 --mem-fraction-static 0.88 \
  --dtype bfloat16 --kv-cache-dtype bfloat16 --mamba-ssm-dtype float32 \
  --max-mamba-cache-size 8 --disable-radix-cache \
  --disable-flashinfer-autotune \
  --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled \
  >> "$study/results/sglang-server.log" 2>&1
