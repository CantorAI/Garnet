#!/usr/bin/env bash
set -euo pipefail
study=/mnt/d/CantorAI/5090-study-20261023
export HF_HUB_OFFLINE=1
export HF_HOME=/opt/cantorai5090/huggingface
export CUDA_VISIBLE_DEVICES=0
export PATH="/opt/cantorai5090/vllm/bin:$PATH"
export CUDA_HOME=/opt/cantorai5090/vllm/lib/python3.12/site-packages/nvidia/cu13
export VLLM_CACHE_ROOT=/opt/cantorai5090/vllm-cache
export TRITON_CACHE_DIR=/opt/cantorai5090/triton-cache
printf '\n=== vLLM initial NVFP4 VLM load, %s ===\n' "$(date -u +%FT%TZ)" >> "$study/results/vllm-server.log"
# First load uses eager mode; this is not the final optimized comparison arm.
exec /opt/cantorai5090/vllm/bin/vllm serve "$study/models/Qwen3.8-27B-NVFP4" \
  --served-model-name qwen38-27b-nvfp4 --host 127.0.0.1 --port 18080 \
  --tensor-parallel-size 1 --max-model-len 4096 --max-num-seqs 8 \
  --max-num-batched-tokens 4096 --gpu-memory-utilization 0.88 \
  --dtype bfloat16 --kv-cache-dtype bfloat16 --mamba-ssm-cache-dtype float32 \
  --no-enable-prefix-caching --enforce-eager \
  >> "$study/results/vllm-server.log" 2>&1
