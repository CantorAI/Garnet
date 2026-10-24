#!/usr/bin/env bash
set -euo pipefail
study=/mnt/d/CantorAI/5090-study-20261023
log=${GARNET_VLLM_LOG:-$study/results/vllm-mtp3-server.log}
export HF_HUB_OFFLINE=1
export HF_HOME=/opt/cantorai5090/huggingface
export CUDA_VISIBLE_DEVICES=0
export PATH="/opt/cantorai5090/vllm/bin:$PATH"
export CUDA_HOME=/opt/cantorai5090/vllm/lib/python3.12/site-packages/nvidia/cu13
export LIBRARY_PATH="/usr/lib/wsl/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export VLLM_CACHE_ROOT=/opt/cantorai5090/vllm-cache
export TRITON_CACHE_DIR=/opt/cantorai5090/triton-cache
printf '\n=== vLLM MTP3 decode-graph NVFP4 VLM candidate, %s ===\n' "$(date -u +%FT%TZ)" >> "$log"
# MTP3 verifies4 query tokens per request; token buckets4/8/16/32 cover C1/2/4/8.
# Candidate only: actual graph replay, admission, quality and timing must pass.
# Explicit cache budget is not evidence of eight full4K admissions; inspect
# resolved cache capacity and test admission before claiming that profile.
export MAX_JOBS=2
export CMAKE_BUILD_PARALLEL_LEVEL=2
export TORCHINDUCTOR_COMPILE_THREADS=1
exec /opt/cantorai5090/vllm/bin/vllm serve "$study/models/Qwen3.8-27B-NVFP4" \
  --served-model-name qwen38-27b-nvfp4 --host 127.0.0.1 --port 18080 \
  --tensor-parallel-size 1 --max-model-len 4096 --max-num-seqs 8 \
  --max-num-batched-tokens 4096 --gpu-memory-utilization 0.88 \
  --dtype bfloat16 --kv-cache-dtype bfloat16 --mamba-ssm-cache-dtype float32 \
  --no-enable-prefix-caching --kv-cache-memory-bytes 4294967296 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --max-cudagraph-capture-size 32 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,16,32]}' \
  >> "$log" 2>&1
