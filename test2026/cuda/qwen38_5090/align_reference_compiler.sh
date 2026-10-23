#!/usr/bin/env bash
set -euo pipefail
engine=${1:?Choose vllm or sglang}
case "$engine" in vllm|sglang) ;; *) exit 2;; esac
runtime=/opt/cantorai5090
study=/mnt/d/CantorAI/5090-study-20261023
export UV_CACHE_DIR="$runtime/cache"
"$runtime/bootstrap/bin/uv" pip install --python "$runtime/$engine/bin/python" \
  -r "$(dirname "$0")/linux-reference-compiler-requirements.txt"
"$runtime/bootstrap/bin/uv" pip freeze --python "$runtime/$engine/bin/python" \
  > "$study/results/references/$engine/freeze-cuda13.0.txt"
"$runtime/$engine/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc" --version \
  > "$study/results/references/$engine/nvcc-cuda13.0.txt"
cat "$study/results/references/$engine/nvcc-cuda13.0.txt"
