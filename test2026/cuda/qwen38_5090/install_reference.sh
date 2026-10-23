#!/usr/bin/env bash
set -euo pipefail
engine=${1:?Choose vllm or sglang}
case "$engine" in
  vllm) requirement='vllm==0.32.0';;
  sglang) requirement='sglang==0.5.22';;
  *) exit 2;;
esac
study=/mnt/d/CantorAI/5090-study-20261023
# /opt is within this study's WSL virtual disk, stored under the fixed Windows folder.
runtime=/opt/cantorai5090
mkdir -p "$runtime" "$study/results/references/$engine"
export UV_CACHE_DIR="$runtime/cache"
export HF_HUB_OFFLINE=1
export HF_HOME="$runtime/huggingface"
if [ ! -x "$runtime/bootstrap/bin/uv" ]; then
  python3 -m venv "$runtime/bootstrap"
  "$runtime/bootstrap/bin/python" -m pip install --no-cache-dir 'uv==0.13.5'
fi
uv="$runtime/bootstrap/bin/uv"
"$uv" venv --python /usr/bin/python3 "$runtime/$engine"
"$uv" pip install --python "$runtime/$engine/bin/python" "$requirement"
"$uv" pip freeze --python "$runtime/$engine/bin/python" > "$study/results/references/$engine/freeze.txt"
"$runtime/$engine/bin/python" - <<'PY'
import json,torch,importlib.metadata
print(json.dumps({'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
 'gpu':torch.cuda.get_device_name(0),'capability':torch.cuda.get_device_capability(0),
 'compiled_architectures':torch.cuda.get_arch_list()}))
x=torch.ones(16,device='cuda'); assert float(x.sum())==16
print('REFERENCE_CUDA_SMOKE_PASS; no model inference or performance claim')
PY
