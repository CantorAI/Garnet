#!/usr/bin/env bash
# Run one homogeneous GPT-OSS TP2 batch with exclusive GPU access.
set -euo pipefail

if [[ $# != 4 ]]; then
    echo "Usage: $0 REQUEST_JSON RESULT_JSON BATCH OUTPUT_TOKENS" >&2
    exit 2
fi
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${CANTORAI_ROOT:-$(dirname "$repo")}
work=${GARNET_BENCH_WORK_DIR:-$root/work}
runtime=${XLANG3_RUNTIME_BIN:-$root/out/build/xlang3/bin/xlang3}
build=${GARNET_BUILD_DIR:-$root/out/build/gpt-oss}
weights=${GARNET_GPT_OSS_WEIGHTS:-$root/models/gpt-oss-120b/original}
cache=${GARNET_GPT_OSS_CACHE:-$work/gpt-oss-tp-cache}
tensorrt=${GARNET_TENSORRT_ROOT:-$root/ThirdPartySDK/TensorRT}

[[ -x $runtime && -d $weights && -f $1 && ! -e $2 ]] || {
    echo 'Runtime, weights or request missing, or result already exists' >&2
    exit 2
}
mkdir -p "$work" "$(dirname "$2")"
exec 9>"$work/gpu-benchmark.lock"
flock -n 9 || { echo 'Another benchmark owns the GPUs' >&2; exit 1; }
if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]; then
    echo 'Another GPU compute process is active' >&2
    exit 1
fi

export XLANG3_PYTHON_LIB=${XLANG3_PYTHON_LIB:-$root/ThirdPartySDK/Python-3.14.0/Lib}
export PYTHONPATH="$build/bin${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$build/bin:$tensorrt/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$runtime" "$repo/tools/gpt_oss/run_tp_batch_throughput.py" \
    "$weights" "$cache" "$1" "$2" "$3" "$4"
