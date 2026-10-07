#!/usr/bin/env bash
# Focused rerun after a strict API-classifier failure; preserve prior evidence.
set -euo pipefail
[[ $# == 1 && ! -e $1 ]] || exit 2
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${CANTORAI_ROOT:-$(dirname "$repo")}
build=${GARNET_BUILD_DIR:-$root/out/build/gpt-oss}
python=${GARNET_VERIFY_PYTHON:-$root/venv-vllm/bin/python}
directory=$1
exec 9>"$root/work/gpu-benchmark.lock"
flock -n 9 || exit 1
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] || exit 1
mkdir -p "$directory"
directory=$(cd "$directory" && pwd)
export LD_LIBRARY_PATH="$build/bin:$root/ThirdPartySDK/TensorRT/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
git -C "$repo" rev-parse HEAD >"$directory/source.txt"
sha256sum "$build/bin/garnet_gpt_oss_tp_direct_benchmark" "$build/bin/libgarnet_gpt_oss.so" >"$directory/binaries.sha256"
nvidia-smi --query-gpu=index,name,uuid,driver_version --format=csv >"$directory/hardware.csv"
for batch in 128 256 512; do
    export GARNET_GPT_OSS_DIRECT_MAX_BATCH=$batch
    GARNET_GPT_OSS_DIRECT_ALLREDUCE=0 "$build/bin/garnet_gpt_oss_tp_direct_benchmark" "$batch" 50 \
        >"$directory/reduction-b$batch-nccl.log" 2>&1
    for ctas in 2 4 8 16 32; do
        export GARNET_GPT_OSS_DIRECT_ALLREDUCE=1 GARNET_GPT_OSS_DIRECT_BATCH_ALLREDUCE=1
        export GARNET_GPT_OSS_DIRECT_LARGE_BATCH_ALLREDUCE=1 GARNET_GPT_OSS_DIRECT_BATCH_CTAS=$ctas
        "$build/bin/garnet_gpt_oss_tp_direct_benchmark" "$batch" 50 \
            >"$directory/reduction-b$batch-direct$ctas.log" 2>&1
    done
    label=reduction-b$batch-direct-memcheck
    compute-sanitizer --tool memcheck --error-exitcode 0 --target-processes all \
        --report-api-errors explicit --xml --print-limit 0 --print-session-details \
        --save "$directory/$label.xml" "$build/bin/garnet_gpt_oss_tp_direct_benchmark" "$batch" 1 \
        >"$directory/$label.log" 2>&1
    "$python" "$repo/tools/gpt_oss/validate_sanitizer_xml.py" "$directory/$label.xml" "$directory/$label.audit.json"
    echo "Direct batch$batch CPU/NCCL/changing-input/reacquire/strict-memory gate passed"
done
echo 'Focused direct gates complete; full pretrained quality/performance and resident admission remain separate.'
