#!/usr/bin/env bash
# OPT46: independent CPU oracle at full I2880/local I1440, no operator edits.
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
mkdir -p "$directory"; directory=$(cd "$directory" && pwd)
export LD_LIBRARY_PATH="$build/bin:$root/ThirdPartySDK/TensorRT/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GARNET_GPT_OSS_MARLIN_MAX_TOKENS=4096 GARNET_GPT_OSS_MARLIN_PREFILL_BLOCK=32
export GARNET_GPT_OSS_MARLIN_DECODE_BLOCK=32 GARNET_GPT_OSS_PARALLEL_MARLIN_METADATA=1
export GARNET_GPT_OSS_MARLIN_CTAS_PER_SM=1 GARNET_GPT_OSS_MARLIN_PREFILL_CTAS_PER_SM=1
git -C "$repo" rev-parse HEAD >"$directory/source.txt"
sha256sum "$build/bin/libgarnet.so" "$build/bin/libgarnet_gpt_oss.so" \
    "$build/bin/garnet_gpt_oss_kernel_parity" >"$directory/binaries.sha256"
for tile in 0 64; do
    export GARNET_GPT_OSS_MARLIN_LARGE_PREFILL_BLOCK=$tile
    for setting in 128/1 64/1 64/2; do
        export GARNET_GPT_OSS_MARLIN_PREFILL_DOWN_K=${setting%/*}
        export GARNET_GPT_OSS_MARLIN_PREFILL_DOWN_CTAS_PER_SM=${setting#*/}
        label=tp-sized-tile$tile-k${setting%/*}-cta${setting#*/}
        timeout --signal=TERM --kill-after=15s 300s "$build/bin/garnet_gpt_oss_kernel_parity" \
            --tp-sized-prefill-down-parity >"$directory/$label.log" 2>&1
        timeout --signal=TERM --kill-after=15s 900s compute-sanitizer --tool memcheck \
            --error-exitcode 0 --target-processes all --report-api-errors explicit --xml \
            --print-limit 0 --print-session-details --save "$directory/$label.xml" \
            "$build/bin/garnet_gpt_oss_kernel_parity" --tp-sized-prefill-down-parity \
            >"$directory/$label.memcheck.log" 2>&1
        "$python" "$repo/tools/gpt_oss/validate_sanitizer_xml.py" "$directory/$label.xml" "$directory/$label.audit.json"
        echo "$label independent CPU/shard/packing and strict memory passed"
    done
done
echo 'TP-sized CPU numerical gate complete; full pretrained and synthetic timing claims remain separate'
