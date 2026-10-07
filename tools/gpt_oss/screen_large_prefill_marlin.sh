#!/usr/bin/env bash
# Isolated synthetic scheduling screen; no serving-throughput claim.
set -euo pipefail
[[ $# == 1 && ! -e $1 ]] || exit 2
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${CANTORAI_ROOT:-$(dirname "$repo")}
build=${GARNET_BUILD_DIR:-$root/out/build/gpt-oss}
directory=$1
exec 9>"$root/work/gpu-benchmark.lock"
flock -n 9 || exit 1
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] || exit 1
mkdir -p "$directory"
directory=$(cd "$directory" && pwd)
export LD_LIBRARY_PATH="$build/bin:$root/ThirdPartySDK/TensorRT/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GARNET_GPT_OSS_MARLIN_MAX_TOKENS=4096 GARNET_GPT_OSS_MARLIN_PREFILL_BLOCK=32
export GARNET_GPT_OSS_MARLIN_DECODE_BLOCK=32 GARNET_GPT_OSS_MARLIN_CTAS_PER_SM=1
export GARNET_GPT_OSS_PARALLEL_MARLIN_METADATA=1 GARNET_GPT_OSS_ROUTER_QUERY_TILE=2
export GARNET_GPT_OSS_PREFILL_ROUTER_TENSORCORE=1
git -C "$repo" rev-parse HEAD >"$directory/source.txt"
sha256sum "$build/bin/garnet_gpt_oss_marlin_decode_benchmark" >"$directory/binary.sha256"
nvidia-smi --query-gpu=index,name,uuid,driver_version --format=csv >"$directory/hardware.csv"
for rows in 1024 4096; do
    for tile in 0 64; do
        export GARNET_GPT_OSS_MARLIN_LARGE_PREFILL_BLOCK=$tile
        for ctas in 1 2 4; do
            export GARNET_GPT_OSS_MARLIN_PREFILL_CTAS_PER_SM=$ctas
            for trial in 1 2 3; do
                label=rows$rows-tile$tile-cta$ctas-trial$trial
                "$build/bin/garnet_gpt_oss_marlin_decode_benchmark" "$rows" prefill \
                    >"$directory/$label.log" 2>&1
                cat "$directory/$label.log"
            done
        done
    done
done
# A large-prefill override must not change batched decode scheduling/output.
for tile in 0 64; do
    export GARNET_GPT_OSS_MARLIN_LARGE_PREFILL_BLOCK=$tile
    "$build/bin/garnet_gpt_oss_marlin_decode_benchmark" 256 decode \
        >"$directory/decode256-tile$tile.log" 2>&1
done
echo 'Synthetic screen complete; choose only separately quality-gated settings for full model tests.'
