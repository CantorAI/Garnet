#!/usr/bin/env bash
# Gate the opt-in prefill router before any pretrained performance experiment.
# Usage: bash verify_prefill_router.sh NEW_EVIDENCE_DIRECTORY
set -euo pipefail
[[ $# == 1 ]] || { echo "Usage: $0 NEW_EVIDENCE_DIRECTORY" >&2; exit 2; }
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${CANTORAI_ROOT:-$(dirname "$repo")}
work=${GARNET_BENCH_WORK_DIR:-$root/work}
build=${GARNET_BUILD_DIR:-$root/out/build/gpt-oss}
runtime=$build/bin/xlang3
python=${GARNET_VERIFY_PYTHON:-$root/venv-vllm/bin/python}
tensorrt=${GARNET_TENSORRT_ROOT:-$root/ThirdPartySDK/TensorRT}
parity=$build/bin/garnet_gpt_oss_kernel_parity
directory=$1
[[ ! -e $directory && -x $runtime && -x $parity && -x $python ]] || {
    echo 'Evidence directory already exists, or required build/runtime is missing' >&2; exit 2;
}
command -v compute-sanitizer >/dev/null
command -v nsys >/dev/null
mkdir -p "$work"
exec 9>"$work/gpu-benchmark.lock"
flock -n 9 || { echo 'Another test/benchmark owns the GPUs' >&2; exit 1; }
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] || {
    echo 'Another GPU compute process is active' >&2; exit 1;
}
mkdir -p "$directory"
directory=$(cd "$directory" && pwd)
export XLANG3_PYTHON_LIB=${XLANG3_PYTHON_LIB:-$root/ThirdPartySDK/Python-3.14.0/Lib}
export PYTHONPATH="$build/bin${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$build/bin:$tensorrt/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Default synthetic fixtures use32 intermediate channels. Enable the new64
# channel partition only for its separate full-model gate below.
export GARNET_GPT_OSS_TP_MOE_INTERMEDIATE_SHARDS=0 GARNET_GPT_OSS_TP_EXPERT_WEIGHT_SHARDS=0
export GARNET_GPT_OSS_PREFILL_ROUTER_TENSORCORE=1
export GARNET_GPT_OSS_ROUTER_QUERY_TILE=2 GARNET_GPT_OSS_PARALLEL_MARLIN_METADATA=1
export GARNET_GPT_OSS_MARLIN_MAX_TOKENS=4096 GARNET_GPT_OSS_MARLIN_PREFILL_BLOCK=32
export GARNET_GPT_OSS_MARLIN_DECODE_BLOCK=32
git -C "$repo" rev-parse HEAD >"$directory/source-commit.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version --format=csv >"$directory/hardware.csv"
"$parity" --tensorcore-router-parity >"$directory/router-parity.log" 2>&1
echo 'Independent router score/sorting/probability gates passed'
compute-sanitizer --tool memcheck --error-exitcode 99 --target-processes all \
    "$parity" --tensorcore-router-parity >"$directory/router-memcheck.log" 2>&1
grep -Fq 'ERROR SUMMARY: 0 errors' "$directory/router-memcheck.log"
echo 'Router memory-safety gate passed'
"$python" "$repo/tools/gpt_oss/verify.py" --runtime-dir "$build/bin" \
    --work-dir "$directory/full-parity" --tensorrt-root "$tensorrt" --multi-gpu \
    >"$directory/full-parity.log" 2>&1
echo 'Full native and compiled multi-GPU gates passed'
"$python" "$repo/test2026/gpt_oss/make_fixture.py" "$directory/fixture64" --intermediate 64 \
    >"$directory/fixture64.log" 2>&1
export GARNET_GPT_OSS_TP_MOE_INTERMEDIATE_SHARDS=1
# Three CPU-prefix tokens times512requests exercises the>=1024-row router.
# It does not establish pretrained batch512 admission or throughput.
for phase in cold warm; do
    command=("$runtime" "$repo/test2026/gpt_oss/tp_teacher_forced.py" "$directory/fixture64"
        "$directory/teacher-cache" "$directory/teacher-$phase.json" 512)
    if [[ $phase == cold ]]; then
        command=(nsys profile --force-overwrite=false --trace=cuda --sample=none
            --cuda-graph-trace=node --output="$directory/teacher-cold" "${command[@]}")
    fi
    "${command[@]}" >"$directory/teacher-$phase.log" 2>&1
    echo "Full-model batch512 teacher-forced $phase gate passed"
done
nsys stats --report cuda_gpu_kern_sum --format csv --output - "$directory/teacher-cold.nsys-rep" \
    >"$directory/teacher-kernels.csv" 2>"$directory/teacher-kernels.log"
grep -Fq 'routeScoresTensorCore' "$directory/teacher-kernels.csv"
echo 'Compiled-model trace confirms the tensor-core router executed'
echo 'All prefill-router gates passed; pretrained quality/performance remains separate'
