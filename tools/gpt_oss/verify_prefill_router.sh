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
export GARNET_GPT_OSS_BF16_PREFILL_ALLREDUCE=0
git -C "$repo" rev-parse HEAD >"$directory/source-commit.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version --format=csv >"$directory/hardware.csv"
"$build/bin/garnet_gpt_oss_tp_workspace_test" >"$directory/collective-workspace.log" 2>&1
echo 'Compiled plugin prefill workspace contract passed'
"$parity" --tensorcore-router-parity >"$directory/router-parity.log" 2>&1
echo 'Independent router score/sorting/probability gates passed'
compute-sanitizer --tool memcheck --error-exitcode 99 --target-processes all \
    --kernel-name kns=routeScoresTensorCore --print-session-details \
    "$parity" --tensorcore-router-parity >"$directory/router-memcheck.log" 2>&1
grep -Fq 'ERROR SUMMARY: 0 errors' "$directory/router-memcheck.log"
echo 'Tensor-core router memory-safety gate passed (scalar references uninstrumented)'
"$python" "$repo/tools/gpt_oss/verify.py" --runtime-dir "$build/bin" \
    --work-dir "$directory/full-parity" --tensorrt-root "$tensorrt" --multi-gpu \
    >"$directory/full-parity.log" 2>&1
echo 'Full native and compiled multi-GPU gates passed'
"$python" "$repo/test2026/gpt_oss/make_fixture.py" "$directory/fixture64" --intermediate 64 \
    >"$directory/fixture64.log" 2>&1
export GARNET_GPT_OSS_TP_MOE_INTERMEDIATE_SHARDS=1
# Three CPU-prefix tokens times512requests exercises the>=1024-row router.
# It does not establish pretrained batch512 admission or throughput.
for bf16 in 0 1; do
    export GARNET_GPT_OSS_BF16_PREFILL_ALLREDUCE=$bf16
    for phase in cold warm; do
        label=teacher-bf$bf16-$phase
        command=("$runtime" "$repo/test2026/gpt_oss/tp_teacher_forced.py" "$directory/fixture64"
            "$directory/teacher-cache" "$directory/$label.json" 512)
        export GARNET_GPT_OSS_PROFILE_TEACHER_STEP=-1
        if [[ $bf16 == 1 && $phase == cold ]]; then
            # The substring selects both packBf16 and unpackBf16. NCCL and
            # reference kernels execute without instrumentation in this check.
            command=(compute-sanitizer --tool memcheck --error-exitcode 99 --target-processes all
                --kernel-name kns=packBf16 --print-session-details "${command[@]}")
        fi
        "${command[@]}" >"$directory/$label.log" 2>&1
        if [[ $bf16 == 1 && $phase == cold ]]; then
            grep -Fq 'ERROR SUMMARY: 0 errors' "$directory/$label.log"
        fi
        echo "Full-model batch512 teacher-forced BF16=$bf16 $phase gate passed"
    done
done
# Keep instrumentation separate from the complete numerical/reload checks.
# Capture one actual execution phase per process, not engine teardown/build.
export GARNET_GPT_OSS_BF16_PREFILL_ALLREDUCE=1
for phase in prefill decode; do
    label=teacher-bf1-$phase-trace
    export GARNET_GPT_OSS_PROFILE_TEACHER_STEP=0
    [[ $phase != decode ]] || export GARNET_GPT_OSS_PROFILE_TEACHER_STEP=1
    nsys profile --force-overwrite=false --trace=cuda --sample=none --cuda-graph-trace=node \
        --capture-range=cudaProfilerApi --capture-range-end=stop --output="$directory/$label" \
        "$runtime" "$repo/test2026/gpt_oss/tp_teacher_forced.py" "$directory/fixture64" \
        "$directory/teacher-cache" "$directory/$label.json" 512 >"$directory/$label.log" 2>&1
done
"$python" - "$directory" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
results=[json.loads((root/f'teacher-bf{mode}-{phase}.json').read_text())
         for mode in (0,1) for phase in ('cold','warm')]
results += [json.loads((root/f'teacher-bf1-{phase}-trace.json').read_text()) for phase in ('prefill','decode')]
baseline=[step['tp_logits'] for step in results[0]['steps']]
assert all([step['tp_logits'] for step in result['steps']]==baseline for result in results), 'BF16 wire or reload changed compiled logits'
print('Every batch512 CPU-prefix logit matches across FP32/BF16 wire and reload')
PY
for phase in prefill decode; do
    nsys stats --report cuda_gpu_kern_sum --format csv --output - "$directory/teacher-bf1-$phase-trace.nsys-rep" \
        >"$directory/teacher-kernels-$phase.csv" 2>"$directory/teacher-kernels-$phase.log"
done
grep -Fq 'routeScoresTensorCore' "$directory/teacher-kernels-prefill.csv"
grep -Fq '::packBf16(' "$directory/teacher-kernels-prefill.csv"
if grep -Fq 'routeScoresTensorCore' "$directory/teacher-kernels-decode.csv" || \
   grep -Fq '::packBf16(' "$directory/teacher-kernels-decode.csv"; then
    echo 'Prefill-only router or BF16 wire ran during batch512 decode' >&2; exit 1
fi
echo 'Traces confirm tensor-core router/BF16 wire in prefill and neither in batch512 decode'
echo 'All prefill-router gates passed; pretrained quality/performance remains separate'
