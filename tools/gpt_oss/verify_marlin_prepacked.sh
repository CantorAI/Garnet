#!/usr/bin/env bash
# Validate storage-only prepacking before pretrained memory/performance tests.
# All source changes are made locally and pulled before invoking this gate.
set -euo pipefail
[[ $# == 1 ]] || { echo "Usage: $0 NEW_EVIDENCE_DIRECTORY" >&2; exit 2; }
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${CANTORAI_ROOT:-$(dirname "$repo")}
build=${GARNET_BUILD_DIR:-$root/out/build/gpt-oss}
work=${GARNET_BENCH_WORK_DIR:-$root/work}
python=${GARNET_VERIFY_PYTHON:-$root/venv-vllm/bin/python}
tensorrt=${GARNET_TENSORRT_ROOT:-$root/ThirdPartySDK/TensorRT}
runtime=$build/bin/xlang3
directory=$1
[[ ! -e $directory && -x $runtime && -x $python ]] || exit 2
command -v compute-sanitizer >/dev/null
mkdir -p "$work"
exec 9>"$work/gpu-benchmark.lock"
flock -n 9 || { echo 'Another test owns the GPUs' >&2; exit 1; }
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] || exit 1
mkdir -p "$directory"
directory=$(cd "$directory" && pwd)
export XLANG3_PYTHON_LIB=${XLANG3_PYTHON_LIB:-$root/ThirdPartySDK/Python-3.14.0/Lib}
export PYTHONPATH="$build/bin${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$build/bin:$tensorrt/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GARNET_GPT_OSS_MARLIN_MAX_TOKENS=4096 GARNET_GPT_OSS_MARLIN_PREFILL_BLOCK=32
export GARNET_GPT_OSS_MARLIN_DECODE_BLOCK=32 GARNET_GPT_OSS_ROUTER_QUERY_TILE=2
export GARNET_GPT_OSS_PARALLEL_MARLIN_METADATA=1 GARNET_GPT_OSS_PREFILL_ROUTER_TENSORCORE=1
export GARNET_GPT_OSS_BF16_PREFILL_ALLREDUCE=1
export GARNET_GPT_OSS_MARLIN_PREPACKED=0 GARNET_GPT_OSS_TP_EXPERT_WEIGHT_SHARDS=0
export GARNET_GPT_OSS_TP_MOE_INTERMEDIATE_SHARDS=0
git -C "$repo" rev-parse HEAD >"$directory/source-commit.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version --format=csv >"$directory/hardware.csv"
"$python" "$repo/tools/gpt_oss/verify.py" --runtime-dir "$build/bin" \
    --work-dir "$directory/full-parity" --tensorrt-root "$tensorrt" --multi-gpu \
    >"$directory/full-parity.log" 2>&1
echo 'Default native, CPU packing/planner, compiled, TP2 and Qwen/API regressions passed'
# All native kernels (including both layouts) are instrumented in this check.
# Preserve every record and application exit; the strict audit rejects any
# unexpected API or memory record, not only the tool's exit status.
compute-sanitizer --tool memcheck --error-exitcode 0 --target-processes all \
    --report-api-errors explicit --xml --print-limit 0 --print-session-details \
    --save "$directory/native.memcheck.xml" "$build/bin/garnet_gpt_oss_kernel_parity" \
    >"$directory/native.memcheck.log" 2>&1
"$python" "$repo/tools/gpt_oss/validate_sanitizer_xml.py" \
    "$directory/native.memcheck.xml" "$directory/native.memcheck-audit.json"
echo 'Complete native memory checking passed mandatory audit'
"$python" "$repo/test2026/gpt_oss/make_fixture.py" "$directory/fixture64" --intermediate 64 \
    >"$directory/fixture64.log" 2>&1
for partition in expert intermediate; do
    if [[ $partition == expert ]]; then
        export GARNET_GPT_OSS_TP_EXPERT_WEIGHT_SHARDS=1 GARNET_GPT_OSS_TP_MOE_INTERMEDIATE_SHARDS=0
        fixture=$directory/full-parity/fixture
        batch=16
    else
        export GARNET_GPT_OSS_TP_EXPERT_WEIGHT_SHARDS=0 GARNET_GPT_OSS_TP_MOE_INTERMEDIATE_SHARDS=1
        fixture=$directory/fixture64
        batch=512
    fi
    for mode in 0 1; do
        export GARNET_GPT_OSS_MARLIN_PREPACKED=$mode
        for phase in cold warm; do
            label=teacher-$partition-packed$mode-$phase
            "$runtime" "$repo/test2026/gpt_oss/tp_teacher_forced.py" "$fixture" \
                "$directory/teacher-$partition-cache" "$directory/$label.json" "$batch" \
                >"$directory/$label.log" 2>&1
            echo "$label CPU-prefix parity passed"
        done
    done
done
"$python" - "$directory" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
report={}
for partition in ('expert','intermediate'):
    results=[json.loads((root/f'teacher-{partition}-packed{mode}-{phase}.json').read_text())
             for mode in (0,1) for phase in ('cold','warm')]
    baseline=[s['tp_logits'] for s in results[0]['steps']]
    assert all([s['tp_logits'] for s in r['steps']]==baseline for r in results), partition
    assert all(s['within_existing_tolerance'] for r in results for s in r['steps'])
    report[partition]=dict(exact_all_logits_across_storage_and_reload=True,
        batch=results[0]['batch'], steps=len(baseline),
        cpu_max_absolute_errors=[max(s['maximum_absolute_error'] for s in r['steps']) for r in results])
(root/'compiled-storage-parity.json').write_text(json.dumps(report,indent=2))
print('All logits match exactly across original/prepacked constants and cold/warm refit')
PY
echo 'Storage gates passed; pretrained correctness, memory and lifecycle remain separate'
