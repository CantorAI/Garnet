#!/usr/bin/env bash
# OPT49 synthetic gates. Run only after the previous GPU controller is terminal.
set -euo pipefail
[[ $# == 1 && ! -e $1 ]] || exit 2
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${CANTORAI_ROOT:-$(dirname "$repo")}
build=${GARNET_BUILD_DIR:-$root/out/build/gpt-oss}
python=${GARNET_VERIFY_PYTHON:-$root/venv-vllm/bin/python}
directory=$1
mkdir -p "$directory"
directory=$(cd "$directory" && pwd)
export GARNET_GPT_OSS_DECODE_ROUTER_TENSORCORE=0
export GARNET_GPT_OSS_BF16_DECODE_ALLREDUCE=0 GARNET_GPT_OSS_MARLIN_BOUNDED_PREFILL=0
bash "$repo/tools/gpt_oss/verify_resident_tp2.sh" "$directory/regressions"
exec 9>"$root/work/gpu-benchmark.lock"
flock -n 9 || { echo 'Another GPU controller owns the benchmark lock' >&2; exit 1; }
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] || exit 1
command -v compute-sanitizer >/dev/null
command -v nsys >/dev/null
export XLANG3_PYTHON_LIB=$root/ThirdPartySDK/Python-3.14.0/Lib
export PYTHONPATH="$build/bin${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$build/bin:$root/ThirdPartySDK/TensorRT/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GARNET_GPT_OSS_MARLIN_MAX_TOKENS=4096 GARNET_GPT_OSS_MARLIN_PREFILL_BLOCK=32
export GARNET_GPT_OSS_MARLIN_DECODE_BLOCK=32 GARNET_GPT_OSS_ROUTER_QUERY_TILE=2
export GARNET_GPT_OSS_ROUTER_THREADS=256 GARNET_GPT_OSS_ROUTER_VECTOR4=1
export GARNET_GPT_OSS_PREFILL_ROUTER_TENSORCORE=1 GARNET_GPT_OSS_PARALLEL_MARLIN_METADATA=1
export GARNET_GPT_OSS_BF16_PREFILL_ALLREDUCE=1 GARNET_GPT_OSS_TEACHER_RESIDENT=1
git -C "$repo" rev-parse HEAD >"$directory/source-commit.txt"
sha256sum "$build/bin/libgarnet.so" "$build/bin/libgarnet_gpt_oss.so" >"$directory/native.sha256"
nvidia-smi --query-gpu=index,name,uuid,driver_version --format=csv >"$directory/hardware.csv"
"$build/bin/garnet_gpt_oss_router_dispatch" >"$directory/dispatch.log" 2>&1

# No kernel filter, no record limit, no API suppression. Preserve application
# exit status and reject incomplete/tool-error reports before strict XML audit.
audit_memory() {
    local label=$1
    if grep -Eq 'No attachable process found|compute-sanitizer timed-out|========= Error:' "$directory/$label.log"; then
        echo "Incomplete sanitizer coverage: $label" >&2; exit 1
    fi
    grep -Fq 'ERROR SUMMARY:' "$directory/$label.log"
    "$python" "$repo/tools/gpt_oss/validate_sanitizer_xml.py" \
        "$directory/$label.xml" "$directory/$label.audit.json"
}
for router in 0 1; do
    export GARNET_GPT_OSS_DECODE_ROUTER_TENSORCORE=$router
    "$build/bin/garnet_gpt_oss_kernel_parity" --tensorcore-router-parity \
        >"$directory/native-router$router.log" 2>&1
done
export GARNET_GPT_OSS_DECODE_ROUTER_TENSORCORE=1
compute-sanitizer --tool memcheck --error-exitcode 0 --target-processes all --launch-timeout 0 \
    --report-api-errors explicit --xml --print-limit 0 --print-session-details \
    --backtrace-short no --strip-paths no --save "$directory/native-router-mem.xml" \
    "$build/bin/garnet_gpt_oss_kernel_parity" --tensorcore-router-parity \
    >"$directory/native-router-mem.log" 2>&1
audit_memory native-router-mem

# Small synthetic teacher models: both TP weight axes, original/prepacked,
# padded tails, cold/warm engine refit and changing-prefix shared KV. This is
# not a pretrained checkpoint or maximum-capacity benchmark.
export GARNET_GPT_OSS_BF16_DECODE_ALLREDUCE=1 GARNET_GPT_OSS_MARLIN_BOUNDED_PREFILL=1
for partition in expert intermediate; do
    if [[ $partition == expert ]]; then
        export GARNET_GPT_OSS_TP_EXPERT_WEIGHT_SHARDS=1 GARNET_GPT_OSS_TP_MOE_INTERMEDIATE_SHARDS=0
        fixture=$directory/regressions/full-parity/fixture; batch=128
    else
        export GARNET_GPT_OSS_TP_EXPERT_WEIGHT_SHARDS=0 GARNET_GPT_OSS_TP_MOE_INTERMEDIATE_SHARDS=1
        fixture=$directory/regressions/fixture64; batch=512
    fi
    for packed in 0 1; do
        export GARNET_GPT_OSS_MARLIN_PREPACKED=$packed
        for pad in 4 8; do
            export GARNET_GPT_OSS_TEACHER_PADDED_TOKENS=$pad
            cache=$directory/$partition-packed$packed-pad$pad-cache
            for router in 0 1; do
                export GARNET_GPT_OSS_DECODE_ROUTER_TENSORCORE=$router
                for phase in cold warm; do
                    label=$partition-packed$packed-pad$pad-router$router-$phase
                    export GARNET_GPT_OSS_PROFILE_TEACHER_STEP=-1
                    "$build/bin/xlang3" "$repo/test2026/gpt_oss/tp_teacher_forced.py" "$fixture" \
                        "$cache" "$directory/$label.json" "$batch" >"$directory/$label.log" 2>&1
                    echo "$label CPU-prefix/shared-KV gate passed"
                done
            done
        done
    done
done
export GARNET_GPT_OSS_DECODE_ROUTER_TENSORCORE=1 GARNET_GPT_OSS_MARLIN_PREPACKED=1
export GARNET_GPT_OSS_TEACHER_PADDED_TOKENS=4 GARNET_GPT_OSS_PROFILE_TEACHER_STEP=-1
fixture=$directory/regressions/fixture64
compute-sanitizer --tool memcheck --error-exitcode 0 --target-processes all --launch-timeout 0 \
    --report-api-errors explicit --xml --print-limit 0 --print-session-details \
    --backtrace-short no --strip-paths no --save "$directory/compiled-router-mem.xml" \
    "$build/bin/xlang3" "$repo/test2026/gpt_oss/tp_teacher_forced.py" "$fixture" \
    "$directory/intermediate-packed1-pad4-cache" "$directory/compiled-router-mem.json" 512 \
    >"$directory/compiled-router-mem.log" 2>&1
audit_memory compiled-router-mem
export GARNET_GPT_OSS_PROFILE_TEACHER_STEP=1
nsys profile --force-overwrite=false --trace=cuda --sample=none --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi --capture-range-end=stop --output="$directory/decode-trace" \
    "$build/bin/xlang3" "$repo/test2026/gpt_oss/tp_teacher_forced.py" "$fixture" \
    "$directory/intermediate-packed1-pad4-cache" "$directory/decode-trace.json" 512 \
    >"$directory/decode-trace.log" 2>&1
nsys stats --report cuda_gpu_kern_sum --format csv --output - "$directory/decode-trace.nsys-rep" \
    >"$directory/decode-kernels.csv" 2>"$directory/decode-kernels.log"
grep -Fq 'routeScoresTensorCore' "$directory/decode-kernels.csv"
"$python" - "$directory" <<'PY'
import json,math,sys
from pathlib import Path
root=Path(sys.argv[1]);report={}
def audit_result(result):
    assert result['resident_phases']
    for step in result['steps']:
        values=step['tp_logits'];expected=step['cpu_logits']*result['batch']
        assert len(values)==len(expected)
        errors=[abs(value-reference) for value,reference in zip(values,expected)]
        assert step['within_existing_tolerance']
        assert all(math.isfinite(value) and error<=.025*(1+abs(reference))
                   for value,reference,error in zip(values,expected,errors))
        assert max(errors)==step['maximum_absolute_error']
for partition in ('expert','intermediate'):
    results=[json.loads((root/f'{partition}-packed{packed}-pad{pad}-router{router}-{phase}.json').read_text())
             for packed in (0,1) for pad in (4,8) for router in (0,1) for phase in ('cold','warm')]
    reference=[step['tp_logits'] for step in results[0]['steps']]
    for result in results: audit_result(result)
    assert all([step['tp_logits'] for step in result['steps']]==reference for result in results),partition
    report[partition]=dict(exact_all_matrices=True,runs=len(results),batch=results[0]['batch'],
        cpu_max_absolute_errors=[max(step['maximum_absolute_error'] for step in result['steps']) for result in results])
for name in ('compiled-router-mem','decode-trace'):
    observed=json.loads((root/f'{name}.json').read_text())
    audit_result(observed)
    baseline=json.loads((root/'intermediate-packed1-pad4-router0-cold.json').read_text())
    assert [step['tp_logits'] for step in observed['steps']]==[step['tp_logits'] for step in baseline['steps']]
(root/'compiled-router-parity.json').write_text(json.dumps(report,indent=2))
print(report)
PY
sha256sum --check "$directory/native.sha256"
echo 'OPT49 synthetic decode-router gates passed; pretrained quality/speed remains unproven'
