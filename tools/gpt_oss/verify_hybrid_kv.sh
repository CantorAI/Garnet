#!/usr/bin/env bash
# OPT50 target correctness only. Existing benchmark controller must be terminal.
set -euo pipefail
[[ $# == 1 && ! -e $1 ]] || exit 2
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${CANTORAI_ROOT:-$(dirname "$repo")}
build=${GARNET_BUILD_DIR:-$root/out/build/gpt-oss}
python=${GARNET_VERIFY_PYTHON:-$root/venv-vllm/bin/python}
directory=$1
mkdir -p "$directory"
directory=$(cd "$directory" && pwd)
export GARNET_GPT_OSS_HYBRID_KV=0 GARNET_GPT_OSS_DECODE_ROUTER_TENSORCORE=0
export GARNET_GPT_OSS_PROFILE_HYBRID_KV=0
export GARNET_GPT_OSS_BF16_DECODE_ALLREDUCE=0 GARNET_GPT_OSS_MARLIN_BOUNDED_PREFILL=0
bash "$repo/tools/gpt_oss/verify_resident_tp2.sh" "$directory/regressions"
exec 9>"$root/work/gpu-benchmark.lock"
flock -n 9 || exit 1
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] || exit 1
export XLANG3_PYTHON_LIB=$root/ThirdPartySDK/Python-3.14.0/Lib
export PYTHONPATH="$build/bin${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$build/bin:$root/ThirdPartySDK/TensorRT/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
git -C "$repo" rev-parse HEAD >"$directory/source.txt"
sha256sum "$build/bin/libgarnet.so" "$build/bin/libgarnet_gpt_oss.so" \
    "$build/bin/garnet_gpt_oss_kernel_parity" >"$directory/native.sha256"
nvidia-smi --query-gpu=index,name,uuid,driver_version --format=csv >"$directory/hardware.csv"

audit_memory() {
    local label=$1
    if grep -Eq 'No attachable process found|compute-sanitizer timed-out|========= Error:' "$directory/$label.log"; then
        echo "Incomplete sanitizer coverage: $label" >&2; exit 1
    fi
    grep -Fq 'ERROR SUMMARY:' "$directory/$label.log"
    "$python" "$repo/tools/gpt_oss/validate_sanitizer_xml.py" \
        "$directory/$label.xml" "$directory/$label.audit.json"
}
export GARNET_GPT_OSS_PREFILL_FLASHINFER=0 GARNET_GPT_OSS_DECODE_FLASHINFER=0
export GARNET_GPT_OSS_PREFILL_FAST_EXP=0 GARNET_GPT_OSS_PREFILL_TILED_64=0
"$build/bin/garnet_gpt_oss_kernel_parity" --hybrid-kv-attention-parity \
    >"$directory/native-ring.log" 2>&1
compute-sanitizer --tool memcheck --error-exitcode 0 --target-processes all --launch-timeout 0 \
    --report-api-errors explicit --xml --print-limit 0 --print-session-details \
    --backtrace-short no --strip-paths no --save "$directory/native-ring-mem.xml" \
    "$build/bin/garnet_gpt_oss_kernel_parity" --hybrid-kv-attention-parity \
    >"$directory/native-ring-mem.log" 2>&1
audit_memory native-ring-mem

# New independent dense CPU fixtures include many ring wraps and a padded
# final tile. Scalar and real FlashInfer TP2 paths get separate cache roots.
"$python" "$repo/test2026/gpt_oss/make_fixture.py" "$directory/scalar-fixture" \
    --intermediate 64 --prompt-length 65 --window 17 >"$directory/scalar-fixture.log" 2>&1
"$python" "$repo/test2026/gpt_oss/make_fixture.py" "$directory/flash-fixture" \
    --intermediate 64 --prompt-length 65 --window 17 --tp-gqa8 >"$directory/flash-fixture.log" 2>&1
export GARNET_GPT_OSS_TP_EXPERT_WEIGHT_SHARDS=0 GARNET_GPT_OSS_TP_MOE_INTERMEDIATE_SHARDS=1
export GARNET_GPT_OSS_MARLIN_PREPACKED=1 GARNET_GPT_OSS_MARLIN_MAX_TOKENS=4096
export GARNET_GPT_OSS_MARLIN_PREFILL_BLOCK=32 GARNET_GPT_OSS_MARLIN_DECODE_BLOCK=32
export GARNET_GPT_OSS_MARLIN_LARGE_PREFILL_BLOCK=0 GARNET_GPT_OSS_BF16_PREFILL_ALLREDUCE=1
export GARNET_GPT_OSS_PREFILL_ROUTER_TENSORCORE=1 GARNET_GPT_OSS_PARALLEL_MARLIN_METADATA=1
for surface in scalar flash; do
    if [[ $surface == flash ]]; then attention=1; else attention=0; fi
    export GARNET_GPT_OSS_PREFILL_FLASHINFER=$attention GARNET_GPT_OSS_DECODE_FLASHINFER=$attention
    for hybrid in 0 1; do
        export GARNET_GPT_OSS_HYBRID_KV=$hybrid
        for chunk in 16 32; do
            for phase in cold warm; do
                label=$surface-hybrid$hybrid-chunk$chunk-$phase
                "$build/bin/xlang3" "$repo/test2026/gpt_oss/tp_hybrid_kv.py" \
                    "$directory/$surface-fixture" "$directory/$surface-hybrid$hybrid-chunk$chunk-cache" \
                    "$directory/$label.json" 16 "$chunk" >"$directory/$label.log" 2>&1
            done
        done
    done
done
export GARNET_GPT_OSS_HYBRID_KV=1 GARNET_GPT_OSS_PREFILL_FLASHINFER=1 GARNET_GPT_OSS_DECODE_FLASHINFER=1
compute-sanitizer --tool memcheck --error-exitcode 0 --target-processes all --launch-timeout 0 \
    --report-api-errors explicit --xml --print-limit 0 --print-session-details \
    --backtrace-short no --strip-paths no --save "$directory/compiled-ring-mem.xml" \
    "$build/bin/xlang3" "$repo/test2026/gpt_oss/tp_hybrid_kv.py" "$directory/flash-fixture" \
    "$directory/flash-hybrid1-chunk32-cache" "$directory/compiled-ring-mem.json" 16 32 \
    >"$directory/compiled-ring-mem.log" 2>&1
audit_memory compiled-ring-mem
export GARNET_GPT_OSS_PROFILE_HYBRID_KV=1
nsys profile --force-overwrite=false --trace=cuda --sample=none --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi --capture-range-end=stop --output="$directory/flash-ring-trace" \
    "$build/bin/xlang3" "$repo/test2026/gpt_oss/tp_hybrid_kv.py" "$directory/flash-fixture" \
    "$directory/flash-hybrid1-chunk32-cache" "$directory/flash-ring-trace.json" 16 32 \
    >"$directory/flash-ring-trace.log" 2>&1
nsys stats --report cuda_gpu_kern_sum --format csv --output - "$directory/flash-ring-trace.nsys-rep" \
    >"$directory/flash-ring-kernels.csv" 2>"$directory/flash-ring-kernels.log"
grep -Fq 'SinkAttention' "$directory/flash-ring-kernels.csv"
"$python" - "$directory" <<'PY'
import csv,json,math,sys
from pathlib import Path
root=Path(sys.argv[1]);report={}
def audit(result):
    assert result['complete_generations']==2 and result['input_tokens']==65
    assert len(result['steps'])==6
    for step in result['steps']:
        values=step['tp_logits'];expected=step['cpu_logits']*result['batch']
        assert len(values)==len(expected)
        errors=[abs(a-b) for a,b in zip(values,expected)]
        assert step['within_existing_tolerance'] and max(errors)==step['maximum_absolute_error']
        assert all(math.isfinite(a) and e<=.025*(1+abs(b)) for a,b,e in zip(values,expected,errors))
    assert [s['tp_logits'] for s in result['steps'][:3]]==[s['tp_logits'] for s in result['steps'][3:]]
for surface in ('scalar','flash'):
    for chunk in (16,32):
        results=[json.loads((root/f'{surface}-hybrid{hybrid}-chunk{chunk}-{phase}.json').read_text())
            for hybrid in (0,1) for phase in ('cold','warm')]
        for result in results:audit(result)
        matrices=[[s['tp_logits'] for s in r['steps']] for r in results]
        assert all(matrix==matrices[0] for matrix in matrices)
        assert all(r['shared_auxiliary_bytes']>0 for r in results[2:])
        assert all(r['shared_kv_bytes']<results[0]['shared_kv_bytes'] for r in results[2:])
        report[f'{surface}-chunk{chunk}']=dict(exact_original_hybrid_cold_warm=True,
            cpu_max_absolute_errors=[max(s['maximum_absolute_error'] for s in r['steps']) for r in results],
            original_kv_bytes=results[0]['shared_kv_bytes'],hybrid_kv_bytes=results[2]['shared_kv_bytes'],
            hybrid_auxiliary_bytes=results[2]['shared_auxiliary_bytes'])
extra=json.loads((root/'compiled-ring-mem.json').read_text());audit(extra)
baseline=json.loads((root/'flash-hybrid0-chunk32-cold.json').read_text())
assert [s['tp_logits'] for s in extra['steps']]==[s['tp_logits'] for s in baseline['steps']]
trace=json.loads((root/'flash-ring-trace.json').read_text());audit(trace)
assert trace['profiled_diagnostic']
assert [s['tp_logits'] for s in trace['steps']]==[s['tp_logits'] for s in baseline['steps']]
header=None;calls=0
for row in csv.reader((root/'flash-ring-kernels.csv').read_text().splitlines()):
    if 'Name' in row and 'Instances' in row:header=row;continue
    if header is not None and len(row)==len(header) and 'SinkAttention' in row[header.index('Name')]:
        calls+=int(row[header.index('Instances')])
assert calls>=8,('Missing both-layer/both-rank last-prefill and first-decode FlashInfer dispatch',calls)
report['compiled_flash_sink_instances']=calls
(root/'independent-ring-parity.json').write_text(json.dumps(report,indent=2))
print(report)
PY
sha256sum --check "$directory/native.sha256"
echo 'OPT50 synthetic bank/wrap/FlashInfer/default/memory gates passed; measured resident profile and pretrained speed/quality remain unproven'
