#!/usr/bin/env bash
# V10 default regressions, then eligible decode transport with shared KV.
# This is synthetic correctness/memory evidence, not pretrained throughput.
set -euo pipefail
[[ $# == 1 && ! -e $1 ]] || exit 2
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${CANTORAI_ROOT:-$(dirname "$repo")}
build=${GARNET_BUILD_DIR:-$root/out/build/gpt-oss}
python=${GARNET_VERIFY_PYTHON:-$root/venv-vllm/bin/python}
directory=$1
mkdir -p "$directory"
directory=$(cd "$directory" && pwd)
export GARNET_GPT_OSS_BF16_DECODE_ALLREDUCE=0
export GARNET_GPT_OSS_MARLIN_BOUNDED_PREFILL=0
bash "$repo/tools/gpt_oss/verify_resident_tp2.sh" "$directory/regressions"
exec 9>"$root/work/gpu-benchmark.lock"
flock -n 9 || exit 1
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] || exit 1
command -v compute-sanitizer >/dev/null
command -v nsys >/dev/null
export XLANG3_PYTHON_LIB=$root/ThirdPartySDK/Python-3.14.0/Lib
export PYTHONPATH="$build/bin${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$build/bin:$root/ThirdPartySDK/TensorRT/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GARNET_GPT_OSS_MARLIN_MAX_TOKENS=4096 GARNET_GPT_OSS_MARLIN_PREFILL_BLOCK=32
export GARNET_GPT_OSS_MARLIN_DECODE_BLOCK=32 GARNET_GPT_OSS_ROUTER_QUERY_TILE=2
export GARNET_GPT_OSS_PARALLEL_MARLIN_METADATA=1 GARNET_GPT_OSS_PREFILL_ROUTER_TENSORCORE=1
export GARNET_GPT_OSS_BF16_PREFILL_ALLREDUCE=1 GARNET_GPT_OSS_TEACHER_RESIDENT=1
git -C "$repo" rev-parse HEAD >"$directory/source-commit.txt"
sha256sum "$build/bin/libgarnet.so" "$build/bin/libgarnet_gpt_oss.so" >"$directory/native.sha256"
"$build/bin/garnet_gpt_oss_tp_workspace_test" >"$directory/workspace.log" 2>&1
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
            # Same engine cache across flags; workspace is NOT runtime-dependent.
            cache=$directory/$partition-packed$packed-pad$pad-cache
            for wire in 0 1; do
                export GARNET_GPT_OSS_BF16_DECODE_ALLREDUCE=$wire
                for phase in cold warm; do
                    label=$partition-packed$packed-pad$pad-wire$wire-$phase
                    export GARNET_GPT_OSS_PROFILE_TEACHER_STEP=-1
                    "$build/bin/xlang3" "$repo/test2026/gpt_oss/tp_teacher_forced.py" "$fixture" \
                        "$cache" "$directory/$label.json" "$batch" >"$directory/$label.log" 2>&1
                    echo "$label independent CPU-prefix/resident/refit check passed"
                done
            done
        done
    done
done
# Unrestricted memory instrumentation of the actual compiled resident candidate.
export GARNET_GPT_OSS_MARLIN_PREPACKED=1 GARNET_GPT_OSS_TEACHER_PADDED_TOKENS=4
export GARNET_GPT_OSS_BF16_DECODE_ALLREDUCE=1 GARNET_GPT_OSS_PROFILE_TEACHER_STEP=-1
fixture=$directory/regressions/fixture64
compute-sanitizer --tool memcheck --error-exitcode 0 --target-processes all \
    --report-api-errors explicit --xml --print-limit 0 --print-session-details \
    --backtrace-short no --strip-paths no --save "$directory/resident-bf16.memcheck.xml" \
    "$build/bin/xlang3" "$repo/test2026/gpt_oss/tp_teacher_forced.py" "$fixture" \
    "$directory/intermediate-packed1-pad4-cache" "$directory/resident-bf16.json" 512 \
    >"$directory/resident-bf16.memcheck.log" 2>&1
"$python" "$repo/tools/gpt_oss/validate_sanitizer_xml.py" \
    "$directory/resident-bf16.memcheck.xml" "$directory/resident-bf16.memcheck-audit.json"
# Capture decode alone to prove candidate pack/reduce/unpack actually executes.
export GARNET_GPT_OSS_PROFILE_TEACHER_STEP=1
nsys profile --force-overwrite=false --trace=cuda --sample=none --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi --capture-range-end=stop --output="$directory/decode-trace" \
    "$build/bin/xlang3" "$repo/test2026/gpt_oss/tp_teacher_forced.py" "$fixture" \
    "$directory/intermediate-packed1-pad4-cache" "$directory/decode-trace.json" 512 \
    >"$directory/decode-trace.log" 2>&1
nsys stats --report cuda_gpu_kern_sum --format csv --output - "$directory/decode-trace.nsys-rep" \
    >"$directory/decode-kernels.csv" 2>"$directory/decode-kernels.log"
grep -Fq 'packBf16' "$directory/decode-kernels.csv"
grep -Fq 'unpackBf16' "$directory/decode-kernels.csv"
"$python" - "$directory" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);report={}
for partition in ('expert','intermediate'):
    results=[json.loads((root/f'{partition}-packed{p}-pad{pad}-wire{w}-{phase}.json').read_text())
             for p in (0,1) for pad in (4,8) for w in (0,1) for phase in ('cold','warm')]
    reference=[s['tp_logits'] for s in results[0]['steps']]
    assert all(r['resident_phases'] and all(s['within_existing_tolerance'] for s in r['steps']) for r in results)
    assert all([s['tp_logits'] for s in r['steps']]==reference for r in results),partition
    report[partition]=dict(exact_all_matrices=True,runs=len(results),batch=results[0]['batch'],
        cpu_max_absolute_errors=[max(s['maximum_absolute_error'] for s in r['steps']) for r in results])
for name in ('resident-bf16','decode-trace'):
    observed=json.loads((root/f'{name}.json').read_text())
    expected=json.loads((root/'intermediate-packed1-pad4-wire0-cold.json').read_text())
    assert [s['tp_logits'] for s in observed['steps']]==[s['tp_logits'] for s in expected['steps']]
(root/'compiled-decode-parity.json').write_text(json.dumps(report,indent=2))
print(report)
PY
echo 'V10 synthetic resident BF16 decode gates passed; pretrained proof still required'
