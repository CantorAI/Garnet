#!/usr/bin/env bash
# V11 regressions/BF16 decode first, then bounded token-local outer prefill.
set -euo pipefail
[[ $# == 1 && ! -e $1 ]] || exit 2
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${CANTORAI_ROOT:-$(dirname "$repo")}
build=${GARNET_BUILD_DIR:-$root/out/build/gpt-oss}
python=${GARNET_VERIFY_PYTHON:-$root/venv-vllm/bin/python}
directory=$1
mkdir -p "$directory"
directory=$(cd "$directory" && pwd)
bash "$repo/tools/gpt_oss/verify_bf16_decode.sh" "$directory/regressions"
exec 9>"$root/work/gpu-benchmark.lock"
flock -n 9 || exit 1
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] || exit 1
export XLANG3_PYTHON_LIB=$root/ThirdPartySDK/Python-3.14.0/Lib
export PYTHONPATH="$build/bin${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$build/bin:$root/ThirdPartySDK/TensorRT/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GARNET_GPT_OSS_MARLIN_MAX_TOKENS=4096 GARNET_GPT_OSS_MARLIN_PREFILL_BLOCK=32
export GARNET_GPT_OSS_MARLIN_LARGE_PREFILL_BLOCK=64 GARNET_GPT_OSS_MARLIN_PREFILL_DOWN_K=128
export GARNET_GPT_OSS_MARLIN_PREFILL_CTAS_PER_SM=1 GARNET_GPT_OSS_MARLIN_DECODE_BLOCK=32
export GARNET_GPT_OSS_ROUTER_QUERY_TILE=2 GARNET_GPT_OSS_PARALLEL_MARLIN_METADATA=1
export GARNET_GPT_OSS_PREFILL_ROUTER_TENSORCORE=1 GARNET_GPT_OSS_MARLIN_BOUNDED_PREFILL=1
git -C "$repo" rev-parse HEAD >"$directory/source-commit.txt"
sha256sum "$build/bin/libgarnet.so" "$build/bin/libgarnet_gpt_oss.so" >"$directory/native.sha256"
"$build/bin/garnet_gpt_oss_marlin_chunks" >"$directory/chunks.log" 2>&1
for spec in '4097 small' '4608 small' '7168 small' '8020 small' '4608 tp' '7168 tp'; do
    read -r rows extent <<<"$spec"
    option=--bounded-prefill-parity
    [[ $extent != tp ]] || option=--bounded-prefill-tp-sized-parity
    label=native-$rows-$extent
    "$build/bin/garnet_gpt_oss_kernel_parity" "$option" "$rows" >"$directory/$label.log" 2>&1
    compute-sanitizer --tool memcheck --error-exitcode 0 --target-processes all \
        --report-api-errors explicit --xml --print-limit 0 --print-session-details \
        --backtrace-short no --strip-paths no --save "$directory/$label.memcheck.xml" \
        "$build/bin/garnet_gpt_oss_kernel_parity" "$option" "$rows" \
        >"$directory/$label.memcheck.log" 2>&1
    "$python" "$repo/tools/gpt_oss/validate_sanitizer_xml.py" \
        "$directory/$label.memcheck.xml" "$directory/$label.memcheck-audit.json"
    echo "$label independent CPU/shards/packing/changing-graph/full memory gates passed"
done
export GARNET_GPT_OSS_TP_EXPERT_WEIGHT_SHARDS=0 GARNET_GPT_OSS_TP_MOE_INTERMEDIATE_SHARDS=1
export GARNET_GPT_OSS_TEACHER_RESIDENT=1 GARNET_GPT_OSS_TEACHER_PADDED_TOKENS=16
export GARNET_GPT_OSS_BF16_PREFILL_ALLREDUCE=1
export GARNET_GPT_OSS_PROFILE_TEACHER_STEP=-1
fixture=$directory/regressions/regressions/fixture64
for packed in 0 1; do
    export GARNET_GPT_OSS_MARLIN_PREPACKED=$packed
    for wire in 0 1; do
        export GARNET_GPT_OSS_BF16_DECODE_ALLREDUCE=$wire
        for phase in cold warm; do
            label=resident-packed$packed-wire$wire-$phase
            "$build/bin/xlang3" "$repo/test2026/gpt_oss/tp_teacher_forced.py" "$fixture" \
                "$directory/resident-packed$packed-cache" "$directory/$label.json" 512 \
                >"$directory/$label.log" 2>&1
        done
    done
done
# Instrument the compiled outer8192-row resident path without kernel filters.
export GARNET_GPT_OSS_MARLIN_PREPACKED=1 GARNET_GPT_OSS_BF16_DECODE_ALLREDUCE=1
compute-sanitizer --tool memcheck --error-exitcode 0 --target-processes all \
    --report-api-errors explicit --xml --print-limit 0 --print-session-details \
    --backtrace-short no --strip-paths no --save "$directory/resident-bounded.memcheck.xml" \
    "$build/bin/xlang3" "$repo/test2026/gpt_oss/tp_teacher_forced.py" "$fixture" \
    "$directory/resident-packed1-cache" "$directory/resident-bounded.json" 512 \
    >"$directory/resident-bounded.memcheck.log" 2>&1
"$python" "$repo/tools/gpt_oss/validate_sanitizer_xml.py" \
    "$directory/resident-bounded.memcheck.xml" "$directory/resident-bounded.memcheck-audit.json"
export GARNET_GPT_OSS_PROFILE_TEACHER_STEP=0
nsys profile --force-overwrite=false --trace=cuda --sample=none --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi --capture-range-end=stop --output="$directory/prefill-trace" \
    "$build/bin/xlang3" "$repo/test2026/gpt_oss/tp_teacher_forced.py" "$fixture" \
    "$directory/resident-packed1-cache" "$directory/prefill-trace.json" 512 \
    >"$directory/prefill-trace.log" 2>&1
nsys stats --report cuda_gpu_kern_sum --format csv --output - "$directory/prefill-trace.nsys-rep" \
    >"$directory/prefill-kernels.csv" 2>"$directory/prefill-kernels.log"
"$python" - "$directory" <<'PY'
import csv,json,sys
from pathlib import Path
root=Path(sys.argv[1])
base=json.loads((root/'regressions/intermediate-packed1-pad8-wire0-warm.json').read_text())
results=[json.loads((root/f'resident-packed{p}-wire{w}-{phase}.json').read_text())
         for p in (0,1) for w in (0,1) for phase in ('cold','warm')]
results += [json.loads((root/f'{name}.json').read_text())
            for name in ('resident-bounded','prefill-trace')]
assert all(all(s['within_existing_tolerance'] for s in r['steps']) for r in results)
matrix=[s['tp_logits'] for s in base['steps']]
assert all([s['tp_logits'] for s in r['steps']]==matrix for r in results)
config=json.loads((root/'regressions/regressions/fixture64/config.json').read_text())
rows=list(csv.reader((root/'prefill-kernels.csv').read_text().splitlines()))
header=next(row for row in rows if 'Name' in row and 'Instances' in row)
name_index=header.index('Name');count_index=header.index('Instances')
subcalls=sum(int(row[count_index]) for row in rows
             if len(row)==len(header) and 'convertInput' in row[name_index])
expected_subcalls=2*2*config['num_hidden_layers'] # two pieces, two TP ranks
assert subcalls==expected_subcalls,(subcalls,expected_subcalls)
(root/'compiled-bounded-parity.json').write_text(json.dumps(dict(exact_vs_unbounded_small_shape=True,
    runs=len(results),batch=512,outer_prefill_rows=8192,native_subcall_max_rows=4096,
    prefill_convert_input_kernel_instances=subcalls,
    cpu_max_absolute_errors=[max(s['maximum_absolute_error'] for s in r['steps']) for r in results]),indent=2))
PY
echo 'Bounded outer-prefill native/resident gates passed; pretrained speed/quality still pending'
