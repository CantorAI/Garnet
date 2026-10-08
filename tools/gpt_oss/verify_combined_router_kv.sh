#!/usr/bin/env bash
# Combined OPT49/50 synthetic prerequisite; prior controller must be terminal.
set -euo pipefail
[[ $# == 1 && ! -e $1 ]] || exit 2
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${CANTORAI_ROOT:-$(dirname "$repo")}
build=${GARNET_BUILD_DIR:-$root/out/build/gpt-oss}
python=${GARNET_VERIFY_PYTHON:-$root/venv-vllm/bin/python}
exec 9>"$root/work/gpu-benchmark.lock"
flock -n 9 || exit 1
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] || exit 1
command -v compute-sanitizer >/dev/null
command -v nsys >/dev/null
[[ -z $(git -C "$repo" status --porcelain) ]] || exit 1
mkdir -p "$1"
directory=$(cd "$1" && pwd)
phase=metadata
terminal() {
    local status=$?
    trap - EXIT
    "$python" - "$directory" "$phase" "$status" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])/'terminal.json';assert not p.exists()
p.write_text(json.dumps(dict(terminal=True,phase=sys.argv[2],actual_exit=int(sys.argv[3])),indent=2)+'\n')
PY
    exit "$status"
}
trap terminal EXIT
# Start from the declared experiment policy, not inherited optimization flags.
while IFS= read -r variable; do
    case "$variable" in
        GARNET_GPT_OSS_*|GARNET_TP_*|GARNET_BATCH_*|GARNET_RESIDENT_*|GARNET_TRT_SYNC_ALLOCATOR|GARNET_TRT_LOG_ENGINE_MEMORY)
            unset "$variable" ;;
    esac
done < <(compgen -e)
export GARNET_TRT_SYNC_ALLOCATOR=1
export XLANG3_PYTHON_LIB=$root/ThirdPartySDK/Python-3.14.0/Lib
export PYTHONPATH="$build/bin${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$build/bin:$root/ThirdPartySDK/TensorRT/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GARNET_GPT_OSS_MARLIN_PREPACKED=1 GARNET_GPT_OSS_MARLIN_MAX_TOKENS=4096
export GARNET_GPT_OSS_MARLIN_PREFILL_BLOCK=32 GARNET_GPT_OSS_MARLIN_DECODE_BLOCK=32
export GARNET_GPT_OSS_MARLIN_LARGE_PREFILL_BLOCK=0 GARNET_GPT_OSS_MARLIN_BOUNDED_PREFILL=0
export GARNET_GPT_OSS_BF16_PREFILL_ALLREDUCE=1 GARNET_GPT_OSS_BF16_DECODE_ALLREDUCE=1
export GARNET_GPT_OSS_PREFILL_ROUTER_TENSORCORE=1 GARNET_GPT_OSS_PARALLEL_MARLIN_METADATA=1
export GARNET_GPT_OSS_ROUTER_QUERY_TILE=2 GARNET_GPT_OSS_ROUTER_THREADS=256 GARNET_GPT_OSS_ROUTER_VECTOR4=1
export GARNET_GPT_OSS_PREFILL_FLASHINFER=1 GARNET_GPT_OSS_DECODE_FLASHINFER=1
export GARNET_GPT_OSS_PREFILL_FAST_EXP=0 GARNET_GPT_OSS_PREFILL_TILED_64=0
export GARNET_GPT_OSS_PROFILE_HYBRID_KV=0
unset GARNET_GPT_OSS_COMPACT_VOCAB_GREEDY
"$python" "$repo/test2026/gpt_oss/make_fixture.py" "$directory/fixture" \
    --intermediate 64 --prompt-length 65 --window 17 --tp-gqa8 >"$directory/fixture.log" 2>&1
mkdir "$directory/native-payload"
cp -- "$build/bin/libgarnet.so" "$build/bin/libgarnet_gpt_oss.so" "$directory/native-payload/"
sha256sum "$build/bin/libgarnet.so" "$build/bin/libgarnet_gpt_oss.so" >"$directory/native.sha256"
"$python" - "$repo" "$build" "$directory" <<'PY'
import hashlib,json,sys,subprocess
from pathlib import Path
repo,build,out=map(Path,sys.argv[1:]);sys.path.insert(0,str(repo/'tools/gpt_oss'))
from resident_budget import native_identity,kernel_environment,hardware_identity
env=kernel_environment()
for key in ('GARNET_GPT_OSS_TP_EXPERT_WEIGHT_SHARDS','GARNET_GPT_OSS_TP_MOE_INTERMEDIATE_SHARDS',
            'GARNET_GPT_OSS_DECODE_ROUTER_TENSORCORE','GARNET_GPT_OSS_HYBRID_KV'):env.pop(key,None)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
(out/'gate-metadata.json').write_text(json.dumps(dict(directory=str(out),
 source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(repo),text=True).strip(),
 native_binaries=native_identity(build),hardware_csv=hardware_identity(),common_environment=env,
 fixture_sha256={n:sha(out/'fixture'/n) for n in ('model.safetensors','config.json','request.json','expected.json')}),indent=2)+'\n')
PY
for axis in expert intermediate; do
    if [[ $axis == expert ]]; then
        export GARNET_GPT_OSS_TP_EXPERT_WEIGHT_SHARDS=1 GARNET_GPT_OSS_TP_MOE_INTERMEDIATE_SHARDS=0
        batch=128; chunk=16
    else
        export GARNET_GPT_OSS_TP_EXPERT_WEIGHT_SHARDS=0 GARNET_GPT_OSS_TP_MOE_INTERMEDIATE_SHARDS=1
        batch=512; chunk=8
    fi
    for router in 0 1; do
        export GARNET_GPT_OSS_DECODE_ROUTER_TENSORCORE=$router
        for hybrid in 0 1; do
            export GARNET_GPT_OSS_HYBRID_KV=$hybrid
            for temperature in cold warm; do
                phase=$axis-router$router-hybrid$hybrid-$temperature
                "$build/bin/xlang3" "$repo/test2026/gpt_oss/tp_hybrid_kv.py" "$directory/fixture" \
                    "$directory/$axis-router$router-hybrid$hybrid-cache" "$directory/$phase.json" \
                    "$batch" "$chunk" >"$directory/$phase.log" 2>&1
            done
        done
    done
done
export GARNET_GPT_OSS_DECODE_ROUTER_TENSORCORE=1 GARNET_GPT_OSS_HYBRID_KV=1
phase=compiled-combined-mem
compute-sanitizer --tool memcheck --error-exitcode 0 --target-processes all --launch-timeout 0 \
    --report-api-errors explicit --xml --print-limit 0 --print-session-details \
    --backtrace-short no --strip-paths no --save "$directory/$phase.xml" \
    "$build/bin/xlang3" "$repo/test2026/gpt_oss/tp_hybrid_kv.py" "$directory/fixture" \
    "$directory/intermediate-router1-hybrid1-cache" "$directory/$phase.json" 512 8 \
    >"$directory/$phase.log" 2>&1
if grep -Eq 'No attachable process found|compute-sanitizer timed-out|========= Error:|Target application returned an error' "$directory/$phase.log"; then
    echo 'Incomplete sanitizer/application coverage' >&2; exit 1
fi
grep -Fq 'ERROR SUMMARY:' "$directory/$phase.log"
"$python" "$repo/tools/gpt_oss/validate_sanitizer_xml.py" "$directory/$phase.xml" "$directory/$phase.audit.json"
phase=combined-trace
export GARNET_GPT_OSS_PROFILE_HYBRID_KV=1
nsys profile --force-overwrite=false --trace=cuda --sample=none --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi --capture-range-end=stop --output="$directory/$phase" \
    "$build/bin/xlang3" "$repo/test2026/gpt_oss/tp_hybrid_kv.py" "$directory/fixture" \
    "$directory/intermediate-router1-hybrid1-cache" "$directory/$phase.json" 512 8 \
    >"$directory/$phase.log" 2>&1
nsys stats --report cuda_gpu_kern_sum --format csv --output - "$directory/$phase.nsys-rep" \
    >"$directory/combined-kernels.csv" 2>"$directory/combined-kernels.log"
grep -Fq 'SinkAttention' "$directory/combined-kernels.csv"
grep -Fq 'routeScoresTensorCore' "$directory/combined-kernels.csv"
sha256sum --check "$directory/native.sha256"
phase=complete-raw-combined
echo 'Combined raw compiled application/memory/trace phases complete; independent audit required; no performance claim'
