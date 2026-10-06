#!/usr/bin/env bash
# Run the saved GPT-OSS TP2 prompts alone on a Linux GPU host.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${CANTORAI_ROOT:-$(dirname "$repo")}
work=${GARNET_BENCH_WORK_DIR:-$root/work}
output=${GARNET_BENCH_OUTPUT:?Set GARNET_BENCH_OUTPUT to a new results directory}
trials=${GARNET_BENCH_TRIALS:-2}
cases=${GARNET_BENCH_CASES:-arithmetic code-tracing instruction-following long-context-retrieval}
runtime=${XLANG3_RUNTIME_BIN:-$root/out/build/xlang3/bin/xlang3}
build=${GARNET_BUILD_DIR:-$root/out/build/gpt-oss}
weights=${GARNET_GPT_OSS_WEIGHTS:-$root/models/gpt-oss-120b/original}
cache=${GARNET_GPT_OSS_CACHE:-$work/gpt-oss-tp-cache}
tensorrt=${GARNET_TENSORRT_ROOT:-$root/ThirdPartySDK/TensorRT}

[[ $trials =~ ^[1-9][0-9]*$ ]] || { echo "Invalid trial count: $trials" >&2; exit 2; }
[[ -x $runtime && -d $weights ]] || { echo 'Runtime or weights are missing' >&2; exit 2; }
[[ ! -e $output ]] || { echo "Results already exist: $output" >&2; exit 2; }
mkdir -p "$work" "$output"
exec 9>"$work/gpu-benchmark.lock"
flock -n 9 || { echo 'Another benchmark owns the GPUs' >&2; exit 1; }
if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]; then
    echo 'Another GPU compute process is active' >&2
    exit 1
fi

export XLANG3_PYTHON_LIB=${XLANG3_PYTHON_LIB:-$root/ThirdPartySDK/Python-3.14.0/Lib}
export PYTHONPATH="$build/bin${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$build/bin:$tensorrt/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
git -C "$repo" rev-parse HEAD >"$output/revision.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv >"$output/gpus.csv"
env | grep -E '^(GARNET_[^=]*|XLANG3_PYTHON_LIB|PYTHONPATH|LD_LIBRARY_PATH)=' | sort >"$output/environment.txt"

for case in $cases; do
    case "$case" in
        arithmetic|code-tracing|instruction-following|long-context-retrieval) ;;
        *) echo "Unknown benchmark case: $case" >&2; exit 2 ;;
    esac
    request="$work/prompt-benchmarks/$case/request.json"
    [[ $case != long-context-retrieval ]] || request="$work/long-prompt-test/request.json"
    [[ -f $request ]] || { echo "Missing request: $request" >&2; exit 2; }
    for ((trial=0; trial<trials; trial++)); do
        result="$output/$case-$trial"
        echo "Starting $case trial $trial"
        "$runtime" "$repo/tools/gpt_oss/run_tp_tokens.py" \
            "$weights" "$cache" "$request" "$result.json" >"$result.log" 2>&1
        python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(sys.argv[1], "tokens", len(r["token_ids"]), "prefill_s", r["prefill_seconds"], "warm_decode_wall_tok_s", r["warm_decode_wall_tokens_per_second"])' "$result.json"
    done
done
