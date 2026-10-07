#!/usr/bin/env bash
# Sequential memory/prefill/CTA screen after recording a matched vLLM reference.
set -euo pipefail
if [[ $# != 6 ]]; then
    echo 'Usage: REQUEST_JSON EXPECTED_JSON VLLM_REFERENCE_JSON RESULT_DIR BATCH OUTPUT_TOKENS' >&2
    exit 2
fi
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${CANTORAI_ROOT:-$(dirname "$repo")}
python=${VLLM_PYTHON:-$root/venv-vllm/bin/python}
tokenizer=${GARNET_GPT_OSS_TOKENIZER:-$root/models/gpt-oss-120b-hf}
request=$1 expected=$2 reference=$3 result_dir=$4 batch=$5 output_tokens=$6
[[ -x $python && -f $expected && -f $reference ]] || exit 2
capacity=$("$python" -c '
import json,sys
reference=json.load(open(sys.argv[1])); request=json.load(open(sys.argv[2]))
assert reference["input_tokens_per_request"] == len(request["input_ids"])
assert reference["input_token_ids"] == request["input_ids"]
assert reference["batch"] == int(sys.argv[3])
assert reference["output_tokens_per_request"] == int(sys.argv[4])
assert reference["settings"]["tensor_parallel_size"] == 2
assert reference["settings"]["dtype"] == "bfloat16"
assert reference["settings"]["kv_cache_dtype"] == "auto"
assert not reference["settings"]["enable_prefix_caching"]
capacity=reference["max_context_tokens_per_request"]
assert len(request["input_ids"]) + int(sys.argv[4]) <= capacity <= 4096
print(capacity)
' "$reference" "$request" "$batch" "$output_tokens")
mkdir -p "$result_dir"
export GARNET_BATCH_CONTEXT_CAPACITY=$capacity
export GARNET_TP_RANK_LOCAL_INPUTS=1
export GARNET_GPT_OSS_TP_EXPERT_WEIGHT_SHARDS=1
export GARNET_GPT_OSS_DIRECT_ALLREDUCE=1
export GARNET_GPT_OSS_DIRECT_BATCH_ALLREDUCE=1
export GARNET_GPT_OSS_DIRECT_BATCH_CTAS=16
export GARNET_GPT_OSS_INLINE_CONTROL_KERNEL=1
export GARNET_GPT_OSS_INLINE_BATCH_CONTROLS=1
export GARNET_GPT_OSS_MARLIN_PREFILL_BLOCK=32
export GARNET_GPT_OSS_MARLIN_MAX_TOKENS=4096
export GARNET_GPT_OSS_PREFILL_TILED_64=1
export GARNET_GPT_OSS_PREFILL_QUERY_TILE=16
export GARNET_GPT_OSS_FUSED_DECODE_ROUTE=1
export GARNET_GPT_OSS_FUSED_MARLIN_LOCK_CLEAR=1
export GARNET_BATCH_PREFILL_WARMUP=1
export GARNET_BATCH_DECODE_TRIALS=3
for mode in full-cta1 chunk128-cta1 chunk128-cta2 chunk128-cta4; do
    case $mode in
        full-cta1) chunk=0; ctas=1 ;;
        chunk128-cta1) chunk=128; ctas=1 ;;
        chunk128-cta2) chunk=128; ctas=2 ;;
        chunk128-cta4) chunk=128; ctas=4 ;;
    esac
    result=$result_dir/$mode-b$batch-o$output_tokens.json
    log=$result_dir/$mode-b$batch-o$output_tokens.log
    # Preserve failures as well as successful artifacts instead of overwriting.
    [[ ! -e $result && ! -e $log ]] || { echo "Evidence already exists: $log" >&2; exit 2; }
    export GARNET_BATCH_PREFILL_CHUNK=$chunk
    export GARNET_GPT_OSS_MARLIN_CTAS_PER_SM=$ctas
    echo "Starting $mode batch=$batch output=$output_tokens context=$capacity"
    bash "$repo/tools/gpt_oss/benchmark_batch_tp2.sh" "$request" "$result" "$batch" "$output_tokens" > "$log" 2>&1
    "$python" "$repo/tools/gpt_oss/validate_batch_results.py" "$tokenizer" "$result" "$expected" "${result%.json}.validation.json"
    "$python" -c '
import json,sys
r=json.load(open(sys.argv[1]))
print("prefill_s",r["prefill_seconds"],"decode_trials_tok_s",[t["decode_aggregate_output_tokens_per_second"] for t in r["decode_trials"]],"full_tok_s",r["full_request_output_tokens_per_second"],"sampled_peak_MiB",r["sampled_peak_gpu_memory_mib"])
' "$result"
done
