#!/usr/bin/env bash
# Screen shared paged GQA decode against a previously recorded reference.
set -euo pipefail
if [[ $# != 7 ]]; then
    echo 'Usage: REQUEST_JSON EXPECTED_JSON VLLM_REFERENCE_JSON RESULT_DIR BATCH OUTPUT_TOKENS PREFILL_CHUNK' >&2
    exit 2
fi
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${CANTORAI_ROOT:-$(dirname "$repo")}
python=${VLLM_PYTHON:-$root/venv-vllm/bin/python}
tokenizer=${GARNET_GPT_OSS_TOKENIZER:-$root/models/gpt-oss-120b-hf}
request=$1 expected=$2 reference=$3 result_dir=$4 batch=$5 output_tokens=$6 chunk=$7
[[ -x $python && -f $expected && -f $reference ]] || exit 2
capacity=$("$python" -c '
import json,sys
ref=json.load(open(sys.argv[1])); request=json.load(open(sys.argv[2]))
batch,output,chunk=map(int,sys.argv[3:6])
assert 128 == batch and 16 <= output <= 512 and 0 < chunk <= len(request["input_ids"])
assert batch*chunk <= 4096 and len(request["input_ids"]) % chunk == 0
assert ref["input_token_ids"] == request["input_ids"]
assert ref["batch"] == batch and ref["output_tokens_per_request"] == output
assert ref["settings"]["tensor_parallel_size"] == 2
assert ref["settings"]["dtype"] == "bfloat16" and ref["settings"]["kv_cache_dtype"] == "auto"
assert not ref["settings"]["enable_prefix_caching"]
assert len(request["input_ids"]) + output <= ref["max_context_tokens_per_request"] <= 4096
print(ref["max_context_tokens_per_request"])
' "$reference" "$request" "$batch" "$output_tokens" "$chunk")
mkdir -p "$result_dir"
export GARNET_BATCH_CONTEXT_CAPACITY=$capacity
export GARNET_BATCH_PREFILL_CHUNK=$chunk
export GARNET_TP_RANK_LOCAL_INPUTS=1
export GARNET_GPT_OSS_TP_EXPERT_WEIGHT_SHARDS=1
export GARNET_GPT_OSS_DIRECT_ALLREDUCE=1
export GARNET_GPT_OSS_DIRECT_BATCH_ALLREDUCE=1
export GARNET_GPT_OSS_DIRECT_LARGE_BATCH_ALLREDUCE=1
export GARNET_GPT_OSS_DIRECT_BATCH_CTAS=32
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
export GARNET_GPT_OSS_MARLIN_DECODE_BLOCK=32
export GARNET_GPT_OSS_MARLIN_CTAS_PER_SM=1
export GARNET_GPT_OSS_DECODE_GQA_TILED=1
for splits in 4 8 16; do
    export GARNET_GPT_OSS_DECODE_GQA_SPLITS=$splits
    result=$result_dir/gqa$splits-b$batch-o$output_tokens.json
    log=${result%.json}.log
    [[ ! -e $result && ! -e $log ]] || { echo "Evidence already exists: $log" >&2; exit 2; }
    echo "Starting GQA splits=$splits decode_block=32 ctas=1 batch=$batch output=$output_tokens context=$capacity chunk=$chunk"
    bash "$repo/tools/gpt_oss/benchmark_batch_tp2.sh" "$request" "$result" "$batch" "$output_tokens" > "$log" 2>&1
    "$python" "$repo/tools/gpt_oss/validate_batch_results.py" "$tokenizer" "$result" "$expected" "${result%.json}.validation.json"
    "$python" -c '
import json,sys
r=json.load(open(sys.argv[1]))
print("prefill_s",r["prefill_seconds"],"decode_trials_tok_s",[t["decode_aggregate_output_tokens_per_second"] for t in r["decode_trials"]],"full_tok_s",r["full_request_output_tokens_per_second"],"sampled_peak_MiB",r["sampled_peak_gpu_memory_mib"],"KV_allocated_bytes",r["kv_cache_allocated_bytes_per_gpu"],"KV_logical_bytes",r["logical_kv_bytes_per_gpu_at_completion"])
' "$result"
done