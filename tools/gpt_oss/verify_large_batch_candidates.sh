#!/usr/bin/env bash
# OPT42: independent candidates first; never use this gate as a model win.
set -euo pipefail
[[ $# == 1 && ! -e $1 ]] || exit 2
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${CANTORAI_ROOT:-$(dirname "$repo")}
build=${GARNET_BUILD_DIR:-$root/out/build/gpt-oss}
python=${GARNET_VERIFY_PYTHON:-$root/venv-vllm/bin/python}
directory=$1
unset GARNET_GPT_OSS_MARLIN_LARGE_PREFILL_BLOCK GARNET_GPT_OSS_MARLIN_PREFILL_CTAS_PER_SM
unset GARNET_GPT_OSS_DIRECT_MAX_BATCH GARNET_GPT_OSS_DIRECT_ALLREDUCE
bash "$repo/tools/gpt_oss/verify_resident_tp2.sh" "$directory"
directory=$(cd "$directory" && pwd)
exec 9>"$root/work/gpu-benchmark.lock"
flock -n 9 || exit 1
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] || exit 1
export XLANG3_PYTHON_LIB=$root/ThirdPartySDK/Python-3.14.0/Lib
export PYTHONPATH="$build/bin${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$build/bin:$root/ThirdPartySDK/TensorRT/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GARNET_GPT_OSS_MARLIN_MAX_TOKENS=4096 GARNET_GPT_OSS_MARLIN_PREFILL_BLOCK=32
export GARNET_GPT_OSS_MARLIN_DECODE_BLOCK=32 GARNET_GPT_OSS_PARALLEL_MARLIN_METADATA=1
git -C "$repo" rev-parse HEAD >"$directory/candidate-source.txt"
sha256sum "$build/bin/libgarnet.so" "$build/bin/libgarnet_gpt_oss.so" \
    "$build/bin/garnet_gpt_oss_kernel_parity" "$build/bin/garnet_gpt_oss_tp_direct_benchmark" \
    >"$directory/candidate-binaries.sha256"
for tile in 0 64; do
    export GARNET_GPT_OSS_MARLIN_LARGE_PREFILL_BLOCK=$tile
    for ctas in 1 2 4; do
        export GARNET_GPT_OSS_MARLIN_PREFILL_CTAS_PER_SM=$ctas
        label=large-prefill-tile$tile-cta$ctas
        "$build/bin/garnet_gpt_oss_kernel_parity" --large-prefill-parity >"$directory/$label.log" 2>&1
        compute-sanitizer --tool memcheck --error-exitcode 0 --target-processes all \
            --report-api-errors explicit --xml --print-limit 0 --print-session-details \
            --save "$directory/$label.memcheck.xml" "$build/bin/garnet_gpt_oss_kernel_parity" \
            --large-prefill-parity >"$directory/$label.memcheck.log" 2>&1
        "$python" "$repo/tools/gpt_oss/validate_sanitizer_xml.py" \
            "$directory/$label.memcheck.xml" "$directory/$label.memcheck-audit.json"
        echo "$label independent CPU/shard/packing and strict memcheck passed"
    done
done
# Full model fixtures exercise resident padded tails, future KV overwrite and
# both TP partitions. Compare every CPU-prefix logit; retain any tile changes.
export GARNET_GPT_OSS_ROUTER_QUERY_TILE=2 GARNET_GPT_OSS_PREFILL_ROUTER_TENSORCORE=1
export GARNET_GPT_OSS_BF16_PREFILL_ALLREDUCE=1 GARNET_GPT_OSS_MARLIN_PREPACKED=1
export GARNET_GPT_OSS_TEACHER_RESIDENT=1
for partition in expert intermediate; do
    if [[ $partition == expert ]]; then
        export GARNET_GPT_OSS_TP_EXPERT_WEIGHT_SHARDS=1 GARNET_GPT_OSS_TP_MOE_INTERMEDIATE_SHARDS=0
        fixture=$directory/full-parity/fixture; batch=256
    else
        export GARNET_GPT_OSS_TP_EXPERT_WEIGHT_SHARDS=0 GARNET_GPT_OSS_TP_MOE_INTERMEDIATE_SHARDS=1
        fixture=$directory/fixture64; batch=512
    fi
    for tile in 0 64; do
        export GARNET_GPT_OSS_MARLIN_LARGE_PREFILL_BLOCK=$tile
        for ctas in 1 2 4; do
            export GARNET_GPT_OSS_MARLIN_PREFILL_CTAS_PER_SM=$ctas
            for tokens in 4 8; do
                export GARNET_GPT_OSS_TEACHER_PADDED_TOKENS=$tokens
                for phase in cold warm; do
                    label=large-resident-$partition-tile$tile-cta$ctas-pad$tokens-$phase
                    "$build/bin/xlang3" "$repo/test2026/gpt_oss/tp_teacher_forced.py" "$fixture" \
                        "$directory/large-$partition-tile$tile-cta$ctas-pad$tokens-cache" \
                        "$directory/$label.json" "$batch" >"$directory/$label.log" 2>&1
                    echo "$label passed"
                done
            done
        done
    done
done
"$python" - "$directory" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); report={}
for partition in ('expert','intermediate'):
    rows={}
    for tile in (0,64):
        for ctas in (1,2,4):
            for pad in (4,8):
                pair=[json.loads((root/f'large-resident-{partition}-tile{tile}-cta{ctas}-pad{pad}-{p}.json').read_text()) for p in ('cold','warm')]
                assert all(s['within_existing_tolerance'] for r in pair for s in r['steps'])
                matrices=[[s['tp_logits'] for s in r['steps']] for r in pair]
                assert matrices[0]==matrices[1], (partition,tile,ctas,pad)
                rows[f'{tile}/{ctas}/{pad}']=dict(exact_cold_warm=True,
                    max_cpu_abs=max(s['maximum_absolute_error'] for r in pair for s in r['steps']),
                    exact_to_tile32_cta1=matrices[0]==rows.get(f'0/1/{pad}',{}).get('matrices',matrices[0]),
                    matrices=matrices[0])
    report[partition]=rows
(root/'large-resident-quality.json').write_text(json.dumps(report,indent=2))
PY
unset GARNET_GPT_OSS_MARLIN_LARGE_PREFILL_BLOCK GARNET_GPT_OSS_MARLIN_PREFILL_CTAS_PER_SM
for batch in 128 256 512; do
    export GARNET_GPT_OSS_DIRECT_MAX_BATCH=$batch
    GARNET_GPT_OSS_DIRECT_ALLREDUCE=0 "$build/bin/garnet_gpt_oss_tp_direct_benchmark" "$batch" 50 \
        >"$directory/reduction-b$batch-nccl.log" 2>&1
    for ctas in 2 4 8 16 32; do
        export GARNET_GPT_OSS_DIRECT_ALLREDUCE=1 GARNET_GPT_OSS_DIRECT_BATCH_ALLREDUCE=1
        export GARNET_GPT_OSS_DIRECT_LARGE_BATCH_ALLREDUCE=1 GARNET_GPT_OSS_DIRECT_BATCH_CTAS=$ctas
        "$build/bin/garnet_gpt_oss_tp_direct_benchmark" "$batch" 50 \
            >"$directory/reduction-b$batch-direct$ctas.log" 2>&1
    done
    label=reduction-b$batch-direct-memcheck
    compute-sanitizer --tool memcheck --error-exitcode 0 --target-processes all \
        --report-api-errors explicit --xml --print-limit 0 --print-session-details \
        --save "$directory/$label.xml" "$build/bin/garnet_gpt_oss_tp_direct_benchmark" "$batch" 1 \
        >"$directory/$label.log" 2>&1
    "$python" "$repo/tools/gpt_oss/validate_sanitizer_xml.py" "$directory/$label.xml" "$directory/$label.audit.json"
done
echo 'Candidate gates complete; pretrained performance, new resident profiles and maximum capacity remain separate.'
