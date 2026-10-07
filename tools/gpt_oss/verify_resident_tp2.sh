#!/usr/bin/env bash
# Complete existing storage regressions, then padded-tail + shared-KV gates.
set -euo pipefail
[[ $# == 1 && ! -e $1 ]] || exit 2
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${CANTORAI_ROOT:-$(dirname "$repo")}
build=${GARNET_BUILD_DIR:-$root/out/build/gpt-oss}
python=${GARNET_VERIFY_PYTHON:-$root/venv-vllm/bin/python}
directory=$1
bash "$repo/tools/gpt_oss/verify_marlin_prepacked.sh" "$directory"
directory=$(cd "$directory" && pwd)
exec 9>"$root/work/gpu-benchmark.lock"
flock -n 9 || exit 1
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] || exit 1
export XLANG3_PYTHON_LIB=$root/ThirdPartySDK/Python-3.14.0/Lib
export PYTHONPATH="$build/bin${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$build/bin:$root/ThirdPartySDK/TensorRT/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GARNET_GPT_OSS_MARLIN_MAX_TOKENS=4096 GARNET_GPT_OSS_MARLIN_PREFILL_BLOCK=32
export GARNET_GPT_OSS_MARLIN_DECODE_BLOCK=32 GARNET_GPT_OSS_ROUTER_QUERY_TILE=2
export GARNET_GPT_OSS_PARALLEL_MARLIN_METADATA=1 GARNET_GPT_OSS_PREFILL_ROUTER_TENSORCORE=1
export GARNET_GPT_OSS_BF16_PREFILL_ALLREDUCE=1 GARNET_GPT_OSS_MARLIN_PREPACKED=1
export GARNET_GPT_OSS_TEACHER_RESIDENT=1
for partition in expert intermediate; do
    if [[ $partition == expert ]]; then
        export GARNET_GPT_OSS_TP_EXPERT_WEIGHT_SHARDS=1 GARNET_GPT_OSS_TP_MOE_INTERMEDIATE_SHARDS=0
        fixture=$directory/full-parity/fixture; batch=16
    else
        export GARNET_GPT_OSS_TP_EXPERT_WEIGHT_SHARDS=0 GARNET_GPT_OSS_TP_MOE_INTERMEDIATE_SHARDS=1
        fixture=$directory/fixture64; batch=512
    fi
    for tokens in 4 8; do
        export GARNET_GPT_OSS_TEACHER_PADDED_TOKENS=$tokens
        for phase in cold warm; do
            label=resident-$partition-pad$tokens-$phase
            "$build/bin/xlang3" "$repo/test2026/gpt_oss/tp_teacher_forced.py" "$fixture" \
                "$directory/resident-$partition-pad$tokens-cache" "$directory/$label.json" "$batch" \
                >"$directory/$label.log" 2>&1
            echo "$label CPU-prefix/shared-KV/future-overwrite gate passed"
        done
    done
done
compute-sanitizer --tool memcheck --error-exitcode 0 --target-processes all \
    --report-api-errors explicit --xml --print-limit 0 --print-session-details \
    --save "$directory/selection.memcheck.xml" "$build/bin/xlang3" \
    "$repo/test2026/gpt_oss/sequence_selection.py" "$directory/selection-sanitizer-cache" \
    >"$directory/selection.memcheck.log" 2>&1
"$python" "$repo/tools/gpt_oss/validate_sanitizer_xml.py" \
    "$directory/selection.memcheck.xml" "$directory/selection.memcheck-audit.json"
"$python" - "$directory" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); report={}
for partition in ('expert','intermediate'):
    baseline=json.loads((root/f'teacher-{partition}-packed1-warm.json').read_text())
    results=[json.loads((root/f'resident-{partition}-pad{tokens}-{phase}.json').read_text())
             for tokens in (4,8) for phase in ('cold','warm')]
    assert all(s['within_existing_tolerance'] for r in results for s in r['steps'])
    for tokens in (4,8):
        cold,warm=[r for r in results if r['padded_prefill_tokens']==tokens]
        assert [s['tp_logits'] for s in cold['steps']]==[s['tp_logits'] for s in warm['steps']]
    report[partition]=dict(batch=baseline['batch'],
        exact_cold_warm=True, cpu_max_absolute_errors=[max(s['maximum_absolute_error'] for s in r['steps']) for r in results],
        exact_vs_unpadded=[all(a['tp_logits']==b['tp_logits'] for a,b in zip(r['steps'],baseline['steps'])) for r in results])
(root/'resident-compiled-parity.json').write_text(json.dumps(report,indent=2))
print(report)
PY
echo 'Resident synthetic, padded future-KV and strict selection memcheck gates passed; pretrained residency/performance still pending'
