#!/usr/bin/env bash
# OPT44 stage1: native transport proof/screen only, no plugin dispatch change.
set -euo pipefail
[[ $# == 1 && ! -e $1 ]] || exit 2
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${CANTORAI_ROOT:-$(dirname "$repo")}
build=${GARNET_BUILD_DIR:-$root/out/build/gpt-oss}
python=${GARNET_VERIFY_PYTHON:-$root/venv-vllm/bin/python}
directory=$1
exec 9>"$root/work/gpu-benchmark.lock"
flock -n 9 || exit 1
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] || exit 1
mkdir -p "$directory"; directory=$(cd "$directory" && pwd)
export LD_LIBRARY_PATH="$build/bin:$root/ThirdPartySDK/TensorRT/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GARNET_GPT_OSS_DIRECT_ALLREDUCE=0
binary=$build/bin/garnet_gpt_oss_tp_bf16_wire_benchmark
git -C "$repo" rev-parse HEAD >"$directory/source.txt"
sha256sum "$binary" >"$directory/binary.sha256"
nvidia-smi --query-gpu=index,name,uuid,driver_version --format=csv >"$directory/hardware.csv"
for batch in 128 256 512; do
    for mode in fp32 bf16; do
        for trial in 1 2 3; do
            timeout --signal=TERM --kill-after=15s 300s "$binary" "$batch" "$mode" 50 >"$directory/b$batch-$mode-trial$trial.log" 2>&1
        done
        label=b$batch-$mode-memcheck
        timeout --signal=TERM --kill-after=15s 900s compute-sanitizer --tool memcheck --error-exitcode 0 --target-processes all \
            --report-api-errors explicit --xml --print-limit 0 --print-session-details \
            --backtrace-short no --strip-paths no \
            --save "$directory/$label.xml" "$binary" "$batch" "$mode" 1 \
            >"$directory/$label.log" 2>&1
        "$python" "$repo/tools/gpt_oss/validate_sanitizer_xml.py" \
            "$directory/$label.xml" "$directory/$label.audit.json"
        echo "Native batch$batch $mode finite-bit/graph/reacquire/strict-memcheck passed"
    done
done
"$python" - "$directory" <<'PY'
import json,re,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]); report={}
for batch in (128,256,512):
    modes={}
    for mode in ('fp32','bf16'):
        cycles=[]
        for trial in (1,2,3):
            text=(root/f'b{batch}-{mode}-trial{trial}.log').read_text()
            medians=[float(x) for x in re.findall(r' median=([0-9.]+)',text)]
            assert len(medians)==2 and text.count('parity=PASS')==2
            cycles.append(medians)
        modes[mode]=dict(cycle_medians_us=cycles,
            median_us=statistics.median(x for row in cycles for x in row))
    report[str(batch)]=dict(modes=modes,
        relative_bf16_reduction=1-modes['bf16']['median_us']/modes['fp32']['median_us'])
(root/'comparison.json').write_text(json.dumps(report,indent=2))
print(report)
PY
echo 'OPT44 native screen complete; plugin/cache/workspace/model integration NOT implemented or validated'
