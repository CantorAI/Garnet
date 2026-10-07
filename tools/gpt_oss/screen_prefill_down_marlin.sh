#!/usr/bin/env bash
# OPT43 isolated scheduling screen; keep cross-mode matrices, not only timings.
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
export GARNET_GPT_OSS_MARLIN_MAX_TOKENS=4096 GARNET_GPT_OSS_MARLIN_PREFILL_BLOCK=32
export GARNET_GPT_OSS_MARLIN_DECODE_BLOCK=32 GARNET_GPT_OSS_MARLIN_CTAS_PER_SM=1
export GARNET_GPT_OSS_MARLIN_PREFILL_CTAS_PER_SM=1 GARNET_GPT_OSS_PARALLEL_MARLIN_METADATA=1
export GARNET_GPT_OSS_ROUTER_QUERY_TILE=2 GARNET_GPT_OSS_PREFILL_ROUTER_TENSORCORE=1
export GARNET_GPT_OSS_DEBUG_MARLIN=1
git -C "$repo" rev-parse HEAD >"$directory/source.txt"
sha256sum "$build/bin/garnet_gpt_oss_marlin_decode_benchmark" >"$directory/binary.sha256"
nvidia-smi --query-gpu=index,name,uuid,driver_version --format=csv >"$directory/hardware.csv"
for rows in 1024 4096; do
    for tile in 0 64; do
        export GARNET_GPT_OSS_MARLIN_LARGE_PREFILL_BLOCK=$tile
        for setting in 128/1 64/1 64/2; do
            export GARNET_GPT_OSS_MARLIN_PREFILL_DOWN_K=${setting%/*}
            export GARNET_GPT_OSS_MARLIN_PREFILL_DOWN_CTAS_PER_SM=${setting#*/}
            for trial in 1 2 3; do
                label=rows$rows-tile$tile-k${setting%/*}-cta${setting#*/}-trial$trial
                output=(); [[ $trial != 1 ]] || output+=("$directory/$label.f32")
                timeout --signal=TERM --kill-after=15s 300s "$build/bin/garnet_gpt_oss_marlin_decode_benchmark" "$rows" prefill "${output[@]}" \
                    >"$directory/$label.log" 2>&1
                tail -2 "$directory/$label.log"
            done
        done
    done
done
# Candidate geometry/grid must leave decode results unchanged.
for setting in 128/1 64/1 64/2; do
    export GARNET_GPT_OSS_MARLIN_PREFILL_DOWN_K=${setting%/*}
    export GARNET_GPT_OSS_MARLIN_PREFILL_DOWN_CTAS_PER_SM=${setting#*/}
    label=decode256-k${setting%/*}-cta${setting#*/}
    timeout --signal=TERM --kill-after=15s 300s "$build/bin/garnet_gpt_oss_marlin_decode_benchmark" 256 decode "$directory/$label.f32" \
        >"$directory/$label.log" 2>&1
done
"$python" - "$directory" <<'PY'
import hashlib,json,re,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]); rows={}
baseline_decode=(root/'decode256-k128-cta1.f32').read_bytes()
for path in sorted(root.glob('*.f32')):
    data=path.read_bytes()
    row=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
    values=np.frombuffer(data,dtype='<f4')
    assert np.isfinite(values).all(),path
    if path.name.startswith('decode'):
        row['exact_to_decode_baseline']=data==baseline_decode
        assert row['exact_to_decode_baseline'],path
    else:
        match=re.fullmatch(r'rows(1024|4096)-tile(0|64)-k(64|128)-cta(1|2|4)-trial1.f32',path.name)
        n,tile,k,ctas=match.groups()
        reference=(root/f'rows{n}-tile{tile}-k128-cta1-trial1.f32').read_bytes()
        expected=np.frombuffer(reference,dtype='<f4')
        assert values.size==int(n)*2880
        row.update(exact_to_same_tile_k128=data==reference,
            max_absolute_vs_same_tile_k128=float(np.max(np.abs(values-expected))),
            scope='Synthetic cross-mode difference, not independent CPU-oracle quality')
    rows[path.name]=row
(root/'synthetic-matrix-comparison.json').write_text(json.dumps(rows,indent=2))
(root/'unsupported-settings.json').write_text(json.dumps(dict(down_k64_cta4=
    'Prior native4096-row hang; now rejected before access/launch. Not a passing or timed setting.'),indent=2))
PY
echo 'Synthetic screen complete; resident/pretrained quality and serving throughput remain separate'
