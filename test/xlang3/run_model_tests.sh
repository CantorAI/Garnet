#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if (( $# < 3 || $# > 5 )); then
    echo "Usage: $0 runtime-path {tensorrt|openvino} work-root [tensorrt-directory] [cuda-directory]" >&2
    exit 2
fi
runtime=$(realpath "$1")
backend=$2
case "$backend" in
    tensorrt|openvino) ;;
    *) echo "Unsupported backend: $backend" >&2; exit 2 ;;
esac
test -x "$runtime"
mkdir -p "$3"
work_root=$(cd "$3" && pwd -P)
work=$(mktemp -d "$work_root/run.XXXXXXXX")
models=$(cd "$(dirname "${BASH_SOURCE[0]}")/models" && pwd -P)
trap 'echo "Garnet model tests failed at line $LINENO; artifacts: $work" >&2' ERR
if [[ -n ${4:-} ]]; then
    trt=$(realpath "$4")
    export PATH="$trt:$PATH"
    export LD_LIBRARY_PATH="$trt:${LD_LIBRARY_PATH:-}"
fi
if [[ -n ${5:-} ]]; then
    export PATH="$(realpath "$5"):$PATH"
fi
cd "$(dirname "$runtime")"
export LD_LIBRARY_PATH="$PWD:${LD_LIBRARY_PATH:-}"
cache="$work/engines"
"$runtime" "$models/run_models.py" "$backend" "$cache" build
"$runtime" "$models/run_models.py" "$backend" "$cache" reload
"$runtime" "$models/run_contracts.py" "$backend" "$work/contracts"
"$runtime" "$models/run_source_isolation.py" "$backend" "$work/isolation"
if [[ $backend == tensorrt ]]; then
    "$runtime" "$models/run_reuse.py" "$work/reuse"
    "$runtime" "$models/run_concurrent.py" "$work/concurrent"
fi
echo "Garnet $backend integration passed; artifacts: $work"
