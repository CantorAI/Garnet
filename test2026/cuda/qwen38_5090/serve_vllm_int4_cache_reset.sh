#!/usr/bin/env bash
set -euo pipefail
study=/mnt/d/CantorAI/5090-study-20261023
export VLLM_SERVER_DEV_MODE=1
export GARNET_VLLM_LOG="$study/results/vllm-int4-cache-reset-server.log"
[[ ! -e "$GARNET_VLLM_LOG" ]] || { echo 'Preserve prior cache-reset server log'; exit 1; }
/usr/bin/python3 - <<'PY'
import socket
with socket.socket() as probe: probe.bind(('127.0.0.1',18080))
PY
exec bash "$(dirname "$0")/serve_vllm_int4.sh"
