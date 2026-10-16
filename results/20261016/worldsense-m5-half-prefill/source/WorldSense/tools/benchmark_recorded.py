"""XLang3 client for repeatable recorded-video-frame Skill experiments.

Usage: xlang3 -B benchmark_recorded.py URL CASES_JSON OUTPUT_JSON LABEL
WORLDSENSE_BENCH_TOKEN supplies authentication; server must explicitly enable
WORLDSENSE_ENABLE_BENCHMARK=1. CASES_JSON has cases (observation_id, start_ms,
end_ms, skill_ref), repeats (1..4), and optional queries (text search planning).
One warmup of every case precedes measured repeats. Run fresh baseline /
candidate / candidate / baseline processes; preserve complete JSON and build
manifests. These timings exclude video decode/candidate scheduling/storage.
Use benchmark_video.py separately for live integration behavior.
"""
import sys
import os
import json
import time
import urllib.request
from pathlib import Path

if sys.implementation.name != 'xlang3':
    raise RuntimeError('Execute this client with XLang3')
base, cases, output, label = sys.argv[1:5]
payload = json.loads(Path(cases).read_text())
req = urllib.request.Request(base.rstrip('/') + '/api/skill-lab/benchmark/run',
    data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json',
    'vega-authorization': 'Bearer ' + os.environ['WORLDSENSE_BENCH_TOKEN']})
started = time.monotonic()
print('WorldSense fixed-frame suite started:', label, flush=True)
with urllib.request.urlopen(req, timeout=1800) as response:
    result = json.loads(response.read().decode())
result['http_wall_seconds'] = time.monotonic() - started
result['label'] = label
result['request'] = payload
result['executor'] = 'XLang3'
Path(output).write_text(json.dumps(result, indent=2))
print('WorldSense fixed-frame suite:', label, 'valid=', result['valid'],
      'frames=', len(result['items']), 'seconds=', result['http_wall_seconds'], flush=True)
if not result['valid']:
    raise RuntimeError('Benchmark failed; inspect saved output')
