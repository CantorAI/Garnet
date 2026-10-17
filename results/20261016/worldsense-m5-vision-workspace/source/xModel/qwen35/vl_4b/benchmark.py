# SPDX-License-Identifier: Apache-2.0
"""Native serving comparison. Run serial and parallel profiles in separate processes."""
import json
import os
import sys
import time
from pathlib import Path
import garnet

if len(sys.argv) < 4:
    raise SystemExit('Usage: xlang3 benchmark.py MODEL_ROOT serial|parallel IMAGE [IMAGE ...]')
model_root, mode = sys.argv[1:3]
if mode not in ('serial', 'parallel'):
    raise ValueError('Select serial or parallel')
xmodel = Path(__file__).resolve().parent
profile = json.loads((xmodel/'serving-profile.json').read_text())
profile['linearChunk'] = mode == 'parallel'
profile['maxOutputTokens'] = 256
name = 'Qwen3.5-4B'
cache = os.path.join(model_root, 'metal-benchmark-cache')
start = time.monotonic()
status = json.loads(garnet.serve_model(model_root, str(xmodel), cache, json.dumps(profile), name))
if not status.get('ready'):
    raise RuntimeError(str(status))
print(json.dumps({'load_seconds': time.monotonic()-start, 'profile': profile}), flush=True)
prompt = ('Return only a compact JSON object with description (one factual sentence), '
          'entities (at most three visible objects), rabbit_visible (boolean), car_visible (boolean), '
          'elephant_visible (boolean). Do not describe absent objects.')
for repeat in range(2):
    for image in sys.argv[3:]:
        result = json.loads(garnet.infer_json(prompt, image, 180, name))
        print(json.dumps({'repeat': repeat, 'image': image, 'call': result}), flush=True)
        if result.get('status') != 'ok':
            raise RuntimeError(str(result))
