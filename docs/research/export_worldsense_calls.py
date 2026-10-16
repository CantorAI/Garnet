"""Export raw WorldSense call rows without pooling, filtering or speed claims.

Usage: python3 export_worldsense_calls.py ARCHIVE OUTPUT_CSV
Only root cohort JSON files with items/search_plans are read. Other manifests,
analyses and nested failed attempts remain untouched; exclusions are explicit.
"""
import csv
import hashlib
import json
from pathlib import Path
import sys

archive, output = map(Path, sys.argv[1:3])
fields = ['artifact', 'artifact_sha256', 'label', 'sample_index', 'phase',
          'policy', 'warmup', 'repetition', 'model_id', 'observation_id',
          'skill_ref', 'query', 'image_sha256', 'skill_sha256', 'status',
          'input_tokens', 'generated_tokens', 'total_seconds', 'ttft_seconds',
          'decode_tokens_per_second', 'native_call_seconds']
rows = []
for path in sorted(archive.glob('*.json')):
    raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict) or 'items' not in data or 'search_plans' not in data:
        continue
    for category, items in [('frame', data['items']), ('search', data['search_plans'])]:
        for index, item in enumerate(items):
            result = item['result']
            calls = result.get('model_response', {}) if category == 'frame' else {
                'search_plan': result.get('benchmark_model_response', {})}
            for phase, call in calls.items():
                if not isinstance(call, dict) or 'model_id' not in call:
                    continue
                row = {key: item.get(key, '') for key in fields}
                row.update(artifact=path.name, artifact_sha256=hashlib.sha256(raw).hexdigest(),
                           label=data.get('label', path.stem), sample_index=index,
                           phase=phase)
                for key in ['model_id', 'status', 'generated_tokens', 'total_seconds', 'ttft_seconds']:
                    row[key] = call.get(key, '')
                row['input_tokens'] = call.get('timings', {}).get('input_tokens', '')
                total, ttft, count = (call.get(k) for k in ['total_seconds', 'ttft_seconds', 'generated_tokens'])
                if all(isinstance(x, (int, float)) for x in [total, ttft, count]) and total > ttft and count > 1:
                    row['decode_tokens_per_second'] = (count - 1) / (total - ttft)
                rows.append(row)
output.parent.mkdir(parents=True, exist_ok=True)
with output.open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({'rows': len(rows), 'output': str(output),
                  'scope': 'All eligible root cohort calls, including warmups; no pooling or independent-sample claim; nested failure attempts excluded'}))
