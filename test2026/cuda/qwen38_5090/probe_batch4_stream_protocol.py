"""Real-worker EOF protocol/reset/row-isolation gate; no timing admission."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root, out = map(Path, sys.argv[1:3])
assert not out.exists()
out.mkdir()
oracle = root / 'results/native-mtp3-http'
keys = ['s3-text-c1-r0-greedy', 's3-image-c1-r0-greedy', 'red', 'green']
saved = {}
for key in keys:
    request = oracle / (key+'-request.json')
    response = oracle / (key+'-response.json')
    saved[key] = (json.loads(request.read_text()), json.loads(response.read_text())['choices'][0])
groups = [(keys, False), (keys, False), ([keys[0]], False),
    ([keys[1]], False), (keys[2:], False), (keys[:3], False), (keys[:2], True)]
frames = []
expected = {}
for index, (names, short_first) in enumerate(groups):
    jobs = []
    for row, name in enumerate(names):
        body = copy.deepcopy(saved[name][0])
        if short_first and row == 0:
            body['max_tokens'] = 2
        key = str(index)+'-'+str(row)
        jobs.append({'id': key, 'body': body})
        expected[key] = {'prompt': saved[name][1]['prompt_token_ids'],
            'generated': saved[name][1]['response_token_ids'][:body['max_tokens']]}
    frames.append({'batch': jobs})
wire = ''.join(json.dumps(frame, separators=(',', ':'))+'\n' for frame in frames).encode()
(out / 'requests.jsonl').write_bytes(wire)
launcher = root / 'serve-native-batch4-shared-worker.bat'
cache = json.loads((root / 'results/study-ssd-engine-cache/receipt.json').read_text())['cache_root']
process = subprocess.run(['cmd.exe', '/d', '/c', str(launcher), 'unused', cache], input=wire,
    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
(out / 'worker-stdout.log').write_bytes(process.stdout)
(out / 'worker-stderr.log').write_bytes(process.stderr)
report = {'status': 'process_failed', 'exit_code': process.returncode, 'cases': [],
    'request_sha256': hashlib.sha256(wire).hexdigest(), 'performance_admitted': False}
def save():
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
save()
assert process.returncode == 0, 'Worker failed; raw output retained'
tokens = {key: [] for key in expected}
completed = set()
ready = []
times = {key: [] for key in expected}
for line in process.stdout.decode('utf8', errors='strict').splitlines():
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        continue
    kind = event.get('event')
    if kind == 'ready':
        ready.append(event)
        continue
    if kind not in ('token', 'completed', 'error'):
        continue
    key = event['id']
    assert key in expected and key not in completed, 'Unknown/dummy/retired row event'
    assert kind != 'error', event
    if kind == 'token':
        tokens[key].append(event['token_id'])
        times[key].append(event['native_monotonic_seconds'])
        assert event['sampled_tokens'] == len(tokens[key])
        assert event['prompt_tokens'] == len(expected[key]['prompt'])
    else:
        assert event['generation_mode'] == 'greedy_target'
        assert event['generated_ids'] == tokens[key] == expected[key]['generated']
        assert event['prompt_ids'] == expected[key]['prompt']
        assert times[key] == sorted(times[key])
        completed.add(key)
        report['cases'].append({'id': key, 'exact': True, 'tokens': len(tokens[key])})
        save()
assert len(ready) == 1 and ready[0]['protocol'] == 'study_batch_v1' and ready[0]['batch'] == 4
assert completed == set(expected) and len(completed) == 17
report['status'] = 'batch_stream_protocol_passed_needs_independent_replay'
report['ready'] = ready[0]
save()
print(json.dumps(report, indent=2))
