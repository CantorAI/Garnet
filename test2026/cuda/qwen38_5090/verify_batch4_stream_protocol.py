"""Replay actual B2 worker transport against independently retained B1 oracles."""
from pathlib import Path
import base64
import hashlib
import json
import math
import zipfile

out = Path(__file__).parent
sha = lambda raw: hashlib.sha256(raw).hexdigest()
archive = out / 'batch4-stream-protocol-attempt1-evidence.zip'
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    evidence = json.loads(z.read('evidence-manifest.json'))
    for name, digest in evidence['files'].items():
        assert sha(z.read(name)) == digest
    for name, digest in evidence['linked_archives'].items():
        assert sha((out / name).read_bytes()) == digest
    doc = lambda name: json.loads(z.read(name))
    base = 'results/batch4-stream-protocol-attempt1/'
    workflow = doc(base+'workflow.json')
    report = doc(base+'native/report.json')
    assert workflow['exit_code'] == report['exit_code'] == 0
    assert report['status'] == 'batch_stream_protocol_passed_needs_independent_replay'
    build = doc('results/native-full-attention-fp8-build-attempt1/manifest.json')
    assert evidence['library_sha256'] == workflow['library_sha256'] == build['library_sha256']
    for name, digest in build['source_sha256'].items():
        assert sha(z.read('Garnet/'+name)) == digest
    prior = doc('results/shared-full-b4-t8-attempt1-verified.json')
    assert prior['status'] == 'shared_full_b4_t8_id_gate_verified'
    assert prior['archive_sha256'] == evidence['linked_archives']['shared-full-b4-t8-attempt1-evidence.zip']
    assert prior['model_gate_passed'] and prior['library_sha256'] == evidence['library_sha256']
    wire = z.read(base+'native/requests.jsonl')
    assert sha(wire) == report['request_sha256']
    frames = [json.loads(line) for line in wire.splitlines()]
    names = ['s3-text-c1-r0-greedy', 's3-image-c1-r0-greedy', 'red', 'green']
    groups = [names, names, [names[0]], [names[1]], names[2:], names[:3], names[:2]]
    assert len(frames) == 7
    expected = {}
    for index, (frame, group) in enumerate(zip(frames, groups)):
        assert len(frame['batch']) == len(group)
        for row, (job, name) in enumerate(zip(frame['batch'], group)):
            reference = 'results/native-mtp3-http/'+name
            body = doc(reference+'-request.json')
            choice = doc(reference+'-response.json')['choices'][0]
            if index == 6 and row == 0:
                body['max_tokens'] = 2
            key = str(index)+'-'+str(row)
            assert job == {'id': key, 'body': body}
            content = body['messages'][0]['content']
            image_hash = None if isinstance(content, str) else sha(base64.b64decode(content[0]['image_url']['url'].split(',', 1)[1], validate=True))
            expected[key] = {'prompt': choice['prompt_token_ids'], 'ids': choice['response_token_ids'][:body['max_tokens']], 'image_hash': image_hash}
    tokens = {key: [] for key in expected}
    completed = set()
    ready = []
    previous = -math.inf
    for line in z.read(base+'native/worker-stdout.log').decode('utf8').splitlines():
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
        assert len(ready) == 1 and kind != 'error'
        key = event['id']
        assert key in expected and key not in completed
        if kind == 'token':
            tokens[key].append(event['token_id'])
            now = event['native_monotonic_seconds']
            assert math.isfinite(now) and now >= previous
            previous = now
            assert event['sampled_tokens'] == len(tokens[key])
            assert event['prompt_tokens'] == len(expected[key]['prompt'])
        else:
            assert event['generation_mode'] == 'greedy_target'
            assert event['generated_ids'] == tokens[key] == expected[key]['ids']
            assert event['prompt_ids'] == expected[key]['prompt']
            assert event['image_sha256'] == expected[key]['image_hash']
            completed.add(key)
    assert completed == set(expected) and len(completed) == 17
    assert ready[0] == report['ready'] and ready[0]['protocol'] == 'study_batch_v1' and ready[0]['batch'] == 4
    assert ready[0]['prefill_chunk'] == ready[0]['chunk'] == 8 and ready[0]['mtp_depth'] == 0
    assert ready[0]['library_sha256'] == evidence['library_sha256'] and ready[0]['shared_compressed_weights']
    assert {(v['id'], v['tokens'], v['exact']) for v in report['cases']} == {(key, len(v['ids']), True) for key, v in expected.items()}
    events = [json.loads(line) for line in z.read(base+'ledger/commands.jsonl').decode().splitlines()]
    assert events[-1]['event'] == 'finished' and events[-1]['exit_code'] == 0
    monitor = doc(base+'resources/resource-monitor.json')
    samples = [json.loads(line) for line in z.read(base+'resources/resource-vram.jsonl').decode().splitlines()]
    assert monitor['child_exit_code'] == 0 and monitor['sampled_memory']['samples'] == len(samples)
    peak = max(v['used_bytes'] for v in samples)
    assert peak == monitor['sampled_memory']['sampled_peak_adapter_used_bytes']
result = {'status': 'batch4_stream_protocol_independently_verified', 'archive_sha256': sha(archive.read_bytes()),
    'files_verified': len(evidence['files']), 'library_sha256': evidence['library_sha256'],
    'groups': 7, 'requests': 17, 'exact_ids_and_prompt_ids': True, 'partial_groups_and_row_isolation': True,
    'image_hashes_verified': True, 'sampled_peak_adapter_used_bytes': peak, 'performance_admitted': False}
(out/'batch4-stream-protocol-attempt1-verified.json').write_text(json.dumps(result, indent=2), encoding='utf8')
print(json.dumps(result, indent=2))
