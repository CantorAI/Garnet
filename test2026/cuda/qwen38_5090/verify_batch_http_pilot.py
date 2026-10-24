"""Replay B2 HTTP measurements, native events and real dispatch membership."""
from pathlib import Path
import base64
import hashlib
import json
import sys
import zipfile

out = Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent
repo = Path(sys.argv[2]) if len(sys.argv)>2 else Path('D:/CantorAI/work/qwen38-5090')
sys.path.insert(0, str(repo/'test2026/cuda/qwen38_5090'))
from compare_http_runs import verify_run, compare
sha = lambda raw: hashlib.sha256(raw).hexdigest()
archive = out/'batch-http-pilot-attempt1-evidence.zip'
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    evidence = json.loads(z.read('evidence-manifest.json'))
    for name, digest in evidence['files'].items(): assert sha(z.read(name)) == digest
    for name, digest in evidence['linked_archives'].items(): assert sha((out/name).read_bytes()) == digest
    doc = lambda name: json.loads(z.read(name))
    base = 'results/batch-http-pilot-attempt1/'
    native = base+'native/'
    quality = doc(native+'quality.json')
    assert quality['status'] == 'passed' and len(quality['cases']) == 8 and len(quality['stream_cases']) == 2
    for case in quality['cases']:
        assert case['exact']
        expected = doc('results/native-mtp3-http/'+case['oracle']+'-response.json')['choices'][0]
        actual = doc(native+case['key']+'-response.json')['choices'][0]
        for field in ('prompt_token_ids', 'response_token_ids'): assert actual[field] == expected[field]
    workers = {}
    tokens = {}
    ready = []
    for line in z.read(native+'worker/worker-stdout.jsonl').decode().splitlines():
        try: event = json.loads(json.loads(line)['line'])
        except ValueError: continue
        kind = event.get('event')
        if kind == 'ready': ready.append(event)
        elif kind == 'token':
            key = event['id']; assert key not in workers
            tokens.setdefault(key, []).append(event['token_id'])
            assert event['sampled_tokens'] == len(tokens[key])
        elif kind == 'completed':
            key = event['id']; assert key not in workers and event['generated_ids'] == tokens[key]
            workers[key] = event
        elif kind == 'error': raise AssertionError(event)
    assert len(ready) == 1 and ready[0]['batch'] == 2 and ready[0]['library_sha256'] == evidence['library_sha256']
    dispatch = [json.loads(line) for line in z.read(native+'worker/batch-dispatch.jsonl').decode().splitlines()]
    dispatched = []
    for group in dispatch:
        assert group['physical_capacity'] == 2 and group['real_rows'] == len(group['request_ids']) in (1,2)
        dispatched.extend(group['request_ids'])
    assert len(dispatched) == len(set(dispatched)) and set(dispatched) == set(workers)
    assert any(group['real_rows'] == 2 for group in dispatch)
    def final(path):
        events = [json.loads(line)['data'] for line in z.read(path).decode().splitlines()]
        assert events[-1] == '[DONE]'
        return next(json.loads(v) for v in reversed(events) if v != '[DONE]')
    for case in quality['stream_cases']:
        name = case['oracle']; event = final(native+'stream-'+name+'-events.jsonl')
        expected = doc('results/native-mtp3-http/'+name+'-response.json')['choices'][0]
        for field in ('prompt_token_ids','response_token_ids'): assert event[field] == expected[field]
    manifests = [name for name in z.namelist() if name.startswith(native+'benchmark/') and name.endswith('-manifest.json')]
    assert len(manifests) == 1
    manifest = doc(manifests[0]); prefix = manifests[0].rsplit('/',1)[0]+'/'
    run = verify_run(manifest, lambda name:z.read(prefix+name))
    assert len(run['cohorts']) == 8
    samples = [json.loads(line) for line in z.read(prefix+manifest['run_id']+'-vram.jsonl').decode().splitlines()]
    measured = 0
    for cohort in manifest['cohorts']:
        assert cohort['concurrency'] in (1,2,4,8) and len(cohort['request_keys']) == 8
        phase = cohort['case']+'-c'+str(cohort['concurrency'])
        rows = [s for s in samples if s['phase'] == phase][:cohort['samples']]
        assert len(rows) == cohort['samples'] and max(s['used_bytes'] for s in rows) == cohort['sampled_peak_adapter_used_bytes']
        for key in cohort['request_keys']:
            event = final(prefix+key+'-events.jsonl'); work = workers[event['id']]
            assert event['response_token_ids'] == work['generated_ids'] and len(work['generated_ids']) == 128
            assert event['prompt_token_ids'] == work['prompt_ids']
            assert event['study']['generation_mode'] == work['generation_mode'] == 'greedy_target'
            body = doc(prefix+key+'-request.json')
            if cohort['case'] == 'image':
                assert sha(base64.b64decode(body['messages'][0]['content'][0]['image_url']['url'].split(',',1)[1])) == work['image_sha256']
            measured += 1
    assert measured == 64
    metadata = doc(native+'metadata.json')
    assert metadata['ready'] == ready[0] and metadata['physical_batch'] == 2 and metadata['grouping_delay_ms'] == 2
    for path, digest in metadata['file_sha256'].items():
        name = path.replace('\\','/').split('5090-study-20261023/',1)[1]
        if name == 'build/bin/garnet.dll': assert digest == evidence['library_sha256']
        else: assert sha(z.read(name)) == digest
    references = []
    for name in ('vllm-graph-linkfix-evidence.zip','sglang-corrected-ipv4-evidence.zip'):
        with zipfile.ZipFile(out/name) as ref:
            names = [n for n in ref.namelist() if n.endswith('-manifest.json') and n.rsplit('/',1)[-1] not in ('evidence-manifest.json','preparation-manifest.json','source-manifest.json')]
            assert len(names) == 1
            m = json.loads(ref.read(names[0])); stem = names[0].rsplit('/',1)[0]+'/'
            references.append(verify_run(m, lambda n:ref.read(stem+n)))
    events = [json.loads(line) for line in z.read(base+'ledger/commands.jsonl').decode().splitlines()]
    assert events[-1]['event'] == 'finished' and events[-1]['exit_code'] == 0
    workflow = doc(base+'workflow.json'); assert workflow['exit_code'] == 0
    build = doc('results/native-full-attention-fp8-build-attempt1/manifest.json')
    assert build['library_sha256'] == workflow['library_sha256'] == evidence['library_sha256']
    for name, digest in build['source_sha256'].items(): assert sha(z.read('Garnet/'+name)) == digest
    resource = doc(base+'resources/resource-monitor.json')
    lifetime = [json.loads(line) for line in z.read(base+'resources/resource-vram.jsonl').decode().splitlines()]
    assert resource['child_exit_code'] == 0 and resource['sampled_memory']['samples'] == len(lifetime)
    assert resource['sampled_memory']['sampled_peak_adapter_used_bytes'] == max(v['used_bytes'] for v in lifetime)
    result = {'status':'batch_http_pilot_independently_verified','archive_sha256':sha(archive.read_bytes()),
        'library_sha256':evidence['library_sha256'],'files_verified':len(evidence['files']),'measured_sse_worker_pairs':measured,'physical_batch':2,
        'dispatch_group_sizes':{str(n):sum(g['real_rows']==n for g in dispatch) for n in (1,2)},
        'run':run,'references':references,'comparison':compare(run,references),'raw_cohorts':manifest['cohorts'],
        'two_x_established':False,'scope':'One B2 MTP0 fixed-batch pilot; C4/8 queue physical B2 groups. Bounded quality; ordinary references, tuning/repeats/4K/broad quality pending.'}
(out/'batch-http-pilot-attempt1-verified.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k not in ('run','references','raw_cohorts')},indent=2))
