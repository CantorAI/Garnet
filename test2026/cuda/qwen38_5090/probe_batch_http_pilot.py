"""Same-worker B2 HTTP quality then exploratory C1/2/4/8 stream measurements."""
import concurrent.futures
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.request import Request, urlopen
from native_batch_http_bridge import BatchWorker, BatchHandler, ThreadingHTTPServer
from benchmark_stream import request as stream_request

root, out = map(Path, sys.argv[1:3])
assert not out.exists()
out.mkdir()
methods = root/'Garnet/test2026/cuda/qwen38_5090'
proof_path = root/'results/batch-stream-protocol-attempt1-verified.json'
proof = json.loads(proof_path.read_text())
assert proof['status'] == 'batch_stream_protocol_independently_verified'
assert proof['library_sha256'] == hashlib.sha256((root/'build/bin/garnet.dll').read_bytes()).hexdigest()
cache = json.loads((root/'results/study-ssd-engine-cache/receipt.json').read_text())['cache_root']
command = ['cmd.exe', '/d', '/c', str(root/'serve-native-batch2-shared-worker.bat'), 'unused', cache]
worker = BatchWorker(command, out/'worker', 2, 2)
server = None
thread = None
quality = {'status': 'running', 'cases': [], 'stream_cases': [], 'performance_admitted': False}
def save():
    (out/'quality.json').write_text(json.dumps(quality, indent=2), encoding='utf8')
save()
try:
    while not worker.healthy():
        assert worker.process.poll() is None and worker.failure is None, worker.failure
        time.sleep(.25)
    server = ThreadingHTTPServer(('127.0.0.1', 18082), BatchHandler)
    server.worker = worker
    server.model = 'qwen38-27b-nvfp4'
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = 'http://127.0.0.1:18082'
    oracle = root/'results/native-mtp3-http'
    names = ['s3-text-c1-r0-greedy', 's3-image-c1-r0-greedy', 'red', 'green']
    groups = [names[:2], names[:2], names[2:], [names[0]], [names[1]]]
    def check(index, row, name, barrier):
        body = json.loads((oracle/(name+'-request.json')).read_text())
        body['stream'] = False
        choice = json.loads((oracle/(name+'-response.json')).read_text())['choices'][0]
        key = 'quality-'+str(index)+'-'+str(row)
        wire = json.dumps(body, separators=(',', ':')).encode()
        (out/(key+'-request.json')).write_bytes(wire)
        barrier.wait(timeout=10)
        with urlopen(Request(url+'/v1/chat/completions', data=wire, headers={'Content-Type':'application/json'}), timeout=300) as response:
            raw = response.read()
        (out/(key+'-response.json')).write_bytes(raw)
        actual = json.loads(raw)['choices'][0]
        assert actual['prompt_token_ids'] == choice['prompt_token_ids']
        assert actual['response_token_ids'] == choice['response_token_ids']
        return {'key': key, 'oracle': name, 'exact': True}
    for index, group in enumerate(groups):
        barrier = threading.Barrier(len(group))
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(group)) as executor:
            futures = [executor.submit(check, index, row, name, barrier) for row, name in enumerate(group)]
            quality['cases'].extend(f.result() for f in futures)
        save()
    for name in names[:2]:
        body = copy.deepcopy(json.loads((oracle/(name+'-request.json')).read_text()))
        body['stream'] = True
        key = 'stream-'+name
        result = stream_request(url, body, out, key, 128)
        assert result['status'] == 'completed' and result['generated_tokens'] == 128
        events = [json.loads(line) for line in (out/(key+'-events.jsonl')).read_text().splitlines()]
        final = [json.loads(e['data']) for e in events if e.get('data') != '[DONE]' and 'response_token_ids' in json.loads(e['data'])]
        assert len(final) == 1
        reference = json.loads((oracle/(name+'-response.json')).read_text())['choices'][0]
        assert final[0]['response_token_ids'] == reference['response_token_ids']
        assert final[0]['prompt_token_ids'] == reference['prompt_token_ids']
        quality['stream_cases'].append({'oracle': name, 'exact': True})
        save()
    quality['status'] = 'passed'
    save()
    paths = [root/'build/bin/garnet.dll', proof_path, Path(__file__), root/'serve-native-batch2-shared-worker.bat']
    paths += [methods/name for name in ('native_batch_stream_worker.py', 'native_batch_http_bridge.py', 'native_http_bridge.py', 'benchmark_stream.py', 'metrics.py')]
    metadata = {'physical_batch': 2, 'queued_client_concurrency': [1,2,4,8], 'mtp_depth': 0,
        'grouping_delay_ms': 2, 'precision': 'mixed NVFP4/FP8; BF16 vision', 'ready': worker.ready_info,
        'quality': quality, 'file_sha256': {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'scope': 'Exploratory pilot after bounded HTTP ID/stream gates; no broad-quality or 2x admission.'}
    metadata_path = out/'metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf8')
    subprocess.run([sys.executable, str(methods/'benchmark_stream.py'), '--url', url, '--engine', 'garnet',
        '--output', str(out/'benchmark'), '--metadata', str(metadata_path), '--concurrency', '1', '2', '4', '8',
        '--requests', '8', '--warmups', '2', '--max-tokens', '128', '--workload-seed', '3',
        '--image-format', 'jpeg', '--image-order', 'image-first', '--image-nonce-pattern', 'binary-grid'], check=True)
finally:
    if server is not None:
        server.shutdown()
        server.server_close()
    if thread is not None:
        thread.join()
    worker.close()
