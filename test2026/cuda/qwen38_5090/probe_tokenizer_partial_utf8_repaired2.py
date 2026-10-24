"""CPU tokenizer-prefix serialization reproduction, no decoder/GPU invocation."""
import hashlib
import json
import os
from pathlib import Path
import sys

root, out = map(Path, sys.argv[1:3])
sys.path.insert(0, str(root/'build/bin'))
sys.path.insert(0, str(root/'Garnet'))
import garnet
from test2026.cuda.qwen38_5090.native_worker_io import emit
assert not out.exists()
out.mkdir()
tokenizer = garnet.tokenizer()
tokenizer.open(str(root/'models/Qwen3.8-27B-NVFP4'))
fixture_path=root/'results/tokenizer-prefix-expected-utf8.json'
fixture=json.loads(fixture_path.read_text())
rows = []
for index, text in enumerate(['\U0001f535', '\U0001f600', '\u4e2d\u6587', '\u2019\u2014', '### \U0001f535 Color', '\u00e9']):
    ids = list(tokenizer.encode(text))
    for count in range(1, len(ids)+1):
        decoded = tokenizer.decode(ids[:count])
        parts = []
        def write(raw):
            parts.append(raw)
            return len(raw)
        error = None
        try:
            emit({'event':'token', 'text':decoded}, write)
        except Exception as failure:
            error = repr(failure)
        stem = str(index)+'-'+str(count)
        raw = decoded.encode('utf8')
        (out/(stem+'-decoded.bin')).write_bytes(raw)
        (out/(stem+'-emitter.jsonl')).write_bytes(b''.join(parts))
        assert hashlib.sha256(raw).hexdigest()==fixture['cases'][stem]['raw_sha256']
        normalized = fixture['cases'][stem]['text']
        safe = (json.dumps({'event':'token','text':normalized}, ensure_ascii=False)+'\n').encode('utf8')
        (out/(stem+'-normalized.jsonl')).write_bytes(safe)
        assert json.loads(safe)['text'] == normalized
        assert error is None,error
        assert json.loads(b''.join(parts))['text']==normalized
        rows.append({'case':index,'prefix_tokens':count,'ids':ids[:count],'emitter_error':error,
            'decoded_bytes':len(raw),'decoded_sha256':hashlib.sha256(raw).hexdigest(),
            'normalized_sha256':hashlib.sha256(safe).hexdigest()})
report = {'status':'tokenizer_partial_utf8_repaired_emitter_completed','cases':rows,
    'expected_fixture_sha256':hashlib.sha256(fixture_path.read_bytes()).hexdigest(),'scope':'CPU-only tokenizer and serialization; no decoder, CUDA execution or performance claim.'}
(out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print(json.dumps(report, indent=2))
