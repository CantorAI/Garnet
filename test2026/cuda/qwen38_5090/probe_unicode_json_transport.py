"""CPU-only old emitter vs UTF-8 JSON control in the actual embedded runtime."""
import hashlib
import json
from pathlib import Path
import sys

root, out = map(Path, sys.argv[1:3])
sys.path.insert(0, str(root/'Garnet'))
from test2026.cuda.qwen38_5090.native_worker_io import emit
assert not out.exists()
out.mkdir()
rows = []
for name, text in [('ascii', 'A "quote"\nline'), ('bmp', '\u4e2d\u6587\u00e9'), ('nonbmp', '\U0001f535\U0001f600')]:
    value = {'event':'token','text':text,'token_id':123}
    parts = []
    def write(raw):
        parts.append(raw)
        return len(raw)
    error = None
    try:
        emit(value, write)
    except Exception as failure:
        error = repr(failure)
    old = b''.join(parts)
    (out/(name+'-old.jsonl')).write_bytes(old)
    candidate = (json.dumps(value, separators=(',', ':'), ensure_ascii=False)+'\n').encode('utf8')
    (out/(name+'-utf8.jsonl')).write_bytes(candidate)
    assert json.loads(candidate) == value
    assert candidate.endswith(b'\n') and candidate.count(b'\n') == 1
    rows.append({'name':name,'old_error':error,'old_bytes':len(old),
        'old_sha256':hashlib.sha256(old).hexdigest(), 'utf8_sha256':hashlib.sha256(candidate).hexdigest(),
        'utf8_round_trip_exact':True})
report = {'status':'unicode_transport_control_completed','cases':rows,'scope':'CPU-only serialization control; no GPU/model/performance claim.'}
(out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print(json.dumps(report, indent=2))
