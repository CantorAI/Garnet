"""Retain CPU control lineage and immutable original HTTP failure before retry."""
from pathlib import Path
import hashlib
import json
import zipfile

root = Path('D:/CantorAI/5090-study-20261023')
results = root/'results'
target = results/'unicode-transport-repair-evidence.zip'
assert not target.exists()
folders = ['unicode-json-transport-attempt1','unicode-json-transport-ledger',
    'tokenizer-partial-utf8-attempt1','tokenizer-partial-utf8-ledger',
    'tokenizer-partial-utf8-repaired-attempt1','tokenizer-partial-utf8-repaired-ledger']
for folder in folders:
    if folder.endswith('ledger'):
        events = [json.loads(line) for line in (results/folder/'commands.jsonl').read_text().splitlines()]
        assert events[-1]['event']=='finished' and events[-1]['exit_code']==0
files = [p for folder in folders for p in (results/folder).rglob('*') if p.is_file()]
files += [root/name for name in ('probe-unicode-json-transport.py','probe-tokenizer-partial-utf8.py',
    'probe-tokenizer-partial-utf8-repaired.py','test-tokenizer-partial-utf8.bat','test-tokenizer-partial-utf8-repaired.bat')]
files += [root/'Garnet/test2026/cuda/qwen38_5090/native_worker_io.py', Path(__file__)]
raw = {p.relative_to(root).as_posix():p.read_bytes() for p in files}
failure = results/'batch-http-pilot-attempt1-evidence.zip'
with zipfile.ZipFile(failure) as old:
    raw['baseline/native_worker_io.py'] = old.read('Garnet/test2026/cuda/qwen38_5090/native_worker_io.py')
sha = lambda b:hashlib.sha256(b).hexdigest()
manifest = {'files':{name:sha(data) for name,data in raw.items()},
    'linked_archives':{failure.name:sha(failure.read_bytes())},
    'runtime_sha256':{name:sha((root/'build/bin'/name).read_bytes()) for name in ('xlang3.exe','garnet.dll')},
    'scope':'CPU tokenizer-prefix transport intervention only; not model performance.'}
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    for name,data in raw.items():z.writestr(name,data)
    z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
print(json.dumps({'bytes':target.stat().st_size,'sha256':sha(target.read_bytes()),'files':len(raw)}))
