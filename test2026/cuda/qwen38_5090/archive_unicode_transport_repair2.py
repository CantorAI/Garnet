"""Keep second repair separate from the independently rejected first repair."""
from pathlib import Path
import hashlib
import json
import zipfile
root=Path('D:/CantorAI/5090-study-20261023');results=root/'results'
target=results/'unicode-transport-repair2-evidence.zip';assert not target.exists()
folder=results/'tokenizer-partial-utf8-repaired-attempt2'
ledger=results/'tokenizer-partial-utf8-repaired2-ledger'
events=[json.loads(line) for line in (ledger/'commands.jsonl').read_text().splitlines()]
assert events[-1]['event']=='finished' and events[-1]['exit_code']==0
files=[p for directory in (folder,ledger) for p in directory.rglob('*') if p.is_file()]
files += [root/'probe-tokenizer-partial-utf8-repaired2.py',root/'test-tokenizer-partial-utf8-repaired2.bat',
    results/'tokenizer-prefix-expected-utf8.json',root/'Garnet/test2026/cuda/qwen38_5090/native_worker_io.py',Path(__file__)]
sha=lambda raw:hashlib.sha256(raw).hexdigest()
raw={p.relative_to(root).as_posix():p.read_bytes() for p in files}
manifest={'files':{name:sha(data) for name,data in raw.items()},
    'linked_archives':{'unicode-transport-repair-evidence.zip':sha((results/'unicode-transport-repair-evidence.zip').read_bytes())},
    'runtime_sha256':{name:sha((root/'build/bin'/name).read_bytes()) for name in ('xlang3.exe','garnet.dll')}}
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    for name,data in raw.items():z.writestr(name,data)
    z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
print(json.dumps({'bytes':target.stat().st_size,'sha256':sha(target.read_bytes()),'files':len(raw)}))
