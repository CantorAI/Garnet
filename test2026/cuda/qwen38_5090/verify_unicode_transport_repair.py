"""Check native prefix failure and repaired bytes using independent CPython UTF-8."""
from pathlib import Path
import hashlib
import json
import zipfile

out = Path(__file__).parent
sha = lambda data:hashlib.sha256(data).hexdigest()
archive = out/'unicode-transport-repair-evidence.zip'
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    evidence = json.loads(z.read('evidence-manifest.json'))
    for name,digest in evidence['files'].items():assert sha(z.read(name))==digest
    for name,digest in evidence['linked_archives'].items():assert sha((out/name).read_bytes())==digest
    with zipfile.ZipFile(out/'batch-http-pilot-attempt1-evidence.zip') as failure:
        assert z.read('baseline/native_worker_io.py')==failure.read('Garnet/test2026/cuda/qwen38_5090/native_worker_io.py')
    oldbase = 'results/tokenizer-partial-utf8-attempt1/'
    newbase = 'results/tokenizer-partial-utf8-repaired-attempt1/'
    old = json.loads(z.read(oldbase+'report.json')); new = json.loads(z.read(newbase+'report.json'))
    assert old['status']=='tokenizer_partial_utf8_control_completed'
    assert new['status']=='tokenizer_partial_utf8_repaired_emitter_completed'
    assert len(old['cases'])==len(new['cases'])
    failures = 0
    for before,after in zip(old['cases'],new['cases']):
        for key in ('case','prefix_tokens','ids','decoded_sha256','normalized_sha256'):assert before[key]==after[key]
        stem = str(before['case'])+'-'+str(before['prefix_tokens'])
        raw = z.read(oldbase+stem+'-decoded.bin')
        assert raw==z.read(newbase+stem+'-decoded.bin') and sha(raw)==before['decoded_sha256']
        normalized = raw.decode('utf8','replace')
        try:raw.decode('utf8');invalid=False
        except UnicodeDecodeError:invalid=True
        if invalid:
            assert before['old_error']=="TypeError('ord() expected a character')"
            failures+=1
        else:
            assert before['old_error'] is None
            assert json.loads(z.read(oldbase+stem+'-old.jsonl'))['text']==normalized
        assert after['emitter_error'] is None
        emitted = z.read(newbase+stem+'-emitter.jsonl')
        assert emitted.endswith(b'\n') and emitted.count(b'\n')==1
        assert json.loads(emitted)=={'event':'token','text':normalized}
    assert failures>0
    source = z.read('Garnet/test2026/cuda/qwen38_5090/native_worker_io.py').decode().replace('\r\n','\n')
    baseline = z.read('baseline/native_worker_io.py').decode().replace('\r\n','\n')
    inserted = "    if isinstance(value.get('text'),str):\n        # The native tokenizer can return an incomplete UTF-8 byte prefix.\n        # Normalize transport text; token IDs/counts and the caller stay intact.\n        value=dict(value)\n        value['text']=value['text'].encode('utf8').decode('utf8','replace')\n"
    assert source.replace(inserted,'',1)==baseline and source.count(inserted)==1
    repair_digest=sha(z.read('Garnet/test2026/cuda/qwen38_5090/native_worker_io.py'))
result = {'status':'unicode_transport_repair_independently_verified','archive_sha256':sha(archive.read_bytes()),
    'files_verified':len(evidence['files']),'prefix_cases':len(old['cases']),'old_invalid_utf8_failures':failures,
    'repaired_all_prefixes_exact':True,'partial_write_bytes':3,'repair_source_sha256':repair_digest,
    'runtime_sha256':evidence['runtime_sha256'],'performance_admitted':False}
(out/'unicode-transport-repair-verified.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
