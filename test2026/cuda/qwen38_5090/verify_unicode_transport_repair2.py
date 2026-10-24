"""Independent maximal-subpart UTF-8 oracle against native actual emitter bytes."""
from pathlib import Path
import hashlib
import json
import zipfile
out=Path(__file__).parent;sha=lambda raw:hashlib.sha256(raw).hexdigest()
archive=out/'unicode-transport-repair2-evidence.zip'
with zipfile.ZipFile(archive) as z,zipfile.ZipFile(out/'unicode-transport-repair-evidence.zip') as old:
    assert z.testzip() is None and old.testzip() is None
    evidence=json.loads(z.read('evidence-manifest.json'));prior=json.loads(old.read('evidence-manifest.json'))
    for name,digest in evidence['files'].items():assert sha(z.read(name))==digest
    for name,digest in prior['files'].items():assert sha(old.read(name))==digest
    for name,digest in evidence['linked_archives'].items():assert sha((out/name).read_bytes())==digest
    assert evidence['runtime_sha256']==prior['runtime_sha256']
    base='results/tokenizer-partial-utf8-repaired-attempt2/'
    oldbase='results/tokenizer-partial-utf8-attempt1/'
    firstbase='results/tokenizer-partial-utf8-repaired-attempt1/'
    report=json.loads(z.read(base+'report.json'));before=json.loads(old.read(oldbase+'report.json'))
    assert report['status']=='tokenizer_partial_utf8_repaired_emitter_completed'
    fixture_raw=z.read('results/tokenizer-prefix-expected-utf8.json');fixture=json.loads(fixture_raw)
    assert sha(fixture_raw)==report['expected_fixture_sha256'] and fixture['source_archive_sha256']==sha((out/'unicode-transport-repair-evidence.zip').read_bytes())
    assert len(report['cases'])==len(before['cases'])
    original_failures=0;first_mismatches=0
    for previous,row in zip(before['cases'],report['cases']):
        for key in ('case','prefix_tokens','ids','decoded_sha256'):assert row[key]==previous[key]
        stem=str(row['case'])+'-'+str(row['prefix_tokens'])
        raw=z.read(base+stem+'-decoded.bin')
        assert raw==old.read(oldbase+stem+'-decoded.bin') and sha(raw)==row['decoded_sha256']
        expected=raw.decode('utf8','replace')
        assert fixture['cases'][stem]=={'raw_sha256':sha(raw),'text':expected}
        actual=z.read(base+stem+'-emitter.jsonl')
        assert actual.count(b'\n')==1 and actual.endswith(b'\n')
        assert json.loads(actual)=={'event':'token','text':expected} and row['emitter_error'] is None
        original_failures+=previous['old_error'] is not None
        first_mismatches+=json.loads(old.read(firstbase+stem+'-emitter.jsonl'))['text']!=expected
    assert original_failures>0 and first_mismatches==3
    events=[json.loads(line) for line in z.read('results/tokenizer-partial-utf8-repaired2-ledger/commands.jsonl').decode().splitlines()]
    assert events[-1]['event']=='finished' and events[-1]['exit_code']==0
    source_hash=sha(z.read('Garnet/test2026/cuda/qwen38_5090/native_worker_io.py'))
result={'status':'canonical_unicode_transport_repair_independently_verified','archive_sha256':sha(archive.read_bytes()),
    'files_verified':len(evidence['files']),'prefix_cases':len(report['cases']),'original_failures':original_failures,
    'first_repair_mismatches':first_mismatches,'all_repaired_bytes_match_cpython':True,
    'repair_source_sha256':source_hash,'runtime_sha256':evidence['runtime_sha256'],'performance_admitted':False}
(out/'unicode-transport-repair2-verified.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
