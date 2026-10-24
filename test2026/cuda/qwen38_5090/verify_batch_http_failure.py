"""Classify the failed HTTP pilot without promoting partial cohorts to success."""
from pathlib import Path
import hashlib
import json
import sys
import zipfile

out = Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent
repo = Path(sys.argv[2]) if len(sys.argv)>2 else Path('D:/CantorAI/work/qwen38-5090')
sys.path.insert(0, str(repo/'test2026/cuda/qwen38_5090'))
from compare_http_runs import verify_run
sha = lambda raw: hashlib.sha256(raw).hexdigest()
archive = out/'batch-http-pilot-attempt1-evidence.zip'
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    evidence = json.loads(z.read('evidence-manifest.json'))
    for name, digest in evidence['files'].items(): assert sha(z.read(name)) == digest
    for name, digest in evidence['linked_archives'].items(): assert sha((out/name).read_bytes()) == digest
    base = 'results/batch-http-pilot-attempt1/'
    doc = lambda name: json.loads(z.read(name))
    workflow = doc(base+'workflow.json')
    assert workflow['status'] == 'process_failed' and workflow['exit_code'] == 1
    quality = doc(base+'native/quality.json')
    assert quality['status'] == 'passed' and len(quality['cases']) == 8 and len(quality['stream_cases']) == 2
    names = [name for name in z.namelist() if name.startswith(base+'native/benchmark/') and name.endswith('-manifest.json')]
    assert len(names) == 1
    manifest = doc(names[0]); prefix = names[0].rsplit('/',1)[0]+'/'
    assert manifest['status'] == 'failed' and len(manifest['cohorts']) == 6
    # Arithmetic replay of saved successful cohorts is diagnostic only. Preserve
    # the original failed manifest; never pass this derived copy as a full run.
    partial = dict(manifest); partial['status'] = 'completed'
    arithmetic = verify_run(partial, lambda name:z.read(prefix+name))
    failures = []
    for name in z.namelist():
        if name.startswith(prefix) and name.endswith('-result.json'):
            record = doc(name)
            if record['status'] != 'completed': failures.append(record)
    assert len(failures) == 6 and all('-s3-image-c4-r' in v['key'] for v in failures)
    stderr = z.read(base+'native/worker/worker-stderr.log')
    assert b'TypeError: ord() expected a character' in stderr
    assert b"when serializing dict item 'text'" in stderr
    events = [json.loads(line) for line in z.read(base+'ledger/commands.jsonl').decode().splitlines()]
    assert events[-1]['event'] == 'finished' and events[-1]['exit_code'] == 1
    resource = doc(base+'resources/resource-monitor.json')
    assert resource['child_exit_code'] == 1
    result = {'status':'batch_http_failed_pilot_evidence_verified', 'archive_sha256':sha(archive.read_bytes()),
        'files_verified':len(evidence['files']), 'library_sha256':evidence['library_sha256'],
        'quality_report_passed':True,'completed_cohorts_replayed_for_diagnosis':len(arithmetic['cohorts']),
        'failed_requests':failures,'observed_terminal_error':'JSON ASCII text encoding raised ord() TypeError',
        'full_run_passed':False,'performance_admitted':False,'two_x_established':False,
        'scope':'Failure accounting and partial metric arithmetic only; does not establish unique cause or a completed benchmark comparison.'}
(out/'batch-http-pilot-attempt1-failure-verified.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
