"""Audit complete validation and the preserved pre-inference log-date failure."""
import hashlib,json,sys,tarfile
from pathlib import Path
base=Path(sys.argv[1]).resolve();proof=json.loads((base/'manifest.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for info in proof['files']:
 p=base/info['path'];assert p.resolve().is_relative_to(base) and p.stat().st_size==info['bytes'] and digest(p)==info['sha256']
assert proof['candidate_sha256']=='bad8c5db120df0e6ac107a0fabc6b19708a379f36785b7b5c8869a30fd14534b'
assert proof['control_sha256']=='887b70f521d5af7f948bcb32a2cf94842377e2ecb828bc9280ffb38af6fe058f'
assert proof['jobs']['prefill-stream-checks-20261015']['exit_code']==0 and proof['jobs']['prefill-stream-native-20261015']['exit_code']==1
stage=base/'prefill-stream-staging';manifest=json.loads((stage/'manifest.json').read_text())
with tarfile.open(stage/'source.tar.gz') as archive:
 assert {m.name for m in archive.getmembers()}==set(manifest)
 for m in archive.getmembers():assert hashlib.sha256(archive.extractfile(m).read()).hexdigest()==manifest[m.name]==digest(base/'source'/m.name)
for name,sha in json.loads((stage/'original.json').read_text()).items():assert digest(base/'logs/prefill-stream-original-source-20261014'/name)==sha
checks=base/'logs/prefill-stream-checks-20261015/jobs';stages=json.loads((checks/'statuses.json').read_text());assert len(stages)==19 and all(s['exit_code']==0 and s['interruption'] is None for s in stages)
for s in stages:
 cases=s['accepted_cases'];assert [v['case_index'] for v in cases]==list(range(8 if s['name'].endswith('-exact-tails') else 1))
 for case in cases:
  folder=checks/case['evidence'];assert json.loads((folder/'statuses.json').read_text())==[case['status']] and case['status']['exit_code']==0
  assert json.loads((folder/(s['name']+'-monitor.json')).read_text())
  if s['name'].endswith('-exact-tails'):assert 'Original dense-tail fixture passed case_index='+str(case['case_index']) in (folder/(s['name']+'.log')).read_text()
assert 'Prefill stream exact dependencies, ownership, abort drain and recovery PASS 27' in (checks/'CTest-LastTest.log').read_text()
folder=base/'logs/prefill-stream-native-20261015/jobs';log=(folder/'Qwen3-0.6B-b1-attempt0.log').read_text();assert 'FileNotFoundError' in log and 'prefill-stream-checks-20261014/jobs/metadata.json' in log
assert not (folder/'Qwen3-0.6B-b1-attempt0/Qwen3-0.6B-b1-control-0').exists()
old=(base/'source/test2026/metal/compare_prefill_stream.py').read_bytes();new=Path(__file__).with_name('compare_prefill_stream.py').read_bytes();assert old.replace(b'prefill-stream-checks-20261014',b'prefill-stream-checks-20261015')==new
result=dict(verified_files=len(proof['files']),original_validation_stages=19,complete_native_requests=0,reason='Wrong validation-log date before any inference worker; comparator path only repaired',scope='Completed numerical validation; no serving performance from failed launch',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
