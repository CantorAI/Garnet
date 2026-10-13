"""Archive extended-profile evidence, including failures and preflight recovery."""
from pathlib import Path
import hashlib,json,tarfile
r=Path('/Users/zsdx/garnet-metal-investigation')
required=['qwen35-extended-prefill-build-retry1-20261013','qwen35-extended-prefill-numerical-retry1-20261013','qwen35-extended-prefill-guard-recovery-20261013','qwen35-extended-prefill-checks-retry2-20261013','qwen35-extended-prefill-native-vl-retry2-20261013','qwen35-extended-prefill-native-summary-retry1-20261013','qwen35-extended-prefill-confirm-20261013','qwen35-extended-prefill-confirm-summary-20261013']
for name in required:assert json.loads((r/'logs'/name/'exit.json').read_text())['exit_code']==0,name
checks=json.loads((r/'logs/qwen35-extended-prefill-checks-retry2-20261013/jobs/job_statuses.json').read_text());assert len(checks)==11 and all(v['exit_code']==0 for v in checks)
for name in ['qwen35-extended-prefill-native-vl-retry2-20261013','qwen35-extended-prefill-confirm-20261013']:
 d=r/'logs'/name/'jobs';summary=json.loads((d/'native_frame_summary.json').read_text());assert len(summary['records'])==24
 for v in summary['records']:
  candidate='-garnet-control-' not in v['folder'];batch=v['case']['batch']
  expected=([80] if batch==1 else [40,40]) if candidate else ([64,16] if batch==1 else [32,32,16])
  assert v['prefill_chunk_plan']==expected
  assert all(v['finished'])
 for folder in d.glob('Qwen3.5-*-garnet-*'):
  ready=json.loads((folder/'protocol/ready.json').read_text());assert ready['extended_admission']['state_preserved']
  case=json.loads((folder/'case.json').read_text())
  rows=[json.loads(line.split(' ',1)[1]) for line in (folder/'output.log').read_text().splitlines() if line.startswith('matched_frame_result ')]
  assert rows and all(v['case']['chunk']==case['chunk'] for v in rows),'Case profile changed after execution'
files=[];omitted=[]
def add(path):
 for p in ([path] if path.is_file() else sorted(path.rglob('*'))):
  if not p.is_file():continue
  if p.suffix in ('.f32','.rgb','.bin','.dylib'):
   omitted.append(dict(path=str(p.relative_to(r)),bytes=p.stat().st_size,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()))
  else:files.append(p)
for p in sorted((r/'logs').glob('qwen35-extended-prefill-*')):
 if p.name!='qwen35-extended-prefill-archive-retry1-20261013':add(p)
for rel in ['src/backends/metal/metal_context.h','src/backends/metal/metal_context.mm','src/backends/metal/compiled_graph.cpp','src/entry/metal_main.cpp','cmake/GarnetMetal.cmake','xModel/qwen35/vl_0_8b/metal_decoder.py','test2026/metal/test_qwen35_extended_prefill_reference.py','test2026/metal/run_chunk_graph.py','test2026/metal/frame_worker_garnet.py','test2026/metal/run_matched_frame_process.py','test2026/metal/summarize_native_frames.py','test2026/metal/compare_vllm_metal.py','test2026/metal/test_worldsense_guard.py']:
 add(r/'source'/rel)
for rel in ['qwen35-extended-prefill-manifest.json','qwen35-extended-fixed-manifest.json','build_qwen35_extended_prefill.py','check_qwen35_extended_prefill.py','check_qwen35_extended_prefill_retry1.py','run_extended_chunk_recovery.py','compare_qwen35_extended_prefill_frames.py','three_model_suite_guarded_fused_mlp.py','archive_qwen35_extended_prefill.py','build-qwen35-extended-prefill/CMakeCache.txt','libgarnet-qwen35-extended-initial.dylib']:
 add(r/rel)
archive=r/'qwen35-extended-prefill-completed-evidence.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for p in sorted(set(files)):t.add(p,arcname=str(p.relative_to(r)),recursive=False)
proof=dict(sha256=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest(),bytes=archive.stat().st_size,files=len(set(files)),omitted=omitted,libraries={name:hashlib.file_digest((r/name/'libgarnet.dylib').open('rb'),'sha256').hexdigest() for name in ['build-qwen35-extended-prefill','build-small-output-decode']})
assert proof['libraries']['build-small-output-decode']=='56bc06c693f3776c994d3f4078dd2e25093b575f93f6f09c4e2087a4917ee8d4'
(r/'qwen35-extended-prefill-completed-proof.json').write_text(json.dumps(proof,indent=2))
print(json.dumps({k:v for k,v in proof.items() if k!='omitted'}),flush=True)
