"""Keep completed norm-cache validation and real native repeated timing."""
from pathlib import Path
import hashlib,json,tarfile
r=Path('/Users/zsdx/garnet-metal-investigation')
names=['delta-norm-cache-build-20261013','delta-norm-cache-checks-20261013','delta-norm-cache-extended-numerical-20261013','delta-norm-cache-native-vl-20261013','delta-norm-cache-native-summary-20261013']
for name in names:assert json.loads((r/'logs'/name/'exit.json').read_text())['exit_code']==0,name
tests=json.loads((r/'logs/delta-norm-cache-checks-20261013/jobs/job_statuses.json').read_text());assert len(tests)==11 and all(x['exit_code']==0 for x in tests)
assert 'Chunk delta sequential reference PASS 48 cases' in (r/'build-delta-norm-cache/Testing/Temporary/LastTest.log').read_text()
files=[];omitted=[]
def add(path):
 for p in ([path] if path.is_file() else sorted(path.rglob('*'))):
  if not p.is_file():continue
  if p.suffix in ('.f32','.rgb','.dylib'):
   omitted.append(dict(path=str(p.relative_to(r)),bytes=p.stat().st_size,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()))
  else:files.append(p)
for name in names:add(r/'logs'/name)
add(r/'logs/delta-norm-cache-original-source-20261013')
for rel in ['src/backends/metal/metal_context.mm','src/entry/metal_main.cpp','cmake/GarnetMetal.cmake','test2026/metal/matmul_test.cpp','test2026/metal/test_qwen35_extended_prefill_reference.py','test2026/metal/run_chunk_graph.py','test2026/metal/frame_worker_garnet.py','test2026/metal/compare_vllm_metal.py']:
 add(r/'source'/rel)
for rel in ['build_delta_norm_cache.py','check_delta_norm_cache_extended.py','compare_delta_norm_cache_frames.py','three_model_suite_guarded_fused_mlp.py','archive_delta_norm_cache.py','delta-norm-cache-manifest.json','build-delta-norm-cache/CMakeCache.txt','build-delta-norm-cache/Testing/Temporary/LastTest.log']:
 add(r/rel)
a=r/'delta-norm-cache-completed-evidence.tar.gz'
with tarfile.open(a,'w:gz') as t:
 for p in sorted(set(files)):t.add(p,arcname=str(p.relative_to(r)),recursive=False)
proof=dict(sha256=hashlib.file_digest(a.open('rb'),'sha256').hexdigest(),bytes=a.stat().st_size,files=len(set(files)),omitted=omitted,libraries={name:hashlib.file_digest((r/name/'libgarnet.dylib').open('rb'),'sha256').hexdigest() for name in ['build-delta-norm-cache','build-qwen35-extended-prefill']})
assert proof['libraries']['build-qwen35-extended-prefill']=='cc5539990dcb4effbece7c611787a656df56b51271f1f40e1c3e8b5dbf90c8f8'
(r/'delta-norm-cache-completed-proof.json').write_text(json.dumps(proof,indent=2));print(json.dumps({k:v for k,v in proof.items() if k!='omitted'}),flush=True)
