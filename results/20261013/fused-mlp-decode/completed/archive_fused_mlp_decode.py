"""Preserve completed numerical, diagnostic and repeated request evidence."""
import hashlib,json,tarfile
from pathlib import Path
r=Path('/Users/zsdx/garnet-metal-investigation')
names=['fused-mlp-decode-component-20261013','fused-mlp-decode-component-confirm-20261013','fused-mlp-decode-build-20261013','fused-mlp-decode-checks-20261013','fused-mlp-decode-native-text-20261013','fused-mlp-decode-native-vl-20261013','fused-mlp-decode-vl-summary-20261013','fused-mlp-decode-actual-operations-20261013','fused-mlp-decode-text-confirm-20261013','fused-mlp-decode-vl-confirm-20261013','fused-mlp-decode-confirm-summary-20261013']
for name in names:assert json.loads((r/'logs'/name/'exit.json').read_text())['exit_code']==0,name
checks=json.loads((r/'logs/fused-mlp-decode-checks-20261013/jobs/job_statuses.json').read_text());assert len(checks)==11 and all(v['exit_code']==0 for v in checks)
for name in names[:2]:
 v=json.loads((r/'logs'/name/'jobs/summary.json').read_text());assert len(v['checks'])==48 and all(x['bits_exact'] for x in v['checks'])
assert len(list((r/'logs/fused-mlp-decode-actual-operations-20261013/jobs').glob('*/actual-fusion-proof.json')))==6
files=[];omitted=[]
def add(path):
 for p in ([path] if path.is_file() else sorted(path.rglob('*'))):
  if not p.is_file():continue
  if p.suffix in ('.f32','.rgb','.bin') or p.name=='fused_mlp_decode_gpu':
   omitted.append(dict(path=str(p.relative_to(r)),bytes=p.stat().st_size,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()))
  else:files.append(p)
for name in names:add(r/'logs'/name)
add(r/'logs/fused-mlp-original-source-20261013')
for rel in ['src/backends/metal/metal_context.mm','src/backends/metal/metal_context.h','src/backends/metal/compiled_graph.cpp','src/backends/metal/compiled_graph.h','src/entry/metal_main.cpp','cmake/GarnetMetal.cmake','test2026/metal/fused_mlp_decode_gpu.mm','test2026/metal/make_fused_mlp_decode_shader.py','test2026/metal/prepare_fused_mlp_decode_probe.py','test2026/metal/run_fused_mlp_decode_probe.py','test2026/metal/compare_vllm_metal.py','test2026/metal/run_matched_garnet.py','test2026/metal/frame_worker_garnet.py','test2026/metal/benchmark_matched_process.py','test2026/metal/summarize_native_frames.py']:
 add(r/'source'/rel)
for rel in ['build_fused_mlp_decode.py','benchmark_fused_mlp_dense.py','profile_fused_mlp_operations.py','three_model_suite_guarded_fused_mlp.py','compare_linear_direct_write_frames.py','fused-mlp-native-manifest.json','fused-mlp-decode-manifest.json','build-fused-mlp-decode/CMakeCache.txt','archive_fused_mlp_decode.py']:
 add(r/rel)
archive=r/'fused-mlp-decode-completed-evidence.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for p in sorted(set(files)):t.add(p,arcname=str(p.relative_to(r)),recursive=False)
proof=dict(sha256=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest(),bytes=archive.stat().st_size,files=len(set(files)),omitted=omitted,libraries={name:hashlib.file_digest((r/name/'libgarnet.dylib').open('rb'),'sha256').hexdigest() for name in ['build-fused-mlp-decode','build-small-output-decode']})
assert proof['libraries']['build-small-output-decode']=='56bc06c693f3776c994d3f4078dd2e25093b575f93f6f09c4e2087a4917ee8d4'
(r/'fused-mlp-decode-completed-proof.json').write_text(json.dumps(proof,indent=2))
print(json.dumps({k:v for k,v in proof.items() if k!='omitted'}),flush=True)
