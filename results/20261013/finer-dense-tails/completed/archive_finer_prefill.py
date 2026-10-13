"""Archive only completed owned evidence; exclude retained private tensor/image bytes."""
import hashlib,json,tarfile
from pathlib import Path
r=Path('/Users/zsdx/garnet-metal-investigation')
names=['dense-intermediate-profile-checks-20261013','finer-dense-tails-native-20261013','finer-prefill-compatibility-20261013','finer-dense-tails-confirm-20261013','finer-prefill-48-requests-20261013','equivalent-f32-finer-prefill-20261013','equivalent-f32-finer-prefill-summary-20261013']
for name in names:assert json.loads((r/'logs'/name/'exit.json').read_text())['exit_code']==0,name
summary=json.loads((r/'logs/equivalent-f32-finer-prefill-20261013/jobs/summary.json').read_text())
assert len(summary['records'])==48 and all(g['exact_outputs'] and g['exact_prompt_ids'] and g['clean_backend_shutdown'] for g in summary['groups'])
assert not summary['counts_as_default_bf16_speedup']
files=[];omitted=[]
def add(path):
 for p in ([path] if path.is_file() else sorted(path.rglob('*'))):
  if not p.is_file():continue
  if p.suffix in ('.f32','.rgb') or p.name=='raw.bin':
   omitted.append(dict(path=str(p.relative_to(r)),bytes=p.stat().st_size,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()))
  else:files.append(p)
for name in names:add(r/'logs'/name)
add(r/'logs/finer-dense-tails-original-source-20261013')
for rel in ['xModel/qwen/metal_dense/decoder.py','xModel/qwen35/vl_0_8b/metal_decoder.py','test2026/metal/run_matched_garnet.py','test2026/metal/run_dense_profile_guard.py','test2026/metal/test_dense_intermediate_reference.py','test2026/metal/run_dense_reference_capture.py','test2026/metal/reference_dense.py','test2026/metal/validate_dense_capture.py','test2026/metal/compare_vllm_metal.py','test2026/metal/run_matched_vllm.py','test2026/metal/benchmark_matched_process.py','test2026/metal/frame_worker_garnet.py','test2026/metal/run_matched_frame_process.py','src/backends/metal/metal_context.mm','cmake/GarnetMetal.cmake']:
 add(r/'source'/rel)
for rel in ['check_dense_intermediate_profiles.py','benchmark_finer_dense_tails.py','check_finer_prefill_compatibility.py','check_finer_prefill_48_requests.py','compare_equivalent_f32_finer_prefill.py','equivalent_f32_summarize_vllm_comparison.py','archive_finer_prefill.py','finer-prefill-manifest.json','build-small-output-decode/CMakeCache.txt']:
 add(r/rel)
for name in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct'):
 add(r/'models-exact-f32'/name/'exact-value-equivalence.json')
archive=r/'finer-prefill-completed-evidence.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for p in sorted(set(files)):t.add(p,arcname=str(p.relative_to(r)),recursive=False)
proof=dict(archive_sha256=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest(),archive_bytes=archive.stat().st_size,files=len(set(files)),omitted_private_assets=omitted,library_sha256=hashlib.file_digest((r/'build-small-output-decode/libgarnet.dylib').open('rb'),'sha256').hexdigest())
(r/'finer-prefill-completed-proof.json').write_text(json.dumps(proof,indent=2))
print(json.dumps({k:v for k,v in proof.items() if k!='omitted_private_assets'}),flush=True)
