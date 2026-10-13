import hashlib,json,tarfile
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation')
names=['exact-f32-'+m+'-20261013' for m in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B')]
names+=['exact-f32-actual-dtypes-20261013','exact-f32-actual-dtypes-fit-20261013','equivalent-f32-text-20261013','equivalent-f32-text-summary-20261013','equivalent-f32-vl-20261013','equivalent-f32-vl-summary-20261013']
with tarfile.open(root/'equivalent-f32-evidence.tar.gz','w:gz') as archive:
 for name in names:
  folder=root/'logs'/name
  expected=1 if name=='exact-f32-actual-dtypes-20261013' else 0
  assert json.loads((folder/'exit.json').read_text())['exit_code']==expected,name
  for path in sorted(folder.rglob('*')):
   if path.is_file() and path.suffix not in ('.f32','.rgb'):archive.add(path,arcname=str(path.relative_to(root/'logs')))
 for name in ['make_exact_f32_checkpoint.py','run_exact_f32_dtype_audit.py','audit_comparator_dtypes.py','run_matched_vllm.py','frame_worker_vllm.py']:
  archive.add(root/'source/test2026/metal'/name,arcname='study-source/'+name)
 for name in ['compare_equivalent_f32.py','queue_equivalent_f32.py','equivalent_f32_summarize_vllm_comparison.py','equivalent_f32_summarize_frame_comparison.py','archive_equivalent_f32.py']:
  archive.add(root/name,arcname='study-source/'+name)
 # The failed initial FP32 audit used the prior complete worker and the
 # initial parent without the larger VL memory allowance. Preserve both.
 archive.add(root/'complete_comparator_dtype_audit_before_budget.py',arcname='study-source/initial-worker.py')
 runner=(root/'source/test2026/metal/run_exact_f32_dtype_audit.py').read_text()
 initial=runner.replace(",dtype,'.45' if model=='Qwen3.5-0.8B' else '.30']",",dtype]")
 path=root/'initial_exact_f32_dtype_parent.py';path.write_bytes(initial.encode())
 recorded=json.loads((root/'logs/exact-f32-actual-dtypes-20261013/jobs/metadata.json').read_text())
 assert hashlib.sha256(path.read_bytes()).hexdigest()==recorded['runner_sha256']
 assert hashlib.sha256((root/'complete_comparator_dtype_audit_before_budget.py').read_bytes()).hexdigest()==recorded['worker_sha256']
 archive.add(path,arcname='study-source/initial-parent.py')
 for m in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
  folder=root/'models-exact-f32'/m
  for path in sorted(folder.glob('*.json')):
   archive.add(path,arcname='equivalent-model-metadata/'+m+'/'+path.name)
  proof=json.loads((folder/'exact-value-equivalence.json').read_text())
  for path,key in [(root/'models'/m/'model.safetensors','source_sha256'),(folder/'model.safetensors','destination_sha256')]:
   with path.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==proof[key]
 print('Original and exact-F32 checkpoints rehashed; all evidence archived without model binaries')
