import hashlib,json,tarfile
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation')
names=['comparator-actual-dtypes-20261013','comparator-actual-dtypes-inproc-20261013','comparator-actual-dtypes-complete-20261013']
versions=['initial_comparator_dtype_audit.py','first_inproc_comparator_dtype_audit.py','complete_comparator_dtype_audit_before_budget.py']
with tarfile.open(root/'comparator-dtype-evidence.tar.gz','w:gz') as archive:
 for i,name in enumerate(names):
  folder=root/'logs'/name
  assert json.loads((folder/'exit.json').read_text())['exit_code']==(1 if i==0 else 0)
  metadata=json.loads((folder/'jobs/metadata.json').read_text())
  assert hashlib.sha256((root/versions[i]).read_bytes()).hexdigest()==metadata['worker_sha256']
  archive.add(root/versions[i],arcname=name+'/actual-worker-source.py')
  for p in sorted(folder.rglob('*')):
   if p.is_file():archive.add(p,arcname=str(p.relative_to(root/'logs')))
 archive.add(root/'source/test2026/metal/run_comparator_dtype_audit.py',arcname='audit-source/run_comparator_dtype_audit.py')
 base=root/'comparators/vllm/venv/lib/python3.12/site-packages/vllm_metal'
 for name in ['v1/model_lifecycle.py','quant/awq_loader.py','compiled_mlp.py','attention/impls/sdpa_wrapper.py','attention/impls/linear.py']:
  archive.add(base/name,arcname='installed-comparator-source/'+name)
 archive.add(root/'archive_comparator_dtype_audit.py',arcname='audit-source/archive_comparator_dtype_audit.py')
print('Archived actual loaded-weight and activation dtypes, plus failed RPC and partial initial coverage')
