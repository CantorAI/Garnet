import json,tarfile
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation')
names=['selected-op-attribution-20261013','vector-slice-build-20261013','vector-slice-checks-20261013','native-ab-vector-slice-20261013','vector-slice-text-summary-20261013','native-frame-ab-vector-slice-20261013','vector-slice-frame-summary-20261013','vector-slice-boundary-checks-20261013']
names+=['vllm-matched-text-vector-slice-20261013','vllm-text-vector-summary-20261013','vllm-matched-vl-vector-slice-20261013','vllm-frame-vector-summary-20261013']
with tarfile.open(root/'vector-slice-evidence.tar.gz','w:gz') as archive:
 for name in names:
  folder=root/'logs'/name
  assert json.loads((folder/'exit.json').read_text())['exit_code']==0,name
  for path in sorted(folder.rglob('*')):
   if path.is_file() and path.suffix not in ('.f32','.rgb'):
    archive.add(path,arcname=str(path.relative_to(root/'logs')))
 for name in ['metal_context.mm']:
  archive.add(root/'source/src/backends/metal'/name,arcname='candidate-source/'+name)
 for name in ['matmul_test.cpp','profile_selected_operations.py']:
  archive.add(root/'source/test2026/metal'/name,arcname='candidate-source/'+name)
 for name in ['build_vector_slice.py','check_vector_boundaries.py','queue_vector_followups.py','queue_vector_matched.py','archive_vector_evidence.py']:
  archive.add(root/name,arcname='candidate-source/'+name)
print('Archived terminal vector slice evidence')
