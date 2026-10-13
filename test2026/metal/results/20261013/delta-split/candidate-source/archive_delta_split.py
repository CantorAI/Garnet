import json,tarfile
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation')
names=['delta-split-build-20261013','delta-split-checks-20261013','native-frame-ab-delta-split-20261013','delta-split-frame-summary-20261013','delta-split-op-attribution-20261013']
names+=['vllm-matched-text-delta-split-20261013','vllm-text-delta-summary-20261013','vllm-matched-vl-delta-split-20261013','vllm-frame-delta-summary-20261013']
with tarfile.open(root/'delta-split-evidence.tar.gz','w:gz') as archive:
 for name in names:
  folder=root/'logs'/name
  assert json.loads((folder/'exit.json').read_text())['exit_code']==0,name
  for path in sorted(folder.rglob('*')):
   if path.is_file() and path.suffix not in ('.f32','.rgb'):archive.add(path,arcname=str(path.relative_to(root/'logs')))
 for name in ['metal_context.mm','metal_context.h','compiled_graph.cpp','compiled_graph.h']:
  archive.add(root/'source/src/backends/metal'/name,arcname='candidate-source/'+name)
 for name in ['matmul_test.cpp','profile_selected_operations.py']:
  archive.add(root/'source/test2026/metal'/name,arcname='candidate-source/'+name)
 for name in ['build_delta_split.py','queue_delta_split.py','queue_delta_matched.py','archive_delta_split.py']:
  archive.add(root/name,arcname='candidate-source/'+name)
print('Archived terminal delta split evidence')
