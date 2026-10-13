import hashlib,json,tarfile
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation')
stages=['parallel-delta-build','parallel-delta-checks','native-frame-ab-parallel-delta','parallel-delta-frame-summary']
with tarfile.open(root/'parallel-delta-evidence.tar.gz','w:gz') as archive:
 for attempt in ('','-retry1','-retry2'):
  for index,stage in enumerate(stages):
   name=stage+attempt+'-20261013';folder=root/'logs'/name
   expected=0 if attempt=='-retry2' or index==0 else 1
   assert json.loads((folder/'exit.json').read_text())['exit_code']==expected,name
   for path in sorted(folder.rglob('*')):
    if path.is_file() and path.suffix not in ('.f32','.rgb'):archive.add(path,arcname=str(path.relative_to(root/'logs')))
 for name in ['native-frame-ab-parallel-delta-confirm-20261013','parallel-delta-confirm-frame-summary-20261013','vllm-matched-parallel-delta-text-20261013','parallel-delta-text-target-summary-20261013','vllm-matched-parallel-delta-vl-20261013','parallel-delta-vl-target-summary-20261013','equivalent-f32-parallel-delta-text-20261013','equivalent-f32-parallel-delta-text-summary-20261013','equivalent-f32-parallel-delta-vl-20261013','equivalent-f32-parallel-delta-vl-summary-20261013','parallel-delta-default-fallback-20261013']:
  folder=root/'logs'/name;assert json.loads((folder/'exit.json').read_text())['exit_code']==0,name
  for path in sorted(folder.rglob('*')):
   if path.is_file() and path.suffix not in ('.f32','.rgb'):archive.add(path,arcname=str(path.relative_to(root/'logs')))
 for name in ['cmake/GarnetMetal.cmake','src/backends/metal/metal_context.mm','test2026/metal/matmul_test.cpp']:
  archive.add(root/'source'/name,arcname='candidate-source/'+name)
 for name in ['build_parallel_delta.py','queue_parallel_delta.py','queue_parallel_delta_retry1.py','queue_parallel_delta_retry2.py','archive_parallel_delta.py','queue_parallel_delta_confirmation.py','build_parallel_delta_fallback.py','parallel-delta-installed.json','parallel-delta-initial-artifact-metadata.json']:
  archive.add(root/name,arcname='candidate-source/'+name)
 archive.add(root/'parallel-delta-initial-source-evidence.tar.gz',arcname='candidate-source/initial-transfer-source-snapshot.tar.gz')
 build=root/'build-parallel-delta';control=root/'build-mps-vision-scores/libgarnet.dylib'
 info=dict(candidate_sha256=hashlib.sha256((build/'libgarnet.dylib').read_bytes()).hexdigest(),control_sha256=hashlib.sha256(control.read_bytes()).hexdigest(),
    cmake_options=[line for line in (build/'CMakeCache.txt').read_text().splitlines() if line.startswith(('GARNET_METAL:BOOL=','GARNET_METAL_XMODEL:BOOL=','GARNET_METAL_EXACT_HALF_PREFILL:BOOL=','GARNET_METAL_MPS_VISION_SCORES:BOOL=','GARNET_METAL_PARALLEL_DELTA_SPLIT:BOOL='))])
 assert info['control_sha256']=='c9f8c76aa2657669d6a7b10eba459481ac797fc64904d25360933b565f9d77d1'
 path=root/'parallel-delta-build-metadata.json';path.write_text(json.dumps(info,indent=2)+'\n');archive.add(path,arcname='candidate-source/build-metadata.json')
 archive.add(build/'Testing/Temporary/LastTest.log',arcname='candidate-source/CTest-LastTest.log')
 archive.add(root/'build-parallel-delta-off/Testing/Temporary/LastTest.log',arcname='candidate-source/default-fallback-CTest-LastTest.log')
print('Archived parallel recurrent reduction evidence with transfer failures and clean rebuild')
