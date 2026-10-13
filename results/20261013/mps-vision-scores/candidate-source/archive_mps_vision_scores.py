import hashlib,json,tarfile
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation')
names=['mps-vision-scores-build-20261013','mps-vision-scores-checks-20261013','native-frame-ab-mps-vision-scores-20261013','mps-vision-scores-frame-summary-20261013','native-frame-ab-mps-vision-confirm-20261013','mps-vision-confirm-frame-summary-20261013','vllm-matched-mps-vision-text-20261013','mps-vision-text-target-summary-20261013','vllm-matched-mps-vision-vl-20261013','mps-vision-vl-target-summary-20261013','mps-vision-default-fallback-20261013']
with tarfile.open(root/'mps-vision-scores-evidence.tar.gz','w:gz') as archive:
 for name in names:
  folder=root/'logs'/name
  assert json.loads((folder/'exit.json').read_text())['exit_code']==0,name
  for path in sorted(folder.rglob('*')):
   if path.is_file() and path.suffix not in ('.f32','.rgb'):archive.add(path,arcname=str(path.relative_to(root/'logs')))
 for name in ['cmake/GarnetMetal.cmake','src/backends/metal/metal_context.mm','test2026/metal/matmul_test.cpp']:
  archive.add(root/'source'/name,arcname='candidate-source/'+name)
 for name in ['build_mps_vision_scores.py','queue_mps_vision_scores.py','queue_mps_vision_confirmation.py','archive_mps_vision_scores.py','build_mps_vision_fallback.py']:
  archive.add(root/name,arcname='candidate-source/'+name)
 build=root/'build-mps-vision-scores'
 info=dict(native_library_sha256=hashlib.sha256((build/'libgarnet.dylib').read_bytes()).hexdigest(),control_library_sha256=hashlib.sha256((root/'build-kv-chunk-vector/libgarnet.dylib').read_bytes()).hexdigest(),cmake_options=[line for line in (build/'CMakeCache.txt').read_text().splitlines() if line.startswith(('GARNET_METAL:BOOL=','GARNET_METAL_XMODEL:BOOL=','GARNET_METAL_EXACT_HALF_PREFILL:BOOL=','GARNET_METAL_MPS_VISION_SCORES:BOOL='))])
 assert info['control_library_sha256']=='564b515ab6b0eb68413ef5a54de6beff47d333f8f1fa6a1188562bf6e5c28ec4'
 path=root/'mps-vision-scores-build-metadata.json';path.write_text(json.dumps(info,indent=2)+'\n');archive.add(path,arcname='candidate-source/build-metadata.json')
 archive.add(build/'Testing/Temporary/LastTest.log',arcname='candidate-source/CTest-LastTest.log')
 archive.add(root/'build-mps-vision-scores-off/Testing/Temporary/LastTest.log',arcname='candidate-source/default-fallback-CTest-LastTest.log')
 archive.add(root/'build-mps-vision-scores-off/CMakeCache.txt',arcname='candidate-source/default-fallback-CMakeCache.txt')
print('Archived complete MPS vision score experiment; selected control preserved')
