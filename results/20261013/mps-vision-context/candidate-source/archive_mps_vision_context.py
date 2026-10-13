import hashlib,json,tarfile
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation')
names=['mps-vision-context-build-20261013','mps-vision-context-checks-20261013','native-frame-ab-mps-vision-context-20261013','mps-vision-context-frame-summary-20261013']
with tarfile.open(root/'mps-vision-context-evidence.tar.gz','w:gz') as archive:
 for name in names:
  folder=root/'logs'/name
  assert json.loads((folder/'exit.json').read_text())['exit_code']==0,name
  for path in sorted(folder.rglob('*')):
   if path.is_file() and path.suffix not in ('.f32','.rgb'):archive.add(path,arcname=str(path.relative_to(root/'logs')))
 for name in ['cmake/GarnetMetal.cmake','src/backends/metal/metal_context.mm','test2026/metal/matmul_test.cpp']:
  archive.add(root/'source'/name,arcname='candidate-source/'+name)
 for name in ['build_mps_vision_context.py','queue_mps_vision_context.py','guard_mps_vision_context.py','archive_mps_vision_context.py']:
  archive.add(root/name,arcname='candidate-source/'+name)
 build=root/'build-mps-vision-context'
 info=dict(native_library_sha256=hashlib.sha256((build/'libgarnet.dylib').read_bytes()).hexdigest(),control_library_sha256=hashlib.sha256((root/'build-mps-vision-scores/libgarnet.dylib').read_bytes()).hexdigest(),cmake_options=[line for line in (build/'CMakeCache.txt').read_text().splitlines() if line.startswith(('GARNET_METAL:BOOL=','GARNET_METAL_XMODEL:BOOL=','GARNET_METAL_EXACT_HALF_PREFILL:BOOL=','GARNET_METAL_MPS_VISION_SCORES:BOOL=','GARNET_METAL_MPS_VISION_CONTEXT:BOOL='))])
 assert info['control_library_sha256']=='c9f8c76aa2657669d6a7b10eba459481ac797fc64904d25360933b565f9d77d1'
 path=root/'mps-vision-context-build-metadata.json';path.write_text(json.dumps(info,indent=2)+'\n');archive.add(path,arcname='candidate-source/build-metadata.json')
 archive.add(build/'Testing/Temporary/LastTest.log',arcname='candidate-source/CTest-LastTest.log')
print('Archived complete MPS vision context experiment; selected control preserved')
