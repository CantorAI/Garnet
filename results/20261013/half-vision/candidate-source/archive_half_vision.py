import hashlib,json,tarfile
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation')
names=['half-vision-preflight-20261013','half-vision-build-20261013','half-vision-checks-20261013','native-frame-ab-half-vision-20261013','half-vision-frame-summary-20261013','half-vision-profile-20261013']
with tarfile.open(root/'half-vision-evidence.tar.gz','w:gz') as archive:
 for name in names:
  folder=root/'logs'/name
  if name!='half-vision-preflight-20261013':assert json.loads((folder/'exit.json').read_text())['exit_code']==0,name
  for path in sorted(folder.rglob('*')):
   if path.is_file() and path.suffix not in ('.f32','.rgb'):archive.add(path,arcname=str(path.relative_to(root/'logs')))
 for name in ['cmake/GarnetMetal.cmake','src/backends/metal/metal_context.mm','src/backends/metal/compiled_graph.cpp','test2026/metal/matmul_test.cpp']:
  archive.add(root/'source'/name,arcname='candidate-source/'+name)
 for name in ['build_half_vision.py','queue_half_vision.py','guard_half_vision_mac.py','archive_half_vision.py']:
  archive.add(root/name,arcname='candidate-source/'+name)
 build=root/'build-half-vision'
 info=dict(native_library_sha256=hashlib.sha256((build/'libgarnet.dylib').read_bytes()).hexdigest(),control_library_sha256=hashlib.sha256((root/'build-kv-chunk-vector/libgarnet.dylib').read_bytes()).hexdigest(),cmake_options=[line for line in (build/'CMakeCache.txt').read_text().splitlines() if line.startswith(('GARNET_METAL:BOOL=','GARNET_METAL_XMODEL:BOOL=','GARNET_METAL_EXACT_HALF_PREFILL:BOOL=','GARNET_METAL_EXACT_HALF_VISION:BOOL='))])
 assert info['control_library_sha256']=='564b515ab6b0eb68413ef5a54de6beff47d333f8f1fa6a1188562bf6e5c28ec4'
 path=root/'half-vision-build-metadata.json';path.write_text(json.dumps(info,indent=2)+'\n');archive.add(path,arcname='candidate-source/build-metadata.json')
print('Archived complete unsuccessful half-vision experiment; selected control hash preserved')
