import hashlib,json,tarfile
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation')
names=['exact-half-prefill-build-20261013','exact-half-prefill-checks-20261013','native-ab-exact-half-prefill-20261013','exact-half-text-summary-20261013','native-frame-ab-exact-half-prefill-20261013','exact-half-frame-summary-20261013','exact-half-op-attribution-20261013']
with tarfile.open(root/'exact-half-prefill-evidence.tar.gz','w:gz') as archive:
 for name in names:
  folder=root/'logs'/name;assert json.loads((folder/'exit.json').read_text())['exit_code']==0,name
  for path in sorted(folder.rglob('*')):
   if path.is_file() and path.suffix not in ('.f32','.rgb'):archive.add(path,arcname=str(path.relative_to(root/'logs')))
 for name in ['metal_context.mm','metal_context.h','compiled_graph.cpp']:
  archive.add(root/'source/src/backends/metal'/name,arcname='candidate-source/'+name)
 archive.add(root/'source/cmake/GarnetMetal.cmake',arcname='candidate-source/GarnetMetal.cmake')
 for name in ['matmul_test.cpp','profile_selected_operations.py']:
  archive.add(root/'source/test2026/metal'/name,arcname='candidate-source/'+name)
 for name in ['build_exact_half_prefill.py','queue_exact_half_prefill.py','archive_exact_half_prefill.py']:
  archive.add(root/name,arcname='candidate-source/'+name)
 build=root/'build-exact-half-prefill'
 info=dict(native_library_sha256=hashlib.sha256((build/'libgarnet.dylib').read_bytes()).hexdigest(),
    cmake_options=[line for line in (build/'CMakeCache.txt').read_text().splitlines() if line.startswith(('GARNET_METAL:BOOL=','GARNET_METAL_XMODEL:BOOL=','GARNET_METAL_EXACT_HALF_PREFILL:BOOL='))])
 path=root/'exact-half-build-metadata.json';path.write_text(json.dumps(info,indent=2)+'\n');archive.add(path,arcname='candidate-source/build-metadata.json')
print('Archived terminal exact-half prefill evidence')
