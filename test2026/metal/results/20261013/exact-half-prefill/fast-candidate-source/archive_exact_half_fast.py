import hashlib,json,tarfile
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation')
source=root/'snapshots/exact-half-fast-before-hoist-20261013'
names=['exact-half-fast-pack-build-20261013','exact-half-fast-pack-checks-20261013','native-ab-exact-half-fast-pack-20261013','exact-half-fast-text-summary-20261013','native-frame-ab-exact-half-fast-pack-20261013','exact-half-fast-frame-summary-20261013','exact-half-full-codec-checks-20261013','exact-half-fast-op-attribution-20261013','vllm-matched-text-exact-half-fast-20261013','vllm-text-exact-half-fast-summary-20261013','vllm-matched-vl-exact-half-fast-20261013','vllm-frame-exact-half-fast-summary-20261013']
with tarfile.open(root/'exact-half-fast-evidence.tar.gz','w:gz') as archive:
 for name in names:
  folder=root/'logs'/name;assert json.loads((folder/'exit.json').read_text())['exit_code']==0,name
  for path in sorted(folder.rglob('*')):
   if path.is_file() and path.suffix not in ('.f32','.rgb'):archive.add(path,arcname=str(path.relative_to(root/'logs')))
 for name in ['metal_context.mm','metal_context.h','compiled_graph.cpp','exact_half_codec.h']:
  archive.add(source/'src/backends/metal'/name,arcname='fast-candidate-source/'+name)
 archive.add(source/'cmake/GarnetMetal.cmake',arcname='fast-candidate-source/GarnetMetal.cmake')
 for name in ['matmul_test.cpp','profile_selected_operations.py']:
  archive.add(source/'test2026/metal'/name,arcname='fast-candidate-source/'+name)
 archive.add(source/'manifest.json',arcname='fast-candidate-source/source-manifest.json')
 for name in ['build_exact_half_fast_pack.py','queue_exact_half_fast_pack.py','check_exact_half_full_codec.py','queue_exact_half_fast_matched.py','archive_exact_half_fast.py']:
  archive.add(root/name,arcname='fast-candidate-source/'+name)
 build=root/'build-exact-half-prefill-fast-pack'
 info=dict(native_library_sha256=hashlib.sha256((build/'libgarnet.dylib').read_bytes()).hexdigest(),
    cmake_options=[line for line in (build/'CMakeCache.txt').read_text().splitlines() if line.startswith(('GARNET_METAL:BOOL=','GARNET_METAL_XMODEL:BOOL=','GARNET_METAL_EXACT_HALF_PREFILL:BOOL='))])
 path=root/'exact-half-fast-build-metadata.json';path.write_text(json.dumps(info,indent=2)+'\n');archive.add(path,arcname='fast-candidate-source/build-metadata.json')
print('Archived terminal exact-half fast-pack evidence')
