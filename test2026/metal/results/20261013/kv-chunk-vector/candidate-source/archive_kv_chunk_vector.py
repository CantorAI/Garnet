import hashlib,json,tarfile
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation')
names=['kv-chunk-vector-build-retry1-20261013','kv-chunk-vector-checks-retry1-20261013','native-ab-kv-chunk-vector-retry1-20261013','kv-chunk-vector-text-summary-retry1-20261013','native-frame-ab-kv-chunk-vector-retry1-20261013','kv-chunk-vector-frame-summary-retry1-20261013','kv-chunk-vector-op-attribution-20261013']
failed=['kv-chunk-vector-build-20261013','kv-chunk-vector-checks-20261013','native-ab-kv-chunk-vector-20261013','kv-chunk-vector-text-summary-20261013','native-frame-ab-kv-chunk-vector-20261013','kv-chunk-vector-frame-summary-20261013']
with tarfile.open(root/'kv-chunk-vector-evidence.tar.gz','w:gz') as archive:
 for name in names+failed:
  folder=root/'logs'/name;status=json.loads((folder/'exit.json').read_text());assert status['exit_code']==(1 if name in failed else 0),name
  for path in sorted(folder.rglob('*')):
   if path.is_file() and path.suffix not in ('.f32','.rgb'):archive.add(path,arcname=str(path.relative_to(root/'logs')))
 for name in ['metal_context.mm','metal_context.h','exact_half_codec.h']:
  archive.add(root/'source/src/backends/metal'/name,arcname='candidate-source/'+name)
 for name in ['matmul_test.cpp','profile_selected_operations.py']:
  archive.add(root/'source/test2026/metal'/name,arcname='candidate-source/'+name)
 for name in ['build_kv_chunk_vector.py','queue_kv_chunk_vector.py','queue_kv_chunk_vector_retry1.py','archive_kv_chunk_vector.py']:
  archive.add(root/name,arcname='candidate-source/'+name)
 build=root/'build-kv-chunk-vector';info=dict(native_library_sha256=hashlib.sha256((build/'libgarnet.dylib').read_bytes()).hexdigest(),
    cmake_options=[line for line in (build/'CMakeCache.txt').read_text().splitlines() if line.startswith(('GARNET_METAL:BOOL=','GARNET_METAL_XMODEL:BOOL=','GARNET_METAL_EXACT_HALF_PREFILL:BOOL='))])
 path=root/'kv-chunk-vector-build-metadata.json';path.write_text(json.dumps(info,indent=2)+'\n');archive.add(path,arcname='candidate-source/build-metadata.json')
print('Archived terminal KV chunk vector evidence')
