import hashlib,json,tarfile
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation')
names=['exact-half-decode-build-retry1-20261013','exact-half-decode-checks-retry1-20261013','native-ab-exact-half-decode-retry1-20261013','exact-half-decode-text-summary-retry1-20261013','native-frame-ab-exact-half-decode-retry1-20261013','exact-half-decode-frame-summary-retry1-20261013','restore-kv-control-after-helper-error-20261013','native-ab-half-decode-confirm2-20261013','half-decode-confirm2-text-summary-20261013']
initial=['exact-half-decode-build-20261013','exact-half-decode-checks-20261013','native-ab-exact-half-decode-20261013','exact-half-decode-text-summary-20261013','native-frame-ab-exact-half-decode-20261013','exact-half-decode-frame-summary-20261013','native-ab-exact-half-decode-confirm-20261013','exact-half-decode-confirm-summary-20261013','native-frame-ab-exact-half-decode-confirm-20261013','exact-half-decode-confirm-frame-summary-20261013','native-frame-ab-half-decode-confirm2-20261013','half-decode-confirm2-frame-summary-20261013']
with tarfile.open(root/'half-decode-final-evidence.tar.gz','w:gz') as archive:
 for name in names+initial:
  folder=root/'logs'/name;status=json.loads((folder/'exit.json').read_text());expected=0 if name in names or name=='exact-half-decode-build-20261013' else 1;assert status['exit_code']==expected,name
  for path in sorted(folder.rglob('*')):
   if path.is_file() and path.suffix not in ('.f32','.rgb'):archive.add(path,arcname=str(path.relative_to(root/'logs')))
 for name in ['metal_context.mm','metal_context.h','exact_half_codec.h']:
  archive.add(root/'source/src/backends/metal'/name,arcname='candidate-source/'+name)
 archive.add(root/'source/cmake/GarnetMetal.cmake',arcname='candidate-source/GarnetMetal.cmake')
 for name in ['build_exact_half_decode.py','queue_exact_half_decode_retry1.py','restore_kv_control.py','queue_half_decode_confirmation.py','inspect_external_runtime.py','archive_half_decode_final.py']:
  archive.add(root/name,arcname='candidate-source/'+name)
 build=root/'build-exact-half-decode';info=dict(native_library_sha256=hashlib.sha256((build/'libgarnet.dylib').read_bytes()).hexdigest(),
    cmake_options=[line for line in (build/'CMakeCache.txt').read_text().splitlines() if line.startswith(('GARNET_METAL:BOOL=','GARNET_METAL_XMODEL:BOOL=','GARNET_METAL_EXACT_HALF_PREFILL:BOOL=','GARNET_METAL_EXACT_HALF_DECODE:BOOL='))])
 path=root/'half-decode-final-build-metadata.json';path.write_text(json.dumps(info,indent=2)+'\n');archive.add(path,arcname='candidate-source/build-metadata.json')
print('Archived complete half body-decode evidence and interrupted confirmations')
