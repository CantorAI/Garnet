import hashlib,json,shutil,subprocess
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation')
shutil.copyfile(root/'kv-control-context.mm',root/'source/src/backends/metal/metal_context.mm')
shutil.copyfile(root/'kv-control-cmake.cmake',root/'source/cmake/GarnetMetal.cmake')
subprocess.run(['/opt/homebrew/bin/python3',str(root/'build_kv_chunk_vector.py')],check=True)
actual=hashlib.sha256((root/'build-kv-chunk-vector/libgarnet.dylib').read_bytes()).hexdigest()
expected='564b515ab6b0eb68413ef5a54de6beff47d333f8f1fa6a1188562bf6e5c28ec4'
print('Restored control library SHA256',actual,flush=True)
assert actual==expected
