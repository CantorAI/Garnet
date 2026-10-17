from pathlib import Path
import subprocess,concurrent.futures,sys
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-isolated-vector-prefill/source'
def build(mode):
 assert mode in ['off','primary','on','fault'];b=r/('out/build/garnet-metal-worldsense-isolated-vector-prefill-'+mode)
 primary='OFF'if mode=='off'else'ON';vector='ON'if mode in ['on','fault']else'OFF';fault='ON'if mode=='fault'else'OFF'
 with(w/('build-'+mode+'.log')).open('w')as log:
  command=['/opt/homebrew/bin/cmake','-S',str(s),'-B',str(b),'-C',str(w/'build-profile/MetalWorldSense4BPrefix.cmake'),'-DGARNET_METAL=ON','-DGARNET_METAL_XMODEL=ON','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_OBJCXX_COMPILER=/usr/bin/c++','-DGARNET_METAL_PRIMARY_BF16_BLOCKS='+primary,'-DGARNET_METAL_PRIMARY_VECTOR_PREFILL='+vector,'-DGARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY='+vector,'-DGARNET_METAL_PRIMARY_TEST_FAULTS='+fault,'-DGARNET_METAL_LAZY_RESIDENT_DEFAULTS=OFF','-DGARNET_METAL_CACHED_MATRIX_METADATA=OFF','-DXLANG3_SDK_ROOT='+str(r/'xlang3/sdk'),'-DGARNET_XLANG3_RUNTIME_LIBRARY='+str(r/'out/build/macos-arm64-release/bin/libxlang3_runtime.dylib')]
  subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
  subprocess.run(['/opt/homebrew/bin/cmake','--build',str(b),'-j','2'],stdout=log,stderr=subprocess.STDOUT,check=True)
 print('BUILD PASS',mode,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:list(pool.map(build,sys.argv[1:]or['off','on','primary','fault']))
