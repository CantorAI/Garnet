from pathlib import Path
import subprocess,concurrent.futures,sys
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-direct-interleaved-prefill/source'
def build(mode):
 assert mode in ['off','primary','on','fault','restore'];b=r/('out/build/garnet-metal-worldsense-direct-interleaved-prefill-'+mode)
 primary='OFF'if mode=='off'else'ON';vector='ON'if mode in ['on','fault','restore']else'OFF';fault='ON'if mode=='fault'else'OFF'
 with(w/('build-'+mode+'.log')).open('w')as log:
  command=['/opt/homebrew/bin/cmake','-S',str(s),'-B',str(b),'-C',str(w/'build-profile/MetalWorldSense4BPrefix.cmake'),'-DGARNET_METAL=ON','-DGARNET_METAL_XMODEL=ON','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_OBJCXX_COMPILER=/usr/bin/c++','-DGARNET_METAL_PRIMARY_BF16_BLOCKS='+primary,'-DGARNET_METAL_WIDE_PRIMARY_STORAGE='+primary,'-DGARNET_METAL_PRIMARY_VECTOR_PREFILL='+vector,'-DGARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY='+vector,'-DGARNET_METAL_DIRECT_FLOAT_WEIGHT_UPLOAD='+vector,'-DGARNET_METAL_PRIMARY_VECTOR_PREFILL_K32_ONLY='+vector,'-DGARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY='+vector,'-DGARNET_METAL_PRIMARY_TEST_FAULTS='+fault,'-DGARNET_METAL_SOLE_INTERLEAVED_MLP='+('ON'if mode in ['on','fault','restore']else'OFF'),'-DGARNET_METAL_SOLE_INTERLEAVED_DIRECT_PREFILL='+('ON'if mode in ['on','fault']else'OFF'),'-DGARNET_METAL_LAZY_RESIDENT_DEFAULTS='+primary,'-DGARNET_METAL_CACHED_MATRIX_METADATA='+vector,'-DXLANG3_SDK_ROOT='+str(r/'xlang3/sdk'),'-DGARNET_XLANG3_RUNTIME_LIBRARY='+str(r/'out/build/macos-arm64-release/bin/libxlang3_runtime.dylib')]
  subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
  subprocess.run(['/opt/homebrew/bin/cmake','--build',str(b),'-j','2'],stdout=log,stderr=subprocess.STDOUT,check=True)
 print('BUILD PASS',mode,flush=True)
for mode in sys.argv[1:]or['off','primary','on','fault','restore']:build(mode)
