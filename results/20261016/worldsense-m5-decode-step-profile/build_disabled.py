from pathlib import Path
import subprocess,concurrent.futures
w=Path(__file__).resolve().parent;source='/Users/shawn/CantorAI/out/research/worldsense-decode-step-profile/source'
def build(mode):
 b='/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-decode-step-profile-'+mode
 with (w/('build-'+mode+'.log')).open('w')as f:
  subprocess.run(['/opt/homebrew/bin/cmake','-S',source,'-B',b,'-C','/Users/shawn/CantorAI/Garnet/cmake/MetalWorldSense4BPrefix.cmake','-DGARNET_METAL=ON','-DGARNET_METAL_XMODEL=ON','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_OBJCXX_COMPILER=/usr/bin/c++','-DGARNET_METAL_LAZY_RESIDENT_DEFAULTS='+'OFF','-DGARNET_METAL_RESIDENT_WORKSPACE='+'OFF','-DGARNET_METAL_GRAPH_WORKSPACE='+'OFF','-DGARNET_METAL_DENSE_PRIVATE_STORAGE='+'OFF','-DGARNET_METAL_VISION_WORKSPACE='+'OFF','-DGARNET_METAL_VISION_WEIGHT_STREAM='+'OFF','-DGARNET_METAL_PRIVATE_EXECUTION='+'OFF','-DGARNET_METAL_DECODE_STEP_PROFILE=OFF','-DXLANG3_SDK_ROOT=/Users/shawn/CantorAI/xlang3/sdk','-DGARNET_XLANG3_RUNTIME_LIBRARY=/Users/shawn/CantorAI/out/build/macos-arm64-release/bin/libxlang3_runtime.dylib'],stdout=f,stderr=subprocess.STDOUT,check=True)
  subprocess.run(['/opt/homebrew/bin/cmake','--build',b,'-j','2'],stdout=f,stderr=subprocess.STDOUT,check=True)
 print('BUILD PASS',mode,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2)as e:list(e.map(build,['disabled']))
