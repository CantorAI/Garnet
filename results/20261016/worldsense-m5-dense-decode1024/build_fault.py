from pathlib import Path
import subprocess
w=Path(__file__).resolve().parent;s='/Users/shawn/CantorAI/out/research/worldsense-dense-decode1024/fault-source';b='/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-dense-decode1024-fault'
with (w/'build-fault.log').open('w')as f:
 subprocess.run(['/opt/homebrew/bin/cmake','-S',s,'-B',b,'-C','/Users/shawn/CantorAI/Garnet/cmake/MetalWorldSense4BPrefix.cmake','-DGARNET_METAL=ON','-DGARNET_METAL_XMODEL=ON','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_OBJCXX_COMPILER=/usr/bin/c++','-DGARNET_METAL_LAZY_RESIDENT_DEFAULTS=ON','-DGARNET_METAL_RESIDENT_WORKSPACE=ON','-DGARNET_METAL_GRAPH_WORKSPACE=ON','-DGARNET_METAL_DENSE_PRIVATE_STORAGE=ON','-DGARNET_METAL_VISION_WORKSPACE=ON','-DGARNET_METAL_VISION_WEIGHT_STREAM=ON','-DGARNET_METAL_PRIVATE_EXECUTION=ON','-DGARNET_METAL_DENSE_DECODE256=OFF','-DGARNET_METAL_DENSE_DECODE512=OFF','-DGARNET_METAL_DENSE_DECODE1024=ON','-DGARNET_METAL_LAZY_DEFAULTS_TEST_FAULTS=ON','-DGARNET_METAL_GRAPH_WORKSPACE_TEST_FAULTS=ON','-DXLANG3_SDK_ROOT=/Users/shawn/CantorAI/xlang3/sdk','-DGARNET_XLANG3_RUNTIME_LIBRARY=/Users/shawn/CantorAI/out/build/macos-arm64-release/bin/libxlang3_runtime.dylib'],stdout=f,stderr=subprocess.STDOUT,check=True)
 subprocess.run(['/opt/homebrew/bin/cmake','--build',b,'-j','2'],stdout=f,stderr=subprocess.STDOUT,check=True)
print('Separate fault build PASS',flush=True)
