import json,subprocess
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation');build=root/'build-split-key-prefill'
commands=[['/opt/homebrew/bin/cmake','-S',str(root/'source'),'-B',str(build),'-DGARNET_METAL=ON','-DGARNET_METAL_XMODEL=ON','-DGARNET_METAL_EXACT_HALF_PREFILL=ON','-DGARNET_METAL_MPS_VISION_SCORES=ON','-DGARNET_METAL_PARALLEL_DELTA_SPLIT=ON','-DGARNET_METAL_FLOAT_DENSE_KEY_PREFILL=ON','-DXLANG3_SDK_ROOT=/Users/zsdx/CantorAI/xlang3/sdk','-DGARNET_XLANG3_RUNTIME_LIBRARY=/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/libxlang3_runtime.dylib','-DCMAKE_BUILD_TYPE=Release'],['/opt/homebrew/bin/cmake','--build',str(build),'--clean-first','--parallel','4']]
for command in commands:
 print('Executing',json.dumps(command),flush=True);subprocess.run(command,check=True)
