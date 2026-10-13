import json,subprocess
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation');build=root/'build-mps-vision-scores-off'
commands=[['/opt/homebrew/bin/cmake','-S',str(root/'source'),'-B',str(build),'-DGARNET_METAL=ON','-DGARNET_METAL_XMODEL=ON','-DGARNET_METAL_EXACT_HALF_PREFILL=ON','-DGARNET_METAL_MPS_VISION_SCORES=OFF','-DXLANG3_SDK_ROOT=/Users/zsdx/CantorAI/xlang3/sdk','-DGARNET_XLANG3_RUNTIME_LIBRARY=/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/libxlang3_runtime.dylib','-DCMAKE_BUILD_TYPE=Release'],['/opt/homebrew/bin/cmake','--build',str(build),'--parallel','4']]
for command in commands:
 print('Executing',json.dumps(command),flush=True);subprocess.run(command,check=True)

subprocess.run(['/opt/homebrew/bin/ctest','--test-dir',str(build),'--output-on-failure'],check=True)
