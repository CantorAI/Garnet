import json,subprocess
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation');build=root/'build-body-codec'
assert json.loads((root/'logs/decoder-codec-gpu-20261013/exit.json').read_text())['exit_code']==0
commands=[['/opt/homebrew/bin/cmake','-S',str(root/'source'),'-B',str(build),'-DGARNET_METAL=ON','-DGARNET_METAL_XMODEL=ON','-DGARNET_METAL_LOSSLESS_DECODE=ON','-DXLANG3_SDK_ROOT=/Users/zsdx/CantorAI/xlang3/sdk','-DGARNET_XLANG3_RUNTIME_LIBRARY=/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/libxlang3_runtime.dylib','-DCMAKE_BUILD_TYPE=Release'],['/opt/homebrew/bin/cmake','--build',str(build),'--parallel','4']]
for command in commands:print('Executing',json.dumps(command),flush=True);subprocess.run(command,check=True)
