import json,subprocess
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation');build=root/'build-exact-half-prefill-fast-pack'
commands=[['/opt/homebrew/bin/cmake','--build',str(build),'--target','garnet_metal_matmul_test','--parallel','4'],['/opt/homebrew/bin/ctest','--test-dir',str(build),'--output-on-failure','-V']]
for command in commands:
 print('Executing',json.dumps(command),flush=True);subprocess.run(command,check=True)
