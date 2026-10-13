import json,os,subprocess,time
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation')
logs=root/'logs/op-profile-20261012';logs.mkdir(parents=True,exist_ok=True)
prior=root/'logs/vllm-matched-mps-prefill-20261012/exit.json'
while not prior.exists():time.sleep(5)
assert json.loads(prior.read_text())['exit_code']==0
build=root/'build-op-profile'
commands=[['/opt/homebrew/bin/cmake','-S',str(root/'source'),'-B',str(build),'-DGARNET_METAL=ON','-DGARNET_METAL_XMODEL=ON','-DXLANG3_SDK_ROOT=/Users/zsdx/CantorAI/xlang3/sdk','-DGARNET_XLANG3_RUNTIME_LIBRARY=/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/libxlang3_runtime.dylib','-DCMAKE_BUILD_TYPE=Release'],['/opt/homebrew/bin/cmake','--build',str(build),'--parallel','4'],['/opt/homebrew/bin/ctest','--test-dir',str(build),'--output-on-failure']]
for command in commands:
 print('Executing',json.dumps(command),flush=True)
 subprocess.run(command,check=True)
base=root/'logs/vllm-matched-visible-cache-20261012/jobs'
case=json.loads((base/'Qwen3-0.6B-b1-case.json').read_text());case.update(requests=1,prefill_projection='shared')
casefile=logs/'case.json';casefile.write_text(json.dumps(case,indent=2)+'\n')
command=['/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3',str(root/'source/test2026/metal/run_matched_garnet.py'),str(build),str(root/'source'),str(root/'models/Qwen3-0.6B'),str(casefile)]
print('Profiling command',json.dumps(command),'with GARNET_METAL_PROFILE_OPS=1; diagnostic timings excluded from speed claims',flush=True)
with (logs/'operations.log').open('w') as stream:
 subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env={**os.environ,'GARNET_METAL_PROFILE_OPS':'1'},check=True)
records=[json.loads(line.split(' ',1)[1]) for line in (logs/'operations.log').read_text().splitlines() if line.startswith('matched_result ')]
assert len(records)==1
expected=json.loads((base/'summary.json').read_text())['records'][0]['normalized_token_ids']
assert records[0]['token_ids']==expected,(records[0]['token_ids'],expected)
print('Serialized profiling retained exact native output tokens',flush=True)
