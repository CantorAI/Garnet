import hashlib,json,subprocess
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation');source=root/'source/test2026/metal/probe_mps_exact_half.mm';output=root/'logs/mps-exact-half-compatibility-20261013/jobs';output.mkdir(parents=True,exist_ok=False)
assert json.loads((root/'logs/exact-half-weight-survey-20261013/exit.json').read_text())['exit_code']==0
binary=root/'mps_exact_half_probe';command=['xcrun','clang++','-std=c++17','-fobjc-arc',str(source),'-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders','-o',str(binary)]
metadata=dict(scope='Toy MPS mixed FP32 activations/output with exact scaled FP16 weight storage; no model inference or speed claim',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),compile_command=command)
(output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');subprocess.run(command,check=True)
command=[str(binary)]
with (output/'probe.log').open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=120)
(output/'status.json').write_text(json.dumps(dict(command=command,exit_code=result.returncode),indent=2)+'\n')
assert result.returncode==0
print('Exact half GPU compatibility probe completed',flush=True)
