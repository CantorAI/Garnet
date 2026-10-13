import hashlib,json,subprocess
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation');source=root/'source/test2026/metal/probe_mps_bf16.mm';output=root/'logs/mps-bf16-compatibility-20261013/jobs';output.mkdir(parents=True,exist_ok=False)
assert json.loads((root/'logs/delta-split-final-archive-20261013/exit.json').read_text())['exit_code']==0
metadata=dict(scope='MPS BF16 compatibility and toy numerical probe only, no model inference or speed claim',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),sdk=subprocess.check_output(['xcrun','--show-sdk-path'],text=True).strip())
binary=root/'mps_bf16_probe';command=['xcrun','clang++','-std=c++17','-fobjc-arc',str(source),'-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders','-o',str(binary)]
metadata['compile_command']=command;(output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
subprocess.run(command,check=True);statuses=[]
for mode in ('fp32-control','mixed','bf16-input','bf16-all'):
 command=[str(binary),mode]
 with (output/(mode+'.log')).open('w') as log:
  result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=120)
 statuses.append(dict(command=command,exit_code=result.returncode));print('Completed compatibility probe',mode,'exit',result.returncode,flush=True)
 (output/'statuses.json').write_text(json.dumps(statuses,indent=2)+'\n')
