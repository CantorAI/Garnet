"""Serialize repeat comparisons against a separately built previous commit.

Arguments: runtime build source control_build control_source output_folder.
Both source checkouts use the same existing models and image inputs. Each
native process performs two reset requests; report request 1 as warm latency.
"""
import hashlib,json,subprocess,sys
from pathlib import Path
runtime,build,source,control_build,control_source,output=sys.argv[1:]
source=Path(source);control_source=Path(control_source);output=Path(output);root=source.parent
output.mkdir(parents=True,exist_ok=True)
models=('Qwen3.5-0.8B','Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct')
metadata=dict(control_commit='f35f58f',runtime=runtime,timing_clock='time.monotonic',
    control_patch='Timing-only replacement of time.time() with time.monotonic(); model/native code unchanged',
    builds={name:dict(build=directory,source=str(checkout),native_sha256=hashlib.sha256((Path(directory)/'libgarnet.dylib').read_bytes()).hexdigest(),
        runner_sha256={n:hashlib.sha256((checkout/'test2026/metal'/n).read_bytes()).hexdigest() for n in ('run_image_decoder.py','run_dense_text.py')})
        for name,directory,checkout in [('control',control_build,control_source),('optimized',build,source)]})
(output/'metadata.json').write_bytes((json.dumps(metadata,indent=2)+'\n').encode())
tasks=[('control-ctest',['/opt/homebrew/bin/ctest','--test-dir',control_build,'--output-on-failure'])]
for batch in (1,2):
 for name in models:
  for suffix,checkout,directory,mode in [('control',control_source,control_build,2),('optimized',source,build,3),('control-repeat',control_source,control_build,2),('repeat',source,build,3)]:
   scripts=checkout/'test2026/metal';model=root/'models'/name
   if name==models[0]:
    command=[sys.executable,str(scripts/'benchmark_image.py'),runtime,directory,str(checkout),str(model),str(root/'images'/('RealWorld-04.batch'+str(batch)+'.json')),'32','16',str(mode)]
   else:
    prompt='What is the capital of France? Answer briefly.' if name==models[1] else 'Write a Python function add(a, b) that returns their sum. Output only code.'
    command=[sys.executable,str(scripts/'benchmark_dense_text.py'),runtime,directory,str(checkout),str(model),str(batch),'64',prompt,'16',str(mode)]
   tasks.append((name+'-b'+str(batch)+'-'+suffix,command))
statuses=[]
for name,command in tasks:
 folder=output/name;folder.mkdir(parents=True,exist_ok=True)
 print('Starting isolated comparison',name,flush=True)
 with (folder/'output.log').open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
 status=dict(name=name,command=command,exit_code=result.returncode);statuses.append(status)
 (folder/'exit.json').write_bytes((json.dumps(status,indent=2)+'\n').encode())
 (output/'job_statuses.json').write_bytes((json.dumps(statuses,indent=2)+'\n').encode())
 print('Completed',name,'exit',result.returncode,flush=True)
 if result.returncode:raise SystemExit(result.returncode)
