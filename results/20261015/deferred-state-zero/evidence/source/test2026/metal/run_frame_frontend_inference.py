"""Two batch profiles, sequential native/vLLM and frontend-policy ABBA."""
import json,subprocess,sys
from pathlib import Path
r=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]).resolve();output.mkdir(parents=True,exist_ok=False);statuses=[];profiles=[]
for batch in (1,2):
    name='b'+str(batch);folder=output/name;command=[sys.executable,str(r/'source/test2026/metal/compare_frame_frontend_inference.py'),str(r),str(folder),'Qwen3.5-0.8B',str(batch)]
    print('Starting sequential frontend inference',name,flush=True)
    with (output/(name+'.log')).open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    statuses.append(dict(batch=batch,command=command,exit_code=result.returncode));(output/'statuses.json').write_text(json.dumps(statuses,indent=2));assert result.returncode==0,name
    summary=json.loads((folder/'summary.json').read_text());profiles.append(dict(batch=batch,evidence=str(folder),policies=summary['policies'],frontend_effects=summary['frontend_effects']))
    (output/'summary.json').write_text(json.dumps(dict(scope=__doc__,profiles=profiles),indent=2));print('Completed exact frontend inference',name,flush=True)
