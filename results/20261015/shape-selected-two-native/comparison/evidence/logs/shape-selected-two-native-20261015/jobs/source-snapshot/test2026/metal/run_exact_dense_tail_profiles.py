"""Four native exact-tail profiles; discard contention and retain only clean ABBA."""
import ast,hashlib,json,os,subprocess,sys,time
from pathlib import Path
r=Path(sys.argv[1]);output=Path(sys.argv[2]);output.mkdir(parents=True,exist_ok=False)
s=r/'source/test2026/metal';background_checks=[];statuses=[];profiles=[];confirm='--confirm' in sys.argv[3:]
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((s/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'guard','exec'),globals())
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct'):
    for batch in (1,2):
        if confirm and model=='Qwen3-0.6B' and batch==1:continue
        for attempt in range(20):
            quiet_since=None;waits=[];deadline=time.monotonic()+900
            while True:
                now=time.monotonic()
                try:guard()
                except RuntimeError as error:quiet_since=None;waits.append(dict(elapsed_seconds=now,reason=str(error)))
                else:
                    if quiet_since is None:quiet_since=now
                    if now-quiet_since>=5:break
                (output/'quiet-waits.json').write_text(json.dumps(waits,indent=2))
                if now>deadline:raise TimeoutError('No five-second quiet guard interval')
                time.sleep(.5)
            name=model+'-b'+str(batch)+'-attempt'+str(attempt);folder=output/name
            command=[sys.executable,str(s/'compare_exact_dense_tail_native.py'),str(r),str(folder),'--model',model,'--batch',str(batch)]
            print('Starting bounded clean profile',name,flush=True)
            with (output/(name+'.log')).open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=900)
            statuses.append(dict(name=name,command=command,exit_code=result.returncode));(output/'statuses.json').write_text(json.dumps(statuses,indent=2))
            if result.returncode:
                log=(output/(name+'.log')).read_text()
                if 'Another inference runtime is active' not in log:raise RuntimeError('Profile failed for non-contention reason: '+name)
                print('Discarded interrupted profile',name,flush=True);continue
            result=json.loads((folder/'summary.json').read_text())['profiles'];assert len(result)==1
            profiles.append(dict(**result[0],evidence=str(folder),reused_clean_profile=False))
            (output/'summary.json').write_text(json.dumps(dict(scope=__doc__,profiles=profiles,guarded_interval_seconds=.5),indent=2));print('Completed clean profile',name,flush=True);break
        else:raise RuntimeError('Repeated external contention exhausted profile attempts')
assert len(profiles)==(3 if confirm else 4)
