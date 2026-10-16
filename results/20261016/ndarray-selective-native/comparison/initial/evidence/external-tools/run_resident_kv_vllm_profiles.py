"""Fresh loaded-dtype audits and clean matched profiles; sequential GPU use."""
import ast,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
r=Path(sys.argv[1]);output=Path(sys.argv[2]);output.mkdir(parents=True,exist_ok=False)
s=r/'source/test2026/metal';background_checks=[];statuses=[];profiles=[];audits=[]
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((s/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'guard','exec'),globals())
py=str(r/'comparators/vllm/venv/bin/python')
def quiet(label):
    clear=None;waits=[];start=time.monotonic()
    while True:
        now=time.monotonic()
        try:guard()
        except RuntimeError as e:clear=None;waits.append(dict(elapsed_seconds=now-start,reason=str(e)))
        else:
            if clear is None:clear=now
            if now-clear>=5:break
        (output/(label+'-wait.json')).write_text(json.dumps(waits,indent=2))
        if now-start>900:raise TimeoutError('No five-second clear inference guard interval')
        time.sleep(.5)
def audit(model):
    for attempt in range(20):
        name=model+'-dtype-'+str(attempt);quiet(name);folder=output/name;folder.mkdir()
        command=[py,str(s/'audit_comparator_dtypes.py'),str(r/'models-exact-f32'/model),'float32','.45' if model=='Qwen3.5-0.8B' else '.30']
        print('Starting actual dtype audit',name,flush=True);start=time.monotonic();samples=[];interrupted=None
        with (folder/'output.log').open('w') as stream:
            process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                while process.poll() is None:
                    guard();samples.append(dict(elapsed_seconds=time.monotonic()-start,guard_passed=True))
                    if time.monotonic()-start>900:raise TimeoutError('Dtype audit exceeded deadline')
                    time.sleep(.5)
            except RuntimeError as e:interrupted=str(e)
            finally:
                if process.poll() is None:
                    assert os.getpgid(process.pid)==process.pid;os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        status=dict(name=name,command=command,exit_code=process.returncode,interruption=interrupted)
        (folder/'exit.json').write_text(json.dumps(status,indent=2));(folder/'monitor.json').write_text(json.dumps(samples,indent=2));statuses.append(status)
        (output/'statuses.json').write_text(json.dumps(statuses,indent=2))
        if interrupted:print('Discarded contended dtype audit',name,flush=True);continue
        assert process.returncode==0,status;guard()
        values=[json.loads(v.split(' ',1)[1]) for v in (folder/'output.log').read_text().splitlines() if v.startswith('actual_dtype_audit ')];assert len(values)==1
        worker=values[0]['workers'][0]
        assert set(worker['actual_parameter_tensors'])=={'mlx.core.float32'} and worker['runner_kv_dtype']=='mlx.core.float32'
        assert worker['activation_dtype_probes'] and set(worker['activation_dtype_probes'].values())=={'mlx.core.float32'}
        audits.append(dict(model=model,evidence=str(folder),report=values[0]));(output/'dtype-summary.json').write_text(json.dumps(audits,indent=2));print('Completed exact-F32 dtype audit',model,flush=True);return
    raise RuntimeError('External contention exhausted dtype audit attempts')
versions=[py,'-c','import importlib.metadata as m,platform,json;print(json.dumps(dict(platform=platform.platform(),versions={n:m.version(n) for n in ("vllm","vllm-metal","mlx","mlx-lm","mlx-vlm","torch","transformers","numpy","Pillow")})))']
(output/'versions.json').write_text(subprocess.check_output(versions,text=True))
initial=r/'logs/resident-kv-vllm-text-b1-20261013'
assert json.loads((initial/'exit.json').read_text())['exit_code']==0
profile=json.loads((initial/'jobs/summary.json').read_text());assert profile['model']=='Qwen3-0.6B' and profile['batch']==1
profiles.append(dict(model=profile['model'],batch=profile['batch'],evidence=str(initial/'jobs'),summary={k:v for k,v in profile.items() if k!='records'}))
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):audit(model)
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    for batch in (1,2):
        if model=='Qwen3-0.6B' and batch==1:continue
        for attempt in range(20):
            name=model+'-b'+str(batch)+'-'+str(attempt);quiet(name);folder=output/name
            command=[sys.executable,str(s/'compare_resident_kv_vllm.py'),str(r),str(folder),model,str(batch)]
            print('Starting sequential matched profile',name,flush=True)
            with (output/(name+'.log')).open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=1800)
            statuses.append(dict(name=name,command=command,exit_code=result.returncode));(output/'statuses.json').write_text(json.dumps(statuses,indent=2))
            if result.returncode:
                text=(output/(name+'.log')).read_text()
                if 'Another inference runtime is active' not in text:raise RuntimeError('Non-contention comparator failure: '+name)
                print('Discarded interrupted comparison',name,flush=True);continue
            value=json.loads((folder/'summary.json').read_text());assert value['exact_complete_outputs']
            profiles.append(dict(model=model,batch=batch,evidence=str(folder),summary={k:v for k,v in value.items() if k!='records'}))
            (output/'summary.json').write_text(json.dumps(dict(profiles=profiles,dtype_audits=audits,counts_as_default_bf16_speedup=False,
                all_extra_f32_targets_met=all(v['summary']['decode_target_met'] and v['summary']['ttft_target_met'] for v in profiles)),indent=2))
            print('Completed clean matched profile',name,flush=True);break
        else:raise RuntimeError('External contention exhausted comparison attempts')
assert len(profiles)==6 and len(audits)==3
