"""Sequential native ABBA, same frozen text/coder/image workloads and full outputs."""
import ast,hashlib,json,os,signal,statistics,subprocess,sys,time
from pathlib import Path
r=Path(sys.argv[1]);output=Path(sys.argv[2]);output.mkdir(parents=True,exist_ok=False)
s=r/'source/test2026/metal';statuses=[];background_checks=[];records=[];profiles=[]
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((s/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'owned_guard','exec'),globals());strict_guard=guard
raw_snapshot=snapshot;owned_group=None
# Only the process group created by this runner is exempt during monitoring.
def snapshot():
    retained=[];owned=0
    for line in raw_snapshot().splitlines():
        f=line.split()
        if owned_group is not None and f and f[-1].lower().endswith('/xlang3'):
            try:group=os.getpgid(int(f[0]))
            except ProcessLookupError:continue
            if group==owned_group:owned+=1;continue
        retained.append(line)
    if owned>1:raise RuntimeError('Multiple native workers in owned benchmark group')
    return '\n'.join(retained)

def guard(label):
    strict_guard()
    for line in snapshot().splitlines():
        f=line.split()
        if f and f[-1].lower().endswith(('/split_half_prefill_probe','/lossless_block_gpu_probe','/lossless_gpu_probe','/prefill_tile_gpu','/parallel_delta_gpu')):
            raise RuntimeError('Other GPU probe PID '+f[0])
    (output/(label+'-processes.txt')).write_text(snapshot())
builds={key:r/name for key,name in [('control','build-qwen35-extended-prefill'),('candidate','build-resident-kv-inplace')]}
metadata=dict(scope=__doc__,library_sha256={key:hashlib.file_digest((p/'libgarnet.dylib').open('rb'),'sha256').hexdigest() for key,p in builds.items()},source_sha256={p.name:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in [Path(__file__),s/'run_matched_garnet.py',s/'run_matched_frame_process.py',s/'frame_worker_garnet.py']},assets={})
assert metadata['library_sha256']['control']=='cc5539990dcb4effbece7c611787a656df56b51271f1f40e1c3e8b5dbf90c8f8'
runtime='/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3';frontend=str(r/'comparators/vllm/venv/bin/python')
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    vl=model=='Qwen3.5-0.8B';baseline=r/'logs'/('equivalent-f32-qwen35-extended-prefill-20261013' if vl else 'equivalent-f32-finer-prefill-20261013')/'jobs'
    folder=r/'models'/model
    metadata['assets'][model]={p.name:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in [folder/'model.safetensors',folder/'config.json',folder/'tokenizer.json']}
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    for batch in (1,2):
        label=model+'-b'+str(batch);case=json.loads((baseline/(label+'-case.json')).read_text());case['requests']=3
        casefile=output/(label+'-case.json');casefile.write_text(json.dumps(case,indent=2))
        if vl:expected=json.loads((baseline/(label+'-garnet-0/protocol/command.json.event-0-frame_result.json')).read_text())
        else:expected=next(json.loads(v.split(' ',1)[1]) for v in (baseline/(label+'-garnet-0/output.log')).read_text().splitlines() if v.startswith('matched_result '))
        for policy,repeat in [('control',0),('candidate',0),('candidate',1),('control',1)]:
            name=label+'-'+policy+'-'+str(repeat);directory=output/name;directory.mkdir();guard(name+'-before')
            if vl:command=[frontend,str(s/'run_matched_frame_process.py'),runtime,str(builds[policy]),str(r/'source'),str(folder),str(casefile),'garnet',str(directory/'protocol')]
            else:command=[runtime,str(s/'run_matched_garnet.py'),str(builds[policy]),str(r/'source'),str(folder),str(casefile)]
            print('Starting exclusive native worker',name,flush=True)
            samples=[];begin=time.monotonic()
            with (directory/'output.log').open('w') as stream:
                process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);owned_group=process.pid
                try:
                    while process.poll() is None:
                        strict_guard();samples.append(dict(elapsed_seconds=time.monotonic()-begin,guard_passed=True))
                        if time.monotonic()-begin>900:raise TimeoutError('Owned native worker exceeded deadline')
                        time.sleep(.5)
                    code=process.returncode
                except Exception as error:
                    status=dict(name=name,command=command,exit_code=1,interruption=str(error));statuses.append(status)
                    (directory/'exit.json').write_text(json.dumps(status,indent=2));(output/'statuses.json').write_text(json.dumps(statuses,indent=2))
                    if process.poll() is None:
                        assert os.getpgid(process.pid)==process.pid;os.killpg(process.pid,signal.SIGTERM)
                        try:process.wait(timeout=15)
                        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                    raise
                finally:
                    owned_group=None;(directory/'monitor.json').write_text(json.dumps(samples,indent=2))
            status=dict(name=name,command=command,exit_code=code);statuses.append(status)
            (directory/'exit.json').write_text(json.dumps(status,indent=2));(output/'statuses.json').write_text(json.dumps(statuses,indent=2));assert code==0,status
            guard(name+'-after');prefix='matched_frame_result ' if vl else 'matched_result '
            values=[json.loads(v.split(' ',1)[1]) for v in (directory/'output.log').read_text().splitlines() if v.startswith(prefix)];assert len(values)==3
            for value in values:
                assert value['prompt_token_ids']==expected['prompt_token_ids'] and value['token_ids']==expected['token_ids'] and value['finished']==expected['finished']
                assert value['case']==case
                if vl:assert value['vision_reencoded'] and value['patch_sha256']==case['image_metadata']['patch_sha256']
                value.update(model_name=model,build_policy=policy,repeat=repeat);records.append(value)
            print('Completed exact native worker',name,flush=True)
        warm={policy:[v for v in records if v['model_name']==model and v['case']['batch']==batch and v['build_policy']==policy and v['request']>0] for policy in builds}
        means={policy:dict(requests=len(values),ttft_ms=statistics.mean(v['ttft_seconds'] for v in values)*1000,decode_tps=statistics.mean(v['decode_wall_steps_per_second'] for v in values),initialization_s=statistics.mean(v['initialization_seconds'] for v in values)) for policy,values in warm.items()}
        assert all(x['requests']==4 for x in means.values())
        profiles.append(dict(model=model,batch=batch,means=means,decode_ratio=means['candidate']['decode_tps']/means['control']['decode_tps'],ttft_fraction=means['candidate']['ttft_ms']/means['control']['ttft_ms'],exact_complete_outputs=True))
        (output/'summary.json').write_text(json.dumps(dict(scope=metadata['scope'],profiles=profiles,records=records),indent=2));print(json.dumps(profiles[-1]),flush=True)
assert len(statuses)==24 and len(records)==72 and len(profiles)==6
